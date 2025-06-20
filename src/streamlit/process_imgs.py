"""Utility functions for running orientation, segmentation and disease models
on chest‑X‑ray images and returning ready‑to‑display Pandas DataFrames.

The module can be imported by Streamlit pages or plain Python scripts.
It contains *no* Streamlit‑specific code – that belongs in the UI layer.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard‑library imports
# ---------------------------------------------------------------------------
import base64
import os
import sys
from io import BytesIO
from pathlib import Path
from typing import Sequence
import cv2  # type: ignore[import]

# ---------------------------------------------------------------------------
# Third‑party imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Project‑level imports – path bootstrapping first
# ---------------------------------------------------------------------------
from src.defs import (  # noqa: E402 – after sys.path tweak
    PROJECT_DIR,
    ModelType as mt,
    class_to_disease_map,
    class_to_orientation_map,
    initDataPaths,
)
from src.models.modelbuilder import capsnet_custom_objects  # noqa: E402
from src.utils.datahelper import DataHelper as dh  # noqa: E402
from src.utils.imgprocessing import ImageProcessor as ip  # noqa: E402

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def image_obj_to_base64_html(
    img: Image.Image,
    max_width: int = 60,
    img_format: str | None = None,
) -> str:
    """Convert *img* to a base‑64 inline ``<img>`` tag suitable for AG Grid.

    Parameters
    ----------
    img : PIL.Image.Image
        The already‑loaded image.
    max_width : int, default 60
        Longest edge of the thumbnail in pixels.
    img_format : str | None, default *None*
        Output format (``"PNG"``, ``"JPEG"`` …). If *None* the function uses
        ``img.format`` or falls back to ``"PNG"``.
    """
    fmt = (img_format or img.format or "PNG").upper()

    thumb = img.copy()
    thumb.thumbnail((max_width, max_width))

    # JPEG cannot store an alpha channel – convert if necessary
    if thumb.mode == "RGBA" and fmt == "JPEG":
        thumb = thumb.convert("RGB")

    buf = BytesIO()
    thumb.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()

    return f'<img src="data:image/{fmt.lower()};base64,{b64}" width="{max_width}"/>'


def image_path_to_base64_html(path: str | os.PathLike[str], max_width: int = 60) -> str:
    """Lightweight wrapper that loads *path* and calls :func:`image_obj_to_base64_html`."""
    img = Image.open(path)
    return image_obj_to_base64_html(img, max_width=max_width)


# ---------------------------------------------------------------------------
# TensorFlow dataset helpers
# ---------------------------------------------------------------------------

def _process_image(path: tf.Tensor, img_size: tuple[int, int], normalize: bool) -> tf.Tensor:
    """Read *path* → resize → (optionally) normalise to [0,1]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    return img / 255.0 if normalize else img


def _get_file_paths(source_dir: str | os.PathLike[str], file_names: Sequence[str]) -> list[str]:
    """Return existing *file_names* under *source_dir* (keeps original order)."""
    return [str(Path(source_dir, fn)) for fn in file_names if Path(source_dir, fn).exists()]


def unlabeled_dataset(
    *,
    file_names: Sequence[str],
    dataset_dir: str | os.PathLike[str],
    target_size: tuple[int, int] = (224, 224),
    normalize: bool = False,
) -> tf.data.Dataset:
    """Create a ready‑to‑predict *tf.data.Dataset* for the selected images."""
    paths = _get_file_paths(dataset_dir, file_names)
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(
        lambda p: _process_image(p, target_size, normalize),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return ds.batch(32).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def process_images(
    *,
    dataset_dir: str | os.PathLike[str],
    df_selected_rows: pd.DataFrame,
    selected_models: dict[str, mt | None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run orientation, segmentation and disease models over user‑selected images.

    Parameters
    ----------
    dataset_dir : str | os.PathLike[str]
        Directory containing the original images.
    df_selected_rows : pd.DataFrame
        DataFrame coming from AG Grid – **must** contain a ``"Filename"`` column.
    selected_models : dict, optional
        Keys ``"orientation"``, ``"segmentation"``, ``"desease_classifier"`` (sic).

    Returns
    -------
    df_orientation : pd.DataFrame
        Per‑image orientation confidences (one column per class + *Filename*).
    df_masked : pd.DataFrame
        Thumbnails of lung‑segmented images, ready for AG Grid.
    df_disease : pd.DataFrame
        Per‑image disease confidences (one column per class + *Filename*).
    """
    selected_models = selected_models or {}

    # ---------------------------------------------------------------------
    # 1. Orientation
    # ---------------------------------------------------------------------
    selected_image_names = df_selected_rows["Filename"].tolist()

    ds = unlabeled_dataset(
        file_names=selected_image_names,
        dataset_dir=dataset_dir,
        target_size=(224, 224),
        normalize=False,
    )

    preprocess_fn = dh.get_preprocess_fn(selected_models.get("orientation"))
    ds = ds.map(preprocess_fn)

    mobnet_path = Path(PROJECT_DIR, "models", "orientation_classifier_224x224_aug_head1_mobnet.keras")
    resnet_path = Path(PROJECT_DIR, "models", "orientation_classifier_224x224_aug_head1_resnet50.keras")

    orientation_model = load_model(
        mobnet_path if selected_models.get("orientation") == mt.MOBILENET else resnet_path,
        compile=False,
    )

    ori_confs = orientation_model.predict(ds, verbose=0)
    ori_cols = [class_to_orientation_map["long"][i] for i in sorted(class_to_orientation_map["long"])]

    df_orientation = (
        pd.DataFrame(np.round(ori_confs, 2), columns=ori_cols)
        .assign(Filename=selected_image_names)
        .loc[:, ["Filename", *ori_cols]]
    )

    # ---------------------------------------------------------------------
    # 2. Segmentation & masking
    # ---------------------------------------------------------------------
    unet_path = Path(PROJECT_DIR, "models", "lung_segmentation_unet.h5")
    gan_path = Path(PROJECT_DIR, "models", "lung_segmentation_gan.h5")

    seg_model_path = unet_path if selected_models.get("segmentation") == mt.UNET else gan_path
    seg_model = load_model(seg_model_path, compile=False)

    masked_imgs, names = ip.generate_masked_images(
        from_dir=dataset_dir,
        model=seg_model,
        ori_confs=ori_confs,
        ori_cols=ori_cols,
        select_imgs=selected_image_names,
        target_size=(256, 256),
        
    )

    masked_pil = [ip.from_np_to_pil(img, make_rgb=True) for img in masked_imgs]

    df_masked = pd.DataFrame(
        {
            "Filename": names,
            "Preview": [image_obj_to_base64_html(img, max_width=200) for img in masked_pil],
        }
    )

    # ---------------------------------------------------------------------
    # 3. Disease classification
    # ---------------------------------------------------------------------
    cnn_path = Path(PROJECT_DIR, "models", "ds_crx_covid19.keras")
    capsnet_path = Path(PROJECT_DIR, "models", "capsnet_lung_disease_classifier_krnl_5.keras")

    def gray_to_rgb(x: tf.Tensor) -> tf.Tensor:
        print(x.shape)
        x = tf.expand_dims(x, -1) # Add channel dimension if missing
        print(x.shape)
        return tf.image.grayscale_to_rgb(x)

    ds_dis = tf.data.Dataset.from_tensor_slices(np.array(masked_imgs))
    if selected_models.get("desease_classifier") == mt.CAPSNET:
        ds_dis = ds_dis.map(gray_to_rgb, num_parallel_calls=tf.data.AUTOTUNE)

    ds_dis = ds_dis.batch(32).prefetch(tf.data.AUTOTUNE)

    disease_model = load_model(
        capsnet_path if selected_models.get("desease_classifier") == mt.CAPSNET else cnn_path,
        custom_objects=capsnet_custom_objects if selected_models.get("desease_classifier") == mt.CAPSNET else None,
        compile=False,
    )

    dis_confs = disease_model.predict(ds_dis, verbose=0)
    
    if selected_models.get("desease_classifier") == mt.CAPSNET:
        dis_confs = dis_confs / np.sum(dis_confs, axis=1, keepdims=True)
        # dis_confs: (n_images, n_classes, 1)  ➜  (n_images, n_classes)
        dis_confs = dis_confs.squeeze(-1)

    dis_cols = [class_to_disease_map[i] for i in sorted(class_to_disease_map)]

    
    
    df_disease = (
        pd.DataFrame(np.round(dis_confs, 2), columns=dis_cols)
        .assign(Filename=selected_image_names)
        .loc[:, ["Filename", *dis_cols]]
    )

    return df_orientation, df_masked, df_disease
