"""Utility functions for running orientation, segmentation and disease models
on chest‑X‑ray images and returning ready‑to‑display Pandas DataFrames.

This module is UI‑agnostic (no Streamlit code). Use from Streamlit pages or
plain Python scripts.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library imports
# ---------------------------------------------------------------------------
import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping, Sequence

# ---------------------------------------------------------------------------
# Third‑party imports
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model

# ---------------------------------------------------------------------------
# Project‑level imports
# ---------------------------------------------------------------------------
from src.defs import (
    PROJECT_DIR,
    ModelType as mt,
    class_to_disease_map,
    class_to_orientation_map,
)
from src.models.modelbuilder import capsnet_custom_objects
from src.utils.datahelper import DataHelper as dh
from src.utils.imgprocessing import ImageProcessor as ip

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BATCH_SIZE = 32
ORI_SIZE = (224, 224)
SEG_SIZE = (256, 256)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def image_obj_to_base64_html(
    img: Image.Image,
    max_width: int = 60,
    img_format: str | None = None,
) -> str:
    """Convert *img* to a base‑64 inline ``<img>`` tag suitable for AG Grid."""
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


def image_path_to_base64_html(path: str | Path, max_width: int = 60) -> str:
    img = Image.open(path)
    return image_obj_to_base64_html(img, max_width=max_width)


def _process_image(path: tf.Tensor, img_size: tuple[int, int], normalize: bool) -> tf.Tensor:
    """Read *path* → resize → (optionally) normalise to [0,1]."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, img_size)
    return img / 255.0 if normalize else img


def _existing_paths(base: str | Path, file_names: Sequence[str]) -> list[str]:
    """Return only files that actually exist under *base* (preserve order)."""
    base = Path(base)
    return [str(base / fn) for fn in file_names if (base / fn).is_file()]


def _dataset_for_files(paths: Iterable[str], *, size: tuple[int, int], normalize: bool = False) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices(list(paths))
    ds = ds.map(lambda p: _process_image(p, size, normalize), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds


def _as_text_labels(mapping: Mapping[int, str]) -> list[str]:
    """Return column labels sorted by numeric class id."""
    return [mapping[i] for i in sorted(mapping)]


def _normalize_selection(value) -> str:
    """Robustly normalize enum/str selections to lowercase strings."""
    try:
        # Enum -> name
        name = value.name
    except AttributeError:
        name = value
    return (str(name) if name is not None else "").strip().lower()


def _choose_model_path(choice: str, *, default: Path, options: dict[str, Path]) -> Path:
    return options.get(choice, default)


@lru_cache(maxsize=8)
def _load_keras_model_cached(path_str: str, *, use_capsnet: bool = False):
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")
    return load_model(
        path,
        custom_objects=capsnet_custom_objects if use_capsnet else None,
        compile=False,
    )


def process_images(
    *,
    dataset_dir: str | Path,
    df_selected_rows: pd.DataFrame,
    selected_models: dict[str, mt | str | None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run orientation, segmentation and disease models on selected images.

    Parameters
    ----------
    dataset_dir : str | Path
        Directory containing the original images.
    df_selected_rows : pd.DataFrame
        Must contain a ``"Filename"`` column.
    selected_models : dict, optional
        Keys: "orientation", "segmentation", "disease_classifier" (backward‑
        compatible with misspelled "desease_classifier"). Values can be an
        enum (mt.*) or strings like "mobilenet", "resnet", "unet", "gan",
        "capsnet", "cnn".
    """
    selected_models = selected_models or {}

    # accept both spellings
    disease_classifier = "disease_classifier" if "disease_classifier" in selected_models else "desease_classifier"

    if "Filename" not in df_selected_rows.columns:
        raise KeyError('df_selected_rows must include a "Filename" column')

    dataset_dir = Path(dataset_dir)
    selected_image_names = df_selected_rows["Filename"].tolist()

    # ------------------------------------------------------------------
    # 1) Orientation
    # ------------------------------------------------------------------
    orient_choice = _normalize_selection(selected_models.get("orientation"))

    paths = _existing_paths(dataset_dir, selected_image_names)
    ds = _dataset_for_files(paths, size=ORI_SIZE, normalize=False)

    preprocess_fn = dh.get_preprocess_fn(selected_models.get("orientation"))
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)

    mobnet_path = Path(PROJECT_DIR, "models", "orientation-classifier-224x224-aug-head1-mobnet", "model.keras")
    resnet_path = Path(PROJECT_DIR, "models", "orientation-classifier-224x224-aug-head1-resnet50", "model.keras")

    ori_model_path = _choose_model_path(
        orient_choice,
        default=resnet_path,
        options={"mobilenet": mobnet_path, "resnet": resnet_path},
    )

    orientation_model = _load_keras_model_cached(str(ori_model_path))
    ori_confs = orientation_model.predict(ds, verbose=0)

    ori_cols = _as_text_labels(class_to_orientation_map["long"])  # type: ignore[index]
    df_orientation = (
        pd.DataFrame(np.round(ori_confs, 2), columns=ori_cols)
        .assign(Filename=selected_image_names)
        .loc[:, ["Filename", *ori_cols]]
    )

    # ------------------------------------------------------------------
    # 2) Segmentation & masking
    # ------------------------------------------------------------------
    seg_choice = _normalize_selection(selected_models.get("segmentation"))
    unet_path = Path(PROJECT_DIR, "models", "lung-segmentation-unet", "model.keras")
    gan_path = Path(PROJECT_DIR, "models", "lung-segmentation-gan", "model.keras")
    seg_model_path = _choose_model_path(
        seg_choice,
        default=gan_path,
        options={"unet": unet_path, "gan": gan_path},
        )
    try:
        seg_model = load_model(seg_model_path, compile=False)
    except Exception as e:
        print(f"Loading segmentation model failed: {e}")

    masked_imgs, names = ip.generate_masked_images(
        from_dir=dataset_dir,
        model=seg_model,
        ori_confs=ori_confs,
        ori_cols=ori_cols,
        select_imgs=selected_image_names,
        target_size=SEG_SIZE,
    )

    masked_pil = [ip.from_np_to_pil(img, make_rgb=True) for img in masked_imgs]
    df_masked = pd.DataFrame(
        {
            "Filename": names,
            "Preview": [image_obj_to_base64_html(img, max_width=200) for img in masked_pil],
        }
    )

    # ------------------------------------------------------------------
    # 3) Disease classification
    # ------------------------------------------------------------------
    disease_choice = _normalize_selection(selected_models.get(disease_classifier))
    cnn_path = Path(PROJECT_DIR, "models", "ds-cxr-covid19","CNN", "data", "model.keras")
    capsnet_path = Path(PROJECT_DIR, "models", "capsnet-4class-lung-disease-classifier", "model.keras")

    ds_dis = tf.data.Dataset.from_tensor_slices(np.asarray(masked_imgs))

    if disease_choice == "capsnet":
        # Ensure 3 channels for CapsNet if upstream produced grayscale
        def gray_to_rgb(x: tf.Tensor) -> tf.Tensor:
            x = tf.convert_to_tensor(x)
            x = tf.cond(tf.equal(tf.rank(x), 2), lambda: tf.expand_dims(x, -1), lambda: x)
            return tf.image.grayscale_to_rgb(x)

        ds_dis = ds_dis.map(gray_to_rgb, num_parallel_calls=tf.data.AUTOTUNE)

    ds_dis = ds_dis.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    disease_model_path = _choose_model_path(
        disease_choice,
        default=cnn_path,
        options={"capsnet": capsnet_path, "cnn": cnn_path},
    )

    disease_model = _load_keras_model_cached(
        str(disease_model_path), use_capsnet=(disease_choice == "capsnet")
    )

    dis_confs = disease_model.predict(ds_dis, verbose=0)

    if disease_choice == "capsnet":
        dis_confs = dis_confs / np.sum(dis_confs, axis=1, keepdims=True)
        dis_confs = dis_confs.squeeze(-1)  # (n, n_classes, 1) -> (n, n_classes)

    dis_cols = _as_text_labels(class_to_disease_map)
    df_disease = (
        pd.DataFrame(np.round(dis_confs, 2), columns=dis_cols)
        .assign(Filename=selected_image_names)
        .loc[:, ["Filename", *dis_cols]]
    )

    return df_orientation, df_masked, df_disease
