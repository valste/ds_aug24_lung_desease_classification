# -*- coding: utf-8 -*-
import os

import pandas as pd
from skimage.feature.texture import graycomatrix, graycoprops

from .image_preprocessor import crop_image


def evaluate_image_texture(filename, image, distance=0, angle=0) -> list:
    """
    evaluate_image_texture function evaluates the
    texture of an image.

    Input:
    filename: str: Name of the image file
    image: np.array: Image as a numpy array
    distance: int: Distance between pixels
    angle: int: Angle between pixels

    Output:
    pd.DataFrame: Image texture metrics
    """

    # Quantize image
    quantized_image = (image / 32).astype("uint8")

    # Compute GLCM
    glcm = graycomatrix(
        quantized_image,
        distances=[distance],
        angles=[angle],
        levels=8,
        symmetric=True,
        normed=True,
    )

    # Extract GLCM features
    contrast = graycoprops(glcm, "contrast")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    entropy = graycoprops(glcm, "entropy")[0, 0]
    return [
        filename,
        energy,
        entropy,
        homogeneity,
        contrast,
        correlation,
    ]


def get_images_texture(images_dir, margin_percentage=0, distance=0, angle=0) -> pd.DataFrame:
    """
    get_images_texture function evaluates the
    texture of an image.

    Input:
    images_dir: str: Path to the images directory
    margin_percentage: int: Percentage of the
    image to be cropped from all sides

    Output:
    pd.DataFrame: Image quality metrics
    """
    data = []
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = crop_image(os.path.join(images_dir, filename), margin_percentage)
            data.append(evaluate_image_texture(filename, image[0], distance, angle))
    return pd.DataFrame(
        data,
        columns=[
            "image",
            "energy",
            "entropy",
            "homogeneity",
            "contrast",
            "correlation",
        ],
    )
