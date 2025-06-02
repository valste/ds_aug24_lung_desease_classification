# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
import pandas as pd
from skimage.restoration import estimate_sigma

from .image_preprocessor import crop_image


def detect_blurriness(image, threshold=100) -> tuple:
    """
    detect_blurriness function detects the blurriness of an image.

    Input:
    image: np.array: Image as a numpy array
    threshold: int: Threshold for the variance

    Output:
    str: Blurriness status of the image
    float: Variance of the image
    """
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    if variance < threshold:
        status = f"Blurry (Variance: {variance:.2f})"
    else:
        status = f"Clear (Variance: {variance:.2f})"

    return status, variance


def detect_noise(image, threshold=20) -> tuple:
    """
    detect_noise function detects the noise in an image.

    Input:
    image: np.array: Image as a numpy array
    threshold: int: Threshold for the variance

    Output:
    str: Noise status of the image
    float: Standard deviation of the noise
    """
    noise = cv2.GaussianBlur(image, (5, 5), 0) - image
    noise_std = np.std(noise)
    if noise_std > threshold:
        status = f"Noisy (Std Dev: {noise_std:.2f})"
    else:
        status = f"Low Noise (Std Dev: {noise_std:.2f})"

    return status, noise_std


def check_brightness(image, low_threshold=50, high_threshold=200) -> tuple:
    """
    check_brightness function checks the brightness of an image.

    Input:
    image: np.array: Image as a numpy array
    low_threshold: int: Low threshold for brightness
    high_threshold: int: High threshold for brightness

    Output:
    str: Brightness status of the image
    float: Brightness of the image
    """
    brightness = np.mean(image)
    if brightness < low_threshold:
        status = f"Too Dark (Brightness: {brightness:.2f})"
    elif brightness > high_threshold:
        status = f"Too Bright (Brightness: {brightness:.2f})"
    else:
        status = f"Good Brightness (Brightness: {brightness:.2f})"

    return status, brightness


def check_contrast(image, threshold=100) -> tuple:
    """
    check_contrast function checks the contrast of an image.

    Input:
    image: np.array: Image as a numpy array
    threshold: int: Threshold for the variance

    Output:
    str: Contrast status of the image
    float: Contrast of the image
    """
    contrast = np.max(image) - np.min(image)
    if contrast < threshold:
        status = f"Low Contrast (Contrast: {contrast})"
    else:
        status = f"Good Contrast (Contrast: {contrast})"

    return status, contrast


def check_resolution(image, min_width=800, min_height=600) -> tuple:
    """
    check_resolution function checks the resolution of an image.

    Input:
    image: np.array: Image as a numpy array
    min_width: int: Minimum width of the image
    min_height: int: Minimum height of the image

    Output:
    str: Contrast status of the image
    float: Contrast of the image
    """
    height, width = image.shape[:2]
    if width < min_width or height < min_height:
        status = f"Low Resolution (Width: {width}, Height: {height})"
    else:
        status = f"Good Resolution (Width: {width}, Height: {height})"

    return status, [height, width]


def detect_exposure(image, dark_threshold=0.2, bright_threshold=0.8) -> str:
    """
    detect_exposure function detects the exposure of an image.

    Input:
    image: np.array: Image as a numpy array
    dark_threshold: float: Threshold for dark pixels
    bright_threshold: float: Threshold for bright pixels

    Output:
    str: Exposure status of the image
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    total_pixels = np.sum(hist)
    dark_pixels = np.sum(hist[:50]) / total_pixels
    bright_pixels = np.sum(hist[205:]) / total_pixels

    if dark_pixels > dark_threshold:
        status = "Underexposed"
    elif bright_pixels > bright_threshold:
        status = "Overexposed"
    else:
        status = "Well Exposed"

    return status


def evaluate_image_quality(filename, image) -> list:
    """
    evaluate_image_quality function evaluates the quality of an image.

    Input:
    filename: str: Name of the image file
    image: np.array: Image as a numpy array

    Output:
    pd.DataFrame: Image quality metrics
    """
    return [
        filename,
        detect_blurriness(image)[1],
        detect_noise(image)[1],
        check_brightness(image)[1],
        check_contrast(image)[1],
    ]


def get_images_quality(images_dir, margin_percentage=0) -> pd.DataFrame:
    """
    get_images_quality function evaluates the
    quality of an image.

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
            data.append(evaluate_image_quality(filename, image[0]))
    return pd.DataFrame(
        data,
        columns=["image", "blurriness", "noise", "brightness", "contrast"],
    )


def calculate_snr(filename, image) -> tuple:
    """
    calculate_snr function calculates the signal-to-noise ratio of an image.

    Input:
    filename: str: Name of the image file
    image: np.array: Image as a numpy array

    Output:
    float: Signal-to-noise ratio of the image
    """

    signal_power = np.mean(image**2)
    # Here, we use the skimage method to estimate noise assuming Gaussian noise
    sigma = estimate_sigma(image, channel_axis=None)  # Noise standard deviation
    noise_power = sigma**2

    snr = 10 * np.log10(signal_power / noise_power)

    return filename, snr


def get_images_snr(images_dir, margin_percentage=0) -> pd.DataFrame:
    """
    get_images_snr function calculates the
    signal-to-noise ratio of an image.

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
            data.append(calculate_snr(filename, image[0]))
    return pd.DataFrame(
        data,
        columns=["image", "snr"],
    )
