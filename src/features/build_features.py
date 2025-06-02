# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
import pandas as pd
from skimage.feature._hog import hog

from src.preprocessing.image_preprocessor import apply_image_mask


def get_detector(detector) -> cv2.Feature2D:
    """
    get_detector function returns the detector method.

    Input:
    detector: str: Name of the detector

    Output:
    detector_method: cv2.Feature2D: Detector method
    """
    match detector:
        case "ORB":
            detector_method = cv2.ORB_create()
        case "SIFT":
            detector_method = cv2.SIFT_create()
        case "Blob":
            detector_method = cv2.SimpleBlobDetector_create()
        case "":
            raise TypeError("Detector not found")
    return detector_method


def get_all_images_features(images_dir, masks_dir=None, method="ORB") -> tuple:
    """
    get_all_images_features function returns the keypoints
    and descriptors of all images in a directory.

    Input:
    images_dir: str: Path to the directory containing images
    method: str: Name of the detector to be used

    Output:
    keyPoints: list: List of keypoints
    descriptors: list: List of descriptors
    data: pd.DataFrame: Dataframe containing image names
    and number of descriptors
    """
    data = []
    keyPoints, descriptors = None, None

    detector_method = get_detector(method)

    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = (
                cv2.imread(os.path.join(images_dir, filename), cv2.COLOR_BGR2GRAY)
                if masks_dir is None
                else apply_image_mask(
                    os.path.join(images_dir, filename),
                    os.path.join(masks_dir, filename),
                )
            )
            keyPoints, descriptors = detector_method.detectAndCompute(image, None)
            data.append([filename[:-4], len(keyPoints)])
    return (
        keyPoints,
        descriptors,
        pd.DataFrame(data, columns=["image", "keyPoints"]),
    )


def get_hog_features(image) -> np.array:
    """
    get_hog_features function returns the HOG feature detector.

    Input:
    image: np.array: Image as a numpy array

    Output:
    hog_features: np.array: HOG feature detector
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=True,
    )


def get_edges(original_image, method="Canny") -> tuple:
    """
    get_edges function returns the edges in the image.

    Input:
    original_image: np.array: Image as a numpy array
    method: str: Name of the method to be used

    Output:
    original_image: np.array: Original image
    image: np.array: Modified image
    image_edges: np.array: Image with edges
    """
    image = original_image.copy()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    match method:
        case "Canny":
            image_edges = cv2.Canny(gray_image, 50, 100)
            image[image_edges == 255] = (0, 255, 0)
        case "Harris":
            image_edges = cv2.dilate(
                cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04),
                None,
            )
            image[image_edges > 0.05 * image_edges.max()] = [0, 255, 0]
        case "Gaussian":
            blur = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=33, sigmaY=33)
            image = cv2.divide(gray_image, blur, scale=255)
            thresh = cv2.threshold(image, 5, 100, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            image_edges = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        case "":
            raise TypeError("Empty method")
        case _:
            raise TypeError(f"{method} not supported")

    return original_image, image, image_edges


def get_features(original_image, method="Good") -> tuple:
    """
    get_features function returns the features in the image.

    Input:
    image: np.array: Image as a numpy array
    method: str: Name of the method to be used

    Output:
    image: np.array: Original image
    """
    image = original_image.copy()

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    match method:
        case "Good":
            image_features = cv2.goodFeaturesToTrack(
                gray_image, maxCorners=50, qualityLevel=0.02, minDistance=20
            )
            for item in image_features:
                x, y = item[0]
                x = int(x)
                y = int(y)
                cv2.circle(image, (x, y), 6, (0, 255, 0), -1)
        case "Fast":
            fast = cv2.FastFeatureDetector_create()
            fast.setNonmaxSuppression(False)
            kp = fast.detect(gray_image, None)
            image = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0))
        case "ORB" | "SIFT":
            detector_method = get_detector(method)
            keyPoints, _ = detector_method.detectAndCompute(image, None)
            image = cv2.drawKeypoints(image, keyPoints, None, color=(0, 255, 0), flags=0)
        case "Blob":
            detector_method = get_detector(method)
            keyPoints = detector_method.detect(image)
            image = cv2.drawKeypoints(image, keyPoints, None, color=(0, 255, 0), flags=0)
        case "":
            raise TypeError("Empty method")
        case _:
            raise TypeError(f"{method} not supported")

    return original_image, image


def add_outline(image, kernel_size=(5, 5), dilations=3, canny_low=30, canny_high=150) -> np.array:
    """
    add_outline function returns the image with an outline.

    Input:
    image: np.array: Image as a numpy array
    kernel_size: tuple: Kernel size
    dilations: int: Number of dilations
    canny_low: int: Canny low threshold
    canny_high: int: Canny high threshold

    Output:
    outlined_image: np.array: Image with an outline
    """
    blurred = cv2.GaussianBlur(image, kernel_size, 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)
    kernel = np.ones(kernel_size, np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    dilated = cv2.dilate(closed_edges, kernel, iterations=dilations)
    dilated = cv2.bitwise_not(dilated)
    _, mask = cv2.threshold(dilated, 0, 255, cv2.THRESH_BINARY)
    outlined_image = cv2.bitwise_and(image, image, mask=mask)

    return outlined_image


def segment_image(image, outlines) -> np.array:
    """
    segment_image function returns the segmented image.

    Input:
    image: np.array: Image as a numpy array
    outlines: np.array: Outlines of the image

    Output:
    np.array: Segmented image
    """
    new_mask = np.zeros_like(image)
    for i, (_, outline_row) in enumerate(zip(image, outlines)):
        bounds = np.where(outline_row > 0)[0]
        new_mask[i, bounds.min() : bounds.max()] = 1
    return image * new_mask
