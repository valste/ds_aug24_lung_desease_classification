# -*- coding: utf-8 -*-
import os

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tqdm import tqdm


# Prediction and saving function
def predict_and_save(image_path, output_path, model, target_size, apply_mask=False) -> None:
    """
    Predict the mask for a given image and save it.

    Inputs:
    - image_path: str, path to the input image.
    - output_path: str, path to save the output mask.
    - model: Keras model, pre-trained model for mask prediction.
    - target_size: tuple, target size for the images (height, width).
    - apply_mask: bool, whether to apply the mask to the original image.

    Outputs:
    - None, but saves the mask as a PNG file.
    """
    img = load_img(image_path, color_mode="grayscale", target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = increase_brightness(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # (1, h, w, 1)
    prediction = model.predict(img_array)
    mask = (prediction[0, :, :, 0] > 0.5).astype(np.uint8) * 255  # Convert to 0-255
    mask = mask = (mask > 127).astype(np.uint8)

    if apply_mask:
        mask = np.asarray(img).copy() * mask

    # Save mask as PNG
    cv2.imwrite(output_path, mask)


def generate_masks(input_folder, output_folder, model, target_size) -> None:
    """
    Generate masks for chest X-ray images using a pre-trained model.

    Inputs:
    - input_folder: str, path to the folder containing input images.
    - output_folder: str, path to the folder where masks will be saved.
    - model_path: str, path to the pre-trained model file.
    - target_size: tuple, target size for the images (height, width).

    Outputs:
    - None, but saves the masks as PNG files in the output folder.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process all images in the folder
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
            predict_and_save(input_path, output_path, model, target_size)

    print("All masks have been predicted and saved to:", output_folder)


def generate_masked_images(input_folder, output_folder, model, target_size) -> None:
    """
    Generate masked images for chest X-ray images using a pre-trained model.

    Inputs:
    - input_folder: str, path to the folder containing input images.
    - output_folder: str, path to the folder where masks will be saved.
    - model_path: str, path to the pre-trained model file.
    - target_size: tuple, target size for the images (height, width).

    Outputs:
    - None, but saves the masked images as PNG files in the output folder.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Process all images in the folder
    for filename in tqdm(os.listdir(input_folder)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
            predict_and_save(input_path, output_path, model, target_size, apply_mask=True)

    print("All masks have been predicted and saved to:", output_folder)


def increase_brightness(img_array, factor=1.5) -> np.ndarray:
    """
    Increase the brightness of an image array.
    Inputs:
    - img_array: np.ndarray, input image array.
    - factor: float, factor by which to increase brightness.
    Outputs:
    - np.ndarray, brightened image array.
    """
    # Multiply and clip to valid range
    bright = img_array * factor
    bright = np.clip(bright, 0, 255)
    return bright
