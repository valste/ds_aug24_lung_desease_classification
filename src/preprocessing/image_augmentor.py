# -*- coding: utf-8 -*-
import os
import random
import shutil

import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import (
    Sequence,
    image_dataset_from_directory,
    img_to_array,
    load_img,
    to_categorical,
)


def create_training_validation_datasers(input_dir, output_dir, split_ratio=0.2) -> None:
    """
    create_training_validation_datasers function splits the dataset into training and validation sets.

    Input:
    input_dir: str: Path to the input images
    output_dir: str: Path to save the augmented images
    split_ratio: float: Ratio of validation data to total data

    Output:
    None
    """
    class_names = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for class_name in class_names:
        class_dir = os.path.join(input_dir, class_name)
        images = [f for f in os.listdir(class_dir) if not f.startswith(".")]

        # Shuffle the images
        if len(images) == 0:
            print(f"No images found in {class_dir}. Skipping this class.")
            continue
        if len(images) < 2:
            print(f"Not enough images to split in {class_dir}. Skipping this class.")
            continue

        random.shuffle(images)

        split_index = int(len(images) * (1 - split_ratio))
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Create output folders
        train_class_dir = os.path.join(output_dir, "train", class_name)
        val_class_dir = os.path.join(output_dir, "val", class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)

        # Copy files
        for img in train_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(train_class_dir, img))
        for img in val_images:
            shutil.copy(os.path.join(class_dir, img), os.path.join(val_class_dir, img))

    print("Dataset split completed!")


def generate_augmented_images(input_folder, output_folder, class_name, n_images, total_images) -> None:
    """
    generate_augmented_images function generates augmented images using ImageDataGenerator.

    Input:
    input_folder: str: Path to the input images
    output_folder: str: Path to save the augmented images
    class_name: str: Name of the class for which to generate augmented images
    n_images: int: Number of augmented images to generate
    total_images: int: Total number of images in the input folder
    """
    # Paths
    input_folder = os.path.join(input_folder, class_name)
    output_folder = os.path.join(output_folder, class_name)
    os.makedirs(output_folder, exist_ok=True)

    # Calculate the number of images to generate by iteration
    n_augment_images = np.ceil(total_images / n_images).astype(int)
    if n_augment_images == 0:
        print(f"Not enough images to generate {n_images} augmented images.")
        return
    else:
        print(f"Generating {n_augment_images} augmented images per original image.")

    # Create the ImageDataGenerator with augmentation options
    datagen = ImageDataGenerator(
        rotation_range=10,  # rotate images randomly up to 10 degrees
        horizontal_flip=True,  # flip horizontally
    )

    # Loop through each image in the input folder
    for img_name in os.listdir(input_folder):
        if img_name.lower().endswith(("png", "jpg", "jpeg")):
            img_path = os.path.join(input_folder, img_name)

            # Load image
            img = load_img(img_path)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            # Generate and save 5 augmented images per original image
            i = 0
            for _ in datagen.flow(
                x, batch_size=1, save_to_dir=output_folder, save_prefix="aug", save_format="png"
            ):
                i += 1
                if i >= n_augment_images:
                    break


def load_dataset_images(path, image_size, batch_size, color_mode="grayscale") -> tuple:
    """
    load_dataset_images function loads images from a directory
    using image_dataset_from_directory.

    Input:
    path: str: Path to the images
    image_size: tuple: Size of the images
    batch_size: int: Number of augmented images to generate
    color_mode: str: Color mode of the images (grayscale or rgb)

    Output:
    train_generator: tf.data.Dataset: Training data generator
    val_generator: tf.data.Dataset: Validation data generator
    class_weight_dict: dict: Class weights for the training data
    """
    # Define an image_dataset_from_directory for augmentation
    train_generator = image_dataset_from_directory(
        os.path.join(path, "train"),
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        color_mode=color_mode,
        seed=1234,
        shuffle=True,
    )

    val_generator = image_dataset_from_directory(
        os.path.join(path, "val"),
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
        color_mode=color_mode,
        seed=1234,
        shuffle=False,
    )

    class_labels = train_generator.class_names
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(class_labels), y=class_labels
    )
    class_weight_dict = dict(enumerate(class_weights))

    print(f"Computed Class Weights:{class_weight_dict} labels: {train_generator.class_names}")

    return train_generator, val_generator, class_weight_dict


class LungMaskGenerator(Sequence):
    """
    LungMaskGenerator is a custom data generator for loading and augmenting images and masks.
    It inherits from the Keras Sequence class to allow for easy integration with Keras models.
    """

    def __init__(
        self, image_paths, mask_paths, labels, batch_size=32, image_size=(256, 256), shuffle=True
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.labels = labels
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.image_paths))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        idxs = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        batch_images = []
        batch_labels = []

        for i in idxs:
            img = load_img(self.image_paths[i], color_mode="grayscale", target_size=self.image_size)
            img = img_to_array(img) / 255.0

            mask = load_img(self.mask_paths[i], color_mode="grayscale", target_size=self.image_size)
            mask = img_to_array(mask) / 255.0

            # Concatenate image and mask: shape will be (H, W, 2)
            combined = np.concatenate([img, mask], axis=-1)
            batch_images.append(combined)
            batch_labels.append(self.labels[i])

        return np.array(batch_images), np.array(batch_labels)

    def get_class_labels(self):
        return np.argmax(self.labels, axis=1)


def get_image_mask_pairs(image_root, mask_root, classes) -> tuple:
    """
    get_image_mask_pairs function takes the root directories of images and masks
    and returns lists of image paths, mask paths, and their corresponding labels.
    Input:
    image_root: str: Root directory for images
    mask_root: str: Root directory for masks
    classes: list: List of class names

    Output:
    image_paths: list: List of image paths
    mask_paths: list: List of mask paths
    labels: list: List of labels corresponding to the images
    """
    # Ensure the root directories exist
    image_paths = []
    mask_paths = []
    labels = []

    for label_idx, class_name in enumerate(classes):
        image_class_dir = os.path.join(image_root, class_name)
        mask_class_dir = os.path.join(mask_root, class_name)

        # List all image files in the class directory
        for fname in os.listdir(image_class_dir):
            image_path = os.path.join(image_class_dir, fname)
            mask_path = os.path.join(mask_class_dir, fname)

            # Check that the corresponding mask exists
            if os.path.exists(mask_path):
                image_paths.append(image_path)
                mask_paths.append(mask_path)
                labels.append(label_idx)
            else:
                print(f"Warning: No mask found for {image_path}")

    return image_paths, mask_paths, labels


def generate_augmented_images_masks(
    train_img, val_img, train_mask, val_mask, train_lbl, val_lbl, classes
) -> tuple:
    """
    generate_augmented_images_masks function generates
    augmented images and masks using ImageDataGenerator.

    Input:
    path: str: Path to the images
    image_size: tuple: Size of the images

    batch_size: int: Number of augmented images to generate
    """

    train_lbl_one_hot = to_categorical(train_lbl, num_classes=4)
    val_lbl_one_hot = to_categorical(val_lbl, num_classes=4)

    # Define an ImageDataGenerator for augmentation
    train_gen = LungMaskGenerator(train_img, train_mask, train_lbl_one_hot)
    val_gen = LungMaskGenerator(val_img, val_mask, val_lbl_one_hot)

    class_weight_dict = compute_class_weight("balanced", classes=np.unique(classes), y=classes)
    class_weight_dict = dict(enumerate(class_weight_dict))
    print(f"Class weights: {class_weight_dict}")

    return train_gen, val_gen, class_weight_dict
