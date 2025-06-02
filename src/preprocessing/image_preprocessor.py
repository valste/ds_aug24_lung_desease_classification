# -*- coding: utf-8 -*-
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from   scipy.stats import f_oneway


def crop_image(image_path, margin_percentage=10) -> tuple:
    """
    crop_image function takes an image path as input
    and returns the cropped image.

    Input:
    image_path: str: Path to the image file
    margin_percentage: int: Percentage of the image
    to be cropped from all sides

    Output:
    cropped_image: np.array: Cropped image as a numpy array
    image: np.array: Original image as a numpy array
    """
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    margin_x = int(width * margin_percentage / 100)
    margin_y = int(height * margin_percentage / 100)

    cropped_image = gray[margin_y : height - margin_y, margin_x : width - margin_x]

    return cropped_image, gray


def apply_image_mask(image_path, mask_path, target="") -> np.array:
    """
    apply_image_mask function plots the image statistics.

    Input:
    image_path: str: Path to the image file
    mask_path: str: Path to the mask file
    target: str: Target to be masked

    Output:
    masked_image: np.array: Masked image as a numpy array
    """
    image = cv2.imread(image_path)

    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, image.shape[:2])
    mask = mask = (mask > 127).astype(np.uint8)

    return mask * image


def calulate_image_statistics(filename, image) -> list:
    """
    calulate_image_statistics function calculates
    the statistics of an image.

    Input:
    image: np.array: Image as a numpy array
    filename: str: Name of the image file

    Output:
    statistics: list: List of image statistics
    """
    return [
        filename[:-4],
        np.min(image),
        np.max(image),
        np.mean(image),
        np.median(image),
        np.std(image),
    ]


def get_images_statistics(images_dir, margin_percentage=0) -> pd.DataFrame:
    """
    get_images_statustics function returns the number
    of images in the directory.

    Input:
    images_dir: str: Path to the directory containing images
    margin_percentage: int: Percentage of the image to
    be cropped from all sides
    
    Output:
    images_data: pd.DataFrame: DataFrame containing
    """
    data = []
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = crop_image(os.path.join(images_dir, filename), margin_percentage)
            data.append(calulate_image_statistics(filename, image[0]))
    return pd.DataFrame(data, columns=["image", "min", "max", "mean", "median", "std"])


def get_edges_images_statistics(images_dir, margin_percentage=0) -> pd.DataFrame:
    """
    get_images_edges_statistics function returns the
    number of images in the directory.

    Input:
    images_dir: str: Path to the directory containing images
    margin_percentage: int: Percentage of the image
    to be cropped from all sides

    Output:
    images_data: pd.DataFrame: DataFrame containing
    """
    data = []
    ddept = cv2.CV_8U
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = crop_image(os.path.join(images_dir, filename), margin_percentage)
            x = cv2.Sobel(image[0], ddept, 1, 0, ksize=3, scale=1)
            y = cv2.Sobel(image[0], ddept, 0, 1, ksize=3, scale=1)
            edge = cv2.addWeighted(cv2.convertScaleAbs(x), 0.5, cv2.convertScaleAbs(y), 0.5, 0)
            data.append(calulate_image_statistics(filename, edge))
    return pd.DataFrame(data, columns=["image", "min", "max", "mean", "median", "std"])


def get_masked_images_statistics(images_dir, mask_dir) -> pd.DataFrame:
    """
    get_masked_images_statistics function returns the
    stastics of masked images in the directory.

    Input:
    images_dir: str: Path to the directory containing images
    mask_dir: str: Path to the directory containing masks
    margin_percentage: int: Percentage of the image
    to be cropped from all sides

    Output:
    images_data: pd.DataFrame: DataFrame containing
    """
    data = []
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = apply_image_mask(
                os.path.join(images_dir, filename),
                os.path.join(mask_dir, filename),
            )
            data.append(calulate_image_statistics(filename, image[0]))
    return pd.DataFrame(data, columns=["image", "min", "max", "mean", "median", "std"])


def store_images_statistics(images_data, csv_filename) -> None:
    """
    store_images_statistics function stores the image
    statistics in a CSV file.

    Input:
    images_data: pd.DataFrame: DataFrame containing
    image statistics
    csv_filename: str: Name of the CSV file

    Output:
    None
    """
    images_data.to_csv(csv_filename)


def plot_images_statistics(dataset, stats, no_of_cols=2) -> None:
    """
    plot_images_statistics function plots the image statistics.

    Input:
    dataset: str: Name of the dataset
    stats: pd.DataFrame: DataFrame containing image statistics
    no_of_cols: int: Number of columns in the plot

    Output:
    None
    """

    # Select only numerical columns
    stats = stats.select_dtypes(exclude=["object"])

    # Calculate the number of rows
    no_of_rows = int(len(stats.columns) // 2 + 1)

    # Plot the image statistics
    _, axs = plt.subplots(no_of_rows, no_of_cols, figsize=(10, 10))
    plt.suptitle(f"{dataset} Image Statistics")

    for i in range(len(stats.columns)):
        fig_index = axs[i // no_of_cols, i % no_of_cols] if no_of_rows > 1 else axs
        sns.histplot(data=stats.iloc[:, i], bins=50, kde=True, ax=fig_index)
        fig_index.set_title(stats.columns[i])
        fig_index.set_xlabel("")

    # Remove empty subplots
    if no_of_rows > 1:
        [plt.delaxes(ax) for ax in axs.flatten() if not ax.has_data()]
    # Display the subplots
    plt.tight_layout()
    plt.show()


def normalize_image(image) -> np.array:
    """
    normalize_image function normalizes the image.

    Input:
    image: np.array: Image as a numpy array

    Output:
    normalized_image: np.array: Normalized image as a numpy array
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image))


def standardize_image(image) -> np.array:
    """
    standardize_image function standardizes the image.

    Input:
    image: np.array: Image as a numpy array

    Output:
    standardized_image: np.array: Standardized image as a numpy array
    """
    return (image - np.mean(image)) / np.std(image)


def get_images_statistics_by_scales(images_dir, margin_percentage=0) -> tuple:
    """
    get_images_statustics function returns the number
    of images in the directory.

    Input:
    images_dir: str: Path to the directory containing images
    margin_percentage: int: Percentage of the image to
    be cropped from all sides

    Output:
    images_data: pd.DataFrame: DataFrame containing
    """
    normalized = []
    standardized = []
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = crop_image(os.path.join(images_dir, filename), margin_percentage)
            normalized_image = normalize_image(image[0])
            normalized.append(calulate_image_statistics(filename, normalized_image))

            standardized_image = standardize_image(image[0])
            standardized.append(calulate_image_statistics(filename, standardized_image))
    return pd.DataFrame(
        normalized, columns=["image", "min", "max", "mean", "median", "std"]
    ), pd.DataFrame(standardized, columns=["image", "min", "max", "mean", "median", "std"])










#########################################################################################################
###############     functions added by Hanna to plot ensembles 




from src.preprocessing.image_quality import *





def get_masked_images_quality(images_dir, mask_dir):
 
    data = []
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = apply_image_mask(
                os.path.join(images_dir, filename),
                os.path.join(mask_dir, filename),
            )
            data.append(evaluate_image_quality(filename, image))
    return pd.DataFrame(
        data,
        columns=["image", "blurriness", "noise", "brightness", "contrast"])
  





def apply_image_counter_mask(image_path, mask_path, target="lungs"):

    image = (
        255 - cv2.imread(image_path)
        if target == "lungs"
        else cv2.imread(image_path)
    )

    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, image.shape[:2])
    mask= 255- mask

    return cv2.subtract(mask, image)





def get_counter_masked_images_statistics(images_dir, mask_dir):
  
    data = []
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = apply_image_counter_mask(
                os.path.join(images_dir, filename),
                os.path.join(mask_dir, filename),
            )
            data.append(calulate_image_statistics(filename, image[0]))
    return pd.DataFrame(
        data, columns=["image", "min", "max", "mean", "median", "std"]
    )



def get_counter_mask_images_quality(images_dir, mask_dir):

    data = []
    for filename in os.listdir(images_dir):
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
            image = apply_image_counter_mask(
                os.path.join(images_dir, filename),
                os.path.join(mask_dir, filename),
            )
            data.append(evaluate_image_quality(filename, image))
    return pd.DataFrame(
        data,
        columns=["image", "blurriness", "noise", "brightness", "contrast"],
    )
  





def plot_image_stats_ensembles(data, labels, groupbyvar, loclegend="center right", sharey=False):

    images=[]


    fig, axes = plt.subplots(1, len(labels), figsize=(5* len(labels),5), sharey=False)


    for i, var in enumerate(labels):
            ax=axes[i]
            sns.histplot(data=data, x=var, hue=groupbyvar, ax=ax,
                        alpha=0.5, kde=True, stat='probability', common_norm=False, bins=50)
            ax.set_title(var)
            ax.set_xlabel('Values')
    #        if i == 0:
    #             ax.set_ylabel("relative occurance") 
        


    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(handles, labels, title= groupbyvar, loc= loclegend )

    plt.tight_layout()
    plt.show()






def print_significance_test_images (labels, stats_data, group_var):

    for i, col in enumerate(labels):

        groups = []

        for group_name, group_data in stats_data[[col, group_var]].groupby(group_var):
            groups.append(group_data[col].values)
        # print(group_data[col].values)
        f_stat, p_value = f_oneway(*groups)
        print(f"F-statistic: {round(f_stat,2)}, p_value: {p_value}")

        if p_value < 0.05:
                print("For the variable " + col + " at least one of the groups has a significant difference in the mean.")
        else:
                print("For the variable " + col + " there is no significant difference between the means in the groups found.")
        
        