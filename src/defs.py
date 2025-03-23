import os
from enum import Enum

BASE_DIR = r"C:\Users\User\DataScience"
METADATA_DIR = r"C:\Users\User\DataScience\area51\metadata\original"

IMAGE_DIRECTORIES = {
    "COVID": {
        "images": os.path.join(BASE_DIR, r"area51\data\COVID-19_Radiography_Dataset\COVID\images"),
        "masks": os.path.join(BASE_DIR, r"area51\data\COVID-19_Radiography_Dataset\COVID\masks"),
        "masked": os.path.join(BASE_DIR, r"area51\data\COVID-19_Radiography_Dataset\COVID\masked")
    },
    "Lung_Opacity": {
        "images": os.path.join(BASE_DIR, r"area51\data\COVID-19_Radiography_Dataset\Lung_Opacity\images"),
        "masks": os.path.join(BASE_DIR, r"area51\data\COVID-19_Radiography_Dataset\Lung_Opacity\masks"),
        "masked": os.path.join(BASE_DIR, r"area51\data\COVID-19_Radiography_Dataset\Lung_Opacity\masked")
    },
    "Normal": {
        "images": os.path.join(BASE_DIR, r"area51\data\COVID-19_Radiography_Dataset\Normal\images"),
        "masks": os.path.join(BASE_DIR, r"area51\data\COVID-19_Radiography_Dataset\Normal\masks"),
        "masked": os.path.join(BASE_DIR, r"area51\data\COVID-19_Radiography_Dataset\Normal\masked")
    },
    "Viral Pneumonia": {
        "images": os.path.join(BASE_DIR, r"area51\data\COVID-19_Radiography_Dataset\Viral Pneumonia\images"),
        "masks": os.path.join(BASE_DIR, r"area51\data\COVID-19_Radiography_Dataset\Viral Pneumonia\masks"),
        "masked": os.path.join(BASE_DIR, r"area51\data\COVID-19_Radiography_Dataset\Viral Pneumonia\masked")
    },
}


class DiseaseCategory(str, Enum):
    # Enum for the different disease categories
    # alligned to file names without extension .png
    VIRAL_PNEUMONIA = "Viral Pneumonia"
    COVID = "COVID"
    LUNG_OPACITY = "Lung_Opacity"
    NORMAL = "Normal"


class ImageType(str, Enum):
    IMAGES = "images"
    MASKS = "masks"
    MASKED = "masked"
