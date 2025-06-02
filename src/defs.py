import os
from enum import Enum
from pprint import pprint
from pathlib import Path

mlruns_path = Path(r"c:\Users\User\DataScience\aug24_cds_int_analysis-of-covid-19-chest-x-rays\mlruns")
print(mlruns_path.as_uri())

def initDataPaths(project_dir):
    # initializes datapaths

    global PROJECT_DIR 
    global METADATA_DIR 
    global IMAGE_DIRECTORIES
    global TRAINIG_DATA_DIR_254_IMG_ORIENTATION
    global TRULY_ROTATED_IMG_224
    global TRAINIG_DATA_DIR_256_MASKED_IMBALANCED
    global TRAINIG_DATA_DIR_256_MASKED_BALANCED
    global MLRUNS_URI
    global MLRUNS_DIR

    PROJECT_DIR = project_dir
    METADATA_DIR = os.path.join(PROJECT_DIR, r"metadata")
    TRAINIG_DATA_DIR_254_IMG_ORIENTATION = os.path.join(PROJECT_DIR, r"data_224x224\train_val_224x224")
    TRULY_ROTATED_IMG_224 = os.path.join(PROJECT_DIR, r"224x224_truly_rotated")
    TRAINIG_DATA_DIR_256_MASKED_IMBALANCED = os.path.join(PROJECT_DIR, r"256x256_masked_images_imbalanced")
    TRAINIG_DATA_DIR_256_MASKED_BALANCED = os.path.join(PROJECT_DIR, r"256x256_masked_images_balanced")
    
    MLRUNS_URI = Path(os.path.abspath(os.path.join(PROJECT_DIR, "mlruns_vst"))).as_uri()
    MLRUNS_DIR = os.path.abspath(os.path.join(PROJECT_DIR, "mlruns_vst"))

    IMAGE_DIRECTORIES = {
        "COVID": {
            "images": os.path.join(PROJECT_DIR, "data", "COVID-19_Radiography_Dataset", "COVID", "images"),
            "masks": os.path.join(PROJECT_DIR, "data", "COVID-19_Radiography_Dataset", "COVID", "masks"),
        },
        "Lung_Opacity": {
            "images": os.path.join(PROJECT_DIR, "data", "COVID-19_Radiography_Dataset", "Lung_Opacity", "images"),
            "masks": os.path.join(PROJECT_DIR, "data", "COVID-19_Radiography_Dataset", "Lung_Opacity", "masks"),
        },
        "Normal": {
            "images": os.path.join(PROJECT_DIR, "data", "COVID-19_Radiography_Dataset", "Normal", "images"),
            "masks": os.path.join(PROJECT_DIR, "data", "COVID-19_Radiography_Dataset", "Normal", "masks"),
        },
        "Viral Pneumonia": {
            "images": os.path.join(PROJECT_DIR, "data", "COVID-19_Radiography_Dataset", "Viral Pneumonia", "images"),
            "masks": os.path.join(PROJECT_DIR, "data", "COVID-19_Radiography_Dataset", "Viral Pneumonia", "masks"),
        },
    }

def checkDataPaths():
    print(
    "\nPROJECT_DIR: ", PROJECT_DIR,
    "\nMETADATA_DIR: ", METADATA_DIR,
    "\nIMAGE_DIRECTORIES: ", IMAGE_DIRECTORIES,
    "\nTRAINIG_DATA_DIR_254_IMG_ORIENTATION: ", TRAINIG_DATA_DIR_254_IMG_ORIENTATION,
    "\nTRULY_ROTATED_IMG_224: ", TRULY_ROTATED_IMG_224,
    "\nTRAINIG_DATA_DIR_256_MASKED_IMBALANCED: ", TRAINIG_DATA_DIR_256_MASKED_IMBALANCED,
    "\nTRAINIG_DATA_DIR_256_MASKED_BALANCED: ", TRAINIG_DATA_DIR_256_MASKED_BALANCED,
    "\nMLRUNS_URI: ", MLRUNS_URI
    )
    print("original dataset directories: ")
    pprint(IMAGE_DIRECTORIES)
    
class _Base(str, Enum):
    def __str__(self):
        return self.value


class DiseaseCategory(_Base):
    # Enum for the different disease categories
    # alligned to file names without extension .png
    VIRAL_PNEUMONIA = "Viral Pneumonia"
    COVID = "COVID"
    LUNG_OPACITY = "Lung_Opacity"
    NORMAL = "Normal"


class ImageType(_Base):
    IMAGES = "images"
    MASKS = "masks"
    MASKED = "masked"


class ModelType(_Base):
    # Enum for the different model types
    RESNET50 = "resnet50"
    MOBILENET = "mobnet"
    CAPSNET = "capsnet"
    
    

class ExperimentName(_Base):
    # mlflow experiment names
    ORIENTATION_CLASSIFIER = "orientation_classifier"
    DESEASE_CLASSIFIER = "desease_classifier"
    
    
# >>>>>IMPORTANT: the mapping must be the same as for the training dataset!!!!<<<<<
# check loaded dataset
class_to_orientation_map = {
    "long": {
        0: 'rotated_0',         
        1: 'rotated_180',       
        2: 'rotated_90',        
        3: 'rotated_minus_90'   
    },
    "short": {
        0 : "0°",
        1 : "180°",
        2 : "90°",
        3 : "-90°",
    },
}

orientation_labels = {
    "short": ["0°","180°","90°","-90°",],
    "long": ['rotated_0','rotated_180','rotated_90','rotated_minus_90']
    }
    
class_to_disease_map = {
        0: 'COVID',          
        1: 'Lung_Opacity',   
        2: 'Normal',         
        3: 'Viral Pneumonia' 
    }

disease_labels = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']


class DatasetType(_Base):
    TRAIN = "train"
    TEST = "test"
    PREDICT = "predict"