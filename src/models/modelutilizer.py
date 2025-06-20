"""
Provides the ModelUtilizer class for loading and using models
"""
# --- Standard Library
import os
import time

# --- Third-Party Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import load_model

# --- Project-Specific
from src.defs import MLRUNS_DIR, ModelType as mt, DatasetType as dt, orientation_labels
from src.utils.datahelper import DataHelper as dh

class ModelUtilizer:
    """
    A class to utilize models for various tasks. 
    """

    def __init__(self, model_type, input_shape, model_name=None, mlruns_uri=None, run_id=None, alt_model_path=None, custom_objects={}):
        """
        Initializes the ModelUtilizer with a model name. A model eithe is taken from a run or from alt_model_path.

        Args:
            mlruns_uri (str): The name of the model to be loaded.
            run_id (str): the run the model should be used from 
            alt_model_path (str): the path to the alternative .keras model file
            
        """
        # model_type vs input shape validation
        if model_type in (mt.MOBILENET, mt.RESNET50,):
            if input_shape != (224,224,3):
                raise Exception(f"input shape for {self.model_name} model must be (224,224,3)") 
        elif model_type == mt.CAPSNET:
            if input_shape != (256,256,3):
                raise Exception(f"input shape for {self.model_name} model must be (256,256,3)")
        else:
            raise Exception(f"Model not supported: {self.model_name}. The model name must contain one substring from {mt.MOBILENET, mt.RESNET50, mt.CAPSNET}")
        
        self.input_shape = input_shape
        self.model_type = model_type
        self.custom_objects = custom_objects # for capsnet model
        
        
        # Load the model from a run and set mlflow tracking uri
        if mlruns_uri and run_id and not alt_model_path:
            self.mlruns_uri = dh.set_mlflow_tracking(mlruns_uri=mlruns_uri) #configure datahelper mlruns_uri
            self.model, self.model_name = dh.get_model_by_run_id(run_id)
        
        # loading model from model-path    
        elif not mlruns_uri and not run_id and alt_model_path:
            self.model = tf.keras.models.load_model(alt_model_path, compile=False)
            self.model_name = model_name or self.model.name
            
        else:
            raise Exception("""Either to specify mlruns_uri AND run_id or alt_model_path to load the model""")
            
        
    
    
    def _get_true_labels_from_img_names(self, img_names, class_to_label_map):
        """
        Returns the true labels for the images based on their filenames.
        Applicable only for data containing 
        
        Parameters:
        - img_names: List of image filenames.
        - class_to_label_map: Mapping from class indices to labels.
        
        Returns:
        - true_labels: List of true labels for the images.
        """
        true_labels = []
        for img_name in img_names:
            for _, label in class_to_label_map.items():
                if label in img_name:
                    true_labels.append(label)
                    break
        
        return true_labels




    def predict_on_dataset(self, dataset_type, dataset_dir, alt_model_path=None, class_to_label_map=None, csv_output_path=None):
        """
        Predicts on a test or unlabeled dataset.
        
        Parameters:
        - dataset_dir: Directory containing test images or images for prediction (one folder, not labeled).
        - image_size: to validate vs model types: 224x224 for resnet/mobnet and 256x256 for capsnet
        - class_to_label_map: Mapping from class indices to labels.
        - csv_output_path: Path to save the predictions as a CSV file. 
        - custom_objects: Custom objects used in the model (if any).
        
        Returns:
        - df: Model predictions on the test dataset with columns: img_name|true_label|true_labels|predicted_labels|confidences
        - prediction_time
        """
        
        if dataset_type not in  (dt.TEST, dt.PREDICT):
            raise Exception(f"Use only with dataset types TRAIN. Given: {dataset_type}")
        
        image_size = self.input_shape[:-1] # only  width x heigth
        self.dataset_dir = dataset_dir
        dataset = None
       
        # loader selector
        match dataset_type:
            case dt.TEST:
                # Loads the labeled data for testing
                loading_params = {
                    "directory": self.dataset_dir,
                    "label_mode": "categorical", #one-hot encoded
                    "labels": "inferred",  # same as subfolder names in training dataset
                    "image_size": image_size,
                    "batch_size": 32,
                    "shuffle": False
                }
                
            case dt.PREDICT:
                # the labels are unknown, all images in one directory
                loading_params = {
                    "directory": dataset_dir,
                    "labels": None,
                    "label_mode": None,
                    "image_size": image_size,
                    "batch_size": 32,
                    "shuffle": False
                }
                
            case _:
                raise Exception(f"DatasetType {dataset_type} not supported")
                
                
        dataset = tf.keras.utils.image_dataset_from_directory(**loading_params)
        print(dataset.class_names)
        # determine preprocess function for the dataset inputs according to the model requirements
        preprocess_fn = dh.get_preprocess_fn(self.model_type)  
        dataset = dataset.map(preprocess_fn)
        # Optimize performance with prefetch
        AUTOTUNE = tf.data.AUTOTUNE
        dataset = dataset.prefetch(AUTOTUNE)      
        
        print("Number of batches in dataset:", dataset.cardinality().numpy())
                
        # Make predictions with model from mlruns or alt_model_path
        start = time.time()
        if not alt_model_path:
            predictions = self.model.predict(dataset)
        else:
            alt_model = tf.keras.models.load_model(alt_model_path, compile=False)
            print(f"Alternative model alt_model: {alt_model.name} is used for predictions")
            predictions = alt_model.predict(dataset)
        end = time.time()
        prediction_time = end - start
        
        print(f"Total prediction time: {prediction_time:.2f} seconds")

        # ---Get the predicted classes---
        # works because shuffle=False in image_dataset_from_directory
        predicted_classes = tf.argmax(predictions, axis=1).numpy()
        
        # --> The output of a capsule (e.g., digit_caps) is a vector or matrix, and:
        # --> Its length (or Frobenius norm) is interpreted as the confidence of a class being present.
        # Normalize each row in predictions by its row max --> capsnet only because also values > 1 possible
        predictions = predictions / np.sum(predictions, axis=1, keepdims=True)  # now the sum of each row is 1 (total confidence=100%)
        confidences = np.max(predictions, axis=1).flatten() 

        predicted_labels = [class_to_label_map[int(i)] for i in predicted_classes]
        img_names = dh.get_file_names(dataset_dir)

        true_labels = []
        # get true labels from image names only for test datasets properly labeled according to the class
        if dataset_type != dt.PREDICT:
            true_labels = self._get_true_labels_from_img_names(img_names, class_to_label_map)
            
        # for prediction dataset use
        elif dataset_type == dt.PREDICT and len(true_labels) == 0:
            true_labels = ["unknown"] * len(img_names)
            
        else:
            raise Exception(f"Dataset type not supported: {dataset_type}")
        
        # Build DataFrame
        df = pd.DataFrame({
            'img_name': img_names,
            'true_label': true_labels,
            'predicted_label': predicted_labels,
            'confidence': confidences,
        })

        # Save DataFrame to CSV if path is provided
        if csv_output_path:
            df.to_csv(csv_output_path, index=False)
            print(f"Predictions saved to {csv_output_path}")

        return df, prediction_time
    
    
    
    @staticmethod
    def get_classreport_and_confusion_matrix(df, output_dict=True, normalize_cm="all", labels=None):
        """
        Generates a classification report and confusion matrix based on the predictions.
        """
        y_true = df["true_label"]
        y_pred = df["predicted_label"]

        # 1. Classification report
        clsrep = classification_report(
            y_true,
            y_pred,
            output_dict=output_dict,
            labels=labels,
            zero_division=0  # suppress undefined metric warnings
        )
        
        # 2. Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize_cm)

        return clsrep, cm


    
    @staticmethod
    def show_classification_report(df, title, output_dict=False, normalize_cm="all", labels=None):
        """
        Displays a classification report based on the predictions.
        
        Parameters:
        - df: DataFrame containing predictions.
        - normalize_cm: for cm normalization. possible values: None, "all", "pred", "true"
        
        Returns:
        - None
        """
        
        # display classification report    
        clsrep, cm = ModelUtilizer.get_classreport_and_confusion_matrix(df, output_dict, normalize_cm="all", labels=None)
        print(f"Classification report for {title}")
        print(clsrep)
        
        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues', xticks_rotation=45);
                
        plt.title(f"Confusion matrix for {title}\nnormalization mode:{normalize_cm}")
        plt.show()
        
        
        
    @staticmethod
    def get_imgs_both_models_identified_as_rotated(mod1_predCsvPath, mod2_predCsvPath):
        """
        Returns a DataFrame with images that both models identified as rotated.
        
        Parameters:
        - mod1_predCsvPath: Path to the first model's predictions CSV file.
        - model2_predCsvPath: Path to the second model's predictions CSV file.
        
        Returns:
        - DataFrame with columns: img_name, true_label, mod1_predicted_label, mod2_predicted_label
        """
        df_m = pd.read_csv(mod1_predCsvPath, index_col=0)
        df_r = pd.read_csv(mod2_predCsvPath, index_col=0)

        # Merge on img_name
        merged_df = df_m.merge(df_r, on="img_name", suffixes=("_1", "_2"))

        # Keep rows where both predictions are not 'rotated_0'
        df_both_preds_as_rotated = merged_df[
            (merged_df["predicted_label_1"] != "rotated_0") &
            (merged_df["predicted_label_2"] != "rotated_0")
        ]

        df_both_preds_as_rotated = df_both_preds_as_rotated[["img_name", "predicted_label_1", "confidence_1", "predicted_label_2", "confidence_2", ]]

        return df_both_preds_as_rotated
    
    
    
    