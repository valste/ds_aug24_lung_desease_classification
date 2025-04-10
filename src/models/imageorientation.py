#############################################################################################################################
# The module provides model specific classes for classifying the orientation of the x-ray images among 4 orientation classes: 
#                                   0: 'rotated_90',        
#                                   1: 'rotated_minus_90',  
#                                   2: 'correct',           
#                                   3: 'rotated_180'        
#############################################################################################################################


#-----------Imports

# Imports for visualizations
import matplotlib.pyplot as plt
from pprint import pprint
import os
import json
import pandas as pd
import numpy as np
from src.defs import DiseaseCategory as dc
from src.utils.img_processing import ImageProcessor

# tensorflow
import tensorflow as tf
from keras.utils import image_dataset_from_directory # enables parallel processing
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications import ResNet50 # functional/ 50 layers/ requires images to be 224x224 in RGB 
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback

# Imports for building the model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D

# Imports for image transformations
#from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import RandomZoom
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.layers import RandomTranslation


class _OrientationEstimatorBase():

    def __init__(self, **kwargs):

        # 1. protected basic variables can be set on construction
        # define allowed parameter keys
        self._basic_keys = {
            "img_resolution",
            "model_prefix",
            "model_dir",
            "data_dir",
            "train_valid_dataset_dir",
            "test_dataset_dir",
            "test_dataset_desc",
            "train_ds",
            "val_ds",
            "history",
            "model",
            "estimated_data_results",
            "estimated_test_results"
        }
        self._img_resolution = None  # (h,w)
        self._model_prefix = None
        self._model_dir = None
        self._data_dir = None
        self._train_valid_dataset_dir = None
        self._test_dataset_dir = None
        self._test_dataset_desc = None
        self._train_ds, self._val_ds = None, None
        self._history = None # adict taken from history.history
        self._model = None
        self._estimated_data_results = None
        self._estimated_test_results = None
                
        # Raise error for unexpected kwargs
        unexpected = set(kwargs) - self._basic_keys
        if unexpected:
            raise ValueError(f"Unexpected kwargs: {unexpected}")
        
        # Assign values from kwargs for basic attributes
        for key in self._basic_keys:
            if key in kwargs:
                setattr(self, f"_{key}", kwargs[key])
      
        # dict objects are set when approriate function is called
        self._data_loading_params = None 
        self._compiling_params = None


    
    def prepareTrainValidData(self, **kwargs):
        """
        1. Real-time Data Loading with image_dataset_from_directory

            * Reads images directly from folders in batches.
            * Useful when datasets don't fit entirely into memory.
            * Automatically assigns labels based on folder structure.
            * process parallelization
            
        train_val_{resolution}/
        ├── {resolution}_rotated_0/
        │   ├── image1.png
        │   ├── image2.png
        │   └── ...
        └── {resolution}_rotated_90/
            ├── image101.png
            ├── image102.png
            └── ...
            ...

        # Before using a pre-trained model like ResNet50, it is essential to ensure that the new input data undergoes the 
        # same preprocessing as was applied to the original dataset # (**ImageNet**) when the model was trained. 
        # This ensures compatibility between our data and the pre-trained model

        """

        # When loading data using image_dataset_from_directory() — labels are automatically integers --> sparse_categorical_crossentropy must be used.
        # But after resNet preprocessing: train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)) the classes are one-hot
        # encoded -->  categorical_crossentropy must be used
        self._train_ds = image_dataset_from_directory(subset="training",  **kwargs)
        self._val_ds = image_dataset_from_directory(subset="validation", **kwargs)
        self._data_loading_params = kwargs
        # Number of batches in the training dataset
        print("Number of batches in train_ds:", self._train_ds.cardinality().numpy())
        # Number of batches in the validation dataset
        print("Number of batches in val_ds:", self._val_ds.cardinality().numpy())

        # Preprocess train/validation data to allign with data the resNet50 was trained on--> ImageNet dataset
        # When you apply preprocess_input(image_tensor), it does:

        # ✅ 1. Converts RGB → BGR (channel order switch)
        #     ResNet50 was trained using images in BGR format (from original Caffe implementation).
        #     So it swaps the channels from [R, G, B] → [B, G, R].
        # ✅ 2. Subtracts the ImageNet mean pixel values (no scaling to [0, 1]!)
        #     It subtracts these values per channel:
        #         Blue: 103.939
        #         Green: 116.779
        #         Red: 123.68
        # This centers the data around zero, similar to how the original model was trained.

        # One-hot encoded labels [1000, 0100, 0010, 0001, ...],  len([...])=batch_size
        @tf.autograph.experimental.do_not_convert
        def preprocess(x, y):
            return preprocess_input(x), y

        self._train_ds = self._train_ds.map(preprocess)
        self._val_ds = self._val_ds.map(preprocess)
       
        return self._train_ds, self._val_ds
    


    def checkTrainValDataStructure(self):

        for images, labels in self._val_ds.take(1):
            print("encoded label samples: ", labels.numpy()[:4])
            print("images.shape: ", images.shape)



    def unfreezeLayers(self, layers_to_unfreeze=0):
        # Unfreeze the last n layers of the model for fine-tuning
        for layer in self._model.layers[-layers_to_unfreeze:]:
            layer.trainable = True 
    


    def getCompiledModel(self, layers_to_unfreeze=0, **kwargs):
        # returns the compiled model ready for training

        raise NotImplementedError("to be imlemented in model specific class!")
        

    
    def train(self, target_val_acc=None, epochs = 10, callbacks=[]):
        
        if target_val_acc:
            stop_at_specified_val_acc = StopAtValAccuracy(target_val_acc)
            callbacks.append(stop_at_specified_val_acc)
        
        history = self._model.fit(self._train_ds, validation_data = self._val_ds, epochs=epochs, callbacks = callbacks) 
        self._history = history.history

        return self._history, self._model



    def storeTrainedModel(self):

        # Save the model
        model_file = f'orientation_classifier_{self._model_prefix}.keras'
        model_path = os.path.join(self._model_dir, model_file)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self._model.save(model_path)

        # Save training history
        history_file = f'training_history_{self._model_prefix}.json'
        history_path = os.path.join(self._model_dir, history_file)
        with open(history_path, 'w') as f:
            json.dump(self._history, f)

        print(model_file, history_file, "stored to ", self._model_dir)

       

    def loadTrainedModel(self):

        # loading model
        model_file = f'orientation_classifier_{self._model_prefix}.keras'
        model_path = os.path.join(self._model_dir, model_file)
        self._model = load_model(model_path)

        # model training history
        history_file = f'training_history_{self._model_prefix}.json'
        history_path = os.path.join(self._model_dir, history_file)
        if os.path.getsize(history_path) == 0:
            print(f"{history_path} is empty!")
        else:
            with open(history_path, 'r') as f:
                self._history = json.load(f)

        print(f"model loaded from {model_path}")
        print(f"model train history loaded from {history_path}")
    
        return self._model, self._history



    def plotModelMetrics(self, include=["history",  "summary"]):

        if "summary" in include:
            #----model summary----
            print(f"model summary for: {self._model_prefix}")
            self._model.summary()
        
        if "history" in include:
            print(f"model accuracy & loss for: {self._model_prefix}")
            pprint(self._history)
            # Plot accuracy and loss
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(self._history['accuracy'], label='Accuracy')
            plt.plot(self._history['val_accuracy'], label='Validation Accuracy')
            plt.legend()
            plt.title('Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(self._history['loss'], label='Loss')
            plt.plot(self._history['val_loss'], label='Validation Loss')
            plt.legend()
            plt.title('Loss')

            plt.show()

        if "param_set" in include:
            print(f"image_dataset_from_directory parameters for: {self._model_prefix}")
            pprint(self.__param_set)



    def estimateImageOrientation(self, dataset, save_to_csv=True):

        # estimates the image orientation for specified dataset: "data" or "test"
        # saves to csv

        # Step 1: Define the estimator function
        def estimate_orientation(img_path, model):
            
            img = image.load_img(img_path, target_size=(self._img_resolution))          # loads in RGB by default, ResNet50 expects 224x224 in RGB --> Shape: (224, 224, 3)
            img_array = image.img_to_array(img)                             # Convert to array
            img_array = np.expand_dims(img_array, axis=0)                   # Shape becomes (1, 224, 224, 3)
            img_array = preprocess_input(img_array)                         # 🔥 Preprocess -->ready for prediction

            # Get predictions
            if model:
                prediction = model.predict(img_array)  # Shape: (1, 4)
            else:
                print("model is not loaded/created yet")

            # Class index (one-hot encoded [0, 0, 1, 0]) mapping from train_generator.class_indices
            index_to_orientation = {
                0: 'rotated_90',       # from folder 
                1: 'rotated_minus_90', # from folder 
                2: 'correct',          # from folder 
                3: 'rotated_180'       # from folder 
            }

            estimated_index = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][estimated_index])
            orientation = index_to_orientation[estimated_index]
            
            return orientation, confidence
        
        # Step 3: Predict orientation for all images in the dataset
        df = pd.DataFrame(columns=["Image","Orientation","Confidence","Disease"])
        results = []
        dis = None

        for disease in dc:

            if dataset == "test":
                image_dir = self._test_dataset_dir
                dis = "test_data"

            elif dataset == "data":
                image_dir = os.path.join(self._data_dir, rf"{disease.value}\downscaled\{self._img_resolution[0]}x{self._img_resolution[1]}")
                dis = disease.value

            else:
                raise Exception(f"Wrong dataset: {dataset}")        
            
            for img_name in os.listdir(image_dir):
                img_path = os.path.join(image_dir, img_name)
                orientation, confidence = estimate_orientation(img_path, self._model)
                results.append({
                    "Image": img_name,
                    "Orientation": orientation,
                    "Confidence": confidence,
                    "Disease": dis
                })

            # results to DataFrame
            df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
            results.clear()

            if dataset == "test":
                break

        # Sort 
        df_estimated = df.sort_values(by=["Disease", "Orientation", "Confidence"], ascending=False)
        self.checkEstimatedData(df_estimated)
        
        # 3: save to CSV and set the corresponding variable
        if save_to_csv:

            file_name = None

            if dataset == "test":
                file_name = f"estimated_orientation_{self._model_prefix}_{dataset}.csv"
                self.estimated_test_results = df_estimated

            elif dataset == "data":
                file_name = f"estimated_orientation_{self._model_prefix}.csv"
                self.estimated_data_results = df_estimated

            else:
                 raise Exception(f"No such dataset: {dataset}")

            output_csv = os.path.join(self._model_dir, file_name)
            df_estimated.to_csv(output_csv, index=False)
            print("estimated results saved to: ", output_csv)

        return df_estimated



    def loadEstimatedData(self, dataset, sort_by=["Disease", "Orientation", "Confidence"]):
        
        file_name = None

        if dataset == "test":
            file_name = f"estimated_orientation_{self._model_prefix}_{dataset}.csv"
        elif dataset == "data":
            file_name = f"estimated_orientation_{self._model_prefix}.csv"
        else:
            raise Exception(f"No such dataset: {dataset}!")

        df_estimated = pd.read_csv(os.path.join(self._model_dir, file_name))
        df_estimated = df_estimated.sort_values(by=sort_by).reset_index(drop=True)

        print("estimated results loaded from: ", file_name)
        
        return df_estimated



    def checkEstimatedData(self, df_estimated):

        df_rotated = df_estimated[df_estimated["Orientation"] != "correct"]
        df_correct = df_estimated[df_estimated["Orientation"] == "correct"]

        print(f"estimated as rotated: {df_rotated.shape[0]}")
        print(f"estimated as correct: {df_correct.shape[0]}")
        print(f"Estimates having confidence < 1: ", df_estimated[df_estimated["Confidence"] < 1].shape[0])
        grouped_counts = df_rotated.groupby(["Disease", "Orientation"]).size().reset_index(name="Count")
        print(grouped_counts)

        print(f"checking for NaNs:")
        print(df_estimated.isna().sum())

        


    # show rotated_90, rotated_minus_90, rotated_180 images only
    def showRotatedImg(self, dataset):
        
        df_estimated = None
        dis = None
        ip = ImageProcessor()

        for disease in dc:

            if dataset == "test":
                image_dir = self._test_dataset_dir
                dis = "test_data"
                print("This are the images from the test dataset!")
                print(f"test set directory: {self._test_dataset_dir}")
                print(self._test_dataset_desc)

            elif dataset == "data":
                dis = disease.value
                image_dir = os.path.join(self._data_dir, rf"{dis}\downscaled\{self._img_resolution[0]}x{self._img_resolution[1]}")

            else:
                raise Exception(f"Wrong dataset: {dataset}")
            
            df_estimated = self.loadEstimatedData(dataset)


            for rot in ("rotated_90", "rotated_minus_90", "rotated_180"):
                            
                i_names = df_estimated["Image"][(df_estimated["Disease"]==dis) & (df_estimated["Orientation"] == rot) ].to_list()
                cnt = len(i_names)
                
                if len(i_names)>0:
                    imgs, img_names = ip.loadImgs(i_names, image_dir)
                    print("")
                    print(dis, f"--> estimated orientation: {rot}", f"--> total detected: {len(i_names)}")
                    ip.plot_images(imgs, img_names, tSize=15, max_img_per_row=5)
                else:
                    print(dis, f"--> estimated orientation: {rot}", f"--> total detected: 0")

            if dataset == "test":
                    break  
            
            



class OrientationEstimatorMobileNet(_OrientationEstimatorBase):

    def getCompiledModel(self, **kwargs):
        # returns the compiled model ready for training

        # Input layer
        inputs = Input(shape=kwargs.pop("input_shape", None))

        # Load MobileNet without top classification layer
        base_model = MobileNet(
            input_tensor=inputs,
            weights='imagenet',
            include_top=False  # add own head
        )

        # Freeze the base model layers
        layers_to_unfreeze = kwargs.pop("layers_to_unfreeze", None)
        if layers_to_unfreeze > 0:
            # Unfreeze the last n layers of the model for fine-tuning
            for layer in base_model.layers[-layers_to_unfreeze:]:
                layer.trainable = True
        else:
            # Freeze base model for transfer learning (optional)
            base_model.trainable = False

        # Apply augmentations 
        x = RandomRotation(0.1)(inputs)                          
        x = RandomTranslation(height_factor=0.1, width_factor=0.1)(x) 
        x = RandomZoom(0.1)(x)  

        # Custom classification head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        estimates = Dense(4, activation='softmax')(x)  # 4 orientation classes

        # Final model
        self._model = Model(inputs=inputs, outputs=estimates)

        # Compile
        self._model.compile(**kwargs)
        self._compiling_params = kwargs
        print(f"The {self._model_prefix} model has been compiled successfully")

        return self._model
    


class OrientationEstimatorResnet50(_OrientationEstimatorBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def getCompiledModel(self, **kwargs):
        # returns the compiled model ready for training
    
        # Model creation using the Functional API
        inputs = Input(shape=(224, 224, 3))

        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

        # freeze the model's layers
        base_model.trainable = False

        # Apply augmentations 
        x = RandomRotation(0.1)(inputs)                          
        x = RandomTranslation(height_factor=0.1, width_factor=0.1)(x) 
        x = RandomZoom(0.1)(x)  
       
        # add custom classification block
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        estimates = Dense(4, activation='softmax')(x) # 4 classes for orientations: 0°, 90°, 180°, 270° --> softmax

        self._model = Model(inputs=inputs, outputs=estimates)
        self._model.compile(**kwargs)
        self._compiling_params = kwargs

        print(f"The {self._model_prefix} model has been compiled successfully")

        return self._model
           

class StopAtValAccuracy(Callback):
    # stopps the training (after a completed epoch) once the target_val_accuracy is reached

    def __init__(self, target_val_accuracy):
        super().__init__()
        self.target_val_accuracy = target_val_accuracy

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get("val_accuracy")
        if val_accuracy is not None:
            if val_accuracy >= self.target_val_accuracy:
                print(f"\nReached target validation accuracy of {self.target_val_accuracy*100:.2f}% at epoch {epoch+1}. Stopping training.")
                self._model.stop_training = True
