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
import time
import json
import pandas as pd
import numpy as np
from src.defs import DiseaseCategory as dc
from src.utils.imgprocessing import ImageProcessor

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
        self._basic_param_keys = {
            "img_resolution", # set on construction
            "model_name", # set on construction
            "model_dir", # set on construction
            "data_dir", # set on construction
            "train_valid_dataset_dir", # set on construction
            "test_dataset_dir",  # set on construction
            "test_dataset_desc", # set on construction
            "train_ds",#cant be set on construction
            "val_ds", #cant be set on construction
            "history", #cant be set on construction
            "model", #can only be loaded oder created by calling the getCompiledModel function
            "estimated_data_results", # cant be set on construction
            "estimated_test_results" #cant be set on construction
        }
        self._img_resolution = None  # (height,width)
        self._model_name = None
        self._model_dir = None
        self._data_dir = None
        self._train_valid_dataset_dir = None
        self._test_dataset_dir = None
        self._test_dataset_desc = None
        self._train_ds, self._val_ds = None, None
        self._history = {} # history.history
        self._model = None
        self._estimated_data_results = None
        self._estimated_test_results = None
        self._training_time = None

        # Raise error for unexpected kwargs
        unexpected = set(kwargs) - self._basic_param_keys
        if unexpected:
            raise ValueError(f"Unexpected kwargs: {unexpected}")
        
        # Assign values from kwargs for basic attributes
        for key in self._basic_param_keys:
            if key in kwargs:
                setattr(self, f"_{key}", kwargs[key])
      
        # value are set when approriate function is called
          
        self._basic_params = kwargs or {} # object to be stored
        self._data_loading_params = {} 
        self._model_params = {}
        self._history = {}

        self._all_parameters = {} # object to be stored
        self._all_parameters["basic_params"] = self._basic_params
        # empty initialization, are set when the corresponding function is called
        self._all_parameters["data_loading_params"] = self._data_loading_params
        self._all_parameters["model_params"] = self._model_params
        self._all_parameters["history"] = self._history


    def isModelSet(self):
        # check if model is set
        if self._model is None:
            raise Exception("model is not loaded/created yet")
        else:
            return True

    
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
        print(f"loading train/validation data from directory: {self._train_valid_dataset_dir}")
        pprint(f"loading data loading params: {kwargs}")
        self._train_ds = image_dataset_from_directory(subset="training",  **kwargs)
        self._val_ds = image_dataset_from_directory(subset="validation", **kwargs)

        # keep the params for later use
        self._data_loading_params = kwargs 
        self._all_parameters["data_loading_params"] = self._data_loading_params
        
        
        
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
    


    def compileModel(self, inputs, outputs, **kwargs):
        # Final model
        self._model = Model(inputs=inputs, outputs=outputs)

        # Compile
        self._model.compile(**kwargs)
        pprint(f"The {self._model_name} model has been compiled successfully: {self._all_parameters["model_params"]}")



    def getCompiledModel(self, **kwargs):
        # returns the compiled model ready for training
        raise NotImplementedError("getCompiledModel() must be implemented in a subclass")

        

    
    def train(self, target_val_acc=None, epochs = 10, callbacks=[]):
        
        self.isModelSet()

        if target_val_acc:
            stop_at_specified_val_acc = StopAtValAccuracy(target_val_acc)
            callbacks.append(stop_at_specified_val_acc)
        
        start = time.time()
        history = self._model.fit(self._train_ds, validation_data = self._val_ds, epochs=epochs, callbacks = callbacks) 
        end = time.time()
        self._history = history.history
        self._training_time = end - start
        self._history["trainig_params"] = {"training_time": self._training_time,"target_val_acc": target_val_acc, "epochs": epochs}
        self._all_parameters["history"] = self._history

        return self._history, self._model



    def storeTrainedModel(self):

        self.isModelSet()

        # Save the model
        model_file = f'orientation_classifier_{self._model_name}.keras'
        model_path = os.path.join(self._model_dir, model_file)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self._model.save(model_path)

        print(model_file, "stored to ", self._model_dir)

       

    def loadTrainedModel(self):

        # loading model
        model_file = f'orientation_classifier_{self._model_name}.keras'
        model_path = os.path.join(self._model_dir, model_file)
        self._model = load_model(model_path)
        print(f"model loaded from {model_path}")
        # and related parameters
        self.loadAllParameters()
    
        return self._model, self._history



    def plotModelMetrics(self, include=["summary", "all_parameters"]):

        if "summary" in include:
            #----model summary----
            print(f"model summary for: {self._model_name}")
            self._model.summary()
        
        if "history" in include:
            if self._history != {}:
                print(f"model accuracy & loss for: {self._model_name}")
                #pprint(self._history)
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

        if "all_parameters" in include:
            print(f"All parameters for model: {self._model_name}")
            pprint(self._all_parameters)



    def estimateImageOrientation(self, dataset, save_to_csv=True):

        # estimates the image orientation for specified dataset: "data" or "test" and saves to csv by dflt

        self.isModelSet()
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
                0: 'correct',            # from folder 
                1: 'rotated_180',          # from folder 
                2: 'rotated_90',          # from folder 
                3: 'rotated_minus_90'      # from folder 
            }

            estimated_index = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][estimated_index])
            orientation = index_to_orientation[estimated_index]

            return orientation, confidence


        # Step 3: Predict orientation for all images in the dataset
        df = pd.DataFrame(columns=["Image","Orientation","Confidence","Disease"])
        results = []
        dis = None
        estimationTime = 0
        start = time.time()
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
        
        # save estimation time
        end = time.time()
        estimationTime = end - start
        print(f"estimation time: {estimationTime:.2f} seconds")
        self._all_parameters["estimation_time"] = estimationTime

        # Sort 
        df_estimated = df.sort_values(by=["Orientation", "Disease", "Confidence"], ascending=True)
        self.checkEstimatedData(df_estimated)
        
        # 3: save to CSV and set the corresponding variable
        if save_to_csv:

            file_name = None

            if dataset == "test":
                file_name = f"estimated_orientation_{self._model_name}_{dataset}.csv"
                self.estimated_test_results = df_estimated

            elif dataset == "data":
                file_name = f"estimated_orientation_{self._model_name}.csv"
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
            file_name = f"estimated_orientation_{self._model_name}_{dataset}.csv"
        elif dataset == "data":
            file_name = f"estimated_orientation_{self._model_name}.csv"
        else:
            raise Exception(f"No such dataset: {dataset}!")
        
        file_path = os.path.join(self._model_dir, file_name)

        df_estimated = pd.read_csv(file_path)
        df_estimated = df_estimated.sort_values(by=sort_by).reset_index(drop=True)

        print("estimated results loaded from: ", file_path )
        
        return df_estimated



    def checkEstimatedData(self, df_estimated):

        df_rotated = df_estimated[df_estimated["Orientation"] != "correct"]
        df_correct = df_estimated[df_estimated["Orientation"] == "correct"]

        print(f"estimated as rotated: {df_rotated.shape[0]}")
        print(f"estimated as correct: {df_correct.shape[0]}")
        print(f"Estimates having confidence == 1: ", df_estimated[df_estimated["Confidence"] == 1].shape[0])
        grouped_counts = df_rotated.groupby(["Disease", "Orientation"]).size().reset_index(name="Count")
        print(grouped_counts)

        #print(f"checking for NaNs:")
        #print(df_estimated.isna().sum())


    # show rotated_90, rotated_minus_90, rotated_180 images only
    def showEstimatedImgOrientation(self, dataset, includeSummary=True, rotated_only=True, text_size=10, max_img_per_row=5):
        
        df_estimated = None
        dis = None
        ip = ImageProcessor()
        image_directories = []

        if dataset == "test":
            image_directories = [("test", self._test_dataset_dir)]
            print(self._test_dataset_desc)
        elif dataset == "data":
            for disease in dc:
                dis = disease.value
                image_directories.append((dis, os.path.join(self._data_dir, rf"{dis}\downscaled\{self._img_resolution[0]}x{self._img_resolution[1]}")))

        else:
            raise Exception(f"Dataset not defined: {dataset}")
        
        df_estimated = self.loadEstimatedData(dataset)
        df_estimated = df_estimated.sort_values(by=["Orientation", "Disease", "Confidence"], ascending=True)
        
        if includeSummary:
            self.checkEstimatedData(df_estimated)

        if rotated_only:
            orientations = df_estimated["Orientation"].unique()[1:] # skip "correct" orientation
            print("showing only rotated images")
        else:
            print("showing all images")    

        orient_imgs_names = []
        
        # collect imgs by orientation
        for o in orientations:

            gimgs, ginames = [], []

            for dirt in image_directories:

                imags_dir = dirt[1]
                dis = dirt[0]

                # select images having orientation o 
                i_names = df_estimated["Image"][(df_estimated["Orientation"] == o) & (df_estimated["Disease"]==dis)].to_list()
                cnt = len(i_names)

                if cnt > 0:
                    imgs, img_names = ip.loadImgs(i_names, imags_dir)
                    gimgs.append(imgs)
                    ginames.append(img_names)
                    
                else:
                    gimgs.append((None,))
                    ginames.append((None,))
                    
                if dis == "test":
                    break
            
            orient_imgs_names.append((o, gimgs, ginames))
            
        # plot images by orientation    
        for oin in orient_imgs_names:
            o, imgs, img_names = oin
            cnt = 0

            # flatten the list of lists into a single list
            imgs = [img for sublist in imgs for img in sublist if img is not None]
            img_names = [name for sublist in img_names for name in sublist if name is not None]
            cnt = len(imgs)

            if cnt > 0:
                
                print(f"estimated orientation: {o}", f"--> total estimated: {cnt}")
                ip.plot_images(imgs, img_names, tSize=text_size, max_img_per_row=max_img_per_row)

            else:
                print(f"--> estimated orientation: {o}", f"--> total estimated: 0")
        



    def saveAllParameters(self):
        # save all parameters to json file
        
        file_path = os.path.join(self._model_dir, f"all_parameters_{self._model_name}.json")
        
        # check if file already exists
        if os.path.exists(file_path):
            # ensure that already stored parameters won't be merged out
            with open(file_path, 'r') as f:
                loaded_params = json.load(f)
                # remove empty {} entries
                loaded_params = {k: v for k, v in loaded_params.items() if v}
                self._all_parameters = {k: v for k, v in self._all_parameters.items() if v}
                # merge with current parameters with priority to self._all_parameters current parameters
                self._all_parameters = {**loaded_params, **self._all_parameters}
            
            s = f"parameters updated in {file_path}"
        # initial save
        else:
            s = f"all parameters saved to {file_path}"

        with open(file_path, 'w') as f:
                json.dump(self._all_parameters, f, indent=4)
        
        print(s)
    


    def loadAllParameters(self):
        # load all parameters from json file
        if self._model is None:
            raise Exception("model is not loaded/created yet")
        
        file_path = os.path.join(self._model_dir, f"all_parameters_{self._model_name}.json")
        # check if file already exists
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self._all_parameters = json.load(f)

            #parse into variables: if empty {} entry is found, set to currently loaded value
            self._basic_params = self._all_parameters.setdefault("basic_params", self._basic_params)
            self._data_loading_params = self._all_parameters.setdefault("data_loading_params", self._data_loading_params)
            self._model_params = self._all_parameters.setdefault("model_params", self._model_params)
            self._history = self._all_parameters.setdefault("history", self._history)
        
            print(f"all parameters loaded from {file_path}")

        else:
            raise Exception(f"file not exists: {file_path}")
        



    def compareToAnotherModel(self, estim2, **kwargs):
        # compares the model with another model (estim2) and plots the results
        # params:
        # estim2: another estimator class holding model data to compare with

        if self._history != {} and estim2._history != {}:

            print(f"model accuracy, loss and learning rate: {self._model_name} VS {estim2._model_name}")
            # Plot accuracy and loss
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(self._history['accuracy'], label= f'{self._model_name} Accuracy')
            plt.plot(self._history['val_accuracy'], label= f'{self._model_name} Validation Accuracy')
            plt.plot(estim2._history['accuracy'], label= f'{estim2._model_name} Accuracy')
            plt.plot(estim2._history['val_accuracy'], label= f'{estim2._model_name} Validation Accuracy')
            plt.xlabel('Epochs')
            plt.legend()
            plt.title('Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(self._history['loss'], label= f'{self._model_name} Accuracy')
            plt.plot(self._history['val_loss'], label= f'{self._model_name} Validation Accuracy')
            plt.plot(estim2._history['loss'], label= f'{estim2._model_name} Accuracy')
            plt.plot(estim2._history['val_loss'], label= f'{estim2._model_name} Validation Accuracy')
            plt.xlabel('Epochs')
            plt.legend()
            plt.title('Loss')
            plt.show()


            plt.figure(figsize=(6, 4))
            plt.plot(self._history['learning_rate'], label=f'{self._model_name}')
            plt.plot(estim2._history['learning_rate'], label=f'{estim2._model_name}')
            plt.xlabel('Epochs')
            plt.title('Learnning Rate Comparison')
            plt.legend(loc='center', bbox_to_anchor=(0.5, 3.16e-1))

            plt.yticks([0, 0.00001, 0.0001, 0.001])
            plt.yscale('log')
            plt.tight_layout()
            plt.show()

        else:
            print("No history available for comparison")

        



        
            

class OrientationEstimatorMobileNet(_OrientationEstimatorBase):

    def __init__(self, **kwargs):
        super(OrientationEstimatorMobileNet, self).__init__(**kwargs)

        
    def getCompiledModel(self, **kwargs):
        # returns the compiled model ready for training
        # save params
        self._all_parameters["model_params"] = kwargs
        
        input_shape = kwargs.pop("input_shape", None)
        inputs = Input(shape=input_shape)
        
        # if no augmentation layer is specified, use the input tensor as input
        x = inputs

        exclude_augmentation_layer = kwargs.pop("exclude_augmentation", True)
        # Apply augmentations 
        if not exclude_augmentation_layer:
            x = RandomRotation(0.1)(inputs)                          
            x = RandomTranslation(height_factor=0.1, width_factor=0.1)(x) 
            x = RandomZoom(0.1)(x)
            print("augmentation layer applied")
        else:
            print("NO augmentation layer applied")
        
        
        
        
        # Load MobileNet without top classification layer
        base_model = MobileNet(
            input_shape=input_shape,  # explicitly set input shape
            input_tensor=x, 
            weights='imagenet',
            include_top=False,  # add own head
            name=self._model_name,
            # alpha=1.0,
            # depth_multiplier=1,
            # dropout=0.001,
            # pooling=None,
            # classes=1000,
            # classifier_activation="softmax",
        )

        # Freeze the base model layers
        layers_to_unfreeze = kwargs.pop("layers_to_unfreeze", 0)

        if layers_to_unfreeze > 0:
            # Unfreeze the last n layers of the model for fine-tuning
            for layer in base_model.layers[-layers_to_unfreeze:]:
                layer.trainable = True
        else:
            # Freeze base model for transfer learning (optional)
            base_model.trainable = False

        ### head1 + no augmentation

        bmo = base_model.output
        x = GlobalAveragePooling2D()(bmo)
        estimates = Dense(4, activation='softmax')(x) # 4 classes for orientations: 0°, 90°, 180°, 270° --> softmax

        # Custom classification head 
        # x = GlobalAveragePooling2D()(bmo)
        # x = Dense(128, activation='relu')(x)
        # estimates = Dense(4, activation='softmax')(x)  # 4 orientation classes

        self.compileModel(inputs, estimates, **kwargs)

        return self._model
    


class OrientationEstimatorResnet50(_OrientationEstimatorBase):

    def __init__(self, **kwargs):
        super(OrientationEstimatorResnet50, self).__init__(**kwargs)



    def getCompiledModel(self, **kwargs):
        # returns the compiled model ready for training
        # save params
        self._all_parameters["model_params"] = kwargs
        
        input_shape = kwargs.pop("input_shape", None)
        inputs = Input(shape=input_shape)
        
        # if no augmentation layer is specified, use the input tensor as input
        x = inputs

        exclude_augmentation_layer = kwargs.pop("exclude_augmentation", True)
        # Apply augmentations 
        if not exclude_augmentation_layer:
            x = RandomRotation(0.1)(inputs)                          
            x = RandomTranslation(height_factor=0.1, width_factor=0.1)(x) 
            x = RandomZoom(0.1)(x) 
        
        
        # Load MobileNet without top classification layer
        base_model = ResNet50(
            input_shape=input_shape,  # explicitly set input shape
            input_tensor=x, 
            weights='imagenet',
            include_top=False,  # add own head
            name=self._model_name,
            # alpha=1.0,
            # depth_multiplier=1,
            # dropout=0.001,
            # pooling=None,
            # classes=1000,
            # classifier_activation="softmax",
        )

        # Freeze the base model layers
        layers_to_unfreeze = kwargs.pop("layers_to_unfreeze", 0)

        if layers_to_unfreeze > 0:
            # Unfreeze the last n layers of the model for fine-tuning
            for layer in base_model.layers[-layers_to_unfreeze:]:
                layer.trainable = True
        else:
            # Freeze base model for transfer learning (optional)
            base_model.trainable = False

        bmo = base_model.output
        x = GlobalAveragePooling2D()(bmo)
        estimates = Dense(4, activation='softmax')(x) # 4 classes for orientations: 0°, 90°, 180°, 270° --> softmax

        # Custom classification head: for resnet50, mobnet, no_aug_layer
        # x = GlobalAveragePooling2D()(bmo)
        # x = Dense(128, activation='relu')(x)
        # estimates = Dense(4, activation='softmax')(x)  # 4 orientation classes

        self.compileModel(inputs, estimates, **kwargs)

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
