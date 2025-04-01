
# The module provides tools for classifying the orientation of the x-ray images among 4 orientation classes: 
# 0: 'rotated_90',        
# 1: 'rotated_minus_90',  
# 2: 'correct',           
# 3: 'rotated_180'        


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
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image 
from tensorflow.keras.applications import ResNet50 # functional/ 50 layers/ requires images to be 224x224 in RGB 
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback


# Imports for building the model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten

# Imports for image transformations
#from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import RandomZoom
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.layers import RandomTranslation





class OrientationEstimatorResnet50():

    def __init__(self, model_name):
        
        # 1. Model prefix and directories for storing results and test data
        self.model_prefix = model_name
        self.model_dir = rf"C:\Users\User\DataScience\area51\models\resnet\{model_name}"
        self.data_dir = r"C:\Users\User\DataScience\area51\data\COVID-19_Radiography_Dataset"
        self.train_valid_dataset_dir    =   r"C:\Users\User\DataScience\area51\data_224x224\train_val_224x224"

        self.test_dataset_dir           =   r"C:\Users\User\DataScience\area51\data_224x224\test_224x224"
        self.test_dataset_desc = rf'''
                                        test dataset dir: {self.test_dataset_dir}

                                        dataset contains images:

                                            # rotation    :       image count
                                            # 0°          :       12
                                            # +90         :       13
                                            # -90         :       13
                                            # 180°        :       14
                                        '''

        # 2. parameters for image_dataset_from_directory
        self.__param_set = {
            "directory" : self.train_valid_dataset_dir,
            "batch_size" : 64,
            "seed": 1,
            "label_mode": "categorical",
            "color_mode": "rgb",        # resNet requires 3 channel otherwise needs to be trained from scratch 
            "image_size": (224, 224),   # img_height, img_width
            "shuffle": True,
            "validation_split": 0.2,
            "labels": "inferred"        #from folder structure
        }

        # 3. Compilation parameters
        self.__compiling_params = {
            "optimizer": 'adam',
            "loss": "categorical_crossentropy", #loss function
            "metrics": ['accuracy']
        }

        # 4. class variables
        self.train_ds, self.val_ds = None, None
        self.history = None
        self.model = None
        self.estimated_data_results = None
        self.estimated_test_results  = None




    def prepareTrainValidData(self):
        """
        1. Real-time Data Loading with image_dataset_from_directory

            * Reads images directly from folders in batches.
            * Useful when datasets don't fit entirely into memory.
            * Automatically assigns labels based on folder structure.
            * process parallelization
            
        train_val_224x224/
        ├── 224x224_rotated_0/
        │   ├── image1.png
        │   ├── image2.png
        │   └── ...
        └── 224x224_rotated_90/
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
        self.train_ds = image_dataset_from_directory(subset="training",  **self.__param_set)
        self.val_ds = image_dataset_from_directory(subset="validation", **self.__param_set)

        # Number of batches in the training dataset
        print("Number of batches in train_ds:", self.train_ds.cardinality().numpy())
        # Number of batches in the validation dataset
        print("Number of batches in val_ds:", self.val_ds.cardinality().numpy())

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

        self.train_ds = self.train_ds.map(preprocess)
        self.val_ds = self.val_ds.map(preprocess)
       
        return self.train_ds, self.val_ds
    


    def checkTrainValDataStructure(self):

        for images, labels in self.val_ds.take(1):
            print("encoded label samples: ", labels.numpy()[:4])
            print("images.shape: ", images.shape)

        
    

    def getCompiledModel(self):

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
        #x = RandomFlip("horizontal")(x)

        # add custom classification block
        x = base_model.output
        x = Flatten()(x)
        predictions = Dense(4, activation='softmax')(x) # 4 classes for orientations: 0°, 90°, 180°, 270° --> softmax

        self.model = Model(inputs=inputs, outputs=predictions)
        self.model.compile(**self.__compiling_params)

        print("model has been compiled successfully")

        return self.model


    
    def train(self, target_val_acc, epochs = 10):

        stop_at_specified_val_acc = StopAtValAccuracy(target_val_acc)
        self.history = self.model.fit(epochs=epochs, train_data = self.train_ds, validation_data = self.val_ds, callbacks = [stop_at_specified_val_acc]) 

        return self.history



    def storeTrainedModel(self):

        # Save the model
        model_file = f'orientation_classifier_{self.model_prefix}.keras'
        model_path = os.path.join(self.model_dir, model_file)
        self.model.save(model_path)

        # save param set for data loading
        param_file = f'param_set_{self.model_prefix}.json'
        param_path = os.path.join(self.model_dir, param_file)
        with open(param_path, 'w') as f:
            json.dump(self.__param_set, f)

        # Save training history: accuracy and loss for each epoch
        history_file = f'training_history_{self.model_prefix}.json'
        history_path = os.path.join(self.model_dir, history_file)
        with open(history_path, 'w') as f:
            json.dump(self.history, f)

        print(model_file, param_file, history_file, "stored to ", self.model_dir)

       

    def loadTrainedModel(self, include=["history", "param_set"]):

        # loading model
        model_file = f'orientation_classifier_{self.model_prefix}.keras'
        model_path = os.path.join(self.model_dir, model_file)
        self.model = load_model(model_path)

        # model training history
        if "history" in include:
            history_file = f'training_history_{self.model_prefix}.json'
            history_path = os.path.join(self.model_dir, history_file)
            if os.path.getsize(history_path) == 0:
                print(f"{history_path} is empty!")
            else:
                with open(history_path, 'r') as f:
                    self.history = json.load(f)

        # loading params image_dataset_from_directory
        if "param_set" in include:
            param_file = f'param_set_{self.model_prefix}.json'
            param_path = os.path.join(self.model_dir, param_file)
            if os.path.getsize(param_path) == 0:
                print(f"{param_path} is empty!")
            else:    
                with open(param_path, 'r') as f:
                    self.__param_set = json.load(f)

        print(f"model loaded from {model_path}")
        print(f"model train history loaded from {history_path}")
        print(f"parameter set for image_dataset_from_directory loaded from {param_path}")

        return self.model, self.__param_set, self.history



    def plotModelMetrics(self, include=["history", "param_set", "summary"]):

        if "summary" in include:
            #----model summary----
            print(f"model summary for: {self.model_prefix}")
            self.model.summary()
        
        if "history" in include:
            print(f"model accuracy & loss for: {self.model_prefix}")
            pprint(self.history)
            # Plot accuracy and loss
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            plt.plot(self.history['accuracy'], label='Accuracy')
            plt.plot(self.history['val_accuracy'], label='Validation Accuracy')
            plt.legend()
            plt.title('Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(self.history['loss'], label='Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
            plt.legend()
            plt.title('Loss')

            plt.show()

        if "param_set" in include:
            print(f"image_dataset_from_directory parameters for: {self.model_prefix}")
            pprint(self.__param_set)



    def estimateImageOrientation(self, dataset, save_to_csv=True):

        # estimates the image orientation for specified dataset: "data" or "test"
        # saves to csv

        # Step 1: Define the estimator function
        def estimate_orientation(img_path, model):
            
            img = image.load_img(img_path, target_size=(224, 224))          # loads in RGB by default, ResNet50 expects 224x224 in RGB --> Shape: (224, 224, 3)
            img_array = image.img_to_array(img)                             # Convert to array
            img_array = np.expand_dims(img_array, axis=0)                   # Shape becomes (1, 224, 224, 3)
            img_array = preprocess_input(img_array)                         # 🔥 Preprocess -->ready for prediction

            # Get predictions
            if model:
                prediction = model.predict(img_array)  # Shape: (1, 4)
            else:
                print("model is not loaded/created yet")

            # Class index (one-hot vector [0, 0, 1, 0]) mapping from train_generator.class_indices
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
                image_dir = self.test_dataset_dir
                dis = "test_data"

            elif dataset == "data":
                image_dir = os.path.join(self.data_dir, rf"{disease.value}\downscaled\224x224")
                dis = disease.value

            else:
                raise Exception(f"Wrong dataset: {dataset}")        
            
            for img_name in os.listdir(image_dir):
                img_path = os.path.join(image_dir, img_name)
                orientation, confidence = estimate_orientation(img_path, self.model)
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
                file_name = f"estimated_orientation_{self.model_prefix}_{dataset}.csv"
                self.estimated_test_results = df_estimated

            elif dataset == "data":
                file_name = f"estimated_orientation_{self.model_prefix}.csv"
                self.estimated_data_results = df_estimated

            else:
                 raise Exception(f"No such dataset: {dataset}")

            output_csv = os.path.join(self.model_dir, file_name)
            df_estimated.to_csv(output_csv, index=False)
            print("estimated results saved to: ", output_csv)

        return df_estimated



    def loadEstimatedData(self, dataset, sort_by=["Disease", "Orientation", "Confidence"]):
        
        file_name = None

        if dataset == "test":
            file_name = f"estimated_orientation_{self.model_prefix}_{dataset}.csv"
        elif dataset == "data":
            file_name = f"estimated_orientation_{self.model_prefix}.csv"
        else:
            raise Exception(f"No such dataset: {dataset}!")

        df_estimated = pd.read_csv(os.path.join(self.model_dir, file_name))
        df_estimated = df_estimated.sort_values(by=sort_by).reset_index(drop=True)

        print("estimated results loaded from: ", file_name)
        
        return df_estimated



    def checkEstimatedData(self, df_estimated):

        print(f"\ntotal rotated found: {df_estimated.shape[0]}\n")
        grouped_counts = df_estimated.groupby(["Disease", "Orientation"]).size().reset_index(name="Count")
        print(grouped_counts)

        print(f"\nchecking for NaNs:")
        print(df_estimated.isna().sum())
        print(f"\nEstimates with Confidence < 1: ", df_estimated[df_estimated["Confidence"] < 1].shape[0])
        print()  # 19 images
        


    # show rotated_90, rotated_minus_90, rotated_180 images only
    def showRotatedImg(self, dataset):
        
        df_estimated = None
        dis = None
        ip = ImageProcessor()

        for disease in dc:

            if dataset == "test":
                image_dir = self.test_dataset_dir
                dis = "test_data"
                print("This are the images from the test dataset!")
                print(self.test_dataset_desc)

            elif dataset == "data":
                dis = disease.value
                image_dir = image_dir = os.path.join(self.data_dir, rf"{dis}\downscaled\224x224")

            else:
                raise Exception(f"Wrong dataset: {dataset}")
            
            df_estimated = self.loadEstimatedData(dataset)


            for rot in ("rotated_90", "rotated_minus_90", "rotated_180"):
                            
                i_names = df_estimated["Image"][(df_estimated["Disease"]==dis) & (df_estimated["Orientation"] == rot) ].to_list()
                cnt = len(i_names)
                
                if len(i_names)>0:
                    imgs, img_names = ip.loadImgs(i_names, image_dir)
                    print("")
                    print(disease.value, f"--> estimated orientation: {rot}", f"--> total detected: {len(i_names)}")
                    ip.plot_images(imgs, img_names, tSize=15, max_img_per_row=5)
                else:
                    print("no rotated images")

            if dataset == "test":
                    break  
            
        return df_estimated



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
                self.model.stop_training = True
