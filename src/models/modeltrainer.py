
from mlflow.tracking import MlflowClient
import mlflow
import tensorflow as tf
from src.defs import ModelType as mt, ExperimentName as en
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, LearningRateScheduler, ModelCheckpoint, Callback
import time



class ModelTrainer():
    """
    A class to train models.
    """

    def __init__(self, model, dataset, mlruns_uri, run_name=None):
        """
        Initializes the ModelTrainer to be used for training the model passed.

        Args:
            model: The model to be trained.
            dataset: object containing the both dataset returned by image_dataset_from_directory.
            mlruns_uri: tracking uri as specified in src.defs.MLRUNS_URI
        """
        self.model = model
        self.dataset = dataset
        
        self.mlruns_uri = mlruns_uri
        self.experiment = None
        self.history = None
        self.run_name = run_name if run_name else self.model.name
        self.run = None
        self.training_time = None        
        
        
        # Setup MLflow tracking    
        mlflow.set_tracking_uri(mlruns_uri) #to be set first before experiment_name
        
        if mt.RESNET50 in self.model.name or mt.MOBILENET in self.model.name:
            self.experiment = mlflow.set_experiment(experiment_name = en.ORIENTATION_CLASSIFIER.value)
        elif mt.CAPSNET in self.model.name:
            self.experiment = mlflow.set_experiment(experiment_name = en.DESEASE_CLASSIFIER.value)
        else:
            raise Exception(f"Can't match any experiment name to model: {self.model.name}")
                
        
    
    
    def train(self, target_val_acc=.95, epochs=30, callbacks=[]):
        #----Start MLflow run
        mlflow.tensorflow.autolog()
        
        with mlflow.start_run(run_name=self.run_name) as run:

            self.run = run
            
            print(f"Experiment name/id: {self.experiment.name}/{self.experiment.experiment_id}")
            print(f"MLflow Run ID/name: {run.info.run_id}/{run.info.run_name}")

            if target_val_acc:
                stop_at_specified_val_acc = StopAtValAccuracy(target_val_acc)
                callbacks.append(stop_at_specified_val_acc)
                
            train_ds, val_ds = self.dataset
            
            start = time.time()
            history = self.model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks = callbacks)
            end = time.time()
            
            self.history = history
            self.training_time = end - start
        
        return self.history, self.model
    
    
    
    def train_with_fine_tuning(self, base_model, target_val_acc=1, warmupLR=1e-4, warmup_epochs=5, fine_tuneLR=1e-5, fine_tune_epochs=15, callbacks=[]):
        
        self.base_model = base_model
        
        # warmup training only custom classifier layer
        self.base_model.trainable = False
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=warmupLR),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.run_name = "warmup_"+self.run_name
        history_1, self.model = self.train(target_val_acc=target_val_acc, epochs=warmup_epochs, callbacks=callbacks)
        
        # finetuning with training model layers
        self.base_model.trainable = True
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tuneLR),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.run_name = self.run_name.replace("warmup", "finetune")
        history_2, self.model = self.train(target_val_acc=target_val_acc, epochs=fine_tune_epochs, callbacks=callbacks)
        
        # merge historie
        history = {
            "accuracy": history_1.history["accuracy"] + history_2.history["accuracy"],
            "val_accuracy": history_1.history["val_accuracy"] + history_2.history["val_accuracy"],
            "loss": history_1.history["loss"] + history_2.history["loss"],
            "val_loss": history_1.history["val_loss"] + history_2.history["val_loss"]
        }
        
        return self.model, history

    
   

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
