"""
Provides utility class for data handling
"""

# --- Standard Library
import os
import urllib.parse
import shutil
from collections import defaultdict

# --- Third-Party Libraries
import cv2
import mlflow
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import preprocess_input as mobnet_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input
from keras.utils import image_dataset_from_directory

# --- Local Imports
from src.defs import ModelType as mt, DatasetType as dt


class DataHelper:
    """
    DataHelper class to handle model related data and datasets.

    """
    mlruns_uri = None
    #mlruns_client = None 
    
    @staticmethod
    def set_mlflow_tracking(mlruns_uri):
        # This sets the default tracking server or directory that MLflow uses globally across your script. 
        mlflow.set_tracking_uri(mlruns_uri) 
        DataHelper.mlruns_uri = mlruns_uri
        # This creates a low-level client object that directly interacts with the MLflow tracking server 
        # (or local directory). It offers fine-grained control
        # DataHelper.mlruns_client = MlflowClient(mlruns_uri) 
        return DataHelper.mlruns_uri
    
    
    @staticmethod
    def check_mlflow_tracking_set():
        if not DataHelper.mlruns_uri:
            raise Exception("mlruns_uri not set")
    
    
    @staticmethod
    def add_prefix_to_files_in_dir(file_dir, prefix, prefix_to_replace=None, file_types=[".png", ".jpeg", ".jpg"]):
        """
        Renames all files in the specified directory by adding a prefix,
        only for files matching the given extensions.

        Parameters:
        - file_dir (str): Path to the directory containing the files.
        - prefix (str): Prefix to prepend to each file name.
        - file_types (list): List of allowed file extensions to process (e.g., [".png", ".jpg"])
        """
        if not os.path.isdir(file_dir):
            raise ValueError(f"The path '{file_dir}' is not a valid directory.")

        file_names = os.listdir(file_dir)

        for filename in file_names:
            
            old_path = os.path.join(file_dir, filename)
            
            #replace prefix
            if prefix_to_replace:
                if prefix_to_replace in filename:
                    filename = filename.removeprefix(prefix_to_replace)
                else:
                    raise Exception(f"the {filename} has no such prefix {prefix_to_replace} to be replaced")

            # Skip non-files or files with disallowed extensions
            if not os.path.isfile(old_path) or not any(filename.lower().endswith(ext) for ext in file_types):
                continue

            new_filename = prefix + filename
            new_path = os.path.join(file_dir, new_filename)

            os.rename(old_path, new_path)
        
        print(f"{len(file_names)} renamed by adding prefix {prefix}")
        
        

    @staticmethod
    def rename_file(file_path, new_name):
        directory = os.path.dirname(file_path)
        new_path = os.path.join(directory, new_name)
        os.rename(file_path, new_path)
        return new_path



    @staticmethod
    def get_file_names(dataset_dir, extensions=('.png', ".jpeg", ".jpg", )):
        """
        returns all PNG filenames in the dataset directory.
        """
        filenames = []
        for root, dirs, files in os.walk(dataset_dir):
            for file in files:
                if file.lower().endswith(extensions):
                    filenames.append(file)

        return filenames
    
    
    
    @staticmethod
    def delete_non_png_files(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and not filename.lower().endswith(".png"):
                os.remove(file_path)
                
                
    
    @staticmethod            
    def copy_to_dir(from_dir, to_dir):
        os.makedirs(to_dir, exist_ok=True)  # Create target directory if it doesn't exist

        for filename in os.listdir(from_dir):
            src_path = os.path.join(from_dir, filename)
            dst_path = os.path.join(to_dir, filename)

            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)  # copy2 preserves metadata
                  
      
    
    
    @staticmethod
    def get_run_ids(experiment_id):
        """
         Retunrs all active runs (status = RUNNING or FINISHED)
         for an experiment_name
         
        Args:
            experiment_name : _description_

        Returns:
            run_ids [] 
        """
        DataHelper.check_mlflow_tracking_set()
        runs_df = mlflow.search_runs(
            experiment_ids=[experiment_id],
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY  # includes RUNNING + FINISHED
        )

        return runs_df["run_id"].tolist()
    
    
    
    @staticmethod
    def get_model_by_path(model_path, custom_objects={}):
        """
        Returns the model name for a given run ID.
        
        Parameters:
        - run_id: The run ID to search for.        
        Returns:
        - model: the model associated with the run ID.
        """
        
        DataHelper.check_mlflow_tracking_set()
        model = load_model(model_path, custom_objects=custom_objects)
        
        return model       
        
        
        

    @staticmethod
    def get_per_epoch_metrics_from_mlruns(experiment_ids, run_ids=[], metric_names=['accuracy', 'val_accuracy', 'val_loss', 'loss', 'learning_rate'], output_csv_path=None):
        """
        Extracts per-epoch metric logs from MLflow runs into a wide-form DataFrame.
        
        Returns:
        - pd.DataFrame: Columns = [run_id, run_name, experiment_id, experiment_name, epoch, <metrics...>]
        """
        DataHelper.check_mlflow_tracking_set()
        client = mlflow.tracking.MlflowClient()
        selected_runs = []

        # Load runs
        if run_ids:
            for run_id in run_ids:
                try:
                    run = client.get_run(run_id)
                    selected_runs.append(run)
                except Exception as e:
                    print(f"Could not fetch run {run_id}: {e}")
        else:
            if experiment_ids is None:
                experiment_ids = [exp.experiment_id for exp in client.search_experiments()]
            for exp_id in experiment_ids:
                try:
                    runs = client.search_runs(experiment_ids=[exp_id])
                    selected_runs.extend(runs)
                except Exception as e:
                    print(f"Could not fetch runs for experiment {exp_id}: {e}")

        all_rows = []

        for run in selected_runs:
            run_id = run.info.run_id
            run_name = run.data.tags.get("mlflow.runName", "")
            experiment_id = run.info.experiment_id
            experiment = client.get_experiment(experiment_id)
            experiment_name = experiment.name if experiment else ""

            # Collect all metric histories into: epoch → {metric_name: value}
            epoch_data = defaultdict(dict)

            for metric in metric_names:
                try:
                    history = client.get_metric_history(run_id, metric)
                    for rec in history:
                        epoch_data[rec.step][metric] = rec.value
                except Exception as e:
                    print(f"Metric {metric} missing or not logged for run {run_id}: {e}")

            for epoch, metrics in epoch_data.items():
                row = {
                    'run_id': run_id,
                    'run_name': run_name,
                    'experiment_id': experiment_id,
                    'experiment_name': experiment_name,
                    'epoch': epoch
                }
                # Add metrics for this epoch
                for metric in metric_names:
                    row[metric] = metrics.get(metric, None)
                all_rows.append(row)

        df = pd.DataFrame(all_rows)

        if output_csv_path:
            df.to_csv(output_csv_path, index=False)
            print(f"Per-epoch wide-format metrics saved to {output_csv_path}")

        return df
    
    

    @staticmethod
    def get_training_history_fig_from_csv(metric_csv_file_path, run_ids="all"):
        """
        Reads training metrics from a CSV file and generates a Plotly figure.

        Parameters:
        - metric_csv_file_path: Path to the CSV file containing training metrics.
        - run_ids: List of run IDs to filter by, or "all" for all runs.

        Returns:
        - Plotly figure object with training metrics in a 2x3 subplot layout.
        """
        
        import pandas as pd
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        import plotly.colors as pc
        
        # Load metrics
        metrics = pd.read_csv(metric_csv_file_path)

        # Metric names (5 individual + 1 for combined history view)
        metric_names = ['accuracy', 'val_accuracy', 'loss', 'val_loss', 'learning_rate']

        # Setup 2x3 subplots
        fig = make_subplots(
            rows=2,
            cols=3,
            specs=[
                [{}, {}, {}],             # Row 1: all separate
                [{}, {"colspan": 2}, None]  # Row 2: cols 2+3 merged
            ],
            subplot_titles=metric_names,
            shared_xaxes=True,
            vertical_spacing=0.15,
            horizontal_spacing=0.05
        )

        # Color map setup
        unique_run_ids = metrics['run_id'].unique()
        colors = pc.qualitative.Plotly
        color_map = {run_id: colors[i % len(colors)] for i, run_id in enumerate(unique_run_ids)}

        # Filter/group runs
        selected_run_ids = run_ids
        grouped = metrics.groupby(['run_id', 'run_name'])

        for idx, metric in enumerate(metric_names):
            row = idx // 3 + 1
            col = idx % 3 + 1

            for (rID, runName), group in grouped:
                if selected_run_ids != "all" and rID not in selected_run_ids:
                    continue
                if metric not in group.columns:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=group['epoch'],
                        y=group[metric],
                        mode='lines+markers',
                        name=f"{runName}",
                        legendgroup=rID,
                        showlegend=(idx == 0),
                        line=dict(color=color_map[rID])
                    ),
                    row=row,
                    col=col
                )

                # Log scale for learning rate
                if metric == "learning_rate":
                    fig.update_yaxes(
                        type="log",
                        tickvals=[1e-7, 1e-6, 1e-5, 1e-4, 1e-3],
                        tickformat=".0e",
                        #title="Learning Rate (log)",
                        row=row,
                        col=2
                    )

        # Final layout
        fig.update_layout(
            height=900,
            width=1400,
            #title_text="Training Metrics by Epoch and Run ID",
            xaxis_title="Epoch",
            legend=dict(
                orientation="h",     # horizontal layout
                yanchor="top",       # anchor the legend’s top
                y=-0.2,              # push it below the plot area
                xanchor="center",
                x=0.5,
                font=dict(
                    size=14  # change this to your desired font size
                )
            )

        )

        return fig


    @staticmethod
    def show_training_history_from_csv(metric_csv_file_path, run_ids="all"):
        DataHelper.get_training_history_fig_from_csv(metric_csv_file_path, run_ids).show()

       
    @staticmethod
    def get_preprocess_fn(model_type):
        """
        Args:
            model_type: resnet50/capsnet
        Returns:
            preprocess function: according to model type used
        """
        @tf.autograph.experimental.do_not_convert
        def preprocess(x, y=None):
            """
            Args:
                raw x : images_batch: A tensor of shape (32, height, width, 3) with dtype tf.float32.
                raw y : labels_batch: A tensor of shape (32, num_classes) with dtype tf.float32.
            Returns:
                preprocessed x : images_batch: A tensor of shape (32, height, width, 3) with dtype tf.float32.
                raw y : labels_batch: A tensor of shape (32, num_classes) with dtype tf.float32.
            """
            if model_type == mt.RESNET50:
                return resnet50_preprocess_input(x), y
            elif model_type == mt.MOBILENET:
                return mobnet_preprocess_input(x), y
            elif model_type == mt.CAPSNET:
                pass
            
            # Return based on whether y is passed (labeled or not)
            return (x, y) if y is not None else x
            
        return preprocess
    


    
    #------dataset loading and preprocessing for training------
    @staticmethod
    def load_training_dataset(model_type, dataset_type=dt.TRAIN, **kwargs):
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
        
        if dataset_type != dt.TRAIN:
            raise Exception(f"Use only with dataset types TRAIN. Given: {dataset_type}")

        # When loading data using image_dataset_from_directory() — labels are automatically integers --> sparse_categorical_crossentropy must be used.
        # But after resNet preprocessing: train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y)) the classes are one-hot
        # encoded -->  categorical_crossentropy must be used
        
        ds = image_dataset_from_directory(**kwargs)
        train_ds, val_ds = ds

        
        # Number of batches in the training dataset
        print("Number of batches in train_ds:", train_ds.cardinality().numpy())
        # Number of batches in the validation dataset
        print("Number of batches in val_ds:", val_ds.cardinality().numpy())

        # Preprocess train/validation data to allign with data the resNet50 was trained on--> ImageNet dataset
        # When you apply preprocess_input(image_tensor), it does:

        # 1. Converts RGB → BGR (channel order switch)
        #     ResNet50 was trained using images in BGR format (from original Caffe implementation).
        #     So it swaps the channels from [R, G, B] → [B, G, R].
        # 2. Subtracts the ImageNet mean pixel values (no scaling to [0, 1]!)
        #     It subtracts these values per channel:
        #         Blue: 103.939
        #         Green: 116.779
        #         Red: 123.68
        # This centers the data around zero, similar to how the original model was trained.

        # preprocess the input according the model used
        preprocess_fn = DataHelper.get_preprocess_fn(model_type)  
        train_ds = train_ds.map(preprocess_fn)
        val_ds = val_ds.map(preprocess_fn)
        
        # Optimize performance with prefetch
        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.prefetch(AUTOTUNE)
        val_ds = val_ds.prefetch(AUTOTUNE)

        return train_ds, val_ds
    
    
    @staticmethod
    def get_file_paths(file_dir, include_types=('.png', '.jpg', '.jpeg')):

        file_paths = []
        
        for dirpath, _, filenames in os.walk(file_dir):
            for file in filenames:
                if file.lower().endswith(include_types):
                    full_path = os.path.join(dirpath, file)
                    file_paths.append(full_path)
    
        return file_paths
    
    
    
    
    @staticmethod
    def get_file_urls(file_dir, include_types=('.png', '.jpg', '.jpeg')):
        """
        Converts a list of local file paths to file:// URL format.

        Args:
            file_paths (list of str): List of local file paths.

        Returns:
            list of str: Corresponding file:// URLs.
        """
        file_paths = DataHelper.get_file_paths(file_dir, include_types)
                
        urls = []
        for path in file_paths:
            # Normalize path and convert to absolute if needed
            abs_path = os.path.abspath(path)
            # Convert to file:// URL
            url = 'file://' + urllib.parse.quote(abs_path.replace("\\", "/"))
            urls.append(url)
        return urls

    
    