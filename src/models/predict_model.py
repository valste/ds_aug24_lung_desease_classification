import os
from pathlib import Path
from datetime import datetime
from tensorflow.keras.models import load_model
from src.preprocessing.image_augmentor import load_dataset_images
from src.visualization.visualize import show_grad_cam_cnn, get_predication_output


# Go up N levels (e.g., 2 levels up from current file)
root_folder = Path(__file__).resolve().parents[2]
# Path to the raw data, preprocessed data, model and store images
raw_data_dir = os.path.join(root_folder, "data", "raw", "dataset", "masked_images_dataset")
model_dir = os.path.join(root_folder, "models", "ds_crx_covid19.keras")
images_dir = os.path.join(root_folder, "reports","images", "grad_cam")
# Resize images to IMG_SIZExIMG_SIZE pixels
IMG_SIZE = 256
# Batch size for training
batch_size = 200

def main():
    """
    Main function to predict and evaluate the model.
    This function loads the dataset, tests the model,
    and evaluates it using various metrics.
    """
    # Load the dataset
    _, val_data, _ = load_dataset_images(raw_data_dir, (IMG_SIZE, IMG_SIZE), batch_size)

    # Load the model
    model = load_model(model_dir, compile=False)

    # Test the model
    val_iter = iter(val_data)
    class_names = val_data.class_names
    images, labels =next(val_iter)

    # get current time
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # Save the images
    show_grad_cam_cnn(images[:4], model, class_names, labels, save_dir=images_dir, image_name=f"cnn_{current_time}", save_image=True)
    
    # Get the prediction output
    print(get_predication_output(images[:100], model, class_names, labels).head(100))

if __name__ == "__main__":
    main()