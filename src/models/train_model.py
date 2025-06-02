import os
from pathlib import Path
from tensorflow.keras.models import load_model
from src.preprocessing.image_augmentor import load_dataset_images
from src.models.build_model import train_advanced_supervised_model, evaluate_model
from src.visualization.visualize import show_loss_accuracy_report, show_confusion_matrix_report

# Go up N levels (e.g., 2 levels up from current file)
root_folder = Path(__file__).resolve().parents[2]
# Path to the raw data and preprocessed data
raw_data_dir = os.path.join(root_folder, "data", "raw", "dataset", "masked_images_dataset")
model_dir = os.path.join(root_folder, "models", "ds_cnn_model.keras")
# Resize images to IMG_SIZExIMG_SIZE pixels
IMG_SIZE = 256
# Batch size for training
batch_size = 16

def main():
    """
    Main function to train and evaluate the model.
    This function loads the dataset, trains the model,
    and evaluates it using various metrics.
    """
    # Load the dataset
    train_data, val_data, _ = load_dataset_images(raw_data_dir, (IMG_SIZE, IMG_SIZE), batch_size)

    # Now, check if there is any intersection (common files)
    overlap = set(train_data.file_paths).intersection(set(val_data.file_paths))

    # Print results
    if overlap:
        print(f"⚠️ Overlap found! {len(overlap)} overlapping files.")
        for file in list(overlap)[:5]:  # print first 5 overlaps
            print(file)
    else:
        print("✅ No overlap between training and validation datasets.")

    # Train the model
    cnn_model, cnn_history = train_advanced_supervised_model(train_data, val_data, IMG_SIZE, 300, 4, None, 
                                                         filter_layers=[32, 64, 128, 256, 512], conv2d_layers=4, dense_layers=[128, 32], 
                                                         attention=True, aspp=True, model_type='CNN', classification_type='categorical')
    
    # Evaluate the model
    cnn_train_loss, cnn_train_acc = cnn_history.history['loss'][-1], cnn_history.history['accuracy'][-1]
    print(f"Train Accuracy: {cnn_train_acc:.4f}, Train Loss: {cnn_train_loss:.4f}")
    
    # Save the model
    cnn_model.save(model_dir)

    # Show loss and accuracy report
    show_loss_accuracy_report(cnn_history)

    # Show confusion matrix report
    show_confusion_matrix_report(cnn_model, val_data)

if __name__ == "__main__":
    main()