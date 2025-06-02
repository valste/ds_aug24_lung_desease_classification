# -*- coding: utf-8 -*-
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model

import src.preprocessing.image_preprocessor as preprocessor


def draw_image_histogram(image):
    """
    draw_image_histogram function takes an image as input
    and displays the histogram of the image.

    Input:
    image: np.array: Image as a numpy array
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # Plot the histogram
    plt.figure()
    plt.title("Grayscale Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.plot(hist_gray)
    plt.xlim([0, 256])
    plt.show()


def draw_image_countours(image_path, display_image=False):
    """
    draw_image_countours function takes an image path as input
    and displays the image with contours.

    Input:
    image_path: str: Path to the image file
    display_image: bool: If True, the image with contours will be displayed
    """
    image = preprocessor.crop_image(image_path, 10)
    blurred_image = cv2.GaussianBlur(image[0], (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

    if display_image:
        cv2.imshow("Contours", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def plot_images(images, labels):
    """
    plot_image function takes an image path as input and displays the image.

    Input:
    images : list : List of images
    labels : list : List of labels
    """
    # Read the image
    _, axs = plt.subplots(1, len(images), figsize=(7, 4))

    for i in range(len(images)):
        axs[i].imshow(images[i], cmap="gray")
        axs[i].set_title(labels[i])

    # Remove ticks from the subplots
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    # Display the subplots
    plt.tight_layout()
    plt.show()


def grad_cam(image, model, layer_name) -> tuple:
    """
    "grad_cam function takes an image, a model, and a layer name as input
    and returns the Grad-CAM heatmap and the predicted class.

    Input:
    image: np.array: Image as a numpy array
    model: tf.keras.Model: Trained Keras model
    layer_name: str: Name of the convolutional layer to visualize

    Output:
    heatmap: np.array: Grad-CAM heatmap
    predicted_class: int: Predicted class index
    """
    # Retrieve the convolutional layer
    layer = model.get_layer(layer_name)

    # Create a model that generates the outputs of the convolutional layer and the predictions
    grad_model = Model(inputs=model.input, outputs=[layer.output, model.output])

    # Add a batch dimension
    image = tf.expand_dims(image, axis=0)

    # Compute the gradients
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        predicted_class = tf.argmax(predictions[0])  # Predicted class
        loss = predictions[:, predicted_class]  # Loss for the predicted class

    # Gradients of the scores with respect to the outputs of the convolutional layer
    grads = tape.gradient(loss, conv_outputs)

    # Weighted average of the gradients for each channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the activations by the calculated gradients
    conv_outputs = conv_outputs[0]  # Remove the batch dimension
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0)  # Focus only on positive values
    heatmap /= tf.math.reduce_max(heatmap)  # Normalize between 0 and 1
    heatmap = heatmap.numpy()  # Convert to numpy array for visualization

    # Resize the heatmap to match the original image size
    heatmap_resized = tf.image.resize(
        heatmap[..., np.newaxis], (image.shape[1], image.shape[2])
    ).numpy()
    heatmap_resized = np.squeeze(
        heatmap_resized, axis=-1
    )  # Remove the singleton dimension at the end of the heatmap_resized array

    # Color the heatmap with a palette (e.g., "jet")
    heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3]  # Get the R, G, B channels
    superimposed_image = heatmap_colored * 0.7 + image[0].numpy() / 255.0

    return np.clip(superimposed_image, 0, 1), predicted_class


def show_grad_cam_cnn(
    images, model, class_names, labels, save_dir="", image_name="", save_image=False
):
    """
    show_grad_cam_cnn function takes a list of images, a model, and class names as input
    and displays the Grad-CAM heatmaps for each image.

    Input:
    images: list: List of images
    model: tf.keras.Model: Trained Keras model
    class_names: list: List of class names
    labels: list: List of labels
    """
    number_of_images = images.shape[0]
    conv_layers = [layer.name for layer in model.layers if isinstance(layer, Conv2D)]

    plt.figure(figsize=(30, 6 * len(conv_layers)))

    for j, layer in enumerate(conv_layers):

        for i in range(number_of_images):

            subplot_index = i + 1 + j * number_of_images
            plt.subplot(len(conv_layers), number_of_images, subplot_index)

            # Get the image with the overlaid heatmap
            grad_cam_image, predicted_class = grad_cam(images[i], model, layer)

            # Display the image with Grad-CAM
            plt.title(
                f"Layer: {layer}\nprediction: {class_names[predicted_class]}\nactual: {class_names[np.where(labels[i] == 1)[0][0]]}",
                fontsize=12,
            )
            plt.imshow(grad_cam_image)
            plt.axis("off")

    # Define folder and filename
    if save_image:
        os.makedirs(save_dir, exist_ok=True)  # Create folder if it doesn't exist
        file_path = os.path.join(save_dir, f"{image_name}.png")
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def get_predication_output(images, model, class_names, labels):
    """
    get_predication_output function takes a model and an image as input
    and returns the predicted class probabilities.

    Input:
    images: np.array: Image as a numpy array
    model: tf.keras.Model: Trained Keras model
    class_names: list: List of class names
    labels: np.array: Labels for the images

    Output:
    predictions: np.array: Predicted class probabilities
    """

    number_of_images = images.shape[0]

    df = pd.DataFrame(columns=["image", "confidence", "predicted_class", "actual_class"])

    for i in range(number_of_images):

        # Predict the class probabilities
        predictions = model.predict(np.expand_dims(images[i], axis=0))

        # Get the predicted class (index of maximum probability)
        predicted_class = np.argmax(predictions)  # Get the index of the predicted class

        # Get the confidence (probability) for the predicted class
        confidence = predictions[0][predicted_class]  # Confidence score for the predicted class
        confidence_percentage = confidence * 100  # Convert to percentage

        # Add the Grad-CAM image and label to the DataFrame
        df.loc[len(df)] = {
            "image": i,
            "confidence": confidence_percentage,
            "predicted_class": class_names[predicted_class],
            "actual_class": class_names[np.where(labels[i] == 1)[0][0]],
        }

    return df


def show_loss_accuracy_report(history):
    """
    show_accuracy_loss_report function takes a history object as input
    and displays the loss and accuracy over epochs.
    Input:
    history: tf.keras.callbacks.History: History object containing training history
    """
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy plot
    ax1.plot(history.history["accuracy"], label="Train Accuracy")
    ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()

    # Loss plot
    ax2.plot(history.history["loss"], label="Train Loss")
    ax2.plot(history.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.tight_layout()
    plt.show()


def show_confusion_matrix_report(model, val_data, n_channels=1):
    """
    show_confusion_matrix_report function takes a confusion matrix as input
    and displays the confusion matrix.
    Input:
    model: tf.keras.Model: Trained Keras model
    val_data: tf.data.Dataset: Validation data
    """
    class_names = val_data.class_names
    y_pred = model.predict(val_data).argmax(axis=1)
    if n_channels == 1:
        val_data = np.concatenate([labels.numpy() for _, labels in val_data])
        val_data = np.argmax(val_data, axis=1)
    else:
        val_data = val_data.get_class_labels()
    cm = confusion_matrix(val_data, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot()
    print(classification_report(val_data, y_pred, target_names=class_names))
