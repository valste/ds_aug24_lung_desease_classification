# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    RandomContrast,
    RandomFlip,
    RandomRotation,
    RandomTranslation,
    RandomZoom,
    Rescaling,
    Reshape,
    Resizing,
    UpSampling2D,
    multiply,
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

from src.common.mlflow_manager import log_model


def train_basic_supervised_model(X_train, y_train, model_type="Logistic Regression") -> object:
    """
    train_basic_supervised_model Trains a model on the data

    Input:
    X_train: np.array: Features
    y_train: np.array: Labels
    model_type: str: Type of model to train

    Output:
    model: model: Trained model
    """

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    print(f"Computed Class Weights:{class_weight_dict} labels: {np.unique(y_train)}")

    match model_type:
        case "Logistic Regression":
            model = LogisticRegression(
                C=0.1,
                class_weight=class_weight_dict,
                max_iter=100,
                penalty="l1",
                solver="liblinear",
            )
        case "Linear Regression":
            sample_weights = np.array([class_weight_dict[label] for label in y_train])
            model = LinearRegression()
            return model.fit(X_train, y_train, sample_weight=sample_weights)
        case "SVM Linear":
            model = SVC(kernel="linear", class_weight=class_weight_dict, probability=True)
        case "SVM RBF":
            model = SVC(kernel="rbf", class_weight=class_weight_dict)
        case "Random Forest":
            model = RandomForestClassifier(
                n_estimators=500,
                max_depth=20,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features="sqrt",
                class_weight=class_weight_dict,
            )
        case "CatBoost":
            model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                class_weights=class_weight_dict,
                loss_function="Logloss",
                verbose=100,
            )
        case "CatBoost_Multi":
            model = CatBoostClassifier(
                iterations=500,
                depth=6,
                learning_rate=0.05,
                class_weights=class_weight_dict,
                loss_function="MultiClass",
                verbose=100,
            )
        case _:
            raise ValueError("Invalid model type")
    return model.fit(X_train, y_train)


def train_advanced_supervised_model(
    X_train,
    y_train,
    image_size,
    epochs,
    num_classes,
    class_weight,
    n_channels=1,
    filter_layers=[32, 64, 128, 256, 512],
    conv2d_layers=4,
    dense_layers=[256, 128, 64, 32],
    augmentation=False,
    attention=False,
    aspp=False,
    model_type="CNN",
    classification_type="binary",
) -> tuple:
    """
    train_advanced_supervised_model Trains a model on the data

    Input:
    X_train: np.array: Features
    y_train: np.array: Labels
    image_size: int: Size of the image
    epochs: int: Number of epochs to train
    num_classes: int: Number of classes
    class_weight: dict: Class weights for the model
    n_channels: int: Number of channels in the image
    filter_layers: list: List of filters for the convolutional layers
    conv2d_layers: int: Number of convolutional layers
    dense_layers: list: List of filters for the dense layers
    augmentation: bool: Whether to apply augmentation
    attention: bool: Whether to apply attention
    aspp: bool: Whether to apply ASPP
    model_type: str: Type of model to train
    classification_type: str: Type of the classification [binary or categorical]

    Output:
    model: model: Trained model
    history: history: Training history
    """

    # Set activation and loss based on class mode
    if classification_type == "binary":
        activation = "sigmoid"
        loss = "binary_crossentropy"
    else:
        activation = "softmax"
        loss = focal_loss()
        # loss = "categorical_crossentropy"

    match model_type:
        case "CNN":
            # 🔹 Input layer Block
            inputs = Input(shape=(image_size, image_size, n_channels))
            x = inputs
            if augmentation:
                # Apply augmentation
                x = RandomTranslation(height_factor=0.2, width_factor=0.2)(x)
                x = RandomZoom(height_factor=0.2, width_factor=0.2)(x)
                x = RandomFlip("horizontal")(x)
                x = RandomRotation(0.2)(x)
                x = RandomContrast(0.2)(x)
            x = Rescaling(1.0 / 255)(x)
            x = Resizing(128, 128)(x)

            # 🔹 Convolutional Block (4 layers + max pooling in the 4th layer)
            filter_list = filter_layers
            for filters in filter_list:
                for _ in range(conv2d_layers):
                    x = conv_block(x, filters)
                x = MaxPooling2D(pool_size=(2, 2))(x)
                x = Dropout(0.2)(x)

            # 🔹 ASPP Block
            if aspp:
                x = aspp_block(x, filters=128)

            # 🔹 Attention Block
            if attention:
                x = se_block(x)

            x = GlobalAveragePooling2D()(x)
            # 🔹 Fully Connected Block (4 layers)
            dense_list = dense_layers
            for dense in dense_list:
                x = Dense(dense, activation="relu")(x)
                x = Dropout(0.2)(x)

            outputs = Dense(num_classes, activation=activation)(x)

            model = Model(inputs=inputs, outputs=outputs)

            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss=loss,
                metrics=["accuracy"],
            )

            # Model Summary
            model.summary()
        case "Transfer Learning":
            base_model = DenseNet121(
                weights="imagenet",
                include_top=False,
                input_shape=(image_size, image_size, 3),
            )
            # Freeze pre-trained layers to retain learned features
            base_model.trainable = False

            # Extract deep features
            x = base_model.output
            x = GlobalAveragePooling2D()(x)

            output = Dense(num_classes, activation=activation)(x)  # Binary classification

            # Define the final model
            model = Model(inputs=base_model.input, outputs=output)

            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss=loss,
                metrics=["accuracy"],
            )

            # Model Summary
            model.summary()
        case _:
            raise ValueError("Invalid model type")

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=1e-9, verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True, verbose=1
    )
    return model, model.fit(
        X_train,
        validation_data=y_train,
        epochs=epochs,
        batch_size=16,
        class_weight=class_weight,
        callbacks=[reduce_lr, early_stop],
    )


def evaluate_model(
    description,
    model,
    X_test,
    y_test,
    n_channels=1,
    model_type="Logistic Regression",
    classification_type="binary",
    parameters=None,
    balanced=False,
    history=None,
) -> tuple:
    """
    evaluate_model Evaluates a model on the data

    Input:
    description: str: Description of the dataset
    model: model: Trained model
    X_test: np.array: Features
    y_test: np.array: Labels
    n_channels: int: Number of channels in the image
    model_type: str: Type of model to train
    classification_type: str: Type of the classification
    parameters: dict: Parameters for the model
    balanced: bool: Whether to use balanced class weights
    history: history: Training history

    Output:
    accuracy: float: Accuracy of the model
    classification_report: str: Classification report
    """
    if balanced:
        average = "macro"
    else:
        average = "weighted"

    match model_type:
        case "Logistic Regression":
            y_pred = model.predict(X_test)
        case "Linear Regression":
            y_pred_binary = model.predict(X_test)
            y_pred = (y_pred_binary > 0.5).astype(int)  # Threshold at 0.5
        case "SVM Linear" | "SVM RBF":
            y_pred = model.predict(X_test)
        case "Random Forest":
            y_pred = model.predict(X_test)
        case "CatBoost":
            y_pred = model.predict(X_test)
        case "CNN" | "Transfer Learning":
            loss, accuracy = model.evaluate(X_test)
            y_pred = model.predict(X_test)
            y_pred = np.argmax(y_pred, axis=-1)

            y_true = np.concatenate([labels.numpy() for _, labels in X_test])
            y_true = np.argmax(y_true, axis=-1)

            f1 = (
                f1_score(y_true, y_pred, average=average)
                if n_channels == 1
                else f1_score(X_test.get_class_labels(), y_pred, average=average)
            )
            precision = (
                precision_score(y_true, y_pred, average=average)
                if n_channels == 1
                else precision_score(X_test.get_class_labels(), y_pred, average=average)
            )
            recall = (
                recall_score(y_true, y_pred, average=average)
                if n_channels == 1
                else recall_score(X_test.get_class_labels(), y_pred, average=average)
            )

            metrics = {
                "loss": loss,
                "accuracy": accuracy,
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
            }
            # Fetting validation data
            if n_channels == 1:
                images = np.concatenate([images.numpy() for images, _ in X_test])
            else:
                for x, _ in X_test:
                    images = x
                    break

            log_model(
                "Advanced Supervised Models",
                "tensorflow",
                description,
                model,
                model_type,
                images,
                metrics,
                classification_type,
                parameters,
                history,
            )
            return loss, accuracy
        case _:
            raise ValueError("Invalid model type")

    # Log the model
    accuracy = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "mse": mse,
        "rmse": np.sqrt(mse),
        "r2": r2_score(y_test, y_pred),
        "accuracy": accuracy,
        "f1_score": f1_score(y_test, y_pred, average=average),
        "precision": precision_score(y_test, y_pred, average=average),
        "recall": recall_score(y_test, y_pred, average=average),
    }

    log_model(
        "Basic Supervised Models",
        "sklearn",
        description,
        model,
        model_type,
        X_test,
        metrics,
        classification_type,
        parameters,
        history=None,
    )

    return accuracy, classification_report(y_test, y_pred)


# Define a block: Conv -> BN -> ReLU
def conv_block(x, filters, kernel_size=3) -> tf.Tensor:
    """Convolutional Block."""
    x = Conv2D(filters, kernel_size, activation="relu", padding="same")(x)
    x = BatchNormalization()(x)
    return x


# Attention Block - Squeeze-and-Excitation Attention Block
def se_block(input_tensor, reduction=16) -> tf.Tensor:
    """Squeeze-and-Excitation (SE) Attention Block."""
    channels = input_tensor.shape[-1]

    # Squeeze
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(channels // reduction, activation="relu")(se)
    se = Dense(channels, activation="sigmoid")(se)
    se = Reshape((1, 1, channels))(se)

    # Scale
    x = multiply([input_tensor, se])
    return x


# ASPP Block
def aspp_block(x, filters=256) -> tf.Tensor:
    """Atrous Spatial Pyramid Pooling (ASPP) Block."""
    shape = x.shape
    y1 = Conv2D(filters, (1, 1), padding="same", activation="relu")(x)
    y2 = Conv2D(filters, (3, 3), dilation_rate=2, padding="same", activation="relu")(x)
    y3 = Conv2D(filters, (3, 3), dilation_rate=4, padding="same", activation="relu")(x)
    y4 = Conv2D(filters, (3, 3), dilation_rate=6, padding="same", activation="relu")(x)
    y5 = GlobalAveragePooling2D()(x)
    y5 = Reshape((1, 1, shape[-1]))(y5)
    y5 = Conv2D(filters, (1, 1), padding="same", activation="relu")(y5)
    y5 = UpSampling2D(size=(shape[1], shape[2]), interpolation="bilinear")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv2D(filters, (1, 1), padding="same", activation="relu")(y)
    return y

# Focal Loss Function
def focal_loss(gamma=2.0, alpha=0.25) -> callable:
    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        cce = CategoricalCrossentropy()
        bce = cce(y_true, y_pred)
        bce_exp = tf.exp(-bce)
        focal_loss = alpha * (1 - bce_exp) ** gamma * bce
        return focal_loss

    return loss_fn
