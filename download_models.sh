#!/bin/bash

#################################################################################
# downloads models from 3 different Hugging Face repos into the models/ directory
#################################################################################

# Exit on error
set -e

# Define target directory
MODEL_DIR="./models"

# Create models directory if not exists
mkdir -p "$MODEL_DIR"

# Array of Hugging Face model URLs
MODEL_URLS=(
  "https://huggingface.co/valste/orientation-classifier-224x224-aug-head1-resnet50"
  "https://huggingface.co/valste/orientation-classifier-224x224-aug-head1-mobnet"
  "https://huggingface.co/maja011235/lung-segmentation-gan"
  "https://huggingface.co/maja011235/lung-segmentation-unet"
  "https://huggingface.co/rehabaam/ds-cxr-covid19"
  "https://huggingface.co/valste/capsnet-4class-lung-disease-classifier"
)

echo "[INFO] Starting model downloads into $MODEL_DIR"

# Clone each model
for URL in "${MODEL_URLS[@]}"; do
    MODEL_NAME=$(basename "$URL")
    DEST="$MODEL_DIR/$MODEL_NAME"

    if [ -d "$DEST" ]; then
        echo "[SKIP] $MODEL_NAME already exists in $MODEL_DIR"
    else
        echo "[CLONE] Cloning $MODEL_NAME..."
        git clone "$URL" "$DEST"
    fi
done

echo "[DONE] All models are downloaded to $MODEL_DIR"
