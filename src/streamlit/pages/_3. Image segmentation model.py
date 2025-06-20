# -*- coding: utf-8 -*-
import os
import streamlit as st
import pandas as pd
from PIL import Image

# 🔧 TensorFlow-Imports
import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from src.defs import PROJECT_DIR
from pathlib import Path

# 🧠 Deaktiviere GPU-Nutzung (optional, falls OOM)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 📋 Streamlit-Seiteneinstellungen
st.set_page_config(page_title="Covid-19 🦠 Detection", page_icon="🦠", layout="wide")
st.title("🫁 Lung Segmentation Model")

# 📊 Modellvergleichstabelle
st.header("📊 Model Comparison")

data = {
    "Model": ["Model 1", "Model 2", "Model 3", "Model 3 (GAN)", "Model 3B1", "Model 3B2"],
    "Val Dice": [0.9874, 0.9822, 0.9873, 0.9824, 0.9603, 0.9594],
    "Val IoU": [0.9752, 0.9650, 0.9750, None, 0.9245, 0.9226],
    "Special Features": [
        "Classic U-Net",
        "U-Net with Dropout",
        "U-Net + ASPP, SE, Dilated Convs",
        "Adversarial Fine-Tuning",
        "GAN + Hard-Example Fine-Tuning",
        "Hard-Example Fine-Tuning without GAN"
    ]
}
df = pd.DataFrame(data)
st.dataframe(df)



# 🖼️ Bildvergleich: X-ray – Ground Truth – GAN Prediction
st.header(" Side-by-Side: X-ray vs. Ground Truth vs. GAN")

# 📁 Absoluter Pfad zum Bildordner
#base_path = r"C:\Users\majas\A COVID PROJEKT\github\giti\aug24_cds_int_analysis-of-covid-19-chest-x-rays\src\streamlit\images"
base_path = Path(PROJECT_DIR, "src","streamlit","images")

# 📄 Erwartete Dateien
required_files = [
    "COVID-2267.png",
    "original COVID-2267.png",
    "gan COVID-2267.png"
]

missing_files = [f for f in required_files if not os.path.exists(os.path.join(base_path, f))]

if missing_files:
    for f in missing_files:
        st.warning(f"⚠️ Datei fehlt: {f}")
else:
    try:
        xray = Image.open(os.path.join(base_path, "COVID-2267.png"))
        mask_gt = Image.open(os.path.join(base_path, "original COVID-2267.png"))
        mask_gan = Image.open(os.path.join(base_path, "gan COVID-2267.png"))

        col1, col2, col3 = st.columns(3)

        with col1:
            st.image(xray, caption=" X-ray", use_container_width=True)

        with col2:
            st.image(mask_gt, caption=" Ground Truth", use_container_width=True)

        with col3:
            st.image(mask_gan, caption=" GAN Prediction", use_container_width=True)

    except Exception as e:
        st.error(f"❌ Error loading images: {e}")

# 📈 Klassifikationsergebnisse
st.header("📈 Classification Performance")

classification_data = {
    "Mask Source": [
        "SegGAN (Model 3B)",
        "Ground-Truth Masks",
        "SegGAN (Model 3A)"
    ],
    "Train Loss": [0.1054, 0.1049, 0.1018],
    "Train Accuracy (%)": [96.46, 96.39, 96.40],
    "Val Loss": [0.1607, 0.1689, 0.1766],
    "Val Accuracy (%)": [94.08, 94.51, 93.79]
}

df_class = pd.DataFrame(classification_data)
st.dataframe(df_class)
