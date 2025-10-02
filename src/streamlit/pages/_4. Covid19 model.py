# -*- coding: utf-8 -*-
import streamlit as st
import io, os
import pandas as pd
from tensorflow.keras.models import load_model
from src.defs import PROJECT_DIR
from pathlib import Path

# Set the path to the model directory
model_dir = Path(PROJECT_DIR, "models",)
train_history_path = Path(PROJECT_DIR, "src", "streamlit", "models","covid19","training_history.json")
# report_data_path = "src/streamlit/data"
demo_data_path = Path(PROJECT_DIR, "src", "streamlit", "data")


st.set_page_config(page_title="Covid-19 🦠 Detection", page_icon="🦠", layout="wide")

st.title("Covid 19 model")
st.markdown(
    """
Although it is well known that using a model pre-trained on a very large dataset (millions of images) typically yields more accurate results for COVID-19 classification, we wanted to test our understanding of how CNN models work and explore the limitations faced when building a model from scratch.

In our project, we experimented with both approaches:

- Developing a custom CNN model from scratch.
- Applying transfer learning using a pre-trained model.

Overall, this experience has been highly rewarding and has significantly deepened our understanding of CNNs for tackling future challenges.
"""
)

model_path = os.path.join(
    PROJECT_DIR,
    "models",
    "ds-cxr-covid19",
    "CNN",
    "data",
    "model.keras"
)

try:
    model = load_model(model_path, compile=False)
except Exception as e:
    st.warning(f"""Processing failed: {e}
                Ensure the model files are available in the 
                `ds_aug24_lung_desease_classification\models` folder.
                Use the the links provided in the Models.md file to download them.
                """)
    

file_path=os.path.join(PROJECT_DIR, "models\ds-cxr-covid19\model_summary.txt")
# Read the file into a variable
with open(file_path, 'r', encoding='utf-8') as file:
    model_summary = file.read()


st.subheader("Model Summary")
with st.expander("Show Model Summary"):
    st.code(model_summary, language='text')

st.subheader("Loss function")
st.markdown(
    """
We tried training the model with categorical_crossentrop function in the beginning but that was not giving us a satisfying result especially for classifying the tricky cases found in the masked lung images. 
After some research, we switched to a custom function focal_loss which focuses more on hard examples and less on easy ones.
"""
)

st.latex(
    r"""
    \begin{align}
    FL(p_t) = -\alpha_t (1 - p_t)^\gamma * \log(p_t) \\
    \end{align}
"""
)

st.markdown(
    """
Where:
- $p_t$ = model predicted probability for the correct class.
- $\gamma$ = focusing parameter (>0).
- $\\alpha_t$ = balancing weight for positive/negative classes.

- When $p_t$ is small (model is wrong), $(1 - p_t)^{\gamma}$ is large, increasing loss. 
- When $p_t$ is large (model is confident/correct), $(1 - p_t)^{\gamma}$ shrinks the loss.
"""
)

st.subheader("Model training history")

cnn_history = pd.read_json(train_history_path, orient='records')
st.line_chart(cnn_history[['accuracy', 'val_accuracy']], use_container_width=True)
st.line_chart(cnn_history[['loss', 'val_loss']])
st.line_chart(cnn_history[['learning_rate']])

st.subheader("Model evaluation")
st.markdown(
    """
The results can be summarized as the following:

- COVID: Good precision and slightly lower recall. Our model is cautious predicting COVID (good, but misses some cases).
- Lung Opacity: Slightly lower precision (0.85), but high recall (0.88). Our model captures most Lung Opacity cases but some confusion happens.
- Normal: Balanced and high as the results were 91% precision, 93% recall, 92% F1. Our model is very good at identifying healthy lungs.
- Viral Pneumonia: near perfect prediction, but we know that this class had mode augmented images than the others
"""
)
cnn_model_cm = pd.read_csv(f"{demo_data_path}/cnn_model_cm.csv", index_col=0)
st.dataframe(cnn_model_cm)

imgs_dir = os.path.join(PROJECT_DIR, "src", "streamlit", "images")

st.subheader("Confusion Matrix")
st.image(os.path.join(imgs_dir, "cnn_cm.png"), caption="Confusion Matrix")

st.subheader("Grad-CAM visualization")
st.image(os.path.join(imgs_dir, "5.5_cnn.png"), caption="Grad-CAM visualization for CNN model")
