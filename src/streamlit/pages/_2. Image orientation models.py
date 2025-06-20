# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import sys
import os
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# sys.path.append(project_root)
# #----setting data paths----
# from src import defs
# defs.initDataPaths(project_root)
#defs.checkDataPaths()

from src.defs import PROJECT_DIR, ModelType
from src.utils.datahelper import DataHelper as dh
from src.utils.imgprocessing import ImageProcessor as ip
from src.defs import ModelType as mt, orientation_labels
from src.models.modelutilizer import ModelUtilizer
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib import pyplot as plt    
import numpy as np

st.set_page_config(page_title="Covid-19 🦠 Detection", page_icon="🦠", layout="wide")

st.title("Image orientation models")
st.markdown(
f"""
The purpose of these models is to identify all rotated images in the original dataset. 
We use two different pretrained model architectures — MobileNet and ResNet50 — to improve the overall
accuracy of orientation classification by combining their results. 
The class **'rotated_0'** represents the correct orientation, while the other three classes **'rotated_180', 'rotated_90', 'rotated_minus_90'** correspond to rotated images.

### Training dataset properties (original dataset)
- image properties: 224x224, 3 channels gray-scaled
- balanced by augmentation
- number of images each class: 6012

    
### Test dataset properties (unknown to the models)
- image properties: 224x224, 3 channels gray-scaled
- balanced by augmentation
- number of images each class: 2935

    

"""
)

st.markdown(
    """
### Complexity of the models (augmentation layer + base model + classification header) in terms of parameters

| Model     | Trainable | Total      |
|-----------|-----------|------------|
| ResNet50  | 8,196     | 23,595,908 |
| MobileNet | 4,100     |  3,232,964 |
"""
)

#---summaries of the models---
# def get_model_summary(model_type):

#     resnet_summary_txt_path = os.path.join(PROJECT_DIR, "models", "summary_orientation_classifier_224x224_aug_head1_resnet50.txt")
#     mobnet_summary_txt_path = os.path.join(PROJECT_DIR, "models", "summary_orientation_classifier_224x224_aug_head1_mobnet.txt")

#     summary_str = None
#     # show summaries from text files
#     if model_type == ModelType.RESNET50:
#         with open(resnet_summary_txt_path, "r", encoding="utf-8") as f:
#             summary_str = f.read()
#     elif model_type == ModelType.MOBILENET:
#         with open(mobnet_summary_txt_path, "r", encoding="utf-8") as f:
#             summary_str = f.read()
            
#     return summary_str if summary_str else "Model summary not found."


st.subheader("Model Architectures")
# used netron graphics instead of standard model.summary prints
resnet_netron_SVG = os.path.join(PROJECT_DIR, "src", "streamlit", "images", "model_architecture", "resnet50.svg")
mobnet_netron_SVG = os.path.join(PROJECT_DIR, "src", "streamlit", "images", "model_architecture", "mobnet_with_attributes_shown.svg")

col1, col2 = st.columns(2)
mobnet, resnet = ("MobileNet", "ResNet50")

for col, f, n in zip((col1, col2), (mobnet_netron_SVG, resnet_netron_SVG), (mobnet, resnet)):
    # Load SVG content from file
    with open(f, "r", encoding="utf-8") as f:
        svg_content = f.read()

    with col:
        # Embed inside a collapsible section
        with st.expander(f"{n}"):
            st.markdown(
            f"""
            <div style='
                display: flex; 
                justify-content: center; 
                transform: scale(1.5); 
                transform-origin: top center;
            '>
                {svg_content}
            </div>
            """,
            unsafe_allow_html=True
            )   
            

#----training history----
st.subheader("Training history")
csv_file_path = os.path.join(PROJECT_DIR, "csv_files", "image_orientation", f"traning_history_orientation_classifier_merged.csv")  
st.plotly_chart(dh.get_training_history_fig_from_csv(csv_file_path), use_container_width=True)


#-----classification report and confusion matrix on unknown dataset-----
st.header("Test results on balanced dataset unknown to the models")

def display_classrep_and_cmatrixcm(st_module, formatted=True, normalize_cm = "all"):
    st = st_module
    # Create two columns: one for each model
    col1, col2 = st.columns(2)
    # Normalization type for confusion matrix
    for col, t, n in zip((col1, col2), (mt.MOBILENET, mt.RESNET50), (mobnet, resnet)):
        with col:
            st.markdown(f"#### Model: {n}")

            # Load data
            csv_path = os.path.join(PROJECT_DIR, "src", "streamlit", "data",
                                    f"test_on_unknown_{t.value}_predicted_orientations.csv")
            df = pd.read_csv(csv_path)

            # Get classification report and confusion matrix
            clsreport, cm = ModelUtilizer.get_classreport_and_confusion_matrix(
                df, 
                output_dict=True,
                normalize_cm=normalize_cm,
                labels=orientation_labels["long"]
            )

            # Create DataFrame
            report_df = pd.DataFrame(clsreport).transpose().round(3)

            
            if formatted:
                def smart_format(x):
                    if pd.isna(x):
                        return ""
                    if isinstance(x, float):
                        return f"{x:.2f}".rstrip("0").rstrip(".")
                    return x

                 # Highlight accuracy row
                def highlight_accuracy_row(row):
                    return ['background-color: #1a1a5a'] * len(row) if row.name == "accuracy" else [''] * len(row)

                styled_df = report_df.style \
                    .format(smart_format) \
                    .apply(highlight_accuracy_row, axis=1) \
                    .set_properties(**{'text-align': 'right'}) \
                    .set_table_styles([{
                        'selector': 'th',
                        'props': [('text-align', 'right')]
                    }])

                st.dataframe(styled_df)
            else:
                st.dataframe(report_df)

            # Confusion matrix
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=orientation_labels["short"])
            disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
            ax.set_title(f"Normalization mode: {normalize_cm}")
            st.pyplot(fig)

display_classrep_and_cmatrixcm(st_module=st, formatted=True, normalize_cm="all")



#-----identifying truly rotated  images from the original dataset--------------

# created a df where the class is one of the rotated across both predictions
mod1_predCsvPath = os.path.join(PROJECT_DIR, "src", "streamlit", "data", "prediction_mobnet_predicted_orientations.csv")
mod2_predCsvPath = os.path.join(PROJECT_DIR, "src", "streamlit", "data", "prediction_resnet50_predicted_orientations.csv")
  
df_both_preds_as_rotated = ModelUtilizer.get_imgs_both_models_identified_as_rotated(mod1_predCsvPath, mod2_predCsvPath)
df_both_preds_as_rotated.shape ##14 images

def plot_images_in_rows(imgs, titles=None, images_per_row=5):
    for i in range(0, len(imgs), images_per_row):
        cols = st.columns(images_per_row)
        for col, img_index in zip(cols, range(i, min(i + images_per_row, len(imgs)))):
            with col:
                st.image(imgs[img_index], caption=titles[img_index] if titles else "", use_container_width=True)


st.header("Detection of truly rotated images in the original dataset")

# display images for visual control
#img_names = df_both_preds_as_rotated["img_name"].tolist()
#from_dir = os.path.join(PROJECT_DIR, "data_224x224", "for_orientation_classification", "original", "unlabeled_to_be_predicted_full_original_224x224")

#imgs, names = ip.load_images(imgNames=img_names, from_dir=from_dir)
#lot_images_in_rows(imgs, titles=names, images_per_row=5)


#identified as truly rotated
#imgs, names = ip.load_images(from_dir=from_dir)
#truly_rotated_index = [2,4,5,6,9,11,12]
#truly_rotated_imgs, truly_rotated_names = [imgs[i] for i in truly_rotated_index], [names[i] for i in truly_rotated_index],

from_dir = os.path.join(PROJECT_DIR, "src", "streamlit", "data", "truly_rotated_detected_confirmed")
truly_rotated_imgs, truly_rotated_names = ip.load_images(from_dir=from_dir)
st.subheader(f"Identified as truly rotated: {len(truly_rotated_imgs)} images")
plot_images_in_rows(truly_rotated_imgs, titles=truly_rotated_names, images_per_row=5)