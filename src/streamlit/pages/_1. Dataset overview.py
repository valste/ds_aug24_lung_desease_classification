# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

report_data_path = "src/streamlit/data"

st.set_page_config(page_title="Covid-19 🦠 Detection", page_icon="🦠", layout="wide")

st.title("Dataset overview")
st.subheader("Dataset exploration")
st.markdown(
    """
The dataset utilized in this project comprises 21165 labeled images paired with 21165 corresponding labeled masks. These images are categorized into four distinct classes:

1.	**COVID-19**: 3616 images with 3616 masks
2.	**Normal**: 10192 images with 10192 masks
3.	**Viral Pneumonia**: 1345 images with 1345 masks
4.	**Lung Opacity**: 6012 images with 6012 masks

Each class comes with a metadata excel sheet which has the following fields:

1.	File name: name of the image file, e.g., COVID-1
2.	Format: format of the image, e.g., PNG
3.	Size: size of the image, e.g., 256x256
4.	URL: source of the image (clinic or institute), e.g., https://sirm.org/category/senza-categoria/covid-19/
This dataset is publicly available and can be downloaded from the following link: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database).
"""
)
st.subheader("Dataset analysis")

st.markdown(
    """
Our analysis is divided into two primary categories:

-	**Metadata Analysis**: Examining the metadata files to identify any meaningful information that can contribute to image analysis or support model development.
-	**Image Analysis**: Investigating the pixel data of each image to extract valuable statistics that will aid in constructing the model for use in subsequent stages of the project.
"""
)

st.markdown(
    """
#### Metadata analysis
The metadata provided with the dataset was combined with the metadata captured from original images and combined to a dataframe. Here some samples:
"""
)
metadata = pd.read_csv(f"{report_data_path}/metadata.csv", index_col=0)
st.dataframe(metadata)

st.markdown(
    """
Observations:

*	The filenames for “Normal” in the metadata file are in uppercase (e.g., NORMAL-123), whereas the corresponding image files in the folder are formatted with title case (e.g., Normal-123).
*	All X-ray images are stored as RGB images, but since the pixel values are identical across the R, G, and B channels, the images effectively appear in grayscale.
*	The detected image resolution is 299 x 299 pixels (detected resolution), which differs from the provided resolution of 256 x 256 pixels (size).
*	The X-ray images originate from multiple independent sources:
    *	COVID: 6 sources.
    *	Pneumonia + Normal: 1 source.
    *	Opacity + Normal: 1 source.

Here is the list of sources grouped by class name:
"""
)
metaclasses = pd.read_csv(f"{report_data_path}/metaclasses.csv", index_col=0)
st.dataframe(metaclasses)

st.markdown(
    """
#### Image analysis
##### Duplicate images
Upon calculating the summary statistics for the X-ray images in the dataset, we identified the presence of duplicate images, as outlined below:
"""
)
duplicates = pd.read_csv(f"{report_data_path}/duplicates.csv", index_col=0)
st.dataframe(duplicates)


st.markdown(
    """
#### Image analysis
Upon calculating the summary statistics for the X-ray images in the dataset, we identified the presence of duplicate images, as outlined below:
"""
)
class_distribution = pd.read_csv(f"{report_data_path}/class_distribution.csv", index_col=0)
st.dataframe(class_distribution)
fig = plt.figure(figsize=(8, 6))
sns.barplot(x="Class", y="Image count", hue="%", data=class_distribution)
st.pyplot(fig)
