import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image



st.set_page_config(page_title="Covid-19 🦠 Detection", page_icon="🦠", layout="wide")

st.title("Dataset exploration")



st.markdown(
    """
By looking at examples of images in our dateset and discussing the characteristic elements of them comparing the diagnostic groups,
we identified a collection of image features as potentially relevant for X-ray image classification. Those were:

"""
)



with st.container():
    st.markdown(
        "<div style='max-width: 800px; margin: auto;'>",
        unsafe_allow_html=True
    )
    cells = [
    ("Min"   ,   "minimum pixel value within an image"  ),
    ("Max"  ,  " minimum pixel value within an image"  ),
    ("Median",  " median pixel value within an image  "),
    ("Mean" ,   "mean pixel value within an image"  ),
    ("Std"  , " standard devation in pixel values within an image " ),
    ("Contrast"   ,   "contrast within an image"  ),
    ("Brightness"  ,  " brightness of an image"  ),
    ("Blurriness",  " level of blurriness of an image  "),
    ("Noise" ,   "noise level of an image"  )
    ]
    

    for a, b in cells:
        col1, col2= st.columns([1, 3])
        with col1:
            col1.markdown(f"**{a}**")
        with col2:
            col2.markdown(b)

    
    st.markdown("</div>", unsafe_allow_html=True)



st.subheader("Ensemble Plots of all the chosen image features within the segment of the lung area")

image = Image.open("src/streamlit/images/ensembleplots1.png")


st.image(image, caption="Plots of probability distribution of image features Min, Max, Median, Mean, Std - grouped by diagnostic classes",  use_container_width=True)


image = Image.open("src/streamlit/images/ensembleplots2.png")


st.image(image, caption="Plots of probability distribution of image features Brightness, Contrast, Blurriness and Noise - grouped by diagnostic classes",  use_container_width=True)

