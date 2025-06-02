# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(page_title="Covid-19 🦠 Detection", page_icon="🦠", layout="wide")

st.title("Product Vision")
st.markdown(
    """
The final aim of the project is to provide a product that equips medical personnel with a tool to predict pulmonary diseases based on a patient's chest X-ray image. 
The service can be hosted on-premise or in the cloud. 
Below is a diagram showing the major components and the workflow of the end-to-end process.
"""
)
st.image("src/streamlit/images/product_vision.png", caption="Product Vision Overview")
st.subheader("Prediction Process")
st.markdown(
    """
Once the product setup is complete, the end user uses the graphical, browser-based web interface (control panel) to submit the X-ray image to the system. The user interface provides all the necessary tools to manage the prediction process.
The system consists of three data pipelines, each utilizing pre-trained machine learning models. Since proper image orientation is crucial for accurate disease prediction, the first model validates the image's orientation. 
The second model generates a mask to isolate the lung region and applies it to the image. The final model predicts the disease.
The result is displayed to the user along with collected metrics, which help indicate the quality of the prediction.
Although the prediction accuracy may be high, it is not a final diagnosis. The result must be reviewed by qualified medical staff using all required diagnostic methods to validate it.
After the final diagnosis is confirmed, the user classifies the image using predefined labels for orientation, mask quality, and disease, and then submits it to the database. This is the final step in the prediction process.
"""
)

st.subheader("The MLOps Process")
st.markdown(
    """
The resulting data from the prediction process—images with attached metrics—is used to further improve the models and overall system quality through continuous training.
An MLOps user initiates training runs with new data for each model. They communicate the training and test results to all relevant stakeholders to ensure the quality and safety of the prediction process.
Each model run produces artifacts for historization. Once the deployment of a newly trained model is approved, the related artifacts are securely stored and protected with restricted access rights, e.g., for legal reasons. These artifacts are made available in read-only mode.
"""
)
