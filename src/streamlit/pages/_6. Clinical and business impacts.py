# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(page_title="Covid-19 🦠 Detection", page_icon="🦠", layout="wide")

st.title("Clinical and Business impacts")
st.subheader("Clinical Impacts")
st.markdown(
    """
The developed chest X-ray classification model demonstrates high performance, achieving over 90% accuracy, precision, recall, and F1-score across four critical classes: Normal, Pneumonia, COVID-19, and Lung Opacity.

This performance indicates strong reliability in identifying abnormalities in chest radiographs.

By integrating this model into clinical workflows:

1. **Faster Diagnosis**: Clinicians can rapidly screen large volumes of chest X-rays, helping prioritize patients who require urgent attention.
2. **Decision Support**: The model provides a second opinion for radiologists, reducing diagnostic errors, especially during times of high workload, such as during pandemics or seasonal surges.
3. **Explainability and Trust**: Using Grad-CAM visualizations, the model highlights regions influencing predictions, allowing clinicians to verify model reasoning and build trust in AI support systems.

Ultimately, the model acts as an augmentative tool, not a replacement, helping doctors and radiologists maintain high diagnostic standards while managing increasing demands.
"""
)

st.subheader("Business Impacts")
st.markdown(
    """
The deployment of an AI-driven chest X-ray classification system offers significant business advantages for healthcare providers:

1. **Adaptability to the business process**: The output of the model should help the customers with their business process if it is meant to be at the beginning or at the end of the process (e.g., The output of the model is used before or after doing all checkups and consultations)
2. **Cost Savings**: Faster diagnosis reduces time per case, leading to operational cost savings.
3. **Increased Capacity**: Hospitals and clinics can handle more patients with the same staffing levels by automating preliminary screenings.
4. **Risk Management**: Reducing misdiagnoses minimizes the risk of malpractice claims, contributing to both financial savings and improved patient safety metrics.
5. **Scalability**: The solution can be scaled across multiple sites or integrated into telemedicine platforms, supporting remote diagnostics and expanding market reach.
"""
)