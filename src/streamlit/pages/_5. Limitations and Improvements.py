# -*- coding: utf-8 -*-
import streamlit as st

st.set_page_config(page_title="Covid-19 🦠 Detection", page_icon="🦠", layout="wide")

st.title("Known limitations and possible improvements")

st.subheader("Limitations due to label reliability")
st.markdown(
    """
Classification accuracy of up to 98% must be interpreted with caution, given the likelihood of systematic label noise stemming from inter-observer variability in radiologic interpretation. 
Without documentation of the annotation protocol—including consensus methods or integration of clinical context—the reliability of the ground truth remains uncertain. 
Moreover, human labeling errors in medical imaging are often non-random: subtle pathologies or borderline cases may be systematically underdiagnosed.
"""
)

st.subheader("Limitations due to dataset size")
st.markdown(
    """
Retraining the model on a larger and more diverse dataset—covering a broader range of patient demographics, imaging conditions, and equipment types—would help improve performance and robustness. 
This step is especially crucial to ensure the model maintains accuracy when deployed in real-world clinical settings where data variability is high.
"""
)

st.subheader("Limitations due to single-label structure")
st.markdown(
    """
The enforced single-label structure of the dataset introduces ambiguity, as some categories—such as COVID-19, viral pneumonia, and lung opacity—may overlap in clinical presentation and imaging features. 
This may lead the model to be penalized for detecting valid but unannotated findings. 
To mitigate these issues, future work could explore techniques such as soft targets, label smoothing, or weak supervision, enabling models to generalize beyond human labeling limitations and potentially exceed radiologist-level performance in edge cases.
"""
)

st.subheader("Improvements for model architecture")
st.markdown(
    """
Misclassifications, particularly between visually similar classes, highlight the need for more discriminative feature extraction methods. 
Exploring advanced architectures like attention-based networks, ensemble models, or Capsule Networks can help capture spatial hierarchies and subtle diagnostic cues more effectively. 
"""
)

st.subheader("Improvements for model interpretability")
st.markdown(
    """
Integrating explainability tools like Grad-CAM, SHAP, and others and incorporating uncertainty estimation will not only improve model transparency but also increase clinician confidence and support practical adoption in healthcare environments.
"""
)