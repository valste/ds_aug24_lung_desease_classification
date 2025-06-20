# -*- coding: utf-8 -*-
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# add project root to pythonpath
sys.path.append(project_root)
#print(project_root)
#----setting data paths----
from src import defs
defs.initDataPaths(project_root)

# Home.py (top of file)
import importlib, sys
for m in list(sys.modules):
    if m.startswith("src.") or m == "process_imgs":
        importlib.reload(sys.modules[m])

import streamlit as st

st.set_page_config(page_title="Covid-19 🦠 Detection", page_icon="🦠", layout="wide")
st.title("Analysis of Covid-19 🦠 chest x-rays")


st.subheader("Contributors")
st.write("This project was developed by the following contributors who attended Aug24 CDS class:")
st.markdown(
    """
* Maja
* Hanna
* Valerian
* Ahmad
"""
)
