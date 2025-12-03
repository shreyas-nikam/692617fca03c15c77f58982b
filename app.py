"""
Main Streamlit application file for the LLM Journey Explorer.
This file handles the overall layout, navigation, and page management.
"""

import streamlit as st
import os

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

st.markdown("""
In this lab, we explore the lifecycle of Large Language Models (LLMs) and Agentic Architectures. 
We will cover the key phases of LLM development: Pre-training, Alignment, and Deployment. 
For each phase, we will demonstrate core concepts and highlight emergent risks such as data bias, hallucinations, and model drift. 
Understanding these stages and risks is crucial for anyone engaging with AI technologies, especially in critical decision-making contexts.
""")

# Navigation
page = st.sidebar.selectbox(label="Navigation", options=["Overview & Pre-training", "Alignment & Hallucinations", "Deployment & Drift"])

if page == "Overview & Pre-training":
    from application_pages.page_1_overview_pretraining import main
    main()
elif page == "Alignment & Hallucinations":
    from application_pages.page_2_alignment_hallucinations import main
    main()
elif page == "Deployment & Drift":
    from application_pages.page_3_deployment_drift import main
    main()
