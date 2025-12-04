import streamlit as st

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

st.markdown("""
In this lab, you will step into the role of a **retail investor** evaluating an AI-powered
research assistant built on top of Large Language Models (LLMs) and Agentic AI
architectures.

The goal of this application is to help you **experience** how LLMs are trained,
aligned, and monitored in deployment, and where critical risks such as **data bias**,
**hallucinations**, and **model drift** can arise.

Use the navigation menu in the sidebar to explore the three main phases:
1. **Overview & Story** – Set the context for the LLM journey.
2. **Pre-training & Data Bias** – Play with synthetic corpora and bias simulations.
3. **Alignment, Loss & RLHF** – See how models are steered with loss functions and
   human feedback.
4. **Hallucinations, Drift & Oversight** – Investigate runtime risks and the role of
   human governance, especially in Agentic AI systems.
""")

page = st.sidebar.selectbox(
    label="Navigation",
    options=[
        "Overview & Story",
        "Pre-training & Data Bias",
        "Alignment, Loss & RLHF",
        "Hallucinations, Drift & Oversight",
    ],
)

if page == "Overview & Story":
    from application_pages.overview_story import main as page_main
    page_main()
elif page == "Pre-training & Data Bias":
    from application_pages.pretraining_bias import main as page_main
    page_main()
elif page == "Alignment, Loss & RLHF":
    from application_pages.alignment_rlhf import main as page_main
    page_main()
elif page == "Hallucinations, Drift & Oversight":
    from application_pages.hallucinations_drift import main as page_main
    page_main()
