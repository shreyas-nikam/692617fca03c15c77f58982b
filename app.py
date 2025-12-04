import streamlit as st

st.set_page_config(page_title="QuLab", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab")
st.divider()

st.markdown(
    r"""
In this lab, you will step into the role of a **risk-aware retail investor** exploring how Large Language Models (LLMs) and Agentic AI systems are built, aligned, and deployed.

The goal of this interactive lab is to help you:

* Understand how each phase of the LLM lifecycle connects to concrete business and investment risks.
* Experiment with sliders and controls to see how hidden technical choices (data bias, learning rates, drift thresholds) can materially change system behavior.
* Build intuition for why **human oversight, governance, and monitoring** are as important as the underlying algorithms.

Use the sidebar navigation to move between phases of the LLM journey:

1. **LLM Overview & Pre-training** - Build and inspect a synthetic corpus, explore word statistics, and see how bias creeps in.
2. **Alignment & Hallucinations** - Simulate loss minimization, RLHF, and visualize hallucination risk.
3. **Deployment, Drift & Oversight** - Monitor performance over time, detect drift, and connect this to ongoing governance.

As you move through the pages, imagine you are evaluating an AI-powered product or company:
What could go wrong at each step, and what controls would you expect management to have in place?
"""
)

page = st.sidebar.selectbox(
    label="Navigation",
    options=[
        "LLM Overview & Pre-training",
        "Alignment & Hallucinations",
        "Deployment, Drift & Oversight",
    ],
)

if page == "LLM Overview & Pre-training":
    from application_pages.page_llm_overview_pretraining import main

    main()
elif page == "Alignment & Hallucinations":
    from application_pages.page_alignment_hallucinations import main

    main()
elif page == "Deployment, Drift & Oversight":
    from application_pages.page_deployment_drift_oversight import main

    main()
