import streamlit as st


def main():
    st.title("🧭 LLM Journey Explorer – Overview & Story")
    st.markdown(r"""
Welcome to **Lab 2: Large Language Models and Agentic Architectures**.

In this interactive lab, you will step into the shoes of a **retail investor** evaluating a new AI-powered research assistant that your broker is planning to deploy. This assistant is driven by a Large Language Model (LLM) and may soon start summarizing reports, answering your questions, and even suggesting what you should look at next.

Before you trust it, you want to **understand its journey**:

1. How was this LLM **trained**?
2. How was it **aligned** with human values and guardrails?
3. How is it **monitored in deployment** to avoid nasty surprises?

Along the way, you will also see where things can go wrong: **data bias, hallucinations, and model drift**, and how these issues can be amplified when the LLM becomes part of a more autonomous **Agentic AI system**.
""")
    st.markdown(r"""
### 🧩 Business Logic of This Lab

You are not building a production model. Instead, you are using **small, transparent simulations** that mirror the logic of real LLM systems:

- **Synthetic pre-training corpus** mimics the huge datasets LLMs see.
- **Word frequency and probabilities** mimic how LLMs learn $P(\text{next word} \mid \text{previous words})$.
- **Loss curves and reward signals** mirror how models learn by minimizing an abstract loss $L$ and maximizing alignment rewards.
- **Bias, hallucination, and drift simulators** illustrate risks that an investor should be aware of when evaluating AI products.

Every chart, slider, and table is there to help you **build intuition** for questions like:

- *"If the training data is biased, what happens to outputs?"*
- *"If alignment is weak, what risks does that create?"*
- *"If performance drifts, how would I even notice as a user?"*

Use the sidebar to navigate through the lifecycle: **Pre-training → Alignment → Deployment & Risks**.
""")

    st.markdown("""
Use this page to ground yourself in the story and objectives. When you are ready,
move to the next pages to interact with simulations for each phase of the LLM
lifecycle.
""")


if __name__ == "__main__":
    main()
