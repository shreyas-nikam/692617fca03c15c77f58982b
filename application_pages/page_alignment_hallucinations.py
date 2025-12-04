import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

sns.set_theme(style="whitegrid")


@st.cache_data(ttl="2h")
def simulate_loss_minimization(epochs, initial_loss, learning_rate):
    """Simulates a conceptual loss value decreasing over epochs during training."""
    loss_history = []
    current_loss = float(initial_loss)
    for _ in range(int(epochs)):
        loss_history.append(current_loss)
        current_loss = current_loss * (1 - float(learning_rate))
        if current_loss < 0.1:
            current_loss = 0.1
    return loss_history


@st.cache_data(ttl="2h")
def plot_loss_curve(loss_history):
    """Generates a line plot of the simulated loss values over time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(loss_history, marker="o", linestyle="-", color="blue")
    ax.set_title("Conceptual Loss Minimization over Epochs", fontsize=14)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss Value", fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    return fig


@st.cache_data(ttl="2h")
def generate_rlhf_feedback_data(num_samples):
    """Creates synthetic data representing human feedback."""
    data = []
    queries = [
        "Tell me about AI.",
        "Explain quantum computing.",
        "What is financial leverage?",
        "Summarize recent market trends.",
        "Give me investment advice.",
    ]
    responses = [
        ["AI is a field of computer science.", "AI is when machines think like humans."],
        ["Quantum computing uses qubits.", "Quantum computing is very fast."],
        ["Financial leverage is using borrowed capital.", "It helps magnify returns."],
        ["Markets are volatile.", "Recent trends show tech growth."],
        ["Invest in stocks.", "Consult a financial advisor for personalized advice."],
    ]
    for _ in range(num_samples):
        query = random.choice(queries)
        resp_a, resp_b = random.choice(responses)
        preferred = random.choice([resp_a, resp_b])
        data.append(
            {
                "query": query,
                "response_A": resp_a,
                "response_B": resp_b,
                "preferred_response": preferred,
            }
        )
    return pd.DataFrame(data)


@st.cache_data(ttl="2h")
def simulate_reward_signal_improvement(initial_reward, feedback_rounds, improvement_factor):
    """Simulates a conceptual reward signal increasing over iterative feedback rounds."""
    reward_history = []
    current_reward = float(initial_reward)
    for _ in range(int(feedback_rounds)):
        reward_history.append(current_reward)
        current_reward *= 1 + float(improvement_factor)
        if current_reward > 1.0:
            current_reward = 1.0
    return reward_history


@st.cache_data(ttl="2h")
def plot_reward_signal(reward_history):
    """Visualizes the conceptual reward signal improvement over feedback rounds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(reward_history, marker="o", linestyle="-", color="green")
    ax.set_title("Conceptual Reward Signal Improvement over RLHF Rounds", fontsize=14)
    ax.set_xlabel("Feedback Round", fontsize=12)
    ax.set_ylabel("Reward Signal", fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    return fig


@st.cache_data(ttl="2h")
def simulate_hallucination_likelihood(input_query, actual_answer, simulated_llm_response, hallucination_score):
    """Assigns a conceptual hallucination score (0.0 to 1.0) to an LLM response."""
    factual_correct = actual_answer.lower().strip() in simulated_llm_response.lower().strip()
    return {
        "query": input_query,
        "response": simulated_llm_response,
        "factual_correctness": factual_correct,
        "hallucination_score": float(hallucination_score),
    }


@st.cache_data(ttl="2h")
def plot_hallucination_meter(hallucination_score_factual, hallucination_score_hallucinated):
    """Creates a bar chart comparing conceptual hallucination scores."""
    labels = ["Factual Response", "Hallucinated Response"]
    scores = [float(hallucination_score_factual), float(hallucination_score_hallucinated)]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = sns.barplot(x=labels, y=scores, palette=["lightgreen", "salmon"], ax=ax)
    ax.set_title("Conceptual Hallucination Meter", fontsize=14)
    ax.set_xlabel("Response Type", fontsize=12)
    ax.set_ylabel("Hallucination Score (0-1)", fontsize=12)
    ax.set_ylim(0, 1.0)
    for bar in bars.patches:
        ax.annotate(
            f"{bar.get_height():.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
            xytext=(0, 5),
            textcoords="offset points",
        )
    plt.tight_layout()
    return fig


def _init_alignment_state():
    defaults = {
        "epochs": 50,
        "initial_loss": 10.0,
        "learning_rate_loss": 0.08,
        "feedback_rounds": 10,
        "improvement_factor": 0.2,
        "factual_hallucination_score": 0.1,
        "hallucinated_hallucination_score": 0.9,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.fragment
def main():
    _init_alignment_state()
    st.header("Section 7: Phase 2: Alignment - Steering LLM Behavior with Human Values")
    st.markdown(
        """
On this page, you will act as the "human in the loop" who helps steer an LLM from a raw pattern-matching engine into a **safer, more investor-friendly assistant**.
"""
    )
    st.markdown(
        """
After pre-training, LLMs are **aligned** to make them more helpful, honest, and harmless. Techniques like Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF) use curated data and human judgment to reshape the model's behavior.
"""
    )
    st.header("Section 8: The Conceptual Loss Function: Guiding Model Learning")
    st.markdown(
        r"""
During training, models learn by minimizing a **loss function** that captures how far predictions are from desired answers. Conceptually, a simple loss function can be written as
$$L = \text{Error}(\text{Predicted Output}, \text{True Output})$$
A lower $L$ means the model is doing a better job.
"""
    )
    with st.expander("Configure Loss Minimization Simulation"):
        st.markdown(
            """
Here you will set the starting error level and how aggressively the optimizer tries to reduce it. Think of this as tuning how quickly the team trains the model.
"""
        )
        epochs = st.slider(
            "Epochs",
            10,
            100,
            int(st.session_state["epochs"]),
            key="epochs",
        )
        initial_loss = st.number_input(
            "Initial Loss",
            1.0,
            20.0,
            float(st.session_state["initial_loss"]),
            step=0.5,
            key="initial_loss",
        )
        learning_rate = st.slider(
            "Learning Rate",
            0.01,
            0.2,
            float(st.session_state["learning_rate_loss"]),
            step=0.01,
            key="learning_rate_loss",
        )
    loss_values = simulate_loss_minimization(epochs, initial_loss, learning_rate)
    st.subheader("Simulated Loss Values Snapshot")
    st.write("First 5:", [f"{l:.2f}" for l in loss_values[:5]])
    st.write("Last 5:", [f"{l:.2f}" for l in loss_values[-5:]])
    st.markdown(
        """
Interpret this as: with each training pass, the team nudges the model away from obviously wrong behavior. The speed and stability of this curve can matter for training cost and reliability.
"""
    )
    st.header("Section 9: Visualizing Loss Function Minimization")
    st.markdown(
        """
A healthy training run shows a smooth, downward-sloping loss curve. Sharp spikes or flat lines can signal optimization or data problems.
"""
    )
    st.pyplot(plot_loss_curve(loss_values))
    st.header("Section 10: Simulating Reinforcement Learning from Human Feedback (RLHF)")
    st.markdown(
        """
RLHF uses human rankings of model outputs to push the system toward more **helpful and safe** behavior. This is especially important in regulated domains like finance.
"""
    )
    with st.expander("Configure RLHF Simulation"):
        st.markdown(
            """
Pretend you are running multiple feedback cycles with domain experts rating answers to investor questions.
"""
        )
        feedback_rounds = st.slider(
            "Feedback Rounds",
            1,
            20,
            int(st.session_state["feedback_rounds"]),
            key="feedback_rounds",
        )
        improvement_factor = st.slider(
            "Reward Improvement Factor",
            0.05,
            0.5,
            float(st.session_state["improvement_factor"]),
            step=0.01,
            key="improvement_factor",
        )
    feedback_data = generate_rlhf_feedback_data(num_samples=5)
    st.subheader("Sample Human Feedback Table")
    st.dataframe(feedback_data, use_container_width=True)
    reward_history = simulate_reward_signal_improvement(0.1, feedback_rounds, improvement_factor)
    st.write("Simulated Reward History (first 5):", [f"{r:.2f}" for r in reward_history[:5]])
    st.markdown(
        """
Each round of feedback is like a governance checkpoint where humans express preferences. Over time, the reward signal rising suggests the model is learning what humans consider "good" behavior.
"""
    )
    st.header("Section 11: Visualizing Reward Signal Improvement")
    st.markdown(
        """
Visualizing the reward trajectory helps you see whether alignment is actually working or has stalled.
"""
    )
    st.pyplot(plot_reward_signal(reward_history))
    st.header("Section 12: Emergent Risk: Hallucinations - Factual Inaccuracies")
    st.warning("⚠️ Emergent Risk: Hallucinations")
    st.markdown(
        """
LLMs sometimes produce confident but wrong statements. In finance, that can mean **invented statistics, fake citations, or misleading recommendations**.
"""
    )
    with st.expander("Configure Hallucination Simulation"):
        st.markdown(
            """
Use these sliders to control conceptual hallucination scores for a factual versus incorrect answer to the same question.
"""
        )
        factual_score = st.slider(
            "Conceptual Hallucination Score (Factual Response)",
            0.0,
            1.0,
            float(st.session_state["factual_hallucination_score"]),
            step=0.05,
            key="factual_hallucination_score",
        )
        hallucinated_score = st.slider(
            "Conceptual Hallucination Score (Hallucinated Response)",
            0.0,
            1.0,
            float(st.session_state["hallucinated_hallucination_score"]),
            step=0.05,
            key="hallucinated_hallucination_score",
        )
    query_example = "What is the capital of France?"
    actual_answer_example = "Paris"
    response_factual_example = "The capital of France is Paris."
    hallucination_info_factual = simulate_hallucination_likelihood(
        query_example,
        actual_answer_example,
        response_factual_example,
        factual_score,
    )
    response_hallucinated_example = "The capital of France is Rome."
    hallucination_info_hallucinated = simulate_hallucination_likelihood(
        query_example,
        actual_answer_example,
        response_hallucinated_example,
        hallucinated_score,
    )
    st.subheader("Response Comparison")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Factual Response Example:")
        st.json(hallucination_info_factual)
    with col2:
        st.write("Hallucinated Response Example:")
        st.json(hallucination_info_hallucinated)
    st.markdown(
        """
Even when the surface form looks polished, the underlying answer can be wrong. As an investor or risk manager, you would look for safeguards like fact-checking pipelines or human review in critical workflows.
"""
    )
    st.header("Section 13: Visualizing Hallucination Likelihood")
    st.markdown(
        """
A simple "hallucination meter" makes the relative risk of two responses easy to compare at a glance.
"""
    )
    st.pyplot(
        plot_hallucination_meter(
            hallucination_info_factual["hallucination_score"],
            hallucination_info_hallucinated["hallucination_score"],
        )
    )
    st.info(
        "In real systems, you rarely see an explicit hallucination score, but monitoring tools and evaluation datasets try to approximate this risk over time."
    )
