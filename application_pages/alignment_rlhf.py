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
    for _ in range(epochs):
        loss_history.append(current_loss)
        current_loss = current_loss * (1 - learning_rate)
        if current_loss < 0.1:
            current_loss = 0.1
    return loss_history


def plot_loss_curve(loss_history):
    """Generates a line plot of simulated loss values over epochs."""
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
    """Creates synthetic data representing human feedback for RLHF."""
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
    """Simulates a conceptual reward signal increasing over feedback rounds."""
    reward_history = []
    current_reward = float(initial_reward)
    for _ in range(feedback_rounds):
        reward_history.append(current_reward)
        current_reward *= 1 + improvement_factor
        if current_reward > 1.0:
            current_reward = 1.0
    return reward_history


def plot_reward_signal(reward_history):
    """Visualizes conceptual reward signal improvement over feedback rounds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(reward_history, marker="o", linestyle="-", color="green")
    ax.set_title("Conceptual Reward Signal Improvement over RLHF Rounds", fontsize=14)
    ax.set_xlabel("Feedback Round", fontsize=12)
    ax.set_ylabel("Reward Signal", fontsize=12)
    ax.grid(True)
    plt.tight_layout()
    return fig


def main():
    st.title("🎯 Phase 2 – Alignment, Loss & Human Feedback")

    st.markdown(
        """
In this phase, your broker works with model providers and internal experts to **steer** the
LLM so that its behavior is more helpful, honest, and safe for investors.
"""
    )

    st.subheader("Step 1️⃣ – Understand the Loss Function")
    st.markdown(r"""
During training, the LLM updates its internal parameters to reduce a **loss function**.
The loss captures how far the model\'s predictions are from the desired outputs:

$$L = \text{Error}(\text{Predicted Output}, \text{True Output})$$

A lower $L$ means the model is making fewer mistakes relative to the target behavior.
""")

    with st.expander("Configure Loss Minimization Simulation", expanded=True):
        epochs = st.slider(
            "Epochs",
            10,
            100,
            50,
            key="epochs",
            help="Number of training passes over the data.",
        )
        initial_loss = st.number_input(
            "Initial Loss",
            1.0,
            20.0,
            10.0,
            step=0.5,
            key="initial_loss",
            help="Starting point of the conceptual loss curve.",
        )
        learning_rate = st.slider(
            "Learning Rate",
            0.01,
            0.2,
            0.08,
            step=0.01,
            key="learning_rate_loss",
            help="Controls how quickly the loss decreases.",
        )

    loss_values = simulate_loss_minimization(epochs, initial_loss, learning_rate)

    st.markdown("#### 🔢 Sample of Simulated Loss Values")
    st.write("First 5:", [f"{l:.2f}" for l in loss_values[:5]])
    st.write("Last 5:", [f"{l:.2f}" for l in loss_values[-5:]])

    st.pyplot(plot_loss_curve(loss_values))

    st.markdown(
        """
A smooth downward curve suggests that the model is successfully learning from
mistakes. In real deployments, you would want to know **how** this loss was
measured and whether it reflects objectives you care about (e.g., factual accuracy,
helpfulness, or risk sensitivity).
"""
    )

    st.subheader("Step 2️⃣ – Explore RLHF (Reinforcement Learning from Human Feedback)")
    st.markdown(
        """
RLHF gives humans a direct voice in model behavior. Human annotators rate or rank
responses, and the model is optimized to produce the kinds of answers humans prefer.
"""
    )

    with st.expander("Configure RLHF Simulation", expanded=True):
        feedback_rounds = st.slider(
            "Feedback Rounds",
            1,
            20,
            10,
            key="feedback_rounds",
            help="Number of cycles where human feedback is used to improve the model.",
        )
        improvement_factor = st.slider(
            "Reward Improvement Factor",
            0.05,
            0.5,
            0.2,
            step=0.01,
            key="improvement_factor",
            help="Controls how quickly the reward signal grows.",
        )

    feedback_data = generate_rlhf_feedback_data(num_samples=5)
    st.markdown("#### 🧑‍⚖️ Example Human Feedback Table")
    st.dataframe(feedback_data, use_container_width=True)

    reward_history = simulate_reward_signal_improvement(
        initial_reward=0.1,
        feedback_rounds=feedback_rounds,
        improvement_factor=improvement_factor,
    )

    st.markdown("#### 📈 Simulated Reward Signal (first 5 values)")
    st.write([f"{r:.2f}" for r in reward_history[:5]])

    st.pyplot(plot_reward_signal(reward_history))

    st.info(
        "In a live system, you would want governance around **who** provides feedback, **how** they are trained, and **what objectives** they optimize for, as these choices shape the model\'s behavior and risk profile."
    )


if __name__ == "__main__":
    main()
