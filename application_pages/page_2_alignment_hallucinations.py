"""This page focuses on the Alignment phase, including loss function minimization, RLHF, and the emergent risk of hallucinations, along with Agentic AI systems.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import streamlit as st

sns.set_theme(style="whitegrid")

def main():
    # Initialize session state for widgets if not already present
    if 'epochs' not in st.session_state:
        st.session_state.epochs = 50
    if 'initial_loss' not in st.session_state:
        st.session_state.initial_loss = 10.0
    if 'learning_rate_loss' not in st.session_state:
        st.session_state.learning_rate_loss = 0.08
    if 'feedback_rounds' not in st.session_state:
        st.session_state.feedback_rounds = 10
    if 'improvement_factor' not in st.session_state:
        st.session_state.improvement_factor = 0.2
    if 'factual_hallucination_score' not in st.session_state:
        st.session_state.factual_hallucination_score = 0.1
    if 'hallucinated_hallucination_score' not in st.session_state:
        st.session_state.hallucinated_hallucination_score = 0.9

    st.header("Section 7: Phase 2: Alignment - Steering LLM Behavior with Human Values")
    st.markdown("""
    After pre-training, LLMs are **aligned** to make them more helpful, honest, and harmless. This critical phase refines the model's behavior to follow instructions, avoid generating harmful content, and generally align with human values. Key techniques include Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF).
    """)
    st.markdown("""
    Alignment is where human judgment plays a direct role in shaping an LLM's ethical and practical behavior.
    """)

    st.header("Section 8: The Conceptual Loss Function: Guiding Model Learning")
    st.markdown(r"""
    During both pre-training and alignment, models learn by iteratively minimizing a **loss function**. This function quantifies the "error" or "discrepancy" between the model's predicted output and the desired (true) output. The goal of training is to adjust the model's internal parameters to make this loss as small as possible. Conceptually, a simple loss function can be expressed as:
    $$L = \text{Error}(\text{Predicted Output}, \text{True Output})$$
    Minimizing $L$ means the model is getting "closer" to generating the desired outputs.
    """)

    with st.expander("Configure Loss Minimization Simulation"):
        st.session_state.epochs = st.slider("Epochs", 10, 100, st.session_state.epochs, key='epochs_slider')
        st.session_state.initial_loss = st.number_input("Initial Loss", 1.0, 20.0, st.session_state.initial_loss, step=0.5, key='initial_loss_input')
        st.session_state.learning_rate_loss = st.slider("Learning Rate", 0.01, 0.2, st.session_state.learning_rate_loss, step=0.01, key='learning_rate_loss_slider')

    def simulate_loss_minimization(epochs, initial_loss, learning_rate):
        """
        Simulates a conceptual loss value decreasing over `epochs` during training.
        The decrease is conceptual, representing the optimization process.
        """
        loss_history = []
        current_loss = initial_loss
        for i in range(epochs):
            loss_history.append(current_loss)
            current_loss = current_loss * (1 - learning_rate) # Simple exponential decay
            if current_loss < 0.1: # Prevent loss from going too low conceptually
                current_loss = 0.1
        return loss_history

    loss_values = simulate_loss_minimization(st.session_state.epochs, st.session_state.initial_loss, st.session_state.learning_rate_loss)
    st.write("Simulated Loss Values (first 5):", [f"{l:.2f}" for l in loss_values[:5]])
    st.write("Simulated Loss Values (last 5):", [f"{l:.2f}" for l in loss_values[-5:]])

    st.markdown("""
    The simulated `loss_values` show a decreasing trend, representing the model's iterative process of learning from data and reducing its errors. This is the fundamental mechanism behind an LLM's ability to improve.
    """)

    st.header("Section 9: Visualizing Loss Function Minimization")
    st.markdown("""
    Visualizing the loss function over time (or training "epochs") helps us understand how effectively the model is learning. A steadily decreasing curve indicates that the model is successfully optimizing its parameters.
    """)

    def plot_loss_curve(loss_history):
        """
        Generates a line plot of the simulated loss values over time.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(loss_history, marker='o', linestyle='-', color='blue')
        ax.set_title("Conceptual Loss Minimization over Epochs", fontsize=14)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss Value", fontsize=12)
        ax.grid(True)
        plt.tight_layout()
        return fig

    st.pyplot(plot_loss_curve(loss_values))

    st.markdown("""
    The downward slope of the curve demonstrates the optimization process. As the LLM processes more data and adjusts its internal weights, the discrepancy between its predictions and the desired outcomes (its "error") decreases.
    """)

    st.header("Section 10: Simulating Reinforcement Learning from Human Feedback (RLHF)")
    st.markdown("""
    Reinforcement Learning from Human Feedback (RLHF) is a powerful alignment technique. It involves:
    1.  An LLM generates multiple responses to a prompt.
    2.  Human annotators rank or rate these responses based on quality, helpfulness, and safety.
    3.  A separate "reward model" is trained on these human preferences.
    4.  The LLM is then fine-tuned using reinforcement learning to maximize the reward signal from this reward model, effectively learning to produce responses that humans prefer. This is an iterative process, continuously refining the model.
    """)
    st.markdown("""
    Alignment is where human judgment plays a direct role in shaping an LLM's ethical and practical behavior.
    """)

    with st.expander("Configure RLHF Simulation"):
        st.session_state.feedback_rounds = st.slider("Feedback Rounds", 1, 20, st.session_state.feedback_rounds, key='feedback_rounds_slider')
        st.session_state.improvement_factor = st.slider("Reward Improvement Factor", 0.05, 0.5, st.session_state.improvement_factor, step=0.01, key='improvement_factor_slider')

    def generate_rlhf_feedback_data(num_samples):
        """
        Creates synthetic data representing human feedback, including a query, two responses, and a preferred choice.
        """
        data = []
        queries = ["Tell me about AI.", "Explain quantum computing.", "What is financial leverage?", "Summarize recent market trends.", "Give me investment advice."]
        responses = [
            ["AI is a field of computer science.", "AI is when machines think like humans."],
            ["Quantum computing uses qubits.", "Quantum computing is very fast."],
            ["Financial leverage is using borrowed capital.", "It helps magnify returns."],
            ["Markets are volatile.", "Recent trends show tech growth."],
            ["Invest in stocks.", "Consult a financial advisor for personalized advice."]
        ]
        
        for i in range(num_samples):
            query = random.choice(queries)
            resp_a, resp_b = random.choice(responses)
            preferred = random.choice([resp_a, resp_b])
            data.append({"query": query, "response_A": resp_a, "response_B": resp_b, "preferred_response": preferred})
        return pd.DataFrame(data)

    def simulate_reward_signal_improvement(initial_reward, feedback_rounds, improvement_factor):
        """
        Simulates a conceptual reward signal increasing over iterative feedback rounds.
        Represents the conceptual improvement in model alignment.
        """
        reward_history = []
        current_reward = initial_reward
        for _ in range(feedback_rounds):
            reward_history.append(current_reward)
            current_reward *= (1 + improvement_factor) # Conceptual increase
            if current_reward > 1.0: # Cap conceptual reward at 1.0
                current_reward = 1.0
        return reward_history

    feedback_data = generate_rlhf_feedback_data(num_samples=5)
    st.dataframe(feedback_data)

    reward_history = simulate_reward_signal_improvement(initial_reward=0.1, feedback_rounds=st.session_state.feedback_rounds, improvement_factor=st.session_state.improvement_factor)
    st.write("\nSimulated Reward History (first 5):", [f"{r:.2f}" for r in reward_history[:5]])

    st.markdown("""
    The `feedback_data` table shows how human evaluators might choose between different LLM outputs. The `reward_history` then conceptually demonstrates how the model's ability to generate preferred responses improves with more rounds of such human feedback, driving alignment.
    """)

    st.header("Section 11: Visualizing Reward Signal Improvement")
    st.markdown("""
    The progress of RLHF can be visualized by observing the improvement in the "reward signal." As the reward model learns to accurately capture human preferences and the LLM learns to maximize this reward, the signal should ideally increase, indicating better alignment.
    """)

    def plot_reward_signal(reward_history):
        """
        Visualizes the conceptual reward signal improvement over feedback rounds.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(reward_history, marker='o', linestyle='-', color='green')
        ax.set_title("Conceptual Reward Signal Improvement over RLHF Rounds", fontsize=14)
        ax.set_xlabel("Feedback Round", fontsize=12)
        ax.set_ylabel("Reward Signal", fontsize=12)
        ax.grid(True)
        plt.tight_layout()
        return fig

    st.pyplot(plot_reward_signal(reward_history))

    st.markdown("""
    This upward-sloping curve signifies the success of the alignment process. Each "feedback round" allows the model to better understand and incorporate human values, leading to more desirable and safer outputs.
    """)

    st.header("Section 12: Emergent Risk: Hallucinations - Factual Inaccuracies")
    st.warning("⚠️ **Emergent Risk: Hallucinations**")
    st.markdown("""
    **Hallucinations** are a critical emergent risk where LLMs generate outputs that are factually incorrect or nonsensical, yet appear credible and fluent. These can range from minor inaccuracies to completely fabricated information. Hallucinations are particularly dangerous in high-stakes applications like financial advice or medical diagnosis.
    """)

    with st.expander("Configure Hallucination Simulation"):
        st.session_state.factual_hallucination_score = st.slider("Conceptual Hallucination Score (Factual Response)", 0.0, 1.0, st.session_state.factual_hallucination_score, step=0.05, key='factual_hallucination_score_slider')
        st.session_state.hallucinated_hallucination_score = st.slider("Conceptual Hallucination Score (Hallucinated Response)", 0.0, 1.0, st.session_state.hallucinated_hallucination_score, step=0.05, key='hallucinated_hallucination_score_slider')

    def simulate_hallucination_likelihood(input_query, actual_answer, simulated_llm_response, hallucination_score):
        """
        Assigns a conceptual hallucination score (0.0 to 1.0) to an LLM response.
        Higher score means higher likelihood of hallucination.
        """
        return {"query": input_query, "response": simulated_llm_response,
                "factual_correctness": (actual_answer.lower().strip() == simulated_llm_response.lower().strip()),
                "hallucination_score": hallucination_score}

    query_example = "What is the capital of France?"
    actual_answer_example = "Paris"

    response_factual_example = "The capital of France is Paris."
    hallucination_info_factual = simulate_hallucination_likelihood(query_example, actual_answer_example, response_factual_example, st.session_state.factual_hallucination_score)
    st.write("Factual Response Example:", hallucination_info_factual)

    response_hallucinated_example = "The capital of France is Rome."
    hallucination_info_hallucinated = simulate_hallucination_likelihood(query_example, actual_answer_example, response_hallucinated_example, st.session_state.hallucinated_hallucination_score)
    st.write("Hallucinated Response Example:", hallucination_info_hallucinated)

    st.markdown("""
    The output shows two simulated LLM responses to the same query, one factual and one hallucinated. The `hallucination_score` is a conceptual metric that helps quantify the model's confidence or reliability. A high score indicates a higher risk of the information being incorrect.
    """)

    st.header("Section 13: Visualizing Hallucination Likelihood")
    st.markdown("""
    A "hallucination meter" can conceptually represent the reliability of an LLM's output. By visualizing a score, users can gauge the uncertainty or potential for inaccuracy, prompting them to verify critical information.
    """)

    def plot_hallucination_meter(hallucination_score_factual, hallucination_score_hallucinated):
        """
        Creates a bar chart comparing the conceptual hallucination scores for two responses.
        Y-axis: Hallucination Score (0-1).
        """
        labels = ['Factual Response', 'Hallucinated Response']
        scores = [hallucination_score_factual, hallucination_score_hallucinated]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = sns.barplot(x=labels, y=scores, palette=['lightgreen', 'salmon'], ax=ax)
        ax.set_title("Conceptual Hallucination Meter", fontsize=14)
        ax.set_xlabel("Response Type", fontsize=12)
        ax.set_ylabel("Hallucination Score (0-1)", fontsize=12)
        ax.set_ylim(0, 1.0)

        for bar in bars.patches:
            ax.annotate(f'{bar.get_height():.2f}', 
                         (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                         ha='center', va='bottom', 
                         xytext=(0, 5), 
                         textcoords='offset points')

        plt.tight_layout()
        return fig

    st.pyplot(plot_hallucination_meter(hallucination_info_factual['hallucination_score'],
                                 hallucination_info_hallucinated['hallucination_score']))

    st.markdown("""
    This visualization makes the difference in reliability stark. The significantly higher hallucination score for the incorrect response serves as a visual warning, underscoring the importance of critical evaluation of LLM outputs.
    """)

    st.header("Section 14: Introduction to Agentic AI Systems and Risk Amplification")
    st.markdown("""
    While LLMs are powerful, their capabilities are greatly expanded in **Agentic AI systems**. These systems are designed to perceive, reason, plan, and act autonomously, often by leveraging LLMs as their "brains" to make decisions and interact with tools and environments.
    This increased autonomy, however, inherently **amplifies risks**. Errors or biases that might be contained within an LLM can cascade into real-world consequences when an agent takes autonomous action. Risks include mis-planned goals, unintended actions, and the potential for magnified errors.
    """)
    st.markdown("""
    Understanding Agentic AI is key because it represents a major shift towards more autonomous systems. While powerful, this autonomy demands even greater vigilance regarding the underlying LLM's reliability and ethical alignment.
    """)
