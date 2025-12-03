"""This page focuses on the Deployment phase, including continuous monitoring, model drift, and the importance of human oversight and accountability.
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
    if 'num_timesteps' not in st.session_state:
        st.session_state.num_timesteps = 50
    if 'baseline_mean_acc' not in st.session_state:
        st.session_state.baseline_mean_acc = 0.85
    if 'baseline_std_acc' not in st.session_state:
        st.session_state.baseline_std_acc = 0.02
    if 'drift_start_time' not in st.session_state:
        st.session_state.drift_start_time = 30
    if 'drift_magnitude' not in st.session_state:
        st.session_state.drift_magnitude = 0.1
    if 'k_multiplier_for_drift' not in st.session_state:
        st.session_state.k_multiplier_for_drift = 3.0

    st.header("Section 15: Phase 3: Deployment - Continuous Monitoring and Adaptation")
    st.markdown("""
    Once an LLM is deployed into a real-world application, the lifecycle continues with **continuous monitoring**. This phase is crucial for ensuring the model remains robust, performs as expected, and adapts to new data distributions or changing user behaviors. Without vigilant monitoring, models can degrade, leading to performance issues and the re-emergence of risks.
    """)
    st.markdown("""
    Deployment is not the end of the LLM journey, but a new beginning of active management and oversight.
    """)

    st.header("Section 16: Emergent Risk: Model Drift - Shifting Performance")
    st.warning("⚠️ **Emergent Risk: Model Drift**")
    st.markdown(r"""
    **Model drift** (or concept drift) occurs when the statistical properties of the target variable, or the relationship between the input variables and the target variable, change over time. In LLMs, this can mean the model's performance degrades because the real-world data it encounters diverges significantly from its training data.
    To detect drift, we can establish a **Drift Threshold** based on the model's baseline performance, often defined using basic statistics:
    $$\text{Drift Threshold} = \mu \pm k \cdot \sigma$$
    where $\mu$ is the mean, $\sigma$ is the standard deviation of the performance metric during a stable baseline period, and $k$ is a multiplier (e.g., 2 or 3 for standard deviations) to define the acceptable range.
    """)

    with st.expander("Configure Model Drift Simulation"):
        # Ensure num_timesteps is defined before drift_start_time for correct range calculation
        st.session_state.num_timesteps = st.slider("Number of Time Steps", 20, 100, st.session_state.num_timesteps, key='num_timesteps_slider')
        st.session_state.baseline_mean_acc = st.slider("Baseline Mean Accuracy", 0.7, 0.99, st.session_state.baseline_mean_acc, step=0.01, key='baseline_mean_acc_slider')
        st.session_state.baseline_std_acc = st.slider("Baseline Std Dev for Accuracy", 0.01, 0.05, st.session_state.baseline_std_acc, step=0.005, key='baseline_std_acc_slider')
        
        # Dynamically set max for drift_start_time
        max_drift_start_time = st.session_state.num_timesteps - 5
        if st.session_state.drift_start_time > max_drift_start_time:
            st.session_state.drift_start_time = max_drift_start_time

        st.session_state.drift_start_time = st.slider("Drift Start Time Step", 10, max_drift_start_time, st.session_state.drift_start_time, key='drift_start_time_slider')
        st.session_state.drift_magnitude = st.slider("Drift Magnitude (Performance Drop)", 0.0, 0.2, st.session_state.drift_magnitude, step=0.01, key='drift_magnitude_slider')
        st.session_state.k_multiplier_for_drift = st.slider(r"Multiplier ($k$) for Drift Threshold ($\mu \pm k \cdot \sigma$)", 1.0, 5.0, st.session_state.k_multiplier_for_drift, step=0.5, key='k_multiplier_for_drift_slider')

    def generate_time_series_performance_data(num_timesteps, baseline_mean, baseline_std, drift_start_time, drift_magnitude):
        """
        Generates synthetic time-series data for a model performance metric (e.g., accuracy),
        introducing a conceptual drift at a specified point.
        """
        np.random.seed(42)
        performance_data = []
        for i in range(num_timesteps):
            if i < drift_start_time:
                # Stable baseline performance
                performance_data.append(np.random.normal(baseline_mean, baseline_std))
            else:
                # Performance drops due to drift
                performance_data.append(np.random.normal(baseline_mean - drift_magnitude, baseline_std * 1.2))
        return performance_data

    def calculate_drift_threshold(mean, std_dev, k_multiplier=3):
        """
        Calculates conceptual upper and lower bounds for detecting model drift.
        Returns (upper_bound, lower_bound).
        """
        upper_bound = mean + k_multiplier * std_dev
        lower_bound = mean - k_multiplier * std_dev
        return (upper_bound, lower_bound)

    def detect_conceptual_drift(performance_data, baseline_mean, baseline_std, k_multiplier=3):
        """
        Checks if the latest performance data point exceeds the calculated drift thresholds.
        Returns True if drift is detected, False otherwise.
        """
        upper, lower = calculate_drift_threshold(baseline_mean, baseline_std, k_multiplier)
        latest_performance = performance_data[-1] if len(performance_data) > 0 else baseline_mean
        return latest_performance < lower or latest_performance > upper

    performance_data_over_time = generate_time_series_performance_data(st.session_state.num_timesteps,
                                                                   st.session_state.baseline_mean_acc,
                                                                   st.session_state.baseline_std_acc,
                                                                   st.session_state.drift_start_time,
                                                                   st.session_state.drift_magnitude)

    upper_bound, lower_bound = calculate_drift_threshold(st.session_state.baseline_mean_acc, st.session_state.baseline_std_acc, st.session_state.k_multiplier_for_drift)
    drift_detected_status = detect_conceptual_drift(performance_data_over_time, st.session_state.baseline_mean_acc,
                                                    st.session_state.baseline_std_acc, st.session_state.k_multiplier_for_drift)

    st.write(f"Baseline Mean Accuracy: {st.session_state.baseline_mean_acc:.2f}, Std Dev: {st.session_state.baseline_std_acc:.2f}")
    st.write(f"Drift Threshold (Lower, Upper): ({lower_bound:.2f}, {upper_bound:.2f})")
    if performance_data_over_time:
        st.write(f"Current Performance: {performance_data_over_time[-1]:.2f}")
    st.write(f"Drift Detected: {drift_detected_status}")

    st.markdown("""
    Here, we've simulated a model's performance over 50 time steps. Initially stable, the performance conceptually drops after a certain point, simulating drift. The calculated drift thresholds provide boundaries, and our detector indicates if the current performance falls outside this acceptable range, signaling potential drift.
    """)

    st.header("Section 17: Visualizing Model Drift")
    st.markdown("""
    A clear visualization of performance over time, alongside the calculated drift thresholds, helps in quickly identifying when a model begins to "drift" and its behavior deviates significantly from its expected baseline.
    """)

    def plot_model_performance_with_drift_threshold(performance_data, baseline_mean, upper_bound, lower_bound, drift_detected, drift_start_time):
        """
        Generates a line plot of model performance over time, including baseline and drift thresholds,
        highlighting if drift is detected.
        """
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.plot(performance_data, label='Model Performance', color='blue', marker='.', linestyle='-')
        ax.axhline(y=baseline_mean, color='green', linestyle='--', label='Baseline Mean')
        ax.axhline(y=upper_bound, color='red', linestyle=':', label='Upper Drift Threshold')
        ax.axhline(y=lower_bound, color='red', linestyle=':', label='Lower Drift Threshold')

        if drift_detected and performance_data:
            drift_actual_start_index = -1
            for i in range(drift_start_time, len(performance_data)):
                if performance_data[i] < lower_bound or performance_data[i] > upper_bound:
                    drift_actual_start_index = i
                    break

            if drift_actual_start_index != -1:
                ax.axvspan(drift_actual_start_index, len(performance_data) - 1, color='red', alpha=0.1, label='Drift Detected')
                if drift_actual_start_index < len(performance_data) - 1:
                    ax.annotate('Drift Detected!', xy=(drift_actual_start_index, performance_data[drift_actual_start_index]),
                                xytext=(drift_actual_start_index + 5, performance_data[drift_actual_start_index] + 0.05),
                                arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='red')

        ax.set_title("Model Performance Over Time with Drift Thresholds", fontsize=16)
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Performance Metric (e.g., Accuracy)", fontsize=12)
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        return fig

    st.pyplot(plot_model_performance_with_drift_threshold(performance_data_over_time, st.session_state.baseline_mean_acc,
                                                    upper_bound, lower_bound, drift_detected_status, st.session_state.drift_start_time))

    st.markdown("""
    The plot visually confirms the model drift. The blue line (performance) drops below the lower drift threshold, clearly indicating a significant deviation from its stable operating behavior. Such a detection would trigger a need for investigation, potential retraining, or other mitigation strategies.
    """)

    st.header("Section 18: The Importance of Human Oversight and Accountability")
    st.markdown("""
    Throughout the LLM lifecycle and especially with the rise of Agentic AI, **human oversight and accountability** are paramount. This involves:
    *   **Human-in-the-Loop (HITL)** checkpoints: Integrating human review and intervention points for critical decisions or actions.
    *   **Transparent processes**: Documenting data, models, and decision-making to enable auditing and explainability.
    *   **Clear responsibilities**: Defining who is accountable for an AI system's outcomes.
    Human feedback and continuous monitoring are not just technical requirements; they are ethical imperatives to ensure AI systems remain beneficial and aligned with societal values.
    """)
    st.markdown("""
    This section reinforces that while AI technology advances, human judgment, ethical considerations, and robust governance frameworks are indispensable for responsible AI development and deployment.
    """)

    st.header("Section 19: Conclusion and Key Takeaways")
    st.markdown("""
    We have journeyed through the lifecycle of Large Language Models, from their fundamental pre-training to their critical alignment with human values, and finally to their deployment and continuous monitoring. We've seen how emergent risks like **data bias**, **hallucinations**, and **model drift** can arise at different stages and how these risks are amplified by the autonomy of **Agentic AI** systems.

    **Key Takeaways**:
    *   LLMs learn patterns from vast data, but this process can embed and amplify societal biases.
    *   Alignment processes like RLHF are crucial for steering LLMs towards helpful and harmless behavior, but human feedback itself requires careful design.
    *   LLMs are prone to "hallucinating" factually incorrect information, especially in high-stakes contexts.
    *   Model performance can degrade over time due to "drift," necessitating continuous monitoring.
    *   Human oversight, transparent processes, and clear accountability are essential for managing AI risks and ensuring trustworthy AI.
    """)
    st.markdown("""
    This concludes our exploration of the LLM Journey. We hope this application has provided you, as a retail investor, with a clearer conceptual understanding of LLMs, their lifecycle, and the critical risks to be aware of in the evolving landscape of AI.
    """)
