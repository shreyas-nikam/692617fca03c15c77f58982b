import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

sns.set_theme(style="whitegrid")


@st.cache_data(ttl="2h")
def generate_time_series_performance_data(num_timesteps, baseline_mean, baseline_std, drift_start_time, drift_magnitude):
    """Generates synthetic time-series data for a model performance metric."""
    np.random.seed(42)
    performance_data = []
    for i in range(int(num_timesteps)):
        if i < int(drift_start_time):
            performance_data.append(float(np.random.normal(baseline_mean, baseline_std)))
        else:
            performance_data.append(float(np.random.normal(baseline_mean - drift_magnitude, baseline_std * 1.2)))
    return performance_data


@st.cache_data(ttl="2h")
def calculate_drift_threshold(mean, std_dev, k_multiplier=3.0):
    """Calculates conceptual upper and lower bounds for detecting model drift."""
    upper_bound = float(mean + k_multiplier * std_dev)
    lower_bound = float(mean - k_multiplier * std_dev)
    return upper_bound, lower_bound


@st.cache_data(ttl="2h")
def detect_conceptual_drift(performance_data, baseline_mean, baseline_std, k_multiplier=3.0):
    """Checks if the latest performance data point exceeds drift thresholds."""
    upper, lower = calculate_drift_threshold(baseline_mean, baseline_std, k_multiplier)
    latest_performance = performance_data[-1] if len(performance_data) > 0 else baseline_mean
    drift_flag = latest_performance < lower or latest_performance > upper
    return bool(drift_flag), float(latest_performance), float(upper), float(lower)


@st.cache_data(ttl="2h")
def plot_model_performance_with_drift_threshold(performance_data, baseline_mean, upper_bound, lower_bound, drift_detected):
    """Generates a line plot of model performance over time with drift thresholds."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(performance_data, label="Model Performance", color="blue", marker=".", linestyle="-")
    ax.axhline(y=baseline_mean, color="green", linestyle="--", label="Baseline Mean")
    ax.axhline(y=upper_bound, color="red", linestyle=":", label="Upper Drift Threshold")
    ax.axhline(y=lower_bound, color="red", linestyle=":", label="Lower Drift Threshold")
    if drift_detected and performance_data:
        drift_start_index = 0
        for i, perf in enumerate(performance_data):
            if perf < lower_bound or perf > upper_bound:
                drift_start_index = i
                break
        ax.axvspan(drift_start_index, len(performance_data) - 1, color="red", alpha=0.1, label="Drift Detected")
        if drift_start_index < len(performance_data) - 1:
            ax.annotate(
                "Drift Detected!",
                xy=(drift_start_index, performance_data[drift_start_index]),
                xytext=(drift_start_index + 2, performance_data[drift_start_index] + 0.05),
                arrowprops=dict(facecolor="black", shrink=0.05),
                fontsize=12,
                color="red",
            )
    ax.set_title("Model Performance Over Time with Drift Thresholds", fontsize=16)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Performance Metric (e.g., Accuracy)", fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def _init_deployment_state():
    defaults = {
        "num_timesteps": 50,
        "baseline_mean_acc": 0.85,
        "baseline_std_acc": 0.02,
        "drift_start_time": 30,
        "drift_magnitude": 0.1,
        "k_multiplier_for_drift": 3.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.fragment
def main():
    _init_deployment_state()
    st.header("Section 14: Agentic AI Systems and Risk Amplification")
    st.markdown(
        """
In many modern products, LLMs do not act alone. They are wrapped into **agentic systems** that can call tools, access data sources, and take actions on behalf of users.
"""
    )
    st.markdown(
        """
This extra autonomy amplifies risks: a biased or drifting model might now **execute transactions, send messages, or trigger workflows** without a human in the loop.
"""
    )
    st.header("Section 15: Phase 3: Deployment - Continuous Monitoring and Adaptation")
    st.markdown(
        """
Once an LLM (or LLM-powered agent) is deployed, the environment keeps changing: markets shift, user behavior evolves, and regulations update. Monitoring is not optional; it is the backbone of responsible AI operations.
"""
    )
    st.header("Section 16: Emergent Risk: Model Drift - Shifting Performance")
    st.warning("⚠️ Emergent Risk: Model Drift")
    st.markdown(
        r"""
**Model drift** occurs when performance degrades because the data the model sees in production no longer matches its training data. A common way to detect drift is to define a threshold based on baseline statistics:
$$\text{Drift Threshold} = \mu \pm k \cdot \sigma$$
where $\mu$ is the baseline mean, $\sigma$ is the baseline standard deviation, and $k$ is a multiplier.
"""
    )
    with st.expander("Configure Model Drift Simulation"):
        st.markdown(
            """
Imagine you are tracking model accuracy over days or weeks in production. Adjust these controls to stress-test your monitoring setup.
"""
        )
        num_timesteps = st.slider(
            "Number of Time Steps",
            20,
            100,
            int(st.session_state["num_timesteps"]),
            key="num_timesteps",
        )
        baseline_mean_acc = st.slider(
            "Baseline Mean Accuracy",
            0.7,
            0.99,
            float(st.session_state["baseline_mean_acc"]),
            step=0.01,
            key="baseline_mean_acc",
        )
        baseline_std_acc = st.slider(
            "Baseline Std Dev for Accuracy",
            0.01,
            0.05,
            float(st.session_state["baseline_std_acc"]),
            step=0.005,
            key="baseline_std_acc",
        )
        drift_start_time = st.slider(
            "Drift Start Time Step",
            10,
            max(15, num_timesteps - 5),
            int(min(st.session_state["drift_start_time"], num_timesteps - 5)),
            key="drift_start_time",
        )
        drift_magnitude = st.slider(
            "Drift Magnitude (Performance Drop)",
            0.0,
            0.2,
            float(st.session_state["drift_magnitude"]),
            step=0.01,
            key="drift_magnitude",
        )
        k_multiplier_for_drift = st.slider(
            "Multiplier (k) for Drift Threshold (mu +/- k * sigma)",
            1.0,
            5.0,
            float(st.session_state["k_multiplier_for_drift"]),
            step=0.5,
            key="k_multiplier_for_drift",
        )
    performance_data_over_time = generate_time_series_performance_data(
        num_timesteps,
        baseline_mean_acc,
        baseline_std_acc,
        drift_start_time,
        drift_magnitude,
    )
    drift_detected, latest_performance, upper_bound, lower_bound = detect_conceptual_drift(
        performance_data_over_time,
        baseline_mean_acc,
        baseline_std_acc,
        k_multiplier_for_drift,
    )
    st.subheader("Drift Statistics Summary")
    st.write(f"Baseline Mean Accuracy: {baseline_mean_acc:.2f}, Std Dev: {baseline_std_acc:.2f}")
    st.write(f"Drift Threshold (Lower, Upper): ({lower_bound:.2f}, {upper_bound:.2f})")
    st.write(f"Current Performance: {latest_performance:.2f}")
    st.write(f"Drift Detected: {drift_detected}")
    if drift_detected:
        st.error("Drift detected! In a production system, this should trigger alerts and investigation.")
    else:
        st.success("No drift detected yet. Continue monitoring over time.")
    st.markdown(
        """
You can think of this as a control chart for your AI system. When metrics slip outside the allowed band, the system should not continue operating blindly.
"""
    )
    st.header("Section 17: Visualizing Model Drift")
    st.markdown(
        """
This chart gives you an operations dashboard view of how the model is behaving over time relative to its expected baseline.
"""
    )
    st.pyplot(
        plot_model_performance_with_drift_threshold(
            performance_data_over_time,
            baseline_mean_acc,
            upper_bound,
            lower_bound,
            drift_detected,
        )
    )
    st.header("Section 18: The Importance of Human Oversight and Accountability")
    st.markdown(
        """
Even the best monitoring and alignment pipelines cannot fully eliminate risk. Human oversight is required to set acceptable thresholds, review incidents, and make final decisions on high-impact actions.
"""
    )
    st.markdown(
        """
As a retail investor or board member, you might ask: Who is accountable when an AI agent misbehaves? What escalation paths and governance structures are in place?
These questions are as critical as the underlying model architecture.
"""
    )
    st.header("Section 19: Conclusion and Key Takeaways")
    st.markdown(
        """
Across this lab, you have:

* Built intuition for how LLMs learn from data and why that learning can inherit bias.
* Seen how alignment and RLHF can improve behavior but cannot fully prevent hallucinations.
* Explored how performance can drift in deployment and why continuous monitoring is essential.
"""
    )
    st.markdown(
        """
When evaluating AI-enabled products or investments, look beyond marketing claims. Ask how the team handles **data quality, alignment, monitoring, and governance** throughout the LLM lifecycle.
"""
    )
