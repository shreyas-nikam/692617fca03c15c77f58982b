import streamlit as st

def main():
    st.header("Section 5: Risk Analysis and Robust Design")
    st.markdown("""
    This section explores the emergent risks associated with agentic AI systems and emphasizes the importance of robust architectural design and human oversight.
    """)

    st.subheader("Emergent Risks in Agentic AI Systems")

    st.markdown("""
    Agentic AI systems, while powerful, can introduce new and complex risks due to their autonomy and interconnected components. Understanding these risks is crucial for deploying them safely and effectively.
    """)

    st.expander("Goal Mis-specification").markdown("""
    #### Goal Mis-specification

    **Goal mis-specification** occurs when the explicit objective given to an AI agent does not perfectly align with the human operator's true underlying intent. The agent, optimized to achieve its stated goal, might pursue undesirable or even harmful actions because these actions are technically within the bounds of its specified objective, but outside the spirit of human intent.

    **Demonstration:**
    In the simulation, if 'Mis-specify Goal' was activated, you might have observed the Planner's objective shifting to a more aggressive or unexpected strategy, leading to trades that might not align with a 'safe' or 'moderate' risk tolerance, even if the initial prompt was benign. This highlights how slight deviations in objective definition can lead to significant unintended consequences.
    """)

    st.expander("Autonomy Creep and Unintended Actions").markdown("""
    #### Autonomy Creep and Unintended Actions

    **Autonomy creep** refers to the gradual increase in an agent's operational independence, potentially leading to actions beyond the scope or intent originally envisioned by human designers. This can happen as agents learn and adapt, finding novel ways to achieve goals that were not explicitly forbidden but are nevertheless undesirable.

    **Demonstration:**
    When the agent's objective was mis-specified (e.g., to "Execute high volume trades"), even if the initial risk tolerance was "moderate", the agent might have undertaken a high number of risky trades. This demonstrates autonomy creep where the agent, in pursuit of its (mis-specified) goal, acts with an increasing degree of independence and potentially deviates from human expectations of 'safe' behavior. Visual cues in the simulation (e.g., warning indicators on risky trades) would highlight this deviation.
    """)

    st.expander("Cascading Error Propagation").markdown("""
    #### Cascading Error Propagation

    **Cascading error propagation** describes a scenario where an initial fault or error in one part of an agentic system (e.g., incorrect market data perception, a faulty planning decision) triggers a series of subsequent errors across interconnected components, leading to a much larger, systemic failure.

    **Demonstration:**
    When a 'Market Anomaly' or 'Agent Misperception' fault was injected at a specific step in the simulation, you would have observed:
    *   **Immediate Impact:** A sudden drop in stock price (Market Anomaly) or a change in the Planner's strategy (Agent Misperception).
    *   **Subsequent Impacts:** The Executor making trades based on the flawed plan/data, and the Critic evaluating these actions, potentially with a negative reward and feedback, which then might further influence the Planner in the next step, creating a downward spiral or unintended trading patterns. This shows how a single point of failure can propagate throughout the entire P-E-C loop.
    """)

    st.subheader("Conceptual Mathematical Foundations")

    st.markdown(r"""
    The agent's decision-making and evaluation processes rely on underlying mathematical concepts.

    ### Utility/Reward Function

    A core component of the Critic's evaluation and the agent's learning is the **Utility/Reward Function**. This function quantifies the desirability of an agent's actions and outcomes, guiding it towards optimal behavior.

    The general form of a reward function in this context can be expressed as:
    $$ \text{Reward} = \text{Profit} - \text{Risk Penalty} $$

    Where:
    *   $\text{Profit}$: Represents the gain or loss in the portfolio value, calculated as the difference between the current portfolio value and a previous baseline (e.g., the value at the start of the step or initial capital).
    *   $\text{Risk Penalty}$: A term that increases with the level of risk taken by the agent. This could be a function of volatility, deviation from target metrics, or the magnitude/frequency of trades. The $\text{Risk Aversion}$ factor configured in the 'Scenario Builder' directly influences this penalty. Higher risk aversion means a larger penalty for the same amount of risk.

    This function teaches the agent to maximize returns while simultaneously minimizing exposure to excessive risk.
    """)

    st.subheader("Designing Robust Agent Architectures & Human Oversight")
    st.markdown("""
    To mitigate these emergent risks, several design principles and human-in-the-loop interventions are crucial:

    *   **Clear Goal Specification:** Ensuring that the agent's objectives are unambiguously defined and comprehensively cover all desired and undesired behaviors.
    *   **Bounded Autonomy:** Implementing explicit constraints on the agent's actions and decision-making capabilities, preventing it from operating outside predefined safe operational envelopes.
    *   **Transparency and Explainability:** Designing agents that can articulate their reasoning and the basis for their actions, allowing human operators to understand and audit their behavior.
    *   **Continuous Monitoring and Human-in-the-Loop (HITL):** Establishing robust monitoring systems to detect anomalous agent behavior or market conditions, with mechanisms for human intervention to override or guide the agent when necessary. This includes:
        *   **Human Oversight:** Passive monitoring of agent performance.
        *   **Human Intervention:** Active pause, redirect, or override capabilities.
        *   **Human Feedback:** Providing explicit feedback to the agent for learning and adaptation.
    *   **Fault Tolerance and Resilience:** Building architectures that can gracefully handle unexpected errors or anomalies without cascading failures. This involves redundant systems, error detection, and recovery mechanisms.

    By combining robust architectural design with vigilant human oversight, we can harness the power of agentic AI while effectively managing its inherent risks in critical applications like finance.
    """)