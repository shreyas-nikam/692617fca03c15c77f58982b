import streamlit as st

def main():
    st.header("Section 1: Introduction to Agentic AI Systems and Their Risks")
    st.markdown("""
    ### Introduction to Agentic AI Systems

    Agentic AI systems represent a significant paradigm shift from traditional AI models. Unlike models that simply process data and produce outputs, agentic systems are designed to **perceive**, **reason**, and **act** autonomously within an environment to achieve specific goals. They exhibit goal-driven behavior, adapting their strategies based on continuous feedback.

    **Key Characteristics:**
    *   **Autonomy:** Ability to operate independently.
    *   **Perception:** Capability to observe and interpret the environment.
    *   **Reasoning:** Processing information to make decisions and form plans.
    *   **Action:** Executing plans and interacting with the environment, often through tool use.
    *   **Adaptation:** Learning and self-correcting based on outcomes and feedback.

    This agentic paradigm enables more complex and dynamic applications, particularly in domains requiring sequential decision-making and interaction, like finance.
    """)

    st.header("Section 2: The Planner-Executor-Critic (P-E-C) Loop Architecture")
    st.markdown("""
    ### Understanding the Planner-Executor-Critic (P-E-C) Loop

    The P-E-C loop is a foundational architectural pattern for agentic AI systems, particularly effective in dynamic environments. It breaks down complex decision-making into three distinct, interconnected components:

    1.  **Planner:** Responsible for generating a high-level strategy or sequence of actions to achieve the agent's objective based on the current environmental state and historical data.
    2.  **Executor:** Takes the plan from the Planner and translates it into concrete actions, interacting with the environment (e.g., executing trades in a market).
    3.  **Critic:** Evaluates the outcomes of the Executor's actions against the agent's objectives and risk tolerances, providing feedback to the Planner for course correction and learning.

    This continuous feedback loop allows the agent to iteratively refine its behavior and improve its performance over time.

    **P-E-C Loop Visualization:**
    """)
    st.image("https://i.imgur.com/example_pec_diagram.png", caption="Conceptual Diagram of the Planner-Executor-Critic (P-E-C) Loop. (Placeholder image)", use_column_width=True)
    st.markdown(r"""
    *   **Planner:** Takes input from the environment and the Critic's feedback, outputs a `Plan`.
    *   **Executor:** Takes the `Plan` from the Planner, interacts with the `Environment`, outputs `Actions` and updates the `Environment`.
    *   **Critic:** Takes the `Environment State` (after Executor's actions) and the `Objective`, evaluates the `Outcome`, and provides `Feedback` to the Planner.

    The **Decision Tree Logic** for planning can be conceptually understood as:
    $$ \text{If Critique Result} = \text{Negative}, \text{then Plan Revision occurs} $$
    This means if the Critic identifies undesirable outcomes or deviations, the Planner adjusts its strategy for subsequent steps.

    ---

    ### ReAct (Reasoning and Acting) Chains

    ReAct is another powerful architectural pattern that interleaves **Reasoning** (Thought) and **Acting** (Action) steps within the agent's operation. This allows agents to perform dynamic reasoning, plan, and adapt, much like human problem-solving.

    **ReAct Chains Visualization:**
    """)
    st.image("https://i.imgur.com/example_react_diagram.png", caption="Conceptual Diagram of ReAct (Reasoning and Acting) Chains. (Placeholder image)", use_column_width=True)
    st.markdown("""
    *   **Thought:** The agent reasons about the current situation, identifies sub-goals, and decides on the next course of action.
    *   **Action:** The agent executes a specific action, which might involve using a tool, interacting with the environment, or gathering more information.
    *   This cycle repeats, allowing for complex, multi-step problem-solving.
    """)