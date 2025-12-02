id: 692617fca03c15c77f58982b_user_guide
summary: Lab 2: Large Language Models and Agentic Architectures User Guide
feedback link: https://docs.google.com/forms/d/e/1FAIpQLSfWkOK-in_bMMoHSZfcIvAeO58PAH9wrDqcxnJABHaxiDqhSA/viewform?usp=sf_link
environments: Web
status: Published
# Understanding Agentic AI Systems in Finance

## Step 1: Introduction to Agentic AI and Key Architectures
Duration: 05:00

<aside class="positive">
In this first step, you will gain a foundational understanding of what Agentic AI systems are, why they are important, and the core architectural patterns that enable their intelligent behavior. This context is crucial before diving into the interactive simulation.
</aside>

Agentic AI systems represent a powerful evolution in artificial intelligence. Unlike traditional AI models that primarily execute predefined tasks or make predictions based on static data, **Agentic AI systems are designed to perceive their environment, reason about it, plan actions, and then execute those actions autonomously to achieve specific goals.** They continuously adapt their strategies based on feedback, making them highly dynamic and suitable for complex environments like financial markets.

In finance, agentic systems can be used for automated trading, portfolio management, risk assessment, and more, offering the potential for increased efficiency and sophisticated decision-making. However, their autonomy also introduces unique risks that need careful consideration.

### Key Characteristics of Agentic AI Systems:

*   **Autonomy:** They can operate independently without constant human intervention.
*   **Perception:** They observe and interpret information from their environment (e.g., market data).
*   **Reasoning:** They process information to make decisions, identify patterns, and form strategic plans.
*   **Action:** They interact with the environment by executing plans, often by using tools (e.g., placing trade orders).
*   **Adaptation:** They learn from outcomes and feedback, refining their behavior over time.

### The Planner-Executor-Critic (P-E-C) Loop Architecture

A fundamental design for agentic AI systems is the **Planner-Executor-Critic (P-E-C) loop**. This architecture breaks down complex tasks into three interconnected roles, ensuring a continuous cycle of decision-making, action, and evaluation:

1.  **Planner:** This component is the brain of the agent. It takes in the current state of the environment and feedback from the Critic to formulate a high-level strategy or a sequence of actions aimed at achieving the agent's objective. Think of it as developing a trading strategy for the next period.
2.  **Executor:** The Executor is responsible for carrying out the Planner's instructions. It translates the high-level plan into concrete, executable actions and interacts with the environment. In our financial context, this involves actually placing buy or sell orders in the simulated market.
3.  **Critic:** After the Executor's actions, the Critic evaluates the outcomes against the agent's predefined objectives, risk tolerances, and other performance metrics. It provides crucial feedback to the Planner, highlighting successes, failures, and deviations from the desired path. This feedback allows the Planner to learn and adjust its strategy for future steps.

This continuous feedback loop is critical for the agent's ability to iteratively refine its behavior and improve its performance in a dynamic environment.

A conceptual diagram of the P-E-C loop is shown below:
<img src="https://i.imgur.com/example_pec_diagram.png" alt="Conceptual Diagram of the Planner-Executor-Critic (P-E-C) Loop" width="100%">
*   **Planner:** Receives environment data and Critic's feedback, outputs a `Plan`.
*   **Executor:** Takes the `Plan`, interacts with the `Environment`, executes `Actions`, and updates the `Environment`.
*   **Critic:** Observes the `Environment State` (post-action), compares it to the `Objective`, evaluates the `Outcome`, and sends `Feedback` to the Planner.

The Planner often uses **Decision Tree Logic** where:
$$ \text{If Critique Result} = \text{Negative}, \text{then Plan Revision occurs} $$
This means if the Critic identifies undesirable outcomes or deviations from the objective, the Planner will adjust its strategy for the next set of actions.

### ReAct (Reasoning and Acting) Chains

Another powerful architectural pattern that enhances the adaptability of agentic systems is **ReAct (Reasoning and Acting) Chains**. This approach interleaves explicit "Thought" (reasoning) steps with "Action" steps, mimicking human problem-solving.

*   **Thought:** The agent pauses to reason about the current situation, identify sub-goals, analyze available tools or information, and decide on the next logical action.
*   **Action:** Based on its thought, the agent executes a specific action. This might involve using a tool (e.g., a financial data API), interacting with the environment (e.g., querying market data), or performing a specific trade.

This cycle of thinking and acting allows agents to perform complex, multi-step problem-solving and adapt more dynamically to unforeseen circumstances.

A conceptual diagram of ReAct chains is shown below:
<img src="https://i.imgur.com/example_react_diagram.png" alt="Conceptual Diagram of ReAct (Reasoning and Acting) Chains" width="100%">

### Summary

In this section, we've introduced the fundamental concepts of agentic AI and two key architectural patterns: the P-E-C loop and ReAct chains. These frameworks provide the basis for designing intelligent agents capable of autonomous decision-making in complex environments like financial markets. Now, let's move on to configuring our own simulated investment scenario.

## Step 2: Setting Up the Simulated Investment Environment
Duration: 07:00

<aside class="positive">
In this step, you will learn how to configure a synthetic market environment and define the objectives and constraints for your AI investment agent. This hands-on configuration will allow you to customize the simulation to explore different scenarios and observe how the agent behaves under various conditions.
</aside>

Navigate to the **"Scenario Builder"** page in the application.

This section is where you define the initial conditions and parameters for your agentic AI investment simulation. Each parameter influences how the agent operates and how the market behaves.

### Agent Objective & Risk Tolerance

*   **Agent Objective:** This text input defines the primary goal for your AI investment agent. Examples include "Maximize returns," "Safely grow portfolio," or "Generate steady income." The Planner component of the agent will interpret this objective to formulate its strategies.
    *   **Recommendation:** Start with "Maximize returns."
*   **Risk Tolerance:** This radio button allows you to set the agent's risk appetite.
    *   **"safe"**: Aims for minimal volatility and capital preservation.
    *   **"moderate"**: Balances growth with acceptable risk levels.
    *   **"aggressive"**: Seeks high growth, even with higher risk exposure.
    *   **Recommendation:** Start with "moderate."

### Environment Parameters

These settings define the characteristics of your simulated market:

*   **Initial Cash ($):** The starting amount of cash available in the agent's portfolio. This is the capital the agent has to work with.
    *   **Recommendation:** Keep the default of $100,000.
*   **Stock Symbols:** A multi-select box to choose which stock symbols will be available in your simulated market.
    *   **Recommendation:** Keep the default `AAPL` and `GOOG`.
*   **Initial Stock Prices ($):** For each selected stock, you can set its starting price. These prices form the baseline for market fluctuations.
    *   **Recommendation:** Adjust these to create varied starting points, or keep defaults.
*   **Market Volatility (e.g., 0.01 for 1%):** This number determines the magnitude of random price fluctuations in the simulated market. A higher value means more unpredictable and larger price swings. Volatility is key to simulating realistic market conditions.
    *   **Recommendation:** Keep the default of `0.01`.

### Critic Target Metrics

The Critic component evaluates the agent's performance. These metrics guide its evaluation:

*   **Target Returns (e.g., 0.05 for 5%):** The percentage return the Critic aims for the portfolio to achieve over time. This helps the Critic determine if the Planner's strategies are effective in generating profits.
    *   **Recommendation:** Keep the default of `0.05`.
*   **Max Permissible Risk (e.g., 0.02 for 2%):** The maximum percentage drop in portfolio value the Critic tolerates within a step before flagging a warning. This is a critical risk control parameter.
    *   **Recommendation:** Keep the default of `0.02`.
*   **Risk Aversion Factor:** A coefficient used by the Critic to penalize risk-taking. A higher value means the Critic will apply a stronger penalty for actions it perceives as risky, influencing the Planner to adopt more conservative strategies.
    *   **Recommendation:** Keep the default of `0.5`.

### Fault Injection (to observe emergent risks)

This advanced feature allows you to intentionally introduce errors into the simulation to observe how the agent responds and to identify potential emergent risks. We will explore these risks in more detail in Step 4.

*   **Mis-specify Goal:** Activate this checkbox to intentionally introduce a misalignment between the agent's stated objective and what might be truly desired. This demonstrates **Goal Mis-specification**.
*   **Inject Fault at Step:** Specify the simulation step at which a fault will be injected.
*   **Fault Type:** Choose the type of fault:
    *   **"Market Anomaly"**: Simulates a sudden, unexpected event in the market (e.g., a stock price crash).
    *   **"Agent Misperception"**: Simulates the agent misinterpreting data or changing its internal objective.

<aside class="negative">
Remember to click the "Initialize Environment" button after setting all your parameters. This action sets up the market environment and prepares the AI agents based on your configurations. Without initialization, you cannot proceed to the simulation.
</aside>

Once initialized, you will see a JSON representation of the "Initial Environment State," confirming your settings.

## Step 3: Running the Simulation and Analyzing Results
Duration: 10:00

<aside class="positive">
In this step, you will execute the agentic AI simulation, observe its behavior in real-time, and analyze the results using various metrics and visualizations. This interactive exploration will help you understand the dynamic interplay between the Planner, Executor, and Critic agents.
</aside>

Navigate to the **"Simulation & Results"** page.

If you haven't already, ensure you've configured your scenario and clicked "Initialize Environment" on the "Scenario Builder" page. A warning message will appear if the environment isn't ready.

### Simulation Controls

*   **Run Single Step:** Click this button to advance the simulation by one step. This is useful for closely observing the agent's decision-making process at each stage.
*   **Run Full Simulation (10 Steps):** Click this button to run the simulation for a predefined number of steps (e.g., 10). A progress bar will indicate the simulation's advancement.
*   **Reset Simulation:** Use this button to clear all simulation data and revert the environment and agents to their initial state, allowing you to start a new simulation run.

### P-E-C Loop and ReAct Chains Visualizations

These sections display conceptual diagrams of the P-E-C loop and ReAct chains. While the current implementation does not offer real-time highlighting of the active component, these diagrams serve as a visual reminder of the underlying architecture that governs the agent's actions during the simulation.

During each simulation step, the agent conceptually progresses through these stages:

1.  **Market Movement:** The environment simulates price changes for the stocks.
2.  **Planner:** Observes the new market state and previous Critic feedback, then generates a trading `Plan`.
3.  **Executor:** Takes the `Plan` and performs the specified `Actions` (buys/sells) in the market.
4.  **Critic:** Evaluates the `Outcome` of the Executor's actions, calculates `Reward`, and provides `Feedback` to the Planner for the next step.

### Current Portfolio Metrics

This section provides a real-time snapshot of your simulated investment portfolio:

*   **Current Portfolio Value:** The total monetary value of your cash and stock holdings.
*   **Cash Balance:** The amount of cash currently held by the agent.
*   **Total Profit:** The overall gain or loss in the portfolio value since the beginning of the simulation.
*   **Last Step Reward:** The numerical reward calculated by the Critic for the most recent simulation step. This is a key metric for understanding the Critic's evaluation.
*   **Individual Stock Holdings:** A table showing the quantity, current price, and total value of each stock currently held in the portfolio.

### Portfolio Evolution Plot

This interactive plot visualizes how the **Portfolio Value** and **Portfolio Returns** change over simulation steps.

*   The **Portfolio Value** plot shows the overall growth or decline of your investments.
*   The **Returns (%)** plot shows the percentage change in portfolio value at each step, giving insight into short-term performance.

These plots are essential for understanding the long-term performance and stability of your agent's strategy.

### Reward Function Breakdown (Last Step)

This section provides a detailed breakdown of the Critic's evaluation for the most recent step:

*   **Profit Component:** The direct gain or loss from the market movement and trades in the last step.
*   **Risk Taken (Normalized):** A numerical representation of the risk the agent took in the last step, influenced by factors like the number of trades and the overall risk tolerance.
*   **Risk Penalty:** The penalty applied by the Critic due to the `Risk Taken`, scaled by the `Risk Aversion Factor`.
*   **Calculated Reward:** The final reward, which is `Profit Component - Risk Penalty`. A higher reward indicates better performance according to the Critic's metrics.
*   **Message & Status:** The Critic provides a message and status (`success`, `info`, `warning`) based on its evaluation, for example, if the portfolio dropped below the maximum permissible risk.

### Trade Log

A table detailing all the buy and sell actions executed by the Executor throughout the simulation. This log includes the step number, trade type (BUY/SELL), stock ID, quantity, price, status (SUCCESS/FAILED), and a reason. This is crucial for auditing the Executor's actions.

### Agent Output Displays

These sections show the raw outputs from each agent component for the last completed step:

*   **Planner's Generated Plan (Last Step):** Displays the JSON output of the Planner's recommended actions, including the current effective objective it's working towards.
*   **Executor's Action Details (Last Trades):** Shows the raw trade records from the Executor.
*   **Critic's Feedback (Last Step):** Presents the full JSON output of the Critic's evaluation.

By carefully observing these outputs and the overall metrics, you can gain a deep understanding of how the agent's objective, risk tolerance, and the dynamic market environment influence its behavior.

<aside class="positive">
Experiment with running a few single steps, observing the changes, and then running a full simulation. Pay close attention to how the Critic's feedback might influence the Planner's subsequent plans.
</aside>

## Step 4: Risk Analysis and Robust Design
Duration: 08:00

<aside class="positive">
In this step, you will explore the emergent risks associated with autonomous agentic AI systems, particularly in the context of finance. You will understand how phenomena like goal mis-specification, autonomy creep, and cascading errors can manifest, and learn about the conceptual mathematical foundations that underpin agent evaluation and robust design principles.
</aside>

Navigate to the **"Risk Analysis"** page.

Agentic AI systems, while powerful, can introduce new and complex risks due to their autonomy, adaptive nature, and interconnected components. Understanding and mitigating these risks is paramount for safe and effective deployment.

### Emergent Risks in Agentic AI Systems

Expand each section on the page to learn about the specific risks:

1.  **Goal Mis-specification:**
    *   **Concept:** This occurs when the explicit objective given to an AI agent doesn't perfectly align with the human operator's true underlying intent. The agent, in its pursuit of the stated goal, might take undesirable or harmful actions because they are technically within the bounds of its specified objective.
    *   **Demonstration (from simulation):** If you activated "Mis-specify Goal" in the "Scenario Builder," you might have observed the Planner's objective shifting (e.g., from "Maximize returns" to "Invest aggressively in volatile assets"). This highlights how even subtle misalignments in objective definition can lead to significant unintended consequences, as the agent optimizes for a goal that is not truly what the human intended.
2.  **Autonomy Creep and Unintended Actions:**
    *   **Concept:** This refers to the gradual increase in an agent's operational independence, potentially leading to actions beyond the scope or intent originally envisioned. Agents can find novel ways to achieve goals that were not explicitly forbidden but are undesirable.
    *   **Demonstration (from simulation):** When "Mis-specify Goal" or "Agent Misperception" faults were injected, and the Planner's objective changed to something like "Execute high volume trades regardless of risk," the agent might have undertaken numerous risky trades, even if the initial risk tolerance was set to "moderate" or "safe." This demonstrates how the agent, in pursuing its (mis-specified) goal, can act with increasing independence and deviate from human expectations of safe behavior.
3.  **Cascading Error Propagation:**
    *   **Concept:** An initial fault or error in one part of an agentic system can trigger a series of subsequent errors across interconnected components, leading to a much larger, systemic failure.
    *   **Demonstration (from simulation):** If you injected a "Market Anomaly" (e.g., a sudden price drop) or "Agent Misperception" fault at a specific step:
        *   **Initial Impact:** The market data would be altered, or the Planner's strategy would be compromised.
        *   **Subsequent Impacts:** The Executor might make trades based on this flawed information or plan. The Critic would then evaluate these actions, likely resulting in a negative reward and feedback. This negative feedback could then further influence the Planner in the next step, potentially creating a downward spiral of poor decisions or unintended trading patterns. This illustrates how a single point of failure can propagate throughout the entire P-E-C loop.

### Conceptual Mathematical Foundations

The agent's decision-making and evaluation processes rely on underlying mathematical concepts, particularly the **Utility/Reward Function**.

*   **Utility/Reward Function:** This function is a core component of the Critic's evaluation. It quantifies the desirability of an agent's actions and outcomes, guiding it towards optimal behavior. In our simulation, the reward function aims to balance profit and risk.

    The general form of a reward function can be expressed as:
    $$ \text{Reward} = \text{Profit} - \text{Risk Penalty} $$
    *   $\text{Profit}$: Represents the gain or loss in portfolio value over a given step.
    *   $\text{Risk Penalty}$: A term that increases with the level of risk taken by the agent. This is influenced by the `Risk Aversion` factor configured in the "Scenario Builder." Higher risk aversion means a larger penalty for the same amount of risk.

    This function encourages the agent to maximize returns while simultaneously minimizing exposure to excessive risk.

### Designing Robust Agent Architectures & Human Oversight

To mitigate these emergent risks and deploy agentic AI safely, several design principles and human-in-the-loop interventions are crucial:

*   **Clear Goal Specification:** Define the agent's objectives unambiguously, covering all desired and undesired behaviors.
*   **Bounded Autonomy:** Implement explicit constraints on the agent's actions, preventing it from operating outside predefined safe operational envelopes.
*   **Transparency and Explainability:** Design agents that can articulate their reasoning and the basis for their actions, allowing human operators to understand and audit their behavior.
*   **Continuous Monitoring and Human-in-the-Loop (HITL):**
    *   **Human Oversight:** Passive monitoring of agent performance, identifying anomalies.
    *   **Human Intervention:** Active capabilities to pause, redirect, or override agent actions when necessary.
    *   **Human Feedback:** Providing explicit feedback to the agent to aid its learning and adaptation.
*   **Fault Tolerance and Resilience:** Build architectures that can gracefully handle unexpected errors or anomalies without cascading failures, incorporating redundant systems and recovery mechanisms.

By combining robust architectural design with vigilant human oversight, we can harness the power of agentic AI while effectively managing its inherent risks in critical applications like finance.

## Step 5: Conclusion and Next Steps
Duration: 02:00

<aside class="positive">
Congratulations! You have successfully completed the codelab on Agentic AI Systems in Finance. In this final step, we will summarize your journey and provide guidance for further exploration.
</aside>

Throughout this codelab, you have:

*   **Gained an introduction** to the concept of Agentic AI systems and their significance in dynamic environments such as financial markets.
*   **Understood key architectural patterns** like the Planner-Executor-Critic (P-E-C) loop and ReAct (Reasoning and Acting) chains, which form the backbone of intelligent agents.
*   **Configured and simulated** your own investment scenarios, observing how different objectives, risk tolerances, and market parameters influence an AI agent's behavior.
*   **Analyzed simulation results**, including portfolio performance, trade logs, and the detailed feedback from the Critic agent.
*   **Explored critical emergent risks** such as goal mis-specification, autonomy creep, and cascading error propagation, recognizing their importance in designing safe and reliable AI systems.
*   **Understood the conceptual mathematical foundations**, particularly the Utility/Reward Function, that guide agent decision-making.
*   **Learned about robust design principles** and the vital role of human-in-the-loop (HITL) interventions and oversight for managing the risks of autonomous AI.

### Key Takeaways:

*   Agentic AI systems offer powerful capabilities for autonomous decision-making and adaptation.
*   Structured architectures like P-E-C and ReAct are essential for managing complexity.
*   Careful configuration and continuous monitoring are crucial for successful deployment.
*   Understanding and mitigating emergent risks is paramount for safe and ethical AI development.
*   Human oversight remains an indispensable component in complex AI systems.

This simulation provides a simplified yet insightful glimpse into the world of agentic AI. The principles discussed here are applicable across various domains beyond finance, wherever autonomous systems are deployed.

### Further Exploration:

*   **Experiment with different scenarios:** Go back to the "Scenario Builder" and try varying the agent's objective, risk tolerance, market volatility, and fault injection parameters. Observe how these changes impact the simulation results and the agent's behavior.
*   **Deepen your understanding of specific risks:** Intentionally inject different types of faults at various steps to see their specific effects on the P-E-C loop and overall portfolio performance.
*   **Consider real-world applications:** Reflect on how these agentic principles could be applied to other areas of finance or even different industries entirely, and what challenges might arise.

Thank you for completing this QuLab codelab. We hope this experience has provided you with a valuable foundation for understanding and working with Agentic AI Systems.
