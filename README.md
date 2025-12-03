# QuLab: Agentic AI Systems in Finance

## Project Title and Description

**QuLab: Agentic AI Systems in Finance** is an interactive Streamlit application designed as a lab project to explore the fundamental concepts, architectures, risks, and robust design principles of Agentic AI systems within a simulated financial investment environment.

This application provides a hands-on experience for users to:
*   Understand the core characteristics of agentic AI and architectural patterns like the **Planner-Executor-Critic (P-E-C) Loop** and **ReAct Chains**.
*   Configure custom investment scenarios with varying objectives, risk tolerances, and market conditions.
*   Simulate the behavior of an AI investment agent in a synthetic market.
*   Visualize portfolio evolution, trade logs, and agent decision-making processes.
*   Identify and analyze emergent risks such as **Goal Mis-specification**, **Autonomy Creep**, and **Cascading Error Propagation** through fault injection mechanisms.
*   Appreciate the importance of robust architectural design and **Human-in-the-Loop (HITL)** oversight in deploying agentic AI systems safely in critical domains like finance.

## Features

This application is structured into four main sections, each offering distinct functionalities:

1.  **Introduction**:
    *   Provides a theoretical overview of Agentic AI systems, their key characteristics, and a comparison with traditional AI models.
    *   Explains the **Planner-Executor-Critic (P-E-C) Loop** architecture with conceptual diagrams.
    *   Introduces **ReAct (Reasoning and Acting) Chains** as another powerful architectural pattern.

2.  **Scenario Builder**:
    *   Allows users to define the AI agent's primary objective and risk tolerance.
    *   Configures initial market parameters: starting cash, stock symbols, initial prices, and market volatility.
    *   Sets up the Critic agent's target metrics for evaluating performance (target returns, maximum permissible risk, risk aversion).
    *   Includes a "Fault Injection" mechanism to intentionally introduce emergent risks like "Goal Mis-specification" or "Market Anomaly" at specific simulation steps.
    *   Initializes the simulated investment environment and AI agents (Planner, Executor, Critic) based on user inputs.

3.  **Simulation & Results**:
    *   Provides controls to run the simulation step-by-step or for a full predefined number of steps.
    *   Displays current portfolio metrics including total value, cash balance, and individual stock holdings.
    *   Visualizes portfolio value and returns over simulation steps using interactive plots.
    *   Offers a detailed breakdown of the Critic's reward function for the last step.
    *   Maintains a comprehensive trade log of all transactions executed by the Executor.
    *   Showcases the internal workings of the agents by displaying the Planner's generated plan, Executor's actions, and Critic's feedback for each step.
    *   Shows LLM-powered analysis including strengths, concerns, recommendations, and objective alignment scores.

4.  **Risk Analysis**:
    *   Delves into common emergent risks in agentic AI systems:
        *   **Goal Mis-specification**: When stated objectives diverge from human intent.
        *   **Autonomy Creep**: Gradual increase in an agent's independence leading to unintended actions.
        *   **Cascading Error Propagation**: How an initial fault can trigger systemic failures.
    *   Provides conceptual mathematical foundations, particularly for the **Utility/Reward Function**.
    *   Discusses strategies for designing robust agent architectures and implementing effective Human-in-the-Loop (HITL) oversight.

## 🤖 LLM Integration (NEW!)

This application now uses **actual AI agents** powered by Google's Gemini API instead of rule-based simulations:

*   **PlannerAgent**: Uses Gemini to analyze market conditions and generate sophisticated trading strategies
*   **ExecutorAgent**: Validates trades with LLM reasoning before execution
*   **CriticAgent**: Provides nuanced performance analysis with strengths, concerns, and recommendations

### Getting a Free Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key and paste it in the sidebar of the application

**Note**: The application includes intelligent fallback to rule-based agents if no API key is provided.

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

*   Python 3.8+
*   `pip` (Python package installer)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/quolab-agentic-ai-finance.git
    cd quolab-agentic-ai-finance
    ```
    (Replace `your-username/quolab-agentic-ai-finance` with the actual repository URL if different)

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    Create a `requirements.txt` file in the root directory of your project with the following content:
    ```
    streamlit>=1.20.0
    pandas>=1.3.0
    numpy>=1.21.0
    matplotlib>=3.4.0
    seaborn>=0.11.0
    ```
    Then install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application**:
    Navigate to the project's root directory in your terminal (where `app.py` is located) and run:
    ```bash
    streamlit run app.py
    ```
    This will open the application in your default web browser.

2.  **Navigate the Application**:
    *   Use the sidebar to switch between the "Introduction", "Scenario Builder", "Simulation & Results", and "Risk Analysis" pages.
    *   Start with the "Introduction" for a theoretical overview.
    *   Proceed to the "Scenario Builder" to configure your simulation parameters. **Remember to click "Initialize Environment" after setting your parameters.**
    *   Go to "Simulation & Results" to run the simulation and observe the agent's behavior and performance.
    *   Finally, visit "Risk Analysis" to understand the potential risks and mitigation strategies.

## Project Structure

```
quolab-agentic-ai-finance/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This README file
└── application_pages/          # Directory containing individual page components
    ├── introduction.py         # Content for the 'Introduction' page
    ├── scenario_builder.py     # Logic and UI for 'Scenario Builder' page
    ├── simulation_results.py   # Logic and UI for 'Simulation & Results' page
    └── risk_analysis.py        # Content for the 'Risk Analysis' page
```

## Technology Stack

*   **Framework**: [Streamlit](https://streamlit.io/)
*   **Language**: Python 3.8+
*   **AI/LLM**: [Google Gemini API](https://ai.google.dev/) (gemini-pro model)
*   **Data Manipulation**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Plotting**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
*   **Conceptual Diagrams**: Placeholder images from Imgur are used (`https://i.imgur.com/example_pec_diagram.png`, `https://i.imgur.com/example_react_diagram.png`). For a production environment, consider hosting these locally or using a more robust image hosting service.

## Contributing

Contributions to this lab project are welcome! If you have suggestions for improvements, new features, or bug fixes, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes and ensure the code adheres to the existing style.
4.  Commit your changes (`git commit -m 'Add new feature'`).
5.  Push to the branch (`git push origin feature/your-feature-name`).
6.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or feedback, please reach out to:

*   **Project Maintainer**: Your Name / Organization Name
*   **Email**: [your.email@example.com](mailto:your.email@example.com)
*   **Website/LinkedIn**: [Optional Link]
