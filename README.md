# QuLab: Interactive LLM Lifecycle and Risk Exploration

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**QuLab: Interactive LLM Lifecycle and Risk Exploration** is a Streamlit-powered educational lab designed for risk-aware individuals (like retail investors or board members) to explore the fundamental phases of Large Language Model (LLM) and Agentic AI system development and deployment. The application provides interactive simulations and visualizations to demonstrate how various technical choices and real-world factors can introduce and amplify risks such as data bias, hallucinations, and model drift.

The primary goal of this lab is to foster intuition around the critical importance of human oversight, robust governance, and continuous monitoring throughout the LLM lifecycle, emphasizing their role in mitigating potential business and investment risks.

## Features

This interactive lab allows users to explore the LLM lifecycle through three main phases, each demonstrating key concepts and emergent risks:

### 1. LLM Overview & Pre-training
*   **Conceptual LLM Lifecycle Timeline**: Visualize the sequential phases of LLM development: Pre-training, Alignment, and Deployment.
*   **Synthetic Text Data Generation**: Configure parameters (number of sentences, vocabulary size, average sentence length) to create a simulated pre-training corpus.
*   **Word Frequency Analysis**: Analyze and visualize the top 10 word frequencies from the generated synthetic corpus, demonstrating how data distribution influences learned patterns.
*   **Data Bias Simulation**: Introduce a controlled bias into a synthetic numerical dataset by shifting feature means for a subset of samples.
*   **Impact of Data Bias Visualization**: Compare conceptual output means between unbiased and biased data scenarios to illustrate how pre-training data bias can lead to systematically different model behavior.

### 2. Alignment & Hallucinations
*   **Conceptual Loss Minimization Simulation**: Adjust epochs, initial loss, and learning rate to simulate how an LLM learns by reducing errors during training.
*   **Loss Curve Visualization**: Observe the conceptual loss value decreasing over epochs, mimicking the optimization process.
*   **Reinforcement Learning from Human Feedback (RLHF) Simulation**: Configure feedback rounds and improvement factors to simulate how human preferences guide model behavior.
*   **Reward Signal Improvement Visualization**: Track the conceptual reward signal increasing over RLHF rounds, indicating improved alignment.
*   **Hallucination Simulation**: Assign conceptual hallucination scores to factual versus deliberately incorrect responses to highlight the risk of LLMs confidently providing wrong information.
*   **Hallucination Meter Visualization**: Compare the conceptual hallucination likelihood between different response types using a clear bar chart.

### 3. Deployment, Drift & Oversight
*   **Agentic AI System Context**: Introduction to how LLMs operate within agentic systems, amplifying risks.
*   **Model Drift Simulation**: Generate synthetic time-series data for a model performance metric (e.g., accuracy) and introduce a gradual performance degradation (drift).
*   **Drift Threshold Calculation and Detection**: Define and apply conceptual statistical thresholds ($\mu \pm k \cdot \sigma$) to detect when model performance deviates significantly from its baseline.
*   **Model Performance & Drift Visualization**: Plot model performance over time, including baseline mean and drift thresholds, clearly indicating when drift is detected.
*   **Human Oversight and Accountability**: Discuss the indispensable role of human intervention, governance, and accountability in managing AI risks in production.

## Getting Started

Follow these instructions to set up and run the Streamlit application on your local machine.

### Prerequisites

Make sure you have the following installed:

*   **Python 3.8+**
*   **pip** (Python package installer)

### Installation

1.  **Clone the repository (or download the code):**
    ```bash
    git clone https://github.com/your_username/your_project_name.git
    cd your_project_name
    ```
    *(Note: Replace `your_username/your_project_name` with the actual repository path if it exists.)*

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required Python packages:**
    Create a `requirements.txt` file in the root directory of your project with the following content:
    ```
    streamlit
    numpy
    pandas
    matplotlib
    seaborn
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**
    Ensure your virtual environment is activated and you are in the project's root directory (`qu_lab_project/`).
    ```bash
    streamlit run app.py
    ```

2.  **Access the application:**
    A new tab will automatically open in your default web browser, pointing to `http://localhost:8501` (or another port if 8501 is in use).

3.  **Navigate and Interact:**
    *   Use the **sidebar navigation** on the left to switch between the three main phases of the LLM lifecycle: "LLM Overview & Pre-training", "Alignment & Hallucinations", and "Deployment, Drift & Oversight".
    *   Within each section, use the **interactive sliders and input fields** to adjust simulation parameters (e.g., number of sentences, bias strength, learning rate, drift magnitude) and observe their immediate impact on the displayed data, charts, and conceptual outcomes.
    *   Read the accompanying explanations and insights to understand the business and risk implications of each simulation.

## Project Structure

The project is organized into the following directories and files:

```
qu_lab_project/
├── app.py                            # Main Streamlit application entry point and navigation.
├── application_pages/                # Directory containing individual Streamlit page logic.
│   ├── __init__.py                   # Makes application_pages a Python package.
│   ├── page_llm_overview_pretraining.py  # Logic for the Pre-training phase and data bias.
│   ├── page_alignment_hallucinations.py # Logic for the Alignment phase and hallucinations.
│   └── page_deployment_drift_oversight.py # Logic for the Deployment phase and model drift.
└── requirements.txt                  # Lists all Python dependencies.
└── README.md                         # This file.
```

## Technology Stack

This application is built using the following technologies:

*   **Python 3.8+**: The primary programming language.
*   **Streamlit**: The framework used for building the interactive web application.
*   **NumPy**: For numerical operations and synthetic data generation.
*   **Pandas**: For data manipulation and structuring.
*   **Matplotlib**: For creating static, animated, and interactive visualizations.
*   **Seaborn**: Built on Matplotlib, providing a high-level interface for drawing attractive and informative statistical graphics.

## Contributing

This project is primarily designed as an educational lab. While active external contributions are not formally sought for this specific lab project, we welcome feedback, suggestions for improvements, or bug reports. If you find any issues or have ideas, please open an issue in the GitHub repository.

## License

This project is licensed under the MIT License. See the `LICENSE` file (if included in the actual project) for details. If no `LICENSE` file is present, the standard MIT boilerplate applies, allowing for free use, modification, and distribution.

## Contact

This QuLab project is provided by QuantUniversity.

For inquiries or further information, you can visit:
*   [QuantUniversity Website](https://www.quantuniversity.com/)
