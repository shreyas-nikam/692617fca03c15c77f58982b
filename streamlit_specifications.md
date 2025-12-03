
# Streamlit Application Specification: LLM Journey Explorer

## 1. Application Overview

The "LLM Journey Explorer" Streamlit application aims to be an interactive educational tool for retail investors. It will demystify complex AI concepts, particularly focusing on the lifecycle of Large Language Models (LLMs) and the inherent risks at each stage.

**Learning Goals for Retail Investors:**
*   Understand the key phases of the LLM lifecycle: pre-training, alignment (SFT, RLHF), and deployment.
*   Grasp the fundamental mechanics of LLMs, such as conceptual word prediction ($P(\text{next word} | \text{previous words})$) and loss function minimization ($L = \text{Error}(\text{Predicted Output}, \text{True Output})$).
*   Identify and comprehend emergent risks: data bias, hallucinations, and model drift, including how drift is detected using basic statistics ($\text{Drift Threshold} = \mu \pm k \cdot \sigma$).
*   Appreciate the role of human feedback and oversight in LLM safety and ethical alignment.
*   Understand how risks are amplified in Agentic AI systems.
*   Emphasize the importance of human oversight and accountability throughout the LLM lifecycle.

## 2. User Interface Requirements

### Layout and Navigation Structure
The application will follow a linear, scrollable layout, mirroring the logical flow of the Jupyter Notebook. Each major section of the LLM lifecycle (Overview, Pre-training, Alignment, Deployment, Conclusion) will be clearly delineated with prominent headers.
A sidebar (`st.sidebar`) will provide quick navigation links (anchors) to these main sections, allowing users to jump directly to topics of interest.

### Input Widgets and Controls
Interactive sliders and number input fields will be used to allow users to manipulate parameters and observe immediate changes in visualizations, enhancing the learning experience.

*   **Phase 1: Pre-training - Data Generation Controls:**
    *   **Synthetic Text Corpus:**
        *   `num_sentences`: `st.slider` (e.g., 100 to 5000, default 1000) for corpus size.
        *   `vocab_size`: `st.slider` (e.g., 50 to 500, default 100) for vocabulary diversity.
        *   `avg_sentence_length`: `st.slider` (e.g., 5 to 30, default 15) for sentence complexity.
    *   **Data Bias Simulation:**
        *   `bias_strength`: `st.slider` (e.g., 0.0 to 0.5, default 0.3) to control the proportion of biased samples.
        *   `biased_feature_mean_shift`: `st.slider` (e.g., 0 to 50, default 20) to control the magnitude of bias.
*   **Phase 2: Alignment - Learning Process Controls:**
    *   **Loss Minimization:**
        *   `epochs`: `st.slider` (e.g., 10 to 100, default 50) for the number of training iterations.
        *   `initial_loss`: `st.number_input` (e.g., 1.0 to 20.0, default 10.0) for the starting loss value.
        *   `learning_rate`: `st.slider` (e.g., 0.01 to 0.2, default 0.08) for the rate of loss reduction.
    *   **RLHF Simulation:**
        *   `feedback_rounds`: `st.slider` (e.g., 1 to 20, default 10) for the number of feedback iterations.
        *   `improvement_factor`: `st.slider` (e.g., 0.05 to 0.5, default 0.2) for the reward signal increase per round.
*   **Emergent Risk: Hallucinations:**
    *   `factual_hallucination_score`: `st.slider` (e.g., 0.0 to 0.5, default 0.1) for the conceptual score of a factual response.
    *   `hallucinated_hallucination_score`: `st.slider` (e.g., 0.5 to 1.0, default 0.9) for the conceptual score of a hallucinated response.
*   **Phase 3: Deployment - Drift Detection Controls:**
    *   `num_timesteps`: `st.slider` (e.g., 20 to 100, default 50) for the length of the performance timeline.
    *   `baseline_mean_acc`: `st.slider` (e.g., 0.7 to 0.99, default 0.85) for the model's expected performance.
    *   `baseline_std_acc`: `st.slider` (e.g., 0.01 to 0.05, default 0.02) for the variability of baseline performance.
    *   `drift_start_time`: `st.slider` (e.g., 10 to `num_timesteps - 5`, default 30) for when drift begins.
    *   `drift_magnitude`: `st.slider` (e.g., 0.0 to 0.2, default 0.1) for the severity of performance drop.
    *   `k_multiplier_for_drift`: `st.slider` (e.g., 1.0 to 5.0, default 3.0) for the multiplier in the drift threshold calculation.

### Visualization Components
All visualizations from the Jupyter Notebook will be rendered using `st.pyplot()` for Matplotlib/Seaborn charts. Tabular data will be displayed using `st.dataframe()` or `st.write()`.

*   **Conceptual LLM Lifecycle Timeline:**
    *   A static `st.pyplot()` visualization of the LLM lifecycle phases: "Pre-training", "Alignment", "Deployment".
*   **Top 10 Word Frequency Distribution:**
    *   A dynamic bar plot (`st.pyplot()`) showing word frequencies based on user-controlled synthetic text generation parameters.
    *   A simple `st.write()` output of the top 10 words and their counts.
*   **Comparison of Conceptual Output Means (Bias):**
    *   A dynamic bar chart (`st.pyplot()`) comparing conceptual output means between unbiased and biased data scenarios. This will visually illustrate the impact of data bias across conceptual groups.
    *   `st.write()` output of the calculated means for both unbiased and biased scenarios.
*   **Conceptual Loss Minimization over Epochs:**
    *   A dynamic line plot (`st.pyplot()`) illustrating the conceptual decrease in loss value over training epochs, based on user-defined parameters.
    *   `st.write()` output of initial and final simulated loss values.
*   **Conceptual Reward Signal Improvement over RLHF Rounds:**
    *   A dynamic line plot (`st.pyplot()`) showing the conceptual increase in the reward signal over RLHF feedback rounds, influenced by user parameters.
    *   `st.dataframe()` displaying a small sample of synthetic human feedback data.
    *   `st.write()` output of initial simulated reward history.
*   **Conceptual Hallucination Meter:**
    *   A dynamic bar chart (`st.pyplot()`) comparing conceptual hallucination scores for factual vs. hallucinated responses, driven by user sliders.
    *   `st.write()` output detailing the conceptual query, responses, and assigned scores.
*   **Model Performance Over Time with Drift Thresholds:**
    *   A dynamic line plot (`st.pyplot()`) displaying simulated model performance over time, including baseline mean and dynamically calculated drift thresholds. It will visually indicate when drift is detected based on user parameters.
    *   `st.write()` output of baseline stats, drift thresholds, current performance, and drift detection status.

### Interactive Elements and Feedback Mechanisms
*   **Dynamic Updates:** All visualizations linked to input widgets will update automatically as slider values or number inputs change.
*   **Explanatory Text:** Rich markdown (`st.markdown`) will accompany each section, providing clear conceptual explanations, risk insights, and interpretations of the visualizations for the retail investor persona.
*   **Risk Indicators:** Where appropriate (e.g., Data Bias, Hallucinations, Model Drift sections), explicit `st.warning()` or `st.info()` messages will highlight the emergent risks and their implications.
*   **Timeline Navigation:** The sidebar will feature `st.sidebar.button` or `st.sidebar.radio` options corresponding to the main sections, enabling quick navigation.

## 3. Additional Requirements

### Annotation and Tooltip Specifications
*   **Charts:** All `st.pyplot()` charts will include clear titles, axis labels, and legends as present in the notebook. For bar charts (e.g., Hallucination Meter, Output Bias Comparison), individual bar values will be annotated to provide precise numerical feedback.
*   **Interactive Inputs:** Each slider or input field will have a descriptive label and an `st.help()` tooltip or an adjacent `st.info()` message explaining its purpose and impact on the simulation, making the interactivity intuitive for a non-technical audience.
*   **Risk Indicators:** Specific sections dealing with emergent risks (Data Bias, Hallucinations, Model Drift) will include bolded warnings and concise explanations of the risk, potentially with `st.expander` for more detailed information.

### Save the States of the Fields Properly so that Changes are Not Lost
The application will utilize Streamlit's `st.session_state` to persist the values of all input widgets across reruns. This ensures that user-defined parameters are retained even if the application refreshes or the user navigates between sections, providing a smooth and consistent interactive experience.

## 4. Notebook Content and Code Requirements

This section extracts the essential Markdown content and Python code stubs from the Jupyter Notebook, detailing how they will be incorporated into the Streamlit application.

```python
# --- Python Library Imports (to be placed at the top of the Streamlit script) ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import streamlit as st

sns.set_theme(style="whitegrid")

# --- Streamlit Configuration (to be placed at the top of the Streamlit script) ---
st.set_page_config(layout="wide", page_title="LLM Journey Explorer")
```

### Application Title and Overview

```python
st.title("LLM Journey Explorer")

st.markdown("""
This application, the "LLM Journey Explorer," is designed for retail investors to demystify Large Language Models (LLMs) and their associated risks. We will journey through the LLM lifecycle, from foundational training to real-world deployment, exploring key concepts and practical demonstrations of potential pitfalls. Our goal is to equip you with a conceptual understanding of how LLMs operate, how they can go wrong, and the importance of human oversight.
""")
```

### Section 2: The LLM Lifecycle: An Overview

```python
st.header("Section 2: The LLM Lifecycle: An Overview")
st.markdown("""
The development of a Large Language Model can be conceptually divided into three main phases:
1.  **Pre-training**: The initial phase where the model learns foundational patterns from vast amounts of text data.
2.  **Alignment**: The process of refining the model's behavior to align with human values and specific tasks.
3.  **Deployment**: When the model is put into real-world use, requiring continuous monitoring and adaptation.

Understanding these phases is critical to identifying and mitigating risks at each stage.
""")

# --- Code Stubs from Notebook ---
def plot_llm_lifecycle_timeline():
    """Generates a conceptual visual timeline of the LLM lifecycle phases."""
    stages = ["Pre-training", "Alignment", "Deployment"]
    times = [1, 2, 3]

    fig, ax = plt.subplots(figsize=(10, 2))

    ax.scatter(times, [1]*len(times), s=200, zorder=2)
    ax.plot([min(times), max(times)], [1, 1], linestyle='-', color='gray', linewidth=2)

    for i, (stage, time) in enumerate(zip(stages, times)):
        ax.annotate(stage, (time, 1.05), textcoords="offset points", xytext=(0,10),
                    ha='center', va='bottom', fontsize=12, color='navy', fontweight='bold')
        if i < len(stages) - 1:
            ax.annotate('' , xy=(times[i+1]-0.2, 1), xytext=(times[i]+0.2, 1),
                        arrowprops=dict(facecolor='darkgreen', shrink=0.05, width=2, headwidth=8))

    ax.set_yticks([])
    ax.set_xticks(times)
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylim(0.8, 1.2)
    ax.set_xlim(0.5, 3.5)
    ax.set_title("Conceptual LLM Lifecycle Timeline", fontsize=14, pad=20)
    plt.tight_layout()
    return fig # Return figure for st.pyplot()

# --- Streamlit Integration ---
st.pyplot(plot_llm_lifecycle_timeline())

st.markdown("""
The visualization above provides a high-level overview of the LLM lifecycle. Each stage involves distinct processes and introduces unique risks, which we will explore in detail.
""")
```

### Section 3: Phase 1: Pre-training - Vast Data Ingestion and Pattern Recognition

```python
st.header("Section 3: Phase 1: Pre-training - Vast Data Ingestion and Pattern Recognition")
st.markdown(r"""
Pre-training is the foundational step where LLMs are exposed to immense quantities of text and code data. During this phase, the model learns grammar, facts, reasoning abilities, and how to predict the next word in a sequence. This is fundamentally a process of identifying statistical relationships and patterns. Conceptually, the model aims to maximize the probability of predicting the correct next word given the preceding words, represented as $P(\text{next word} | \text{previous words})$.
""")

# --- Input Widgets for Synthetic Text Data ---
with st.expander("Configure Synthetic Text Data Generation"):
    num_sentences = st.slider("Number of Synthetic Sentences", 100, 5000, 1000, key='num_sentences')
    vocab_size = st.slider("Vocabulary Size", 50, 500, 100, key='vocab_size')
    avg_sentence_length = st.slider("Average Sentence Length", 5, 30, 15, key='avg_sentence_length')

# --- Code Stubs from Notebook ---
def generate_synthetic_text_data(num_sentences, vocab_size, avg_sentence_length):
    """
    Creates a list of synthetic sentences to simulate a vast pre-training corpus.
    Words are generated randomly from a fixed vocabulary.
    """
    vocabulary = [f"word_{i}" for i in range(vocab_size)]
    text_data = []
    for _ in range(num_sentences):
        sentence_length = random.randint(max(1, avg_sentence_length - 5), avg_sentence_length + 5)
        sentence = ' '.join(random.choice(vocabulary) for _ in range(sentence_length))
        text_data.append(sentence)
    return text_data

def analyze_word_frequency(text_data, top_n=10):
    """
    Calculates and returns the top `n` most frequent words from the `text_data`.
    """
    word_counts = {}
    for sentence in text_data:
        for word in sentence.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    return dict(sorted_word_counts[:top_n])

# --- Streamlit Integration ---
synthetic_text = generate_synthetic_text_data(num_sentences, vocab_size, avg_sentence_length)
word_freqs = analyze_word_frequency(synthetic_text, top_n=10)
st.write("Top 10 Synthetic Word Frequencies:", word_freqs)

st.markdown("""
The `generate_synthetic_text_data` function simulates the input data an LLM might encounter during pre-training. By analyzing word frequencies, we get a glimpse into the patterns and common co-occurrences that an LLM would learn, forming its understanding of language structure and word relationships.
""")
```

### Section 4: Visualizing Pre-training: Word Probabilities

```python
st.header("Section 4: Visualizing Pre-training: Word Probabilities")
st.markdown(r"""
The core of pre-training is learning the conditional probability $P(\text{next word} | \text{previous words})$. This means for any given sequence of words, the model learns which words are most likely to follow. A higher frequency implies a higher learned probability.
""")

# --- Code Stubs from Notebook ---
def plot_word_frequency_distribution(word_frequencies):
    """
    Visualizes the distribution of word frequencies using a bar plot.
    """
    words = list(word_frequencies.keys())
    frequencies = list(word_frequencies.values())

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=words, y=frequencies, palette="viridis", ax=ax)
    ax.set_title("Top 10 Word Frequency Distribution in Synthetic Corpus", fontsize=14)
    ax.set_xlabel("Words", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

# --- Streamlit Integration ---
st.pyplot(plot_word_frequency_distribution(word_freqs))

st.markdown("""
This bar chart visually represents the learned statistical regularities in our synthetic data. In a real LLM, these frequencies would translate into probabilities guiding its text generation, enabling it to produce coherent and contextually relevant responses.
""")
```

### Section 5: Emergent Risk: Data Bias during Pre-training

```python
st.header("Section 5: Emergent Risk: Data Bias during Pre-training")
st.warning("⚠️ **Emergent Risk: Data Bias**")
st.markdown("""
One of the most significant emergent risks during the pre-training phase is **data bias**. LLMs learn from the vast, often unfiltered, data of the internet. If this data contains societal biases (e.g., gender stereotypes, racial discrimination, specific economic viewpoints), the LLM will inadvertently encode and amplify these biases in its outputs. This can lead to skewed, unfair, or discriminatory responses, impacting critical decisions.
""")

# --- Input Widgets for Data Bias Simulation ---
with st.expander("Configure Data Bias Simulation"):
    bias_strength = st.slider("Bias Strength (Proportion of Biased Samples)", 0.0, 0.5, 0.3, key='bias_strength')
    biased_feature_mean_shift = st.slider("Biased Feature Mean Shift", 0, 50, 20, key='biased_feature_mean_shift')

# --- Code Stubs from Notebook ---
def generate_data_with_bias(num_samples, feature_dist_mean_unbiased, feature_dist_std_unbiased, bias_strength, biased_feature_mean_shift):
    """
    Generates synthetic numerical data, introducing a controlled bias in a specified feature's distribution for a subset of data.
    Returns a DataFrame with a 'feature' column and a 'conceptual_output' column.
    """
    np.random.seed(42)
    
    # Unbiased data
    unbiased_feature = np.random.normal(loc=feature_dist_mean_unbiased, scale=feature_dist_std_unbiased, size=num_samples)
    unbiased_output = 0.5 * unbiased_feature + np.random.normal(0, 5, num_samples)
    
    df = pd.DataFrame({'feature': unbiased_feature, 'conceptual_output': unbiased_output})
    
    if bias_strength > 0:
        num_biased_samples = int(num_samples * bias_strength)
        
        # Create a subset of biased samples by shifting the feature mean
        biased_feature = np.random.normal(loc=feature_dist_mean_unbiased + biased_feature_mean_shift, 
                                          scale=feature_dist_std_unbiased, size=num_biased_samples)
        biased_output = 0.5 * biased_feature + np.random.normal(0, 5, num_biased_samples) + (biased_feature_mean_shift / 2)
        
        # Randomly replace a subset of unbiased samples with biased ones
        replace_indices = np.random.choice(num_samples, num_biased_samples, replace=False)
        df.loc[replace_indices, 'feature'] = biased_feature
        df.loc[replace_indices, 'conceptual_output'] = biased_output
        
    return df

def analyze_output_distribution_by_group(data, feature_col, output_col, threshold=50):
    """
    Calculates the mean and standard deviation of an `output_col` for different groups
    based on values in a `feature_col`.
    """
    group1 = data[data[feature_col] < threshold]
    group2 = data[data[feature_col] >= threshold]
    
    return {
        "group1_mean": group1[output_col].mean(),
        "group1_std": group1[output_col].std(),
        "group2_mean": group2[output_col].mean(),
        "group2_std": group2[output_col].std()
    }

# --- Streamlit Integration ---
# Generate unbiased data
unbiased_data_df = generate_data_with_bias(num_samples=1000, feature_dist_mean_unbiased=50,
                                           feature_dist_std_unbiased=10, bias_strength=0.0,
                                           biased_feature_mean_shift=0)
unbiased_outputs = analyze_output_distribution_by_group(unbiased_data_df, 'feature', 'conceptual_output')

# Generate biased data
biased_data_df = generate_data_with_bias(num_samples=1000, feature_dist_mean_unbiased=50,
                                         feature_dist_std_unbiased=10, bias_strength=bias_strength,
                                         biased_feature_mean_shift=biased_feature_mean_shift)
biased_outputs = analyze_output_distribution_by_group(biased_data_df, 'feature', 'conceptual_output')

st.write("Unbiased Outputs (Group 1 Mean, Group 2 Mean):", unbiased_outputs)
st.write("Biased Outputs (Group 1 Mean, Group 2 Mean):", biased_outputs)

st.markdown("""
Here, we've simulated two scenarios: one with unbiased data and another where an artificial bias is introduced into a "feature." Observe how the `conceptual_output` means differ significantly between the groups in the biased scenario, mimicking how a real LLM would propagate and amplify these input biases into its generated content.
""")
```

### Section 6: Visualizing the Impact of Data Bias

```python
st.header("Section 6: Visualizing the Impact of Data Bias")
st.markdown("""
Visualizing the output differences between groups is essential to highlight the impact of data bias. This directly shows how skewed inputs can lead to skewed, unfair, or misrepresentative outputs from an LLM.
""")

# --- Code Stubs from Notebook ---
def plot_output_bias_comparison(unbiased_g1_mean, unbiased_g2_mean, biased_g1_mean, biased_g2_mean, group_labels):
    """
    Compares the mean outputs of unbiased and biased data scenarios using bar plots.
    """
    labels = group_labels
    unbiased_means = [unbiased_g1_mean, unbiased_g2_mean]
    biased_means = [biased_g1_mean, biased_g2_mean]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, unbiased_means, width, label='Unbiased Data', color='skyblue')
    rects2 = ax.bar(x + width/2, biased_means, width, label='Biased Data', color='lightcoral')

    ax.set_xlabel("Groups", fontsize=12)
    ax.set_ylabel("Conceptual Output Mean", fontsize=12)
    ax.set_title("Comparison of Conceptual Output Means: Unbiased vs. Biased Data", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend()
    ax.set_ylim(min(min(unbiased_means), min(biased_means)) * 0.8, max(max(unbiased_means), max(biased_means)) * 1.2)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    return fig

# --- Streamlit Integration ---
st.pyplot(plot_output_bias_comparison(unbiased_outputs['group1_mean'], unbiased_outputs['group2_mean'],
                                biased_outputs['group1_mean'], biased_outputs['group2_mean'],
                                group_labels=["Group A (Feature < 50)", "Group B (Feature >= 50)"]))

st.markdown("""
This chart clearly illustrates how a bias in the training data can lead to disparate impacts in the model's conceptual outputs across different groups. For example, if 'Group A' represents a demographic group and 'conceptual_output' is a score, the biased model might consistently give lower scores to Group A.
""")
```

### Section 7: Phase 2: Alignment - Steering LLM Behavior with Human Values

```python
st.header("Section 7: Phase 2: Alignment - Steering LLM Behavior with Human Values")
st.markdown("""
After pre-training, LLMs are **aligned** to make them more helpful, honest, and harmless. This critical phase refines the model's behavior to follow instructions, avoid generating harmful content, and generally align with human values. Key techniques include Supervised Fine-Tuning (SFT) and Reinforcement Learning from Human Feedback (RLHF).
""")
st.markdown("""
Alignment is where human judgment plays a direct role in shaping an LLM's ethical and practical behavior.
""")
```

### Section 8: The Conceptual Loss Function: Guiding Model Learning

```python
st.header("Section 8: The Conceptual Loss Function: Guiding Model Learning")
st.markdown(r"""
During both pre-training and alignment, models learn by iteratively minimizing a **loss function**. This function quantifies the "error" or "discrepancy" between the model's predicted output and the desired (true) output. The goal of training is to adjust the model's internal parameters to make this loss as small as possible. Conceptually, a simple loss function can be expressed as:
$$L = \text{Error}(\text{Predicted Output}, \text{True Output})$$
Minimizing $L$ means the model is getting "closer" to generating the desired outputs.
""")

# --- Input Widgets for Loss Minimization Simulation ---
with st.expander("Configure Loss Minimization Simulation"):
    epochs = st.slider("Epochs", 10, 100, 50, key='epochs')
    initial_loss = st.number_input("Initial Loss", 1.0, 20.0, 10.0, step=0.5, key='initial_loss')
    learning_rate = st.slider("Learning Rate", 0.01, 0.2, 0.08, step=0.01, key='learning_rate_loss')

# --- Code Stubs from Notebook ---
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

# --- Streamlit Integration ---
loss_values = simulate_loss_minimization(epochs, initial_loss, learning_rate)
st.write("Simulated Loss Values (first 5):", [f"{l:.2f}" for l in loss_values[:5]])
st.write("Simulated Loss Values (last 5):", [f"{l:.2f}" for l in loss_values[-5:]])

st.markdown("""
The simulated `loss_values` show a decreasing trend, representing the model's iterative process of learning from data and reducing its errors. This is the fundamental mechanism behind an LLM's ability to improve.
""")
```

### Section 9: Visualizing Loss Function Minimization

```python
st.header("Section 9: Visualizing Loss Function Minimization")
st.markdown("""
Visualizing the loss function over time (or training "epochs") helps us understand how effectively the model is learning. A steadily decreasing curve indicates that the model is successfully optimizing its parameters.
""")

# --- Code Stubs from Notebook ---
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

# --- Streamlit Integration ---
st.pyplot(plot_loss_curve(loss_values))

st.markdown("""
The downward slope of the curve demonstrates the optimization process. As the LLM processes more data and adjusts its internal weights, the discrepancy between its predictions and the desired outcomes (its "error") decreases.
""")
```

### Section 10: Simulating Reinforcement Learning from Human Feedback (RLHF)

```python
st.header("Section 10: Simulating Reinforcement Learning from Human Feedback (RLHF)")
st.markdown("""
Reinforcement Learning from Human Feedback (RLHF) is a powerful alignment technique. It involves:
1.  An LLM generates multiple responses to a prompt.
2.  Human annotators rank or rate these responses based on quality, helpfulness, and safety.
3.  A separate "reward model" is trained on these human preferences.
4.  The LLM is then fine-tuned using reinforcement learning to maximize the reward signal from this reward model, effectively learning to produce responses that humans prefer. This is an iterative process, continuously refining the model.
""")

# --- Input Widgets for RLHF Simulation ---
with st.expander("Configure RLHF Simulation"):
    feedback_rounds = st.slider("Feedback Rounds", 1, 20, 10, key='feedback_rounds')
    improvement_factor = st.slider("Reward Improvement Factor", 0.05, 0.5, 0.2, step=0.01, key='improvement_factor')

# --- Code Stubs from Notebook ---
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

# --- Streamlit Integration ---
feedback_data = generate_rlhf_feedback_data(num_samples=5) # Fixed samples for display
st.dataframe(feedback_data)

reward_history = simulate_reward_signal_improvement(initial_reward=0.1, feedback_rounds=feedback_rounds, improvement_factor=improvement_factor)
st.write("\nSimulated Reward History (first 5):", [f"{r:.2f}" for r in reward_history[:5]])

st.markdown("""
The `feedback_data` table shows how human evaluators might choose between different LLM outputs. The `reward_history` then conceptually demonstrates how the model's ability to generate preferred responses improves with more rounds of such human feedback, driving alignment.
""")
```

### Section 11: Visualizing Reward Signal Improvement

```python
st.header("Section 11: Visualizing Reward Signal Improvement")
st.markdown("""
The progress of RLHF can be visualized by observing the improvement in the "reward signal." As the reward model learns to accurately capture human preferences and the LLM learns to maximize this reward, the signal should ideally increase, indicating better alignment.
""")

# --- Code Stubs from Notebook ---
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

# --- Streamlit Integration ---
st.pyplot(plot_reward_signal(reward_history))

st.markdown("""
This upward-sloping curve signifies the success of the alignment process. Each "feedback round" allows the model to better understand and incorporate human values, leading to more desirable and safer outputs.
""")
```

### Section 12: Emergent Risk: Hallucinations - Factual Inaccuracies

```python
st.header("Section 12: Emergent Risk: Hallucinations - Factual Inaccuracies")
st.warning("⚠️ **Emergent Risk: Hallucinations**")
st.markdown("""
**Hallucinations** are a critical emergent risk where LLMs generate outputs that are factually incorrect or nonsensical, yet appear credible and fluent. These can range from minor inaccuracies to completely fabricated information. Hallucinations are particularly dangerous in high-stakes applications like financial advice or medical diagnosis.
""")

# --- Input Widgets for Hallucination Simulation ---
with st.expander("Configure Hallucination Simulation"):
    factual_score = st.slider("Conceptual Hallucination Score (Factual Response)", 0.0, 1.0, 0.1, step=0.05, key='factual_hallucination_score')
    hallucinated_score = st.slider("Conceptual Hallucination Score (Hallucinated Response)", 0.0, 1.0, 0.9, step=0.05, key='hallucinated_hallucination_score')

# --- Code Stubs from Notebook ---
def simulate_hallucination_likelihood(input_query, actual_answer, simulated_llm_response, hallucination_score):
    """
    Assigns a conceptual hallucination score (0.0 to 1.0) to an LLM response.
    Higher score means higher likelihood of hallucination.
    """
    return {"query": input_query, "response": simulated_llm_response,
            "factual_correctness": (actual_answer.lower().strip() == simulated_llm_response.lower().strip()),
            "hallucination_score": hallucination_score}

# --- Streamlit Integration ---
query_example = "What is the capital of France?"
actual_answer_example = "Paris"

# Simulate a factual response
response_factual_example = "The capital of France is Paris."
hallucination_info_factual = simulate_hallucination_likelihood(query_example, actual_answer_example, response_factual_example, factual_score)
st.write("Factual Response Example:", hallucination_info_factual)

# Simulate a hallucinated response
response_hallucinated_example = "The capital of France is Rome."
hallucination_info_hallucinated = simulate_hallucination_likelihood(query_example, actual_answer_example, response_hallucinated_example, hallucinated_score)
st.write("Hallucinated Response Example:", hallucination_info_hallucinated)

st.markdown("""
The output shows two simulated LLM responses to the same query, one factual and one hallucinated. The `hallucination_score` is a conceptual metric that helps quantify the model's confidence or reliability. A high score indicates a higher risk of the information being incorrect.
""")
```

### Section 13: Visualizing Hallucination Likelihood

```python
st.header("Section 13: Visualizing Hallucination Likelihood")
st.markdown("""
A "hallucination meter" can conceptually represent the reliability of an LLM's output. By visualizing a score, users can gauge the uncertainty or potential for inaccuracy, prompting them to verify critical information.
""")

# --- Code Stubs from Notebook ---
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

# --- Streamlit Integration ---
st.pyplot(plot_hallucination_meter(hallucination_info_factual['hallucination_score'],
                             hallucination_info_hallucinated['hallucination_score']))

st.markdown("""
This visualization makes the difference in reliability stark. The significantly higher hallucination score for the incorrect response serves as a visual warning, underscoring the importance of critical evaluation of LLM outputs.
""")
```

### Section 14: Introduction to Agentic AI Systems and Risk Amplification

```python
st.header("Section 14: Introduction to Agentic AI Systems and Risk Amplification")
st.markdown("""
While LLMs are powerful, their capabilities are greatly expanded in **Agentic AI systems**. These systems are designed to perceive, reason, plan, and act autonomously, often by leveraging LLMs as their "brains" to make decisions and interact with tools and environments.
This increased autonomy, however, inherently **amplifies risks**. Errors or biases that might be contained within an LLM can cascade into real-world consequences when an agent takes autonomous action. Risks include mis-planned goals, unintended actions, and the potential for magnified errors.
""")
st.markdown("""
Understanding Agentic AI is key because it represents a major shift towards more autonomous systems. While powerful, this autonomy demands even greater vigilance regarding the underlying LLM's reliability and ethical alignment.
""")
```

### Section 15: Phase 3: Deployment - Continuous Monitoring and Adaptation

```python
st.header("Section 15: Phase 3: Deployment - Continuous Monitoring and Adaptation")
st.markdown("""
Once an LLM is deployed into a real-world application, the lifecycle continues with **continuous monitoring**. This phase is crucial for ensuring the model remains robust, performs as expected, and adapts to new data distributions or changing user behaviors. Without vigilant monitoring, models can degrade, leading to performance issues and the re-emergence of risks.
""")
st.markdown("""
Deployment is not the end of the LLM journey, but a new beginning of active management and oversight.
""")
```

### Section 16: Emergent Risk: Model Drift - Shifting Performance

```python
st.header("Section 16: Emergent Risk: Model Drift - Shifting Performance")
st.warning("⚠️ **Emergent Risk: Model Drift**")
st.markdown(r"""
**Model drift** (or concept drift) occurs when the statistical properties of the target variable, or the relationship between the input variables and the target variable, change over time. In LLMs, this can mean the model's performance degrades because the real-world data it encounters diverges significantly from its training data.
To detect drift, we can establish a **Drift Threshold** based on the model's baseline performance, often defined using basic statistics:
$$\text{Drift Threshold} = \mu \pm k \cdot \sigma$$
where $\mu$ is the mean, $\sigma$ is the standard deviation of the performance metric during a stable baseline period, and $k$ is a multiplier (e.g., 2 or 3 for standard deviations) to define the acceptable range.
""")

# --- Input Widgets for Model Drift Simulation ---
with st.expander("Configure Model Drift Simulation"):
    num_timesteps = st.slider("Number of Time Steps", 20, 100, 50, key='num_timesteps')
    baseline_mean_acc = st.slider("Baseline Mean Accuracy", 0.7, 0.99, 0.85, step=0.01, key='baseline_mean_acc')
    baseline_std_acc = st.slider("Baseline Std Dev for Accuracy", 0.01, 0.05, 0.02, step=0.005, key='baseline_std_acc')
    drift_start_time = st.slider("Drift Start Time Step", 10, num_timesteps - 5, 30, key='drift_start_time')
    drift_magnitude = st.slider("Drift Magnitude (Performance Drop)", 0.0, 0.2, 0.1, step=0.01, key='drift_magnitude')
    k_multiplier_for_drift = st.slider(r"Multiplier ($k$) for Drift Threshold ($\mu \pm k \cdot \sigma$)", 1.0, 5.0, 3.0, step=0.5, key='k_multiplier_for_drift')


# --- Code Stubs from Notebook ---
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
    latest_performance = performance_data[-1] if len(performance_data) > 0 else baseline_mean # Handle empty data
    return latest_performance < lower or latest_performance > upper

# --- Streamlit Integration ---
performance_data_over_time = generate_time_series_performance_data(num_timesteps,
                                                                   baseline_mean_acc,
                                                                   baseline_std_acc,
                                                                   drift_start_time,
                                                                   drift_magnitude)

upper_bound, lower_bound = calculate_drift_threshold(baseline_mean_acc, baseline_std_acc, k_multiplier_for_drift)
drift_detected_status = detect_conceptual_drift(performance_data_over_time, baseline_mean_acc,
                                                baseline_std_acc, k_multiplier_for_drift)

st.write(f"Baseline Mean Accuracy: {baseline_mean_acc:.2f}, Std Dev: {baseline_std_acc:.2f}")
st.write(f"Drift Threshold (Lower, Upper): ({lower_bound:.2f}, {upper_bound:.2f})")
if performance_data_over_time:
    st.write(f"Current Performance: {performance_data_over_time[-1]:.2f}")
st.write(f"Drift Detected: {drift_detected_status}")

st.markdown("""
Here, we've simulated a model's performance over 50 time steps. Initially stable, the performance conceptually drops after a certain point, simulating drift. The calculated drift thresholds provide boundaries, and our detector indicates if the current performance falls outside this acceptable range, signaling potential drift.
""")
```

### Section 17: Visualizing Model Drift

```python
st.header("Section 17: Visualizing Model Drift")
st.markdown("""
A clear visualization of performance over time, alongside the calculated drift thresholds, helps in quickly identifying when a model begins to "drift" and its behavior deviates significantly from its expected baseline.
""")

# --- Code Stubs from Notebook ---
def plot_model_performance_with_drift_threshold(performance_data, baseline_mean, upper_bound, lower_bound, drift_detected):
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
        # Highlight the area where drift occurred, conceptually from drift_start_time
        drift_start_index = 0
        for i, perf in enumerate(performance_data):
            if perf < lower_bound or perf > upper_bound:
                drift_start_index = i
                break
        ax.axvspan(drift_start_index, len(performance_data) - 1, color='red', alpha=0.1, label='Drift Detected')
        # Only annotate if there's enough space
        if drift_start_index < len(performance_data) - 1:
            ax.annotate('Drift Detected!', xy=(drift_start_index, performance_data[drift_start_index]),
                        xytext=(drift_start_index + 5, performance_data[drift_start_index] + 0.05),
                        arrowprops=dict(facecolor='black', shrink=0.05), fontsize=12, color='red')

    ax.set_title("Model Performance Over Time with Drift Thresholds", fontsize=16)
    ax.set_xlabel("Time Step", fontsize=12)
    ax.set_ylabel("Performance Metric (e.g., Accuracy)", fontsize=12)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig

# --- Streamlit Integration ---
st.pyplot(plot_model_performance_with_drift_threshold(performance_data_over_time, baseline_mean_acc,
                                                upper_bound, lower_bound, drift_detected_status))

st.markdown("""
The plot visually confirms the model drift. The blue line (performance) drops below the lower drift threshold, clearly indicating a significant deviation from its stable operating behavior. Such a detection would trigger a need for investigation, potential retraining, or other mitigation strategies.
""")
```

### Section 18: The Importance of Human Oversight and Accountability

```python
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
```

### Section 19: Conclusion and Key Takeaways

```python
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
```
