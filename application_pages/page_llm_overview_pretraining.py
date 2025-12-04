import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

sns.set_theme(style="whitegrid")


@st.cache_data(ttl="2h")
def generate_synthetic_text_data(num_sentences, vocab_size, avg_sentence_length):
    """Creates a list of synthetic sentences to simulate a vast pre-training corpus."""
    vocabulary = [f"word_{i}" for i in range(vocab_size)]
    text_data = []
    for _ in range(num_sentences):
        sentence_length = random.randint(max(1, avg_sentence_length - 5), avg_sentence_length + 5)
        sentence = " ".join(random.choice(vocabulary) for _ in range(sentence_length))
        text_data.append(sentence)
    return text_data


@st.cache_data(ttl="2h")
def analyze_word_frequency(text_data, top_n=10):
    """Calculates and returns the top `n` most frequent words from the `text_data`."""
    word_counts = {}
    for sentence in text_data:
        for word in sentence.split():
            word_counts[word] = word_counts.get(word, 0) + 1
    sorted_word_counts = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    return dict(sorted_word_counts[:top_n])


@st.cache_data(ttl="2h")
def generate_data_with_bias(num_samples, feature_dist_mean_unbiased, feature_dist_std_unbiased, bias_strength, biased_feature_mean_shift):
    """Generates synthetic numerical data, introducing a controlled bias in a specified feature's distribution."""
    np.random.seed(42)
    unbiased_feature = np.random.normal(loc=feature_dist_mean_unbiased, scale=feature_dist_std_unbiased, size=num_samples)
    unbiased_output = 0.5 * unbiased_feature + np.random.normal(0, 5, num_samples)
    df = pd.DataFrame({"feature": unbiased_feature, "conceptual_output": unbiased_output})
    if bias_strength > 0:
        num_biased_samples = int(num_samples * bias_strength)
        biased_feature = np.random.normal(
            loc=feature_dist_mean_unbiased + biased_feature_mean_shift,
            scale=feature_dist_std_unbiased,
            size=num_biased_samples,
        )
        biased_output = 0.5 * biased_feature + np.random.normal(0, 5, num_biased_samples) + (biased_feature_mean_shift / 2)
        replace_indices = np.random.choice(num_samples, num_biased_samples, replace=False)
        df.loc[replace_indices, "feature"] = biased_feature
        df.loc[replace_indices, "conceptual_output"] = biased_output
    return df


@st.cache_data(ttl="2h")
def analyze_output_distribution_by_group(data, feature_col, output_col, threshold=50):
    """Calculates mean and std of an output column for groups split by a feature threshold."""
    group1 = data[data[feature_col] < threshold]
    group2 = data[data[feature_col] >= threshold]
    return {
        "group1_mean": group1[output_col].mean(),
        "group1_std": group1[output_col].std(),
        "group2_mean": group2[output_col].mean(),
        "group2_std": group2[output_col].std(),
    }


@st.cache_data(ttl="2h")
def plot_llm_lifecycle_timeline():
    """Generates a conceptual visual timeline of the LLM lifecycle phases."""
    stages = ["Pre-training", "Alignment", "Deployment"]
    times = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.scatter(times, [1] * len(times), s=200, zorder=2)
    ax.plot([min(times), max(times)], [1, 1], linestyle="-", color="gray", linewidth=2)
    for i, (stage, time) in enumerate(zip(stages, times)):
        ax.annotate(
            stage,
            (time, 1.05),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            va="bottom",
            fontsize=12,
            color="navy",
            fontweight="bold",
        )
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(times[i + 1] - 0.2, 1),
                xytext=(times[i] + 0.2, 1),
                arrowprops=dict(facecolor="darkgreen", shrink=0.05, width=2, headwidth=8),
            )
    ax.set_yticks([])
    ax.set_xticks(times)
    ax.set_xticklabels(stages, fontsize=10)
    ax.set_ylim(0.8, 1.2)
    ax.set_xlim(0.5, 3.5)
    ax.set_title("Conceptual LLM Lifecycle Timeline", fontsize=14, pad=20)
    plt.tight_layout()
    return fig


@st.cache_data(ttl="2h")
def plot_word_frequency_distribution(word_frequencies):
    """Visualizes the distribution of word frequencies using a bar plot."""
    words = list(word_frequencies.keys())
    frequencies = list(word_frequencies.values())
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=words, y=frequencies, palette="viridis", ax=ax)
    ax.set_title("Top 10 Word Frequency Distribution in Synthetic Corpus", fontsize=14)
    ax.set_xlabel("Words", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    return fig


@st.cache_data(ttl="2h")
def plot_output_bias_comparison(unbiased_g1_mean, unbiased_g2_mean, biased_g1_mean, biased_g2_mean, group_labels):
    """Compares the mean outputs of unbiased and biased data scenarios using bar plots."""
    labels = group_labels
    unbiased_means = [unbiased_g1_mean, unbiased_g2_mean]
    biased_means = [biased_g1_mean, biased_g2_mean]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width / 2, unbiased_means, width, label="Unbiased Data", color="skyblue")
    rects2 = ax.bar(x + width / 2, biased_means, width, label="Biased Data", color="lightcoral")
    ax.set_xlabel("Groups", fontsize=12)
    ax.set_ylabel("Conceptual Output Mean", fontsize=12)
    ax.set_title("Comparison of Conceptual Output Means: Unbiased vs. Biased Data", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.legend()
    ax.set_ylim(
        min(min(unbiased_means), min(biased_means)) * 0.8,
        max(max(unbiased_means), max(biased_means)) * 1.2,
    )

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )

    autolabel(rects1)
    autolabel(rects2)
    plt.tight_layout()
    return fig


def _init_pretraining_state():
    defaults = {
        "num_sentences": 1000,
        "vocab_size": 100,
        "avg_sentence_length": 15,
        "bias_strength": 0.3,
        "biased_feature_mean_shift": 20,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


@st.fragment
def main():
    _init_pretraining_state()
    st.header("Section 2: The LLM Lifecycle: An Overview")
    st.markdown(
        """
You are evaluating a fintech startup that claims to use an "advanced LLM stack" for portfolio insights. Before trusting their pitch, you want to understand **where risks can sneak in** across the model lifecycle.

This page walks you through the **Pre-training** phase and the first emergent risk: **data bias**.
"""
    )
    st.markdown(
        """
The development of a Large Language Model can be conceptually divided into three main phases:

1. **Pre-training**: The initial phase where the model learns foundational patterns from vast amounts of text data.
2. **Alignment**: The process of refining the model's behavior to align with human values and specific tasks.
3. **Deployment**: When the model is put into real-world use, requiring continuous monitoring and adaptation.

Understanding these phases is critical to identifying and mitigating risks at each stage.
"""
    )
    st.pyplot(plot_llm_lifecycle_timeline())
    st.markdown(
        """
The visualization above provides a high-level overview of the LLM lifecycle. Each stage introduces different **risk surfaces**: data issues during pre-training, value alignment during fine-tuning, and real-world drift during deployment.
"""
    )
    st.header("Section 3: Phase 1: Pre-training - Vast Data Ingestion and Pattern Recognition")
    st.markdown(
        r"""
Pre-training is the foundational step where LLMs are exposed to immense quantities of text and code data. Conceptually, the model aims to maximize the probability of predicting the correct next word given the preceding words, represented as $P(\text{next word} | \text{previous words})$.
"""
    )
    with st.expander("Configure Synthetic Text Data Generation"):
        st.markdown(
            """
Use these controls to **shape a synthetic corpus**. Imagine this as a rough proxy for the internet text an LLM might see:

* More sentences = more training data.
* Larger vocabulary = more linguistic variety.
* Longer sentences = more complex reasoning chains.
"""
        )
        num_sentences = st.slider(
            "Number of Synthetic Sentences",
            100,
            5000,
            int(st.session_state["num_sentences"]),
            key="num_sentences",
        )
        vocab_size = st.slider(
            "Vocabulary Size",
            50,
            500,
            int(st.session_state["vocab_size"]),
            key="vocab_size",
        )
        avg_sentence_length = st.slider(
            "Average Sentence Length",
            5,
            30,
            int(st.session_state["avg_sentence_length"]),
            key="avg_sentence_length",
        )
    synthetic_text = generate_synthetic_text_data(num_sentences, vocab_size, avg_sentence_length)
    word_freqs = analyze_word_frequency(synthetic_text, top_n=10)
    st.subheader("Top 10 Synthetic Word Frequencies")
    st.write(word_freqs)
    st.markdown(
        """
**Business intuition:** these frequencies represent which concepts the model sees most often. If certain topics dominate the corpus, the model will over-index on them in its outputs.
"""
    )
    st.header("Section 4: Visualizing Pre-training: Word Probabilities")
    st.markdown(
        r"""
The core of pre-training is learning the conditional probability $P(\text{next word} | \text{previous words})$. More frequent words and phrases lead to **higher learned probabilities**, which then shape what the model tends to say.
"""
    )
    st.pyplot(plot_word_frequency_distribution(word_freqs))
    st.markdown(
        """
Try adjusting the sliders above and watch how the bar chart changes. This mirrors how changing the underlying data distribution changes what an LLM learns to emphasize.
"""
    )
    st.header("Section 5: Emergent Risk: Data Bias during Pre-training")
    st.warning("⚠️ Emergent Risk: Data Bias")
    st.markdown(
        """
If the pre-training data over-represents certain viewpoints or demographics, the model will **bake those preferences into its behavior**. For an investor, this can mean models that systematically favor or disfavor certain clients, products, or narratives.
"""
    )
    with st.expander("Configure Data Bias Simulation"):
        st.markdown(
            """
Here you will create two synthetic datasets:

* One **unbiased** benchmark.
* One where a subset of samples has its feature values shifted, simulating a **systematic skew** in the data.
"""
        )
        bias_strength = st.slider(
            "Bias Strength (Proportion of Biased Samples)",
            0.0,
            0.5,
            float(st.session_state["bias_strength"]),
            key="bias_strength",
        )
        biased_feature_mean_shift = st.slider(
            "Biased Feature Mean Shift",
            0,
            50,
            int(st.session_state["biased_feature_mean_shift"]),
            key="biased_feature_mean_shift",
        )
    unbiased_data_df = generate_data_with_bias(
        num_samples=1000,
        feature_dist_mean_unbiased=50,
        feature_dist_std_unbiased=10,
        bias_strength=0.0,
        biased_feature_mean_shift=0,
    )
    unbiased_outputs = analyze_output_distribution_by_group(
        unbiased_data_df,
        "feature",
        "conceptual_output",
    )
    biased_data_df = generate_data_with_bias(
        num_samples=1000,
        feature_dist_mean_unbiased=50,
        feature_dist_std_unbiased=10,
        bias_strength=bias_strength,
        biased_feature_mean_shift=biased_feature_mean_shift,
    )
    biased_outputs = analyze_output_distribution_by_group(
        biased_data_df,
        "feature",
        "conceptual_output",
    )
    st.subheader("Group-wise Conceptual Outputs")
    st.write("Unbiased scenario:", unbiased_outputs)
    st.write("Biased scenario:", biased_outputs)
    st.markdown(
        """
Notice how even a modest bias strength can **shift the group means**. In a real system, this could translate into **systematically different scores or recommendations** for different customer segments.
"""
    )
    st.header("Section 6: Visualizing the Impact of Data Bias")
    st.markdown(
        """
Visualizing the impact side by side helps you see where fairness or regulatory red flags might appear.
"""
    )
    st.pyplot(
        plot_output_bias_comparison(
            unbiased_outputs["group1_mean"],
            unbiased_outputs["group2_mean"],
            biased_outputs["group1_mean"],
            biased_outputs["group2_mean"],
            group_labels=["Group A (Feature < 50)", "Group B (Feature >= 50)"],
        )
    )
    st.info(
        "Try increasing the bias strength and mean shift until the red bars clearly diverge from the blue bars. This is the kind of effect that, in production, would call for bias audits and mitigation."
    )
