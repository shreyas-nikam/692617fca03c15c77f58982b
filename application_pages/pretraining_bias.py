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
def generate_data_with_bias(num_samples, feature_dist_mean_unbiased, feature_dist_std_unbiased, bias_strength, biased_feature_mean_shift):
    """Generates synthetic numerical data, introducing a controlled bias in a feature."""
    np.random.seed(42)

    unbiased_feature = np.random.normal(
        loc=feature_dist_mean_unbiased,
        scale=feature_dist_std_unbiased,
        size=num_samples,
    )
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
    """Calculates the mean and std of an output column for two groups split by a feature."""
    group1 = data[data[feature_col] < threshold]
    group2 = data[data[feature_col] >= threshold]

    return {
        "group1_mean": float(group1[output_col].mean()),
        "group1_std": float(group1[output_col].std()),
        "group2_mean": float(group2[output_col].mean()),
        "group2_std": float(group2[output_col].std()),
    }


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


def main():
    st.title("📚 Phase 1 – Pre-training & Data Bias")

    st.markdown(r"""
### 🏦 Scenario: Training Your Broker\'s Research Assistant

Your broker wants to train an LLM on **millions of financial documents**: news, analyst reports, blogs, and social media posts. This is the **pre-training** phase, where the LLM learns raw patterns of language and finance.

In this page you will:

1. Generate a **synthetic text corpus** and inspect word frequencies.
2. See how this connects to the probability view $P(\text{next word} \mid \text{previous words})$.
3. Simulate **data bias** and see how it skews conceptual outputs.
""")

    st.subheader("Step 1️⃣ – Configure Synthetic Pre-training Corpus")
    with st.expander("Configure Synthetic Text Data Generation", expanded=True):
        num_sentences = st.slider(
            "Number of Synthetic Sentences",
            100,
            5000,
            1000,
            key="num_sentences",
            help="Controls how many synthetic sentences are generated to mimic a pre-training corpus.",
        )
        vocab_size = st.slider(
            "Vocabulary Size",
            50,
            500,
            100,
            key="vocab_size",
            help="A larger vocabulary introduces more unique 'words' into the synthetic corpus.",
        )
        avg_sentence_length = st.slider(
            "Average Sentence Length",
            5,
            30,
            15,
            key="avg_sentence_length",
            help="Longer sentences conceptually mean more context for next-word prediction.",
        )

    synthetic_text = generate_synthetic_text_data(num_sentences, vocab_size, avg_sentence_length)
    word_freqs = analyze_word_frequency(synthetic_text, top_n=10)

    st.markdown("#### 🔍 Top 10 Synthetic Word Frequencies")
    st.write(word_freqs)

    st.markdown(r"""
💡 **What\'s happening here?**

You just generated a toy version of the massive corpora that real LLMs see. The frequency with which words appear becomes a crude proxy for the model\'s sense of **what is typical**. In a real LLM, these observations feed into a much richer estimate of conditional probabilities like $P(\text{next word} \mid \text{previous words})$.
""")

    st.subheader("Step 2️⃣ – Visualize Word Frequency Distribution")
    st.pyplot(plot_word_frequency_distribution(word_freqs))

    st.markdown(r"""
The bars above show which tokens dominate your synthetic corpus. If certain terms appear far more often, the model will be more inclined to predict them, shaping how it completes your prompts.
""")

    st.subheader("Step 3️⃣ – Simulate Data Bias in Pre-training")
    st.warning("⚠️ **Emergent Risk: Data Bias** – Skewed training data leads to skewed model behavior.")

    st.markdown(
        """
Imagine that many of the documents your broker uses for training systematically favor a
particular sector, geography, or investor profile. Over time, the LLM may internalize
this skew, leading to **unequal or unfair treatment** of different groups of investors.
"""
    )

    with st.expander("Configure Data Bias Simulation", expanded=True):
        bias_strength = st.slider(
            "Bias Strength (Proportion of Biased Samples)",
            0.0,
            0.5,
            0.3,
            key="bias_strength",
            help="Higher values mean a larger share of the dataset is biased.",
        )
        biased_feature_mean_shift = st.slider(
            "Biased Feature Mean Shift",
            0,
            50,
            20,
            key="biased_feature_mean_shift",
            help="Controls how far the biased subgroup is shifted from the baseline.",
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

    st.markdown("#### 📊 Group-wise Conceptual Outputs")
    st.write("Unbiased Scenario:", unbiased_outputs)
    st.write("Biased Scenario:", biased_outputs)

    st.markdown(
        """
Here we split the data into two groups based on a feature threshold (for example,
income level, geography, or portfolio size). When bias is introduced, the **average
output** for one group can diverge significantly from the other.
"""
    )

    st.subheader("Step 4️⃣ – Visualize Impact of Data Bias")
    fig_bias = plot_output_bias_comparison(
        unbiased_outputs["group1_mean"],
        unbiased_outputs["group2_mean"],
        biased_outputs["group1_mean"],
        biased_outputs["group2_mean"],
        group_labels=["Group A (Feature < 50)", "Group B (Feature >= 50)"],
    )
    st.pyplot(fig_bias)

    st.info(
        "Bias in training data can silently translate into **systematic differences** in how an AI system evaluates or responds to different groups. As an investor, you want assurances that such effects are monitored and mitigated."
    )


if __name__ == "__main__":
    main()
