"""This page covers the LLM Lifecycle Overview and the Pre-training phase, including data ingestion, word probabilities, and data bias.
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
    if 'num_sentences' not in st.session_state:
        st.session_state.num_sentences = 1000
    if 'vocab_size' not in st.session_state:
        st.session_state.vocab_size = 100
    if 'avg_sentence_length' not in st.session_state:
        st.session_state.avg_sentence_length = 15
    if 'bias_strength' not in st.session_state:
        st.session_state.bias_strength = 0.3
    if 'biased_feature_mean_shift' not in st.session_state:
        st.session_state.biased_feature_mean_shift = 20

    st.title("LLM Journey Explorer")

    st.markdown("""
    This application, the "LLM Journey Explorer," is designed for retail investors to demystify Large Language Models (LLMs) and their associated risks. We will journey through the LLM lifecycle, from foundational training to real-world deployment, exploring key concepts and practical demonstrations of potential pitfalls. Our goal is to equip you with a conceptual understanding of how LLMs operate, how they can go wrong, and the importance of human oversight.
    """)

    st.header("Section 2: The LLM Lifecycle: An Overview")
    st.markdown("""
    The development of a Large Language Model can be conceptually divided into three main phases:
    1.  **Pre-training**: The initial phase where the model learns foundational patterns from vast amounts of text data.
    2.  **Alignment**: The process of refining the model's behavior to align with human values and specific tasks.
    3.  **Deployment**: When the model is put into real-world use, requiring continuous monitoring and adaptation.

    Understanding these phases is critical to identifying and mitigating risks at each stage.
    """)

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
        return fig

    st.pyplot(plot_llm_lifecycle_timeline())

    st.markdown("""
    The visualization above provides a high-level overview of the LLM lifecycle. Each stage involves distinct processes and introduces unique risks, which we will explore in detail.
    """)

    st.header("Section 3: Phase 1: Pre-training - Vast Data Ingestion and Pattern Recognition")
    st.markdown(r"""
    Pre-training is the foundational step where LLMs are exposed to immense quantities of text and code data. During this phase, the model learns grammar, facts, reasoning abilities, and how to predict the next word in a sequence. This is fundamentally a process of identifying statistical relationships and patterns. Conceptually, the model aims to maximize the probability of predicting the correct next word given the preceding words, represented as $P(\text{next word} | \text{previous words})$.
    """)

    with st.expander("Configure Synthetic Text Data Generation"):
        st.session_state.num_sentences = st.slider("Number of Synthetic Sentences", 100, 5000, st.session_state.num_sentences, key='num_sentences_slider')
        st.session_state.vocab_size = st.slider("Vocabulary Size", 50, 500, st.session_state.vocab_size, key='vocab_size_slider')
        st.session_state.avg_sentence_length = st.slider("Average Sentence Length", 5, 30, st.session_state.avg_sentence_length, key='avg_sentence_length_slider')

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

    synthetic_text = generate_synthetic_text_data(st.session_state.num_sentences, st.session_state.vocab_size, st.session_state.avg_sentence_length)
    word_freqs = analyze_word_frequency(synthetic_text, top_n=10)
    st.write("Top 10 Synthetic Word Frequencies:", word_freqs)

    st.markdown("""
    The `generate_synthetic_text_data` function simulates the input data an LLM might encounter during pre-training. By analyzing word frequencies, we get a glimpse into the patterns and common co-occurrences that an LLM would learn, forming its understanding of language structure and word relationships.
    """)

    st.header("Section 4: Visualizing Pre-training: Word Probabilities")
    st.markdown(r"""
    The core of pre-training is learning the conditional probability $P(\text{next word} | \text{previous words})$. This means for any given sequence of words, the model learns which words are most likely to follow. A higher frequency implies a higher learned probability.
    """)

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

    st.pyplot(plot_word_frequency_distribution(word_freqs))

    st.markdown("""
    This bar chart visually represents the learned statistical regularities in our synthetic data. In a real LLM, these frequencies would translate into probabilities guiding its text generation, enabling it to produce coherent and contextually relevant responses.
    """)

    st.header("Section 5: Emergent Risk: Data Bias during Pre-training")
    st.warning("⚠️ **Emergent Risk: Data Bias**")
    st.markdown("""
    One of the most significant emergent risks during the pre-training phase is **data bias**. LLMs learn from the vast, often unfiltered, data of the internet. If this data contains societal biases (e.g., gender stereotypes, racial discrimination, specific economic viewpoints), the LLM will inadvertently encode and amplify these biases in its outputs. This can lead to skewed, unfair, or discriminatory responses, impacting critical decisions.
    """)

    with st.expander("Configure Data Bias Simulation"):
        st.session_state.bias_strength = st.slider("Bias Strength (Proportion of Biased Samples)", 0.0, 0.5, st.session_state.bias_strength, key='bias_strength_slider')
        st.session_state.biased_feature_mean_shift = st.slider("Biased Feature Mean Shift", 0, 50, st.session_state.biased_feature_mean_shift, key='biased_feature_mean_shift_slider')

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

    unbiased_data_df = generate_data_with_bias(num_samples=1000, feature_dist_mean_unbiased=50,
                                               feature_dist_std_unbiased=10, bias_strength=0.0,
                                               biased_feature_mean_shift=0)
    unbiased_outputs = analyze_output_distribution_by_group(unbiased_data_df, 'feature', 'conceptual_output')

    biased_data_df = generate_data_with_bias(num_samples=1000, feature_dist_mean_unbiased=50,
                                             feature_dist_std_unbiased=10, bias_strength=st.session_state.bias_strength,
                                             biased_feature_mean_shift=st.session_state.biased_feature_mean_shift)
    biased_outputs = analyze_output_distribution_by_group(biased_data_df, 'feature', 'conceptual_output')

    st.write("Unbiased Outputs (Group 1 Mean, Group 2 Mean):", unbiased_outputs)
    st.write("Biased Outputs (Group 1 Mean, Group 2 Mean):", biased_outputs)

    st.markdown("""
    Here, we've simulated two scenarios: one with unbiased data and another where an artificial bias is introduced into a "feature." Observe how the `conceptual_output` means differ significantly between the groups in the biased scenario, mimicking how a real LLM would propagate and amplify these input biases into its generated content.
    """)

    st.header("Section 6: Visualizing the Impact of Data Bias")
    st.markdown("""
    Visualizing the output differences between groups is essential to highlight the impact of data bias. This directly shows how skewed inputs can lead to skewed, unfair, or misrepresentative outputs from an LLM.
    """)

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

    st.pyplot(plot_output_bias_comparison(unbiased_outputs['group1_mean'], unbiased_outputs['group2_mean'],
                                    biased_outputs['group1_mean'], biased_outputs['group2_mean'],
                                    group_labels=["Group A (Feature < 50)", "Group B (Feature >= 50)"]))

    st.markdown("""
    This chart clearly illustrates how a bias in the training data can lead to disparate impacts in the model's conceptual outputs across different groups. For example, if 'Group A' represents a demographic group and 'conceptual_output' is a score, the biased model might consistently give lower scores to Group A.
    """)
