"""
Created on 21/12/2024

@author: Aryan

Filename: analyse_dataset.py

Relative Path: analyse_dataset.py
"""

import matplotlib.pyplot as plt
from collections import Counter


def analyze_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # Calculate total length (characters)
    total_length = len(data)

    # Calculate vocabulary size (unique tokens)
    tokens = list(data)  # Treat each character as a token
    vocabulary_size = len(set(tokens))

    # Frequency analysis of tokens
    token_counts = Counter(tokens)
    most_common = token_counts.most_common(5)
    least_common = token_counts.most_common()[:-6:-1]

    # Sequence-level analysis
    sequences = data.split('\n')  # Split by lines as sequences
    sequence_lengths = [len(seq) for seq in sequences if seq.strip()]
    avg_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    median_sequence_length = sorted(sequence_lengths)[
        len(sequence_lengths) // 2]

    # Get a brief preview of the content
    preview = data[:500]  # First 500 characters as a sample

    # Output results
    print(f"File: {file_path}")
    print(f"Total Length (characters): {total_length}")
    print(f"Vocabulary Size: {vocabulary_size}")
    print(f"Most Common Tokens: {most_common}")
    print(f"Least Common Tokens: {least_common}")
    print(f"Average Sequence Length: {avg_sequence_length:.2f}")
    print(f"Median Sequence Length: {median_sequence_length}\n")
    print(f"Preview:\n{preview}\n")

    # Visualization of token frequencies
    plot_token_frequencies(token_counts, "output/token_frequencies.png")
    # Visualization of sequence lengths
    plot_sequence_lengths(sequence_lengths, "output/sequence_lengths.png")


def plot_token_frequencies(token_counts, save_path):
    tokens, counts = zip(*token_counts.most_common(10))  # Top 10 tokens
    plt.bar(tokens, counts)
    plt.title("Top 10 Token Frequencies")
    plt.xlabel("Tokens")
    plt.ylabel("Frequency")
    plt.savefig(save_path, format='png', dpi=300)  # Save plot as PNG
    plt.show()


def plot_sequence_lengths(sequence_lengths, save_path):
    plt.hist(sequence_lengths, bins=20, edgecolor='black')
    plt.title("Sequence Length Distribution")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.savefig(save_path, format='png', dpi=300)  # Save plot as PNG
    plt.show()


# Example usage:
# analyze_dataset("data/inputMelodies.txt")
if __name__ == "__main__":
    analyze_dataset("data/inputMelodiesAugmented.txt")
