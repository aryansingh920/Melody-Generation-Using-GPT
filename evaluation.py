"""
Created on 21/12/2024

@author: Aryan

Filename: evaluation.py

Relative Path: evaluation.py

Description:
------------
Evaluates GPT-based melody generation models on various test sets.
Adds BLEU scoring as an optional metric.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, List

# BLEU imports
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Local imports
from gpt import ModelConfig, GPTLanguageModel, load_data


def load_model_and_metrics(model_name: str, save_dir='saved_models'):
    """
    Loads saved model weights, config, and metrics from disk.

    :param model_name: Name (prefix) of the model, e.g., 'Original', 'Deeper_Thinner'
    :param save_dir:   Directory where model weights and metrics are stored
    :return:           (model, config, encode, decode, saved_data)
    """
    metrics_path = os.path.join(save_dir, f'{model_name}_metrics.json')
    with open(metrics_path, 'r') as f:
        saved_data = json.load(f)

    # Convert saved config to a ModelConfig object
    config_dict = saved_data['config']
    if 'device' in config_dict:
        config_dict['device'] = torch.device(config_dict['device'])
    config = ModelConfig(**{
        k: v for k, v in config_dict.items()
        if k in ModelConfig.__dataclass_fields__
    })

    # Load data to get vocab_size
    train_data, val_data, encode, decode, vocab_size = load_data(config)

    # Initialize model & load weights
    model = GPTLanguageModel(config, vocab_size)
    model_path = os.path.join(save_dir, f'{model_name}_model.pt')
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    model.eval()

    return model, config, encode, decode, saved_data


def calculate_baseline_loss(data: torch.Tensor) -> float:
    """
    Baseline: always predicting the most frequent token.

    :param data: All tokens in the test set
    :return:     Negative log probability of the majority class
    """
    token_counts = torch.bincount(data.flatten())
    most_common_token = torch.argmax(token_counts)
    total_tokens = len(data.flatten())

    prob_most_common = token_counts[most_common_token].item() / total_tokens
    baseline_loss = -np.log(prob_most_common)
    return baseline_loss


def evaluate_on_test_set(model: GPTLanguageModel,
                         config: ModelConfig,
                         test_file: str,
                         use_bleu: bool = True) -> Dict[str, float]:
    """
    Evaluates a GPT model on a test file, computing:
      - Average cross-entropy loss
      - Baseline cross-entropy loss
      - Perplexity & baseline perplexity
      - BLEU score (optional)
      - Sample generated text

    :param model:     GPTLanguageModel instance
    :param config:    ModelConfig
    :param test_file: Path to the test data file
    :param use_bleu:  Whether to compute BLEU or not
    :return:          dict of evaluation results
    """
    # Save original file_name to restore later
    original_file = config.file_name

    # Override to load the test file
    config.file_name = test_file
    test_data, _, encode, decode, _ = load_data(config)

    # 1) Baseline loss
    baseline_loss = calculate_baseline_loss(test_data)

    # 2) Model cross-entropy loss
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for i in range(0, len(test_data) - config.block_size, config.block_size):
            x = test_data[i: i +
                          config.block_size].unsqueeze(0).to(config.device)
            y = test_data[i + 1: i + config.block_size +
                          1].unsqueeze(0).to(config.device)
            _, loss = model(x, y)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')

    # 3) Generate a sample
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated_tokens = model.generate(context, max_new_tokens=200)[0].tolist()
    generated_text = decode(generated_tokens)

    # 4) BLEU (optional)
    bleu_score = None
    if use_bleu and len(test_data) > 0:
        # We'll treat the entire test_data as a single reference for demonstration.
        # A better approach is to have short reference phrases that match your generated phrase length.
        reference_sequence = test_data.tolist()[:len(generated_tokens)]
        # Converting everything to lists of tokens because sentence_bleu expects tokenized input
        references = [reference_sequence]  # single reference
        candidate = generated_tokens  # the model's output
        # Compute BLEU
        bleu_score = sentence_bleu(references, candidate)

    # 5) Perplexities & improvement ratio
    test_perplexity = float(np.exp(avg_loss)) if avg_loss != float(
        'inf') else float('inf')
    baseline_perplexity = float(np.exp(baseline_loss))
    improvement_ratio = baseline_loss / \
        avg_loss if avg_loss != 0 else float('inf')

    # Restore original
    config.file_name = original_file

    results = {
        'test_loss': avg_loss,
        'baseline_loss': baseline_loss,
        'perplexity': test_perplexity,
        'baseline_perplexity': baseline_perplexity,
        'improvement_ratio': improvement_ratio,
        'generated_text': generated_text
    }
    if bleu_score is not None:
        results['bleu_score'] = bleu_score

    return results


def main():
    # Evaluate on these test files
    test_files = [
        "input_childSpeech_testSet.txt",
        "input_shakespeare.txt"
    ]

    # Your saved model names
    model_names = ["Original", "Deeper_Thinner", "Wider_Shallower"]

    all_results = {}
    for model_name in model_names:
        print(f"\nEvaluating {model_name} model...")
        model, config, encode, decode, saved_metrics = load_model_and_metrics(
            model_name)

        model_results = {}
        for test_file in test_files:
            print(f"\nTesting on {test_file} ...")
            # Evaluate with BLEU on
            results = evaluate_on_test_set(
                model, config, test_file, use_bleu=True)

            print(f"Test Loss:           {results['test_loss']:.4f}")
            print(f"Baseline Loss:       {results['baseline_loss']:.4f}")
            print(f"Perplexity:          {results['perplexity']:.2f}")
            print(f"Baseline Perplexity: {results['baseline_perplexity']:.2f}")
            print(f"Improvement Ratio:   {results['improvement_ratio']:.2f}x")
            if 'bleu_score' in results:
                print(f"BLEU Score:          {results['bleu_score']:.4f}")
            print("\nSample Generation:\n",
                  results['generated_text'][:200], "...")

            model_results[test_file] = results
        all_results[model_name] = model_results

    # Dump results to JSON
    with open('evaluation_results.json', 'w') as f:
        json_results = {}
        for model_nm, model_res in all_results.items():
            json_results[model_nm] = {}
            for test_file, metrics in model_res.items():
                # Skip storing the generated text in JSON or store it if you want
                json_results[model_nm][test_file] = {}
                for k, v in metrics.items():
                    if k == 'generated_text':
                        continue
                    # Convert to float if needed
                    if isinstance(v, (np.float32, np.float64, float, int)):
                        json_results[model_nm][test_file][k] = float(v)
                    else:
                        json_results[model_nm][test_file][k] = v
        json.dump(json_results, f, indent=4)

    return all_results


if __name__ == "__main__":
    results = main()
