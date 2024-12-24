"""
Created on 24/12/2024

@author: Aryan

Filename: gpt_downsizing.py

Relative Path: gpt_downsizing.py
"""

from MelodyDataset import ModelConfig

# gpt_downsizing.py - Updated for melody generation dataset

# Initial downsize configuration


def create_custom_config():
    return ModelConfig(
        batch_size=16,
        block_size=32,
        n_embd=32,
        n_head=2,
        n_layer=2,
        dropout=0.0,
        learning_rate=1e-3,
        max_iters=10000,
        eval_interval=100,
        eval_iters=50,
        file_name="data/inputMelodiesAugmented.txt"  # Updated to use melody dataset
    )

# Deeper but thinner configuration


def create_custom_config_1():
    return ModelConfig(
        batch_size=16,
        block_size=32,
        n_embd=24,
        n_head=3,
        n_layer=3,
        dropout=0.1,
        learning_rate=1e-3,
        max_iters=5000,
        eval_interval=100,
        eval_iters=50,
        file_name="data/inputMelodiesAugmented.txt"  # Updated to use melody dataset
    )

# Wider but shallower configuration


def create_custom_config_2():
    return ModelConfig(
        batch_size=16,
        block_size=32,
        n_embd=48,
        n_head=2,
        n_layer=1,
        dropout=0.0,
        learning_rate=1e-3,
        max_iters=5000,
        eval_interval=100,
        eval_iters=50,
        file_name="data/inputMelodiesAugmented.txt"  # Updated to use melody dataset
    )


def create_custom_config_3():
    return ModelConfig(
        batch_size=32,         # Larger batch size
        block_size=64,         # Longer sequence length
        n_embd=48,             # Wider embeddings
        n_head=4,              # More attention heads
        n_layer=3,             # Deeper network
        dropout=0.1,           # Add dropout for regularization
        learning_rate=1e-4,    # Lower learning rate
        max_iters=10_000,      # More training iterations
        eval_interval=100,
        eval_iters=50,
        file_name="data/inputMelodiesAugmented.txt"  # Melody dataset
    )
