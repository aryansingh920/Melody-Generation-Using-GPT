"""
Created on 22/12/2024

@author: Aryan

Filename: MelodyDataset.py

Relative Path: MelodyDataset.py
"""

from dataclasses import dataclass
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset


@dataclass
class ModelConfig:
    batch_size: int = 1654
    block_size: int = 32
    n_embd: int = 32
    n_head: int = 2
    n_layer: int = 2
    dropout: float = 0.0
    learning_rate: float = 1e-3
    max_iters: int = 5000
    eval_interval: int = 100
    eval_iters: int = 50
    device: torch.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    file_name: str = "input_childSpeech_trainingSet.txt"


class MelodyDataset(Dataset):
    def __init__(self, file_path, block_size):
        # Load the augmented dataset
        with open(file_path, 'r') as f:
            data = f.read()

        # Tokenize data
        self.tokens = list(data)
        self.vocab = sorted(set(self.tokens))
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

        self.data = [self.stoi[ch] for ch in self.tokens]
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y
