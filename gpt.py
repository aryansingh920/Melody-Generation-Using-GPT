"""
Created on 23/12/2024

@author: Aryan

Filename: gpt.py

Relative Path: gpt.py
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import wandb

# We'll import the ModelConfig from MelodyDataset for consistent usage
from MelodyDataset import ModelConfig

# Default hyperparameters (overwritten by config if needed)
batch_size = 16
block_size = 32
n_embd = 32
n_head = 2
n_layer = 2
dropout = 0.0
learning_rate = 1e-3
max_iters = 1
eval_interval = 100
eval_iters = 50
device = torch.device("mps") if torch.backends.mps.is_available(
) else torch.device("cuda" if torch.cuda.is_available() else "cpu")
fileName = "data/inputMelodiesAugmented.txt"


def update_model_config(config: ModelConfig):
    """Update global model configuration variables from a ModelConfig."""
    global batch_size, block_size, max_iters, eval_interval, learning_rate
    global n_embd, n_head, n_layer, dropout, device, fileName

    batch_size = config.batch_size
    block_size = config.block_size
    n_embd = config.n_embd
    n_head = config.n_head
    n_layer = config.n_layer
    dropout = config.dropout
    learning_rate = config.learning_rate
    max_iters = config.max_iters
    eval_interval = config.eval_interval
    device = config.device
    fileName = config.file_name


def load_data(config: ModelConfig):
    """
    Load data from the file specified in config, and split into train and val sets.
    Returns (train_data, val_data, encode, decode, vocab_size).
    """
    with open(config.file_name, 'r', encoding='utf-8') as f:
        text = f.read()

    # Identify the unique characters
    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str):
        return [stoi[c] for c in s]

    def decode(l: list):
        return ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data, encode, decode, vocab_size


# Data loading for training (global variables for convenience)
with open(fileName, 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}


def encode(s: str):
    return [stoi[c] for c in s]


def decode(l: list):
    return ''.join([itos[i] for i in l])


data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split='train'):
    """Generate a small batch of data of inputs x and targets y."""
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix])
    y = torch.stack([d[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """Compute average loss on train and val sets for eval_iters iterations each."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B,T,hs)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):
    def __init__(self, config: ModelConfig = None, vocab_size_override: int = None):
        """
        If config is provided, we can set model dimensions from config.
        If vocab_size_override is provided, use that for final layer. Otherwise, use global vocab_size.
        """
        super().__init__()

        if config is not None:
            update_model_config(config)

        vsz = vocab_size_override if vocab_size_override is not None else vocab_size

        self.token_embedding_table = nn.Embedding(vsz, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)  # final layer norm
        self.lm_head = nn.Linear(n_embd, vsz)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=idx.device))  # (T,C)
        x = tok_emb + pos_emb  # (B,T,C)
        x = self.blocks(x)  # (B,T,C)
        x = self.ln_f(x)  # (B,T,C)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100):
        """
        Generate new tokens from the model, given a prompt idx of shape (B, T).
        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]  # Focus on last time step
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def generate_melody(prompt, max_new_tokens=100):
    """Generate a melody given a text prompt (e.g., 'R F G A ')."""
    model.eval()
    context = torch.tensor([stoi[ch] for ch in prompt if ch in stoi],
                           dtype=torch.long).unsqueeze(0).to(device)
    generated = model.generate(context, max_new_tokens)
    return decode(generated[0].tolist())


# Initialize the global model for training usage below
wandb.init(project="gpt-melody-generation", config={
    "batch_size": batch_size,
    "block_size": block_size,
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "dropout": dropout,
    "learning_rate": learning_rate,
    "max_iters": max_iters,
    "eval_interval": eval_interval
})

model = GPTLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop with tqdm
pbar = tqdm(range(max_iters), desc="Training")
for iter in pbar:
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        wandb.log({
            "train_loss": losses['train'],
            "val_loss": losses['val'],
            "step": iter
        })
        pbar.set_postfix({
            'train_loss': f"{losses['train']:.4f}",
            'val_loss': f"{losses['val']:.4f}"
        })

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    wandb.log({"batch_loss": loss.item(), "step": iter})

# Final generation after training
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(m.generate(context, max_new_tokens=500)[0].tolist())
print(generated_text)
wandb.log({"generated_sample": generated_text})

# Example generation from a prompt
prompt = "R F G A "
generated_melody = generate_melody(prompt)
print("Generated Melody:", generated_melody)
wandb.log({"prompt_generated_melody": generated_melody})

output_file = "output/generated_melody.txt"
with open(output_file, 'w') as f:
    f.write(generated_melody)

print(f"Generated melody saved to {output_file}")
wandb.finish()
