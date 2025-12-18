#!/usr/bin/env python3
"""
CS242 Final Project — AlphaEvolve-style search + full training on WikiText-2.

Modes:
  --mode ea           : evolutionary search with short training to find good configs
  --mode train_best   : load best_config.json and train it to convergence (more epochs)
  --mode train_manual : train a hand-designed baseline transformer config

Example usage:

  # search
  python 1.py --mode ea --ea_pop_size 4 --ea_generations 3 --ea_train_steps 100

  # train best evolved config for 10 epochs
  python 1.py --mode train_best --epochs 10

  # train manual baseline for 10 epochs
  python 1.py --mode train_manual --epochs 10
"""

import math
import copy
import random
import time
from dataclasses import dataclass
from typing import Literal

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# -----------------------------
# Dataset utilities
# -----------------------------

def load_wikitext2(base_dir: str = "data/wikitext2"):
    """
    Load WikiText-2 from a local arrow dataset directory.
    If not found, download and save it.
    """
    import os
    import glob
    # Check if the actual data files exist, not just the directory structure
    train_arrow = glob.glob(os.path.join(base_dir, "train", "*.arrow"))
    if not os.path.exists(base_dir) or not train_arrow:
        print(f"Dataset not found at {base_dir}, downloading from HuggingFace...")
        from datasets import load_dataset
        dataset = load_dataset("wikitext", "wikitext-2-v1")
        dataset.save_to_disk(base_dir)
        print(f"Dataset saved to {base_dir}")
        return dataset
    dataset = load_from_disk(base_dir)
    return dataset


def load_wikitext103(base_dir: str = "data/wikitext103"):
    """
    Load WikiText-103 from disk if present; otherwise download from HuggingFace
    and save to `base_dir`.

    Returns a DatasetDict with 'train', 'validation', 'test' splits.
    """
    import os
    import glob
    # Check if the actual data files exist, not just the directory structure
    train_arrow = glob.glob(os.path.join(base_dir, "train", "*.arrow"))
    if not os.path.exists(base_dir) or not train_arrow:
        print(f"Dataset not found at {base_dir}, downloading WikiText-103 from HuggingFace...")
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-v1")
        ds.save_to_disk(base_dir)
        print(f"Dataset saved to {base_dir}")
        return ds
    print(f"Found WikiText-103 on disk at {base_dir}, loading...")
    return load_from_disk(base_dir)


def tokenize_wikitext2(raw_datasets, tokenizer, block_size: int):
    """
    Tokenize text with a GPT-2 tokenizer and group into fixed-length blocks.
    Each block is of length block_size; labels are next-token targets.
    """
    block_size = min(block_size, tokenizer.model_max_length)

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
    )

    def group_texts(examples):
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {}
        for k, t in concatenated.items():
            result[k] = [t[i:i + block_size] for i in range(0, total_length, block_size)]
        # For LM, labels are just shifted input_ids (we shift in the loss)
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized.map(
        group_texts,
        batched=True,
        desc=f"Grouping into blocks of size {block_size}",
    )
    return lm_datasets


def make_dataloaders(lm_datasets, batch_size: int):
    """
    Create PyTorch dataloaders from grouped LM dataset.
    """
    train_ds = lm_datasets["train"]
    val_ds = lm_datasets["validation"]

    train_ds.set_format(type="torch", columns=["input_ids"])
    val_ds.set_format(type="torch", columns=["input_ids"])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# -----------------------------
# Model: Transformer with configurable attention
# -----------------------------

# class EvoMultiheadSelfAttention(nn.Module):
#     """
#     Multi-head self-attention with two algorithms:
#       - 'full': standard full causal attention over T
#       - 'chunked': local causal attention over chunks of size chunk_size
#     """

#     def __init__(
#         self,
#         d_model: int,
#         n_heads: int,
#         attention_type: Literal["full", "chunked"] = "full",
#         chunk_size: int = 64,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
#         assert attention_type in ("full", "chunked")

#         self.d_model = d_model
#         self.n_heads = n_heads
#         self.head_dim = d_model // n_heads
#         self.attention_type = attention_type
#         self.chunk_size = chunk_size

#         self.dropout = nn.Dropout(dropout)
#         self.q_proj = nn.Linear(d_model, d_model)
#         self.k_proj = nn.Linear(d_model, d_model)
#         self.v_proj = nn.Linear(d_model, d_model)
#         self.out_proj = nn.Linear(d_model, d_model)

class EvoMultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention with three algorithms:
      - 'full': standard full causal attention over T
      - 'chunked': local causal attention over chunks of size chunk_size
      - 'hybrid': learned gate between full and chunked attention
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        attention_type: Literal["full", "chunked", "hybrid"] = "full",
        chunk_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0

        assert attention_type in ("full", "chunked", "hybrid")
        self.attention_type = attention_type
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if self.attention_type == "hybrid":
            # Learned scalar controlling how much *extra local* attention to add.
            # Start near "almost full attention" (sigmoid(-2) ≈ 0.12).
            self.hybrid_gate = nn.Parameter(torch.tensor(-2.0))



    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, _ = x.size()
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # (B, T, H, D) -> (B, H, T, D)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # if self.attention_type == "full":
        #     out = self._full_attention(q, k, v, causal)
        # else:
        #     out = self._chunked_attention(q, k, v, causal)
        if self.attention_type == "full":
            out = self._full_attention(q, k, v, causal)
        elif self.attention_type == "chunked":
            out = self._chunked_attention(q, k, v, causal)
        else:  # 'hybrid'
            full_out = self._full_attention(q, k, v, causal)
            chunk_out = self._chunked_attention(q, k, v, causal)
            chunk_scale = torch.sigmoid(self.hybrid_gate)  # scalar in (0,1)
            out = full_out + chunk_scale * chunk_out



        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.out_proj(out)
        return out

    def _full_attention(self, q, k, v, causal: bool):
        B, H, T, D = q.size()
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
        if causal:
            mask = torch.triu(
                torch.ones(T, T, device=scores.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        return out

    def _chunked_attention(self, q, k, v, causal: bool):
        """
        Sliding-window causal attention (window = self.chunk_size), computed in chunks
        for speed. Unlike the old hard partition, tokens can attend across chunk
        boundaries within the window.
        """
        B, H, T, D = q.size()
        window = min(self.chunk_size, T)
        step = window  # compute in blocks, but allow overlap via masking
        outputs = []

        for start in range(0, T, step):
            end = min(start + step, T)

            # Provide keys/values that include up to `window-1` tokens before `start`
            k_start = max(0, start - (window - 1))

            q_chunk = q[:, :, start:end, :]            # (B,H,Q,D)
            k_chunk = k[:, :, k_start:end, :]          # (B,H,K,D)
            v_chunk = v[:, :, k_start:end, :]          # (B,H,K,D)

            scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / math.sqrt(D)

            if causal:
                # Build a mask that enforces:
                # 1) causal: key_pos <= query_pos
                # 2) window: key_pos >= query_pos - (window-1)
                q_pos = torch.arange(start, end, device=scores.device)[:, None]          # (Q,1)
                k_pos = torch.arange(k_start, end, device=scores.device)[None, :]        # (1,K)

                mask = (k_pos > q_pos) | (k_pos < (q_pos - (window - 1)))
                scores = scores.masked_fill(mask, float("-inf"))

            attn = torch.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            out_chunk = torch.matmul(attn, v_chunk)  # (B,H,Q,D)
            outputs.append(out_chunk)

        out = torch.cat(outputs, dim=2)  # (B,H,T,D)
        return out



class EvoTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        attention_type: Literal["full", "chunked", "hybrid"],
        chunk_size: int,
    ):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = EvoMultiheadSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            attention_type=attention_type,
            chunk_size=chunk_size,
            dropout=dropout,
        )
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class EvoTransformerLM(nn.Module):
    """
    GPT-like Transformer language model driven by EvoConfig.
    """

    def __init__(self, vocab_size: int, block_size: int, config: "EvoConfig"):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.config = config

        self.token_embed = nn.Embedding(vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(block_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [
                EvoTransformerBlock(
                    d_model=config.d_model,
                    n_heads=config.n_heads,
                    d_ff=config.d_ff,
                    dropout=config.dropout,
                    attention_type=config.attention_type,
                    chunk_size=config.chunk_size,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight


    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        if T > self.block_size:
            idx = idx[:, -self.block_size :]
            T = self.block_size

        positions = torch.arange(0, T, device=idx.device, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(B, T)

        x = self.token_embed(idx) + self.pos_embed(positions)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


# -----------------------------
# Evolution config & search
# -----------------------------

@dataclass
class EvoConfig:
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    dropout: float
    attention_type: Literal["full", "chunked", "hybrid"]
    chunk_size: int
    batch_size: int

# class EvoConfig:
#     d_model: int
#     n_heads: int
#     n_layers: int
#     d_ff: int
#     dropout: float
#     attention_type: Literal["full", "chunked"]
#     chunk_size: int
#     batch_size: int


D_MODEL_CHOICES = [128, 192, 256]
N_HEAD_CHOICES = [2, 4, 8]
N_LAYER_CHOICES = [2, 3, 4]
D_FF_CHOICES = [256, 384, 512]
DROPOUT_CHOICES = [0.0, 0.1, 0.2]
ATTN_TYPE_CHOICES = ["full", "chunked", "hybrid"]

# ATTN_TYPE_CHOICES = ["full", "chunked"]
CHUNK_CHOICES = [16, 32, 64]
BATCH_CHOICES = [8, 16, 32]


def _sample_n_heads(d_model: int) -> int:
    valid = [h for h in N_HEAD_CHOICES if d_model % h == 0]
    return random.choice(valid)


def random_config() -> EvoConfig:
    d_model = random.choice(D_MODEL_CHOICES)
    n_heads = _sample_n_heads(d_model)
    return EvoConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=random.choice(N_LAYER_CHOICES),
        d_ff=random.choice(D_FF_CHOICES),
        dropout=random.choice(DROPOUT_CHOICES),
        attention_type=random.choice(ATTN_TYPE_CHOICES),
        chunk_size=random.choice(CHUNK_CHOICES),
        batch_size=random.choice(BATCH_CHOICES),
    )


def mutate(cfg: EvoConfig) -> EvoConfig:
    new = EvoConfig(**cfg.__dict__)
    field = random.choice(
        [
            "d_model",
            "n_heads",
            "n_layers",
            "d_ff",
            "dropout",
            "attention_type",
            "chunk_size",
            "batch_size",
        ]
    )
    if field == "d_model":
        new.d_model = random.choice(D_MODEL_CHOICES)
        new.n_heads = _sample_n_heads(new.d_model)
    elif field == "n_heads":
        new.n_heads = _sample_n_heads(new.d_model)
    elif field == "n_layers":
        new.n_layers = random.choice(N_LAYER_CHOICES)
    elif field == "d_ff":
        new.d_ff = random.choice(D_FF_CHOICES)
    elif field == "dropout":
        new.dropout = random.choice(DROPOUT_CHOICES)
    elif field == "attention_type":
        new.attention_type = random.choice(ATTN_TYPE_CHOICES)
    elif field == "chunk_size":
        new.chunk_size = random.choice(CHUNK_CHOICES)
    elif field == "batch_size":
        new.batch_size = random.choice(BATCH_CHOICES)
    return new


def crossover(a: EvoConfig, b: EvoConfig) -> EvoConfig:
    d_model = random.choice([a.d_model, b.d_model])
    n_heads = _sample_n_heads(d_model)
    return EvoConfig(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=random.choice([a.n_layers, b.n_layers]),
        d_ff=random.choice([a.d_ff, b.d_ff]),
        dropout=random.choice([a.dropout, b.dropout]),
        attention_type=random.choice([a.attention_type, b.attention_type]),
        chunk_size=random.choice([a.chunk_size, b.chunk_size]),
        batch_size=random.choice([a.batch_size, b.batch_size]),
    )


def hamming_neighbors(cfg: EvoConfig) -> list[EvoConfig]:
    """
    Generate all Hamming-1 neighbors within the discrete search space.
    Keeps n_heads compatible with d_model.
    """
    neighbors = []

    # d_model
    for dm in D_MODEL_CHOICES:
        if dm != cfg.d_model:
            new = copy.deepcopy(cfg)
            new.d_model = dm
            valid_heads = [h for h in N_HEAD_CHOICES if dm % h == 0]
            if not valid_heads:
                continue
            if new.n_heads not in valid_heads:
                new.n_heads = random.choice(valid_heads)
            neighbors.append(new)

    # n_heads (compatible with current d_model)
    for h in N_HEAD_CHOICES:
        if h != cfg.n_heads and cfg.d_model % h == 0:
            new = copy.deepcopy(cfg)
            new.n_heads = h
            neighbors.append(new)

    # n_layers
    for nl in N_LAYER_CHOICES:
        if nl != cfg.n_layers:
            new = copy.deepcopy(cfg)
            new.n_layers = nl
            neighbors.append(new)

    # d_ff
    for ff in D_FF_CHOICES:
        if ff != cfg.d_ff:
            new = copy.deepcopy(cfg)
            new.d_ff = ff
            neighbors.append(new)

    # dropout
    for dr in DROPOUT_CHOICES:
        if dr != cfg.dropout:
            new = copy.deepcopy(cfg)
            new.dropout = dr
            neighbors.append(new)

    # attention_type
    for attn in ATTN_TYPE_CHOICES:
        if attn != cfg.attention_type:
            new = copy.deepcopy(cfg)
            new.attention_type = attn
            neighbors.append(new)

    # chunk_size
    for cs in CHUNK_CHOICES:
        if cs != cfg.chunk_size:
            new = copy.deepcopy(cfg)
            new.chunk_size = cs
            neighbors.append(new)

    # batch_size
    for bs in BATCH_CHOICES:
        if bs != cfg.batch_size:
            new = copy.deepcopy(cfg)
            new.batch_size = bs
            neighbors.append(new)

    return neighbors


device = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Short training for EA
# -----------------------------

# def train_and_eval_short(
#     config: EvoConfig,
#     lm_datasets,
#     tokenizer,
#     vocab_size: int,
#     block_size: int,
#     max_train_steps: int = 100,
#     max_eval_batches: int = 40,
# ):
#     """
#     Train a candidate model for a small number of steps and return:
#       fitness, val_loss, perplexity, train_time, tokens_per_second
#     """
#     train_loader, val_loader = make_dataloaders(lm_datasets, config.batch_size)

#     model = EvoTransformerLM(
#         vocab_size=vocab_size, block_size=block_size, config=config
#     ).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

#     pad_token_id = tokenizer.pad_token_id
#     if pad_token_id is None:
#         pad_token_id = (
#             tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
#         )

#     model.train()
#     start_time = time.perf_counter()
#     total_tokens = 0
#     steps = 0

#     for batch in train_loader:
#         input_ids = batch["input_ids"].to(device)
#         B, T = input_ids.size()

#         if T > block_size:
#             input_ids = input_ids[:, :block_size]
#         elif T < block_size:
#             pad_len = block_size - T
#             pad = torch.full(
#                 (B, pad_len), pad_token_id, device=device, dtype=input_ids.dtype
#             )
#             input_ids = torch.cat([input_ids, pad], dim=1)

#         logits = model(input_ids)
#         shift_logits = logits[:, :-1, :].contiguous()
#         shift_labels = input_ids[:, 1:].contiguous()

#         loss = F.cross_entropy(
#             shift_logits.view(-1, vocab_size), shift_labels.view(-1)
#         )

#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()

#         total_tokens += input_ids.numel()
#         steps += 1
#         if steps >= max_train_steps:
#             break

#     train_time = time.perf_counter() - start_time
#     tokens_per_second = total_tokens / max(train_time, 1e-6)

#     # Validation
#     model.eval()
#     total_loss = 0.0
#     total_eval_tokens = 0
#     with torch.no_grad():
#         for i, batch in enumerate(val_loader):
#             input_ids = batch["input_ids"].to(device)
#             B, T = input_ids.size()

#             if T > block_size:
#                 input_ids = input_ids[:, :block_size]
#             elif T < block_size:
#                 pad_len = block_size - T
#                 pad = torch.full(
#                     (B, pad_len),
#                     pad_token_id,
#                     device=device,
#                     dtype=input_ids.dtype,
#                 )
#                 input_ids = torch.cat([input_ids, pad], dim=1)

#             logits = model(input_ids)
#             shift_logits = logits[:, :-1, :].contiguous()
#             shift_labels = input_ids[:, 1:].contiguous()

#             loss = F.cross_entropy(
#                 shift_logits.view(-1, vocab_size),
#                 shift_labels.view(-1),
#                 reduction="sum",
#             )
#             total_loss += loss.item()
#             total_eval_tokens += shift_labels.numel()

#             if i + 1 >= max_eval_batches:
#                 break

#     val_loss = total_loss / max(total_eval_tokens, 1)
#     perplexity = math.exp(min(val_loss, 20.0))

#     # Fitness: lower loss and lower time are better
#     time_penalty_weight = 0.0005
#     fitness = -val_loss - time_penalty_weight * train_time

#     return fitness, val_loss, perplexity, train_time, tokens_per_second


def train_and_eval_short(
    config: EvoConfig,
    lm_datasets,
    tokenizer,
    vocab_size: int,
    block_size: int,
    max_train_steps: int = 100,
    max_eval_batches: int = 40,
):
    """
    Train a candidate model for a small number of steps and return:
      fitness, val_loss, perplexity, train_time, tokens_per_second, num_params
    """
    train_loader, val_loader = make_dataloaders(lm_datasets, config.batch_size)

    model = EvoTransformerLM(
        vocab_size=vocab_size,
        block_size=block_size,
        config=config,
    ).to(device)

    # Count parameters once
    num_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    model.train()
    start_time = time.perf_counter()
    total_tokens = 0
    steps = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        B, T = input_ids.size()

        if T > block_size:
            input_ids = input_ids[:, :block_size]
        elif T < block_size:
            pad_len = block_size - T
            pad = torch.full(
                (B, pad_len),
                pad_token_id,
                device=device,
                dtype=input_ids.dtype,
            )
            input_ids = torch.cat([input_ids, pad], dim=1)

        logits = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        shift_labels = shift_labels.masked_fill(shift_labels == pad_token_id, -100)

        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )


        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_tokens += (shift_labels != -100).sum().item()

        steps += 1
        if steps >= max_train_steps:
            break

    train_time = time.perf_counter() - start_time
    tokens_per_second = total_tokens / max(train_time, 1e-6)

    # Validation
    model.eval()
    total_loss = 0.0
    total_eval_tokens = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            B, T = input_ids.size()

            if T > block_size:
                input_ids = input_ids[:, :block_size]
            elif T < block_size:
                pad_len = block_size - T
                pad = torch.full(
                    (B, pad_len),
                    pad_token_id,
                    device=device,
                    dtype=input_ids.dtype,
                )
                input_ids = torch.cat([input_ids, pad], dim=1)

            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            shift_labels = shift_labels.masked_fill(shift_labels == pad_token_id, -100)

            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                reduction="sum",
                ignore_index=-100,
            )
            total_loss += loss.item()
            total_eval_tokens += (shift_labels != -100).sum().item()


            if i + 1 >= max_eval_batches:
                break

    val_loss = total_loss / max(total_eval_tokens, 1)
    perplexity = math.exp(min(val_loss, 20.0))

    # Keep scalar fitness for logging if you want
    time_penalty_weight = 0.0005
    fitness = -val_loss - time_penalty_weight * train_time

    return fitness, val_loss, perplexity, train_time, tokens_per_second, num_params


# def train_and_eval_short(
#     config: EvoConfig,
#     lm_datasets,
#     tokenizer,
#     vocab_size: int,
#     block_size: int,
#     max_train_steps: int = 100,
#     max_eval_batches: int = 40,
# ):
#     """
#     Train a candidate model for a small number of steps and return:
#       fitness, val_loss, perplexity, train_time, tokens_per_second, num_params
#     """
#     train_loader, val_loader = make_dataloaders(lm_datasets, config.batch_size)

#     model = EvoTransformerLM(
#         vocab_size=vocab_size, block_size=block_size, config=config
#     ).to(device)

#     # Count trainable parameters (for analysis / Pareto plots)
#     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

#     optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

#     pad_token_id = tokenizer.pad_token_id
#     if pad_token_id is None:
#         pad_token_id = (
#             tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
#         )

#     model.train()
#     start_time = time.perf_counter()
#     total_tokens = 0
#     steps = 0

#     for batch in train_loader:
#         input_ids = batch["input_ids"].to(device)
#         B, T = input_ids.size()

#         if T > block_size:
#             input_ids = input_ids[:, :block_size]
#         elif T < block_size:
#             pad_len = block_size - T
#             pad = torch.full(
#                 (B, pad_len), pad_token_id, device=device, dtype=input_ids.dtype
#             )
#             input_ids = torch.cat([input_ids, pad], dim=1)

#         logits = model(input_ids)
#         shift_logits = logits[:, :-1, :].contiguous()
#         shift_labels = input_ids[:, 1:].contiguous()

#         loss = F.cross_entropy(
#             shift_logits.view(-1, vocab_size), shift_labels.view(-1)
#         )

#         optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optimizer.step()

#         total_tokens += input_ids.numel()
#         steps += 1
#         if steps >= max_train_steps:
#             break

#     train_time = time.perf_counter() - start_time
#     tokens_per_second = total_tokens / max(train_time, 1e-6)

#     # Validation
#     model.eval()
#     total_loss = 0.0
#     total_eval_tokens = 0
#     with torch.no_grad():
#         for i, batch in enumerate(val_loader):
#             input_ids = batch["input_ids"].to(device)
#             B, T = input_ids.size()

#             if T > block_size:
#                 input_ids = input_ids[:, :block_size]
#             elif T < block_size:
#                 pad_len = block_size - T
#                 pad = torch.full(
#                     (B, pad_len),
#                     pad_token_id,
#                     device=device,
#                     dtype=input_ids.dtype,
#                 )
#                 input_ids = torch.cat([input_ids, pad], dim=1)

#             logits = model(input_ids)
#             shift_logits = logits[:, :-1, :].contiguous()
#             shift_labels = input_ids[:, 1:].contiguous()

#             loss = F.cross_entropy(
#                 shift_logits.view(-1, vocab_size),
#                 shift_labels.view(-1),
#                 reduction="sum",
#             )
#             total_loss += loss.item()
#             total_eval_tokens += shift_labels.numel()

#             if i + 1 >= max_eval_batches:
#                 break

#     val_loss = total_loss / max(total_eval_tokens, 1)
#     perplexity = math.exp(min(val_loss, 20.0))

#     # Fitness: lower loss and lower time are better
#     time_penalty_weight = 0.0005
#     fitness = -val_loss - time_penalty_weight * train_time

#     return fitness, val_loss, perplexity, train_time, tokens_per_second, num_params

def dominates(a, b):
    """
    Return True if candidate a Pareto-dominates candidate b
    on (val_loss, tokens_per_second).
    Lower loss is better, higher tokens/s is better.
    """
    better_or_equal_loss = a["val_loss"] <= b["val_loss"]
    better_or_equal_speed = a["tokens_per_second"] >= b["tokens_per_second"]
    strictly_better = (
        a["val_loss"] < b["val_loss"]
        or a["tokens_per_second"] > b["tokens_per_second"]
    )
    return better_or_equal_loss and better_or_equal_speed and strictly_better


def evolutionary_search(
    lm_datasets,
    tokenizer,
    vocab_size: int,
    block_size: int = 128,
    pop_size: int = 4,
    generations: int = 3,
    max_train_steps: int = 100,
    max_eval_batches: int = 40,
    log_path: str = "ea_candidates.jsonl",
):
    """
    Evolutionary search over EvoConfig space.

    This version logs:
      - arch_id: unique integer ID for each architecture
      - parent_ids: list of parent arch_ids (empty for gen-0)
      - created_via: "init", "crossover", "crossover+mutate"

    Every evaluated candidate is written to `log_path` as one JSON line.
    The returned `global_best` also carries `arch_id` and `ea_total_time`.
    """
    # --- initial population + lineage meta ---
    population = []
    population_meta = []   # parallel to population
    next_arch_id = 0

    for _ in range(pop_size):
        cfg = random_config()
        population.append(cfg)
        population_meta.append(
            {
                "arch_id": next_arch_id,
                "parent_ids": [],
                "generation": 0,
                "created_via": "init",
            }
        )
        next_arch_id += 1

    history = []
    global_best = None
    loss_by_arch_id = {}
    improve_transitions = 0
    total_transitions = 0

    # overwrite log file each run
    with open(log_path, "w") as f_log:
        total_ea_start = time.perf_counter()

        for gen in range(generations):
            print(f"\n=== Generation {gen} ===")
            scored = []

            # map config object -> meta (for parent lookup later)
            cfg_to_meta = {id(cfg): meta for cfg, meta in zip(population, population_meta)}

            for idx, (cfg, meta) in enumerate(zip(population, population_meta)):
                print(f"\nEvaluating individual {idx} with config: {cfg}")
                (
                    fitness,
                    val_loss,
                    ppl,
                    train_time,
                    tps,
                    num_params,
                ) = train_and_eval_short(
                    cfg,
                    lm_datasets,
                    tokenizer,
                    vocab_size,
                    block_size,
                    max_train_steps=max_train_steps,
                    max_eval_batches=max_eval_batches,
                )

                scored.append(
                    (fitness, cfg, val_loss, ppl, train_time, tps, num_params)
                )

                # lineage-aware improvement counting
                loss_by_arch_id[meta["arch_id"]] = float(val_loss)
                if meta["parent_ids"]:
                    parent_losses = [
                        loss_by_arch_id[p]
                        for p in meta["parent_ids"]
                        if p in loss_by_arch_id
                    ]
                    if parent_losses:
                        total_transitions += 1
                        if val_loss < min(parent_losses):
                            improve_transitions += 1

                rec = {
                    "search_type": "ea",
                    "generation": gen,
                    "index": idx,
                    "arch_id": meta["arch_id"],
                    "parent_ids": meta["parent_ids"],
                    "created_via": meta["created_via"],
                    "config": cfg.__dict__,
                    "fitness": float(fitness),
                    "val_loss": float(val_loss),
                    "perplexity": float(ppl),
                    "train_time": float(train_time),
                    "tokens_per_second": float(tps),
                    "num_params": int(num_params),
                }
                f_log.write(json.dumps(rec) + "\n")

                print(
                    f"  -> fitness={fitness:.4f} | val_loss={val_loss:.4f} "
                    f"| ppl={ppl:.2f} | train_time={train_time:.1f}s "
                    f"| tokens/s={tps:.1f} | params={num_params}"
                )

            # pick best by val_loss for reporting
            scored.sort(key=lambda x: x[2])  # sort by val_loss
            (
                best_f,
                best_cfg,
                best_loss,
                best_ppl,
                best_time,
                best_tps,
                best_params,
            ) = scored[0]
            best_meta = cfg_to_meta[id(best_cfg)]

            total_ea_time = time.perf_counter() - total_ea_start

            history.append(
                {
                    "generation": gen,
                    "arch_id": best_meta["arch_id"],
                    "config": best_cfg.__dict__,
                    "fitness": float(best_f),
                    "val_loss": float(best_loss),
                    "perplexity": float(best_ppl),
                    "train_time": float(best_time),
                    "tokens_per_second": float(best_tps),
                    "num_params": int(best_params),
                    "ea_total_time": float(total_ea_time),
                }
            )

            if (global_best is None) or (best_loss < global_best["val_loss"]):
                global_best = history[-1]

            print(
                f"\n>>> Best in generation {gen}: {best_cfg}\n"
                f"    fitness={best_f:.4f}, val_loss={best_loss:.4f}, "
                f"ppl={best_ppl:.2f}, train_time={best_time:.1f}s, "
                f"tokens/s={best_tps:.1f}, params={best_params}, "
                f"arch_id={best_meta['arch_id']}"
            )

            # ---------- Pareto SELECTION on (val_loss, tokens_per_second) ----------
            pareto_records = []
            for (fitness, cfg, val_loss, ppl, train_time, tps, num_params) in scored:
                pareto_records.append(
                    {
                        "cfg": cfg,
                        "val_loss": float(val_loss),
                        "tokens_per_second": float(tps),
                        "num_params": int(num_params),
                    }
                )

            pareto_indices = []
            for i, a in enumerate(pareto_records):
                dominated = False
                for j, b in enumerate(pareto_records):
                    if i == j:
                        continue
                    if dominates(b, a):
                        dominated = True
                        break
                if not dominated:
                    pareto_indices.append(i)

            parents_cfgs = [pareto_records[i]["cfg"] for i in pareto_indices]

            # fallback if too few parents
            if len(parents_cfgs) < 2:
                parents_cfgs = [
                    cfg for (_, cfg, *_rest) in scored[: max(2, pop_size // 2)]
                ]
            # ensure we have at least 2 parents (allow sampling with replacement)
            if len(parents_cfgs) == 1:
                parents_cfgs = parents_cfgs * 2

            # ---------- Reproduce next generation + lineage meta ----------
            new_population = []
            new_population_meta = []

            while len(new_population) < pop_size:
                if len(parents_cfgs) >= 2:
                    a, b = random.sample(parents_cfgs, 2)
                else:
                    # degenerate case: reuse the single parent twice
                    a = b = parents_cfgs[0]
                meta_a = cfg_to_meta[id(a)]
                meta_b = cfg_to_meta[id(b)]

                child = crossover(a, b)
                created_via = "crossover"

                if random.random() < 0.3:
                    child = mutate(child)
                    created_via = "crossover+mutate"

                child_meta = {
                    "arch_id": next_arch_id,
                    "parent_ids": [meta_a["arch_id"], meta_b["arch_id"]],
                    "generation": gen + 1,
                    "created_via": created_via,
                }
                next_arch_id += 1

                new_population.append(child)
                new_population_meta.append(child_meta)

            population = new_population
            population_meta = new_population_meta

    # attach total EA runtime to global_best for convenience
    if global_best is not None:
        global_best["ea_total_time"] = float(total_ea_time)

    if total_transitions > 0:
        alpha_min_hat = improve_transitions / total_transitions
        print(
            f"\n[EA] alpha_min proxy (P(child improves over best parent)): "
            f"{alpha_min_hat:.4f} ({improve_transitions}/{total_transitions})"
        )
    else:
        print("\n[EA] alpha_min proxy: N/A (no parent losses recorded)")

    return history, global_best




# def evolutionary_search(
#     lm_datasets,
#     tokenizer,
#     vocab_size: int,
#     block_size: int = 128,
#     pop_size: int = 4,
#     generations: int = 3,
#     max_train_steps: int = 100,
#     max_eval_batches: int = 40,
# ):
#     """
#     Evolutionary search over EvoConfig space.

#     Logs every evaluated candidate to ea_candidates.jsonl
#     and stores total EA runtime in the returned global_best record
#     under key 'ea_total_time'.
#     """
#     population = [random_config() for _ in range(pop_size)]
#     history = []
#     global_best = None

#     ea_start = time.perf_counter()

#     with open("ea_candidates.jsonl", "w") as log_f:
#         for gen in range(generations):
#             print(f"\n=== Generation {gen} ===")
#             scored = []

#             for idx, cfg in enumerate(population):
#                 print(f"\nEvaluating individual {idx} with config: {cfg}")
#                 (
#                     fitness,
#                     val_loss,
#                     ppl,
#                     train_time,
#                     tps,
#                     num_params,
#                 ) = train_and_eval_short(
#                     cfg,
#                     lm_datasets,
#                     tokenizer,
#                     vocab_size,
#                     block_size,
#                     max_train_steps=max_train_steps,
#                     max_eval_batches=max_eval_batches,
#                 )

#                 # Log this candidate for later analysis / Pareto front
#                 record = {
#                     "search_type": "ea",
#                     "generation": gen,
#                     "individual": idx,
#                     "config": cfg.__dict__,
#                     "fitness": fitness,
#                     "val_loss": val_loss,
#                     "perplexity": ppl,
#                     "train_time": train_time,
#                     "tokens_per_second": tps,
#                     "num_params": num_params,
#                 }
#                 log_f.write(json.dumps(record) + "\n")
#                 log_f.flush()

#                 scored.append(
#                     (fitness, cfg, val_loss, ppl, train_time, tps, num_params)
#                 )
#                 print(
#                     f"  -> fitness={fitness:.4f} | val_loss={val_loss:.4f} "
#                     f"| ppl={ppl:.2f} | train_time={train_time:.1f}s "
#                     f"| tokens/s={tps:.1f} | params={num_params}"
#                 )

#             scored.sort(key=lambda x: x[0], reverse=True)
#             best_f, best_cfg, best_loss, best_ppl, best_time, best_tps, best_params = scored[0]

#             history.append(
#                 {
#                     "generation": gen,
#                     "config": best_cfg.__dict__,
#                     "fitness": best_f,
#                     "val_loss": best_loss,
#                     "perplexity": best_ppl,
#                     "train_time": best_time,
#                     "tokens_per_second": best_tps,
#                     "num_params": best_params,
#                 }
#             )

#             if (global_best is None) or (best_loss < global_best["val_loss"]):
#                 global_best = history[-1]

#             print(
#                 f"\n>>> Best in generation {gen}: {best_cfg}\n"
#                 f"    fitness={best_f:.4f}, val_loss={best_loss:.4f}, "
#                 f"ppl={best_ppl:.2f}, train_time={best_time:.1f}s, "
#                 f"tokens/s={best_tps:.1f}, params={best_params}"
#             )

#             # Selection + reproduction
#             num_parents = max(2, pop_size // 2)
#             parents = [cfg for (_, cfg, *_rest) in scored[:num_parents]]

#             new_population = []
#             while len(new_population) < pop_size:
#                 a, b = random.sample(parents, 2)
#                 child = crossover(a, b)
#                 if random.random() < 0.3:
#                     child = mutate(child)
#                 new_population.append(child)
#             population = new_population

#     total_ea_time = time.perf_counter() - ea_start
#     print(f"\nTotal EA runtime: {total_ea_time:.1f}s")

#     if global_best is not None:
#         global_best["ea_total_time"] = total_ea_time

#     return history, global_best


def random_search(
    lm_datasets,
    tokenizer,
    vocab_size: int,
    block_size: int = 128,
    num_candidates: int = 12,
    max_train_steps: int = 100,
    max_eval_batches: int = 40,
    log_path: str = "random_candidates.jsonl",
):
    """
    Random search baseline: samples num_candidates random configs
    from EvoConfig space, with the same short training budget used by EA.

    Logs every candidate to random_candidates.jsonl and returns
    (history, global_best) where global_best has key 'search_total_time'.
    """
    history = []
    global_best = None

    search_start = time.perf_counter()

    with open(log_path, "w") as log_f:
        for idx in range(num_candidates):
            cfg = random_config()
            print(f"\n[Random] Evaluating candidate {idx} with config: {cfg}")
            (
                fitness,
                val_loss,
                ppl,
                train_time,
                tps,
                num_params,
            ) = train_and_eval_short(
                cfg,
                lm_datasets,
                tokenizer,
                vocab_size,
                block_size,
                max_train_steps=max_train_steps,
                max_eval_batches=max_eval_batches,
            )

            record = {
                "search_type": "random",
                "candidate": idx,
                "config": cfg.__dict__,
                "fitness": fitness,
                "val_loss": val_loss,
                "perplexity": ppl,
                "train_time": train_time,
                "tokens_per_second": tps,
                "num_params": num_params,
            }
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()

            history.append(record)
            print(
                f"  -> fitness={fitness:.4f} | val_loss={val_loss:.4f} "
                f"| ppl={ppl:.2f} | train_time={train_time:.1f}s "
                f"| tokens/s={tps:.1f} | params={num_params}"
            )

            if (global_best is None) or (val_loss < global_best["val_loss"]):
                global_best = record

    total_time = time.perf_counter() - search_start
    print(f"\nTotal random-search runtime: {total_time:.1f}s")

    if global_best is not None:
        global_best["search_total_time"] = total_time

    return history, global_best



def full_eval_from_log(
    log_path: str,
    lm_datasets,
    tokenizer,
    vocab_size: int,
    block_size: int,
    epochs: int,
    output_path: str,
):
    """
    For each architecture in `log_path` (EA / random / BO-style JSONL),
    run full training for `epochs` and log:

      - val_loss_short, ppl_short  (if present in original log)
      - val_loss_full,  ppl_full   (new full-run numbers)

    Writes one JSON line per architecture to `output_path`.
    """
    from copy import deepcopy

    with open(log_path, "r") as f:
        records = [json.loads(line) for line in f]

    print(f"[full_eval_from_log] Loaded {len(records)} candidates from {log_path}")

    with open(output_path, "w") as out_f:
        for i, rec in enumerate(records):
            cfg_dict = rec["config"]
            cfg = EvoConfig(**cfg_dict)

            print(
                f"\n[full_eval_from_log] ({i+1}/{len(records)}) "
                f"arch_id={rec.get('arch_id')} search_type={rec.get('search_type')} "
                f"val_loss_short={rec.get('val_loss')}"
            )

            best_val_loss, best_ppl, _ = train_full(
                cfg,
                lm_datasets,
                tokenizer,
                vocab_size,
                block_size=block_size,
                epochs=epochs,
            )

            out_rec = {
                "arch_id": rec.get("arch_id"),
                "search_type": rec.get("search_type"),
                "generation": rec.get("generation"),
                "index": rec.get("index", rec.get("candidate", rec.get("trial"))),
                "config": deepcopy(cfg_dict),
                "val_loss_short": rec.get("val_loss"),
                "perplexity_short": rec.get("perplexity"),
                "val_loss_full": float(best_val_loss),
                "perplexity_full": float(best_ppl),
            }
            out_f.write(json.dumps(out_rec) + "\n")
            out_f.flush()

    print(f"[full_eval_from_log] Wrote full-run metrics to {output_path}")


def sample_neighbors_from_log(
    log_path: str,
    lm_datasets,
    tokenizer,
    vocab_size: int,
    block_size: int,
    max_train_steps: int,
    max_eval_batches: int,
    num_neighbors: int,
    output_path: str,
):
    """
    Take the *best* architecture in `log_path` (by short-run val_loss),
    generate `num_neighbors` 1-step EA mutations using `mutate(cfg)`,
    and short-train all of them to get neighbor losses.

    This gives you local neighborhood data around x* for Δ(x*) / κ(D) proxies.
    """
    with open(log_path, "r") as f:
        records = [json.loads(line) for line in f]

    if not records:
        raise ValueError(f"No records in {log_path}")

    # pick best by short-run val_loss
    best_rec = min(
        records,
        key=lambda r: float("inf") if r.get("val_loss") is None else r["val_loss"],
    )
    base_cfg = EvoConfig(**best_rec["config"])
    base_arch_id = best_rec.get("arch_id")

    print(
        f"[neighbors] Using best candidate arch_id={base_arch_id}, "
        f"val_loss={best_rec.get('val_loss')}, search_type={best_rec.get('search_type')}"
    )

    neighbors = hamming_neighbors(base_cfg)
    if not neighbors:
        print("[neighbors] No Hamming-1 neighbors generated (check search space).")
        return

    if num_neighbors < len(neighbors):
        neighbors = random.sample(neighbors, num_neighbors)

    best_loss = best_rec.get("val_loss", float("inf"))
    neighbor_losses = []
    with open(output_path, "w") as out_f:
        for i, cfg in enumerate(neighbors):
            (
                fitness,
                val_loss,
                ppl,
                train_time,
                tps,
                num_params,
            ) = train_and_eval_short(
                cfg,
                lm_datasets,
                tokenizer,
                vocab_size,
                block_size,
                max_train_steps=max_train_steps,
                max_eval_batches=max_eval_batches,
            )

            rec = {
                "search_type": "neighbor",
                "probe": "hamming1",
                "origin_arch_id": base_arch_id,
                "neighbor_index": i,
                "config": cfg.__dict__,
                "fitness": float(fitness),
                "val_loss": float(val_loss),
                "perplexity": float(ppl),
                "train_time": float(train_time),
                "tokens_per_second": float(tps),
                "num_params": int(num_params),
            }
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            neighbor_losses.append(float(val_loss))

            print(
                f"[neighbors] neighbor {i}: val_loss={val_loss:.4f}, "
                f"ppl={ppl:.2f}, params={num_params}"
            )

    # simple valley-depth proxy
    if neighbor_losses and best_loss is not None:
        delta_hat = min(neighbor_losses) - best_loss
        try:
            print(f"[neighbors] valley-depth proxy Δ̂(x*): {delta_hat:.4f}")
        except Exception:
            pass

    print(f"[neighbors] Wrote neighbor evaluations to {output_path}")


def repeat_eval_from_log(
    log_path: str,
    lm_datasets,
    tokenizer,
    vocab_size: int,
    block_size: int,
    max_train_steps: int,
    max_eval_batches: int,
    target_arch_id: int | None,
    n_runs: int,
    output_path: str,
):
    """
    Re-evaluate the SAME architecture n_runs times with the short-run
    training budget, to estimate evaluation noise / variance σ².

    If target_arch_id is None, we use the best val_loss in the log.
    """
    with open(log_path, "r") as f:
        records = [json.loads(line) for line in f]

    if not records:
        raise ValueError(f"No records in {log_path}")

    if target_arch_id is not None:
        candidates = [r for r in records if r.get("arch_id") == target_arch_id]
        if not candidates:
            raise ValueError(f"arch_id={target_arch_id} not found in {log_path}")
        base_rec = candidates[0]
    else:
        base_rec = min(
            records,
            key=lambda r: float("inf") if r.get("val_loss") is None else r["val_loss"],
        )

    cfg = EvoConfig(**base_rec["config"])
    arch_id = base_rec.get("arch_id")

    print(
        f"[repeat_eval] Repeating short-run eval for arch_id={arch_id}, "
        f"val_loss_short={base_rec.get('val_loss')}, n_runs={n_runs}"
    )

    with open(output_path, "w") as out_f:
        for i in range(n_runs):
            (
                fitness,
                val_loss,
                ppl,
                train_time,
                tps,
                num_params,
            ) = train_and_eval_short(
                cfg,
                lm_datasets,
                tokenizer,
                vocab_size,
                block_size,
                max_train_steps=max_train_steps,
                max_eval_batches=max_eval_batches,
            )

            rec = {
                "search_type": "repeat_eval",
                "arch_id": arch_id,
                "run_index": i,
                "config": cfg.__dict__,
                "fitness": float(fitness),
                "val_loss": float(val_loss),
                "perplexity": float(ppl),
                "train_time": float(train_time),
                "tokens_per_second": float(tps),
                "num_params": int(num_params),
            }
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()

            print(
                f"[repeat_eval] run {i}: val_loss={val_loss:.4f}, "
                f"ppl={ppl:.2f}, params={num_params}"
            )

    print(f"[repeat_eval] Wrote repeated evals to {output_path}")

# -----------------------------
# Full training for a given config
# -----------------------------

def train_full(
    config: EvoConfig,
    lm_datasets,
    tokenizer,
    vocab_size: int,
    block_size: int,
    epochs: int = 10,
    lr: float = 3e-4,
    max_grad_norm: float = 1.0,
):
    """
    Train a given config for multiple epochs over the full training set.
    Returns (best_val_loss, best_val_ppl, best_state_dict).
    """
    train_loader, val_loader = make_dataloaders(lm_datasets, config.batch_size)

    model = EvoTransformerLM(
        vocab_size=vocab_size, block_size=block_size, config=config
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        )

    train_ds = lm_datasets["train"]
    tokens_per_epoch = len(train_ds) * block_size
    print(f"Approx tokens/epoch: {tokens_per_epoch}")

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        start = time.perf_counter()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            B, T = input_ids.size()

            if T > block_size:
                input_ids = input_ids[:, :block_size]
            elif T < block_size:
                pad_len = block_size - T
                pad = torch.full(
                    (B, pad_len),
                    pad_token_id,
                    device=device,
                    dtype=input_ids.dtype,
                )
                input_ids = torch.cat([input_ids, pad], dim=1)

            logits = model(input_ids)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            shift_labels = shift_labels.masked_fill(shift_labels == pad_token_id, -100)

            loss = F.cross_entropy(
                shift_logits.view(-1, vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )


            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            n_tokens = (shift_labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens


        scheduler.step()
        train_time = time.perf_counter() - start
        train_loss = total_loss / max(total_tokens, 1)
        train_ppl = math.exp(min(train_loss, 20.0))

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_tokens = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                B, T = input_ids.size()

                if T > block_size:
                    input_ids = input_ids[:, :block_size]
                elif T < block_size:
                    pad_len = block_size - T
                    pad = torch.full(
                        (B, pad_len),
                        pad_token_id,
                        device=device,
                        dtype=input_ids.dtype,
                    )
                    input_ids = torch.cat([input_ids, pad], dim=1)

                logits = model(input_ids)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = input_ids[:, 1:].contiguous()

                shift_labels = shift_labels.masked_fill(shift_labels == pad_token_id, -100)

                loss = F.cross_entropy(
                    shift_logits.view(-1, vocab_size),
                    shift_labels.view(-1),
                    reduction="sum",
                    ignore_index=-100,
                )
                val_loss_total += loss.item()
                val_tokens += (shift_labels != -100).sum().item()


        val_loss = val_loss_total / max(val_tokens, 1)
        val_ppl = math.exp(min(val_loss, 20.0))
        tokens_per_sec = total_tokens / max(train_time, 1e-6)

        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
            f"(ppl={train_ppl:.2f}) | val_loss={val_loss:.4f} "
            f"(ppl={val_ppl:.2f}) | time={train_time:.1f}s | "
            f"tokens/s={tokens_per_sec:.1f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    return best_val_loss, math.exp(min(best_val_loss, 20.0)), best_state


# -----------------------------
# Config save/load
# -----------------------------

def save_best_config(path: str, best_record: dict):
    cfg = best_record["config"]
    obj = {
        "config": cfg,
        "val_loss": best_record["val_loss"],
        "perplexity": best_record["perplexity"],
    }
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_config_from_json(path: str) -> EvoConfig:
    with open(path, "r") as f:
        obj = json.load(f)
    cfg = obj["config"]
    return EvoConfig(**cfg)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="ea",
        choices=[
            "ea",
            "random",
            "train_best",
            "train_manual",
            "full_eval",
            "neighbors",
            "repeat_eval",
        ],
        help=(
            "ea = evolutionary search; random = random search baseline; "
            "train_best = train evolved config; train_manual = train baseline config; "
            "full_eval = full training for all configs in a log; "
            "neighbors = sample neighbors around best config from a log; "
            "repeat_eval = repeated short evals for one config from a log"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wt2",
        choices=["wt2", "wt103"],
        help="wt2 = WikiText-2, wt103 = WikiText-103",
    )
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--ea_pop_size", type=int, default=4)
    parser.add_argument("--ea_generations", type=int, default=3)
    parser.add_argument("--ea_train_steps", type=int, default=100)
    parser.add_argument("--ea_eval_batches", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--config_json", type=str, default="best_config.json")

    # extra args for new modes
    parser.add_argument(
        "--log_jsonl",
        type=str,
        default="",
        help="Path to candidate log (.jsonl) for full_eval / neighbors / repeat_eval",
    )
    parser.add_argument(
        "--neighbors_k",
        type=int,
        default=16,
        help="Number of neighbors to sample in 'neighbors' mode",
    )
    parser.add_argument(
        "--repeat_target_arch_id",
        type=int,
        default=None,
        help="arch_id to re-evaluate in 'repeat_eval' mode (default: best in log)",
    )
    parser.add_argument(
        "--repeat_runs",
        type=int,
        default=5,
        help="Number of repeated runs in 'repeat_eval' mode",
    )

    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)

    # ---------------- Dataset ----------------
    if args.dataset == "wt2":
        base_dir = "data/wikitext2"
        print(f"Loading WikiText-2 from {base_dir} ...")
        raw_datasets = load_wikitext2(base_dir)
    elif args.dataset == "wt103":
        base_dir = "data/wikitext103"
        print(f"Loading WikiText-103 from {base_dir} ...")
        raw_datasets = load_wikitext103(base_dir)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print("Loading tokenizer (GPT-2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    vocab_size = len(tokenizer)
    print(f"Vocab size (after adding pad): {vocab_size}")

    log_suffix = args.dataset
    ea_log_path = f"ea_candidates_{log_suffix}.jsonl"
    random_log_path = f"random_candidates_{log_suffix}.jsonl"

    print("Tokenizing + grouping dataset...")
    lm_datasets = tokenize_wikitext2(
        raw_datasets, tokenizer, block_size=args.block_size
    )

    # ---------------- Modes ----------------
    if args.mode == "ea":
        print("Running evolutionary search (short training)...")
        history, global_best = evolutionary_search(
            lm_datasets,
            tokenizer,
            vocab_size,
            block_size=args.block_size,
            pop_size=args.ea_pop_size,
            generations=args.ea_generations,
            max_train_steps=args.ea_train_steps,
            max_eval_batches=args.ea_eval_batches,
            log_path=ea_log_path,
        )
        print("\nBest over all generations:")
        print(global_best)
        save_best_config(args.config_json, global_best)
        print(f"\nSaved best config to {args.config_json}")

    elif args.mode == "random":
        print("Running random search baseline (short training)...")
        history, global_best = random_search(
            lm_datasets,
            tokenizer,
            vocab_size,
            block_size=args.block_size,
            num_candidates=args.ea_pop_size * args.ea_generations,
            max_train_steps=args.ea_train_steps,
            max_eval_batches=args.ea_eval_batches,
            log_path=random_log_path,
        )
        print("\nBest over all random-search candidates:")
        print(global_best)
        save_best_config("best_random_config.json", global_best)
        print("\nSaved best random-search config to best_random_config.json")

    elif args.mode == "train_best":
        print(f"Loading best config from {args.config_json}")
        cfg = load_config_from_json(args.config_json)
        print(f"Best config: {cfg}")
        best_val_loss, best_ppl, best_state = train_full(
            cfg,
            lm_datasets,
            tokenizer,
            vocab_size,
            block_size=args.block_size,
            epochs=args.epochs,
        )
        torch.save(
            {
                "config": cfg.__dict__,
                "state_dict": best_state,
            },
            "evolved_best_model.pt",
        )
        print(f"\nFinal best val_loss={best_val_loss:.4f}, ppl={best_ppl:.2f}")
        print("Saved evolved model to evolved_best_model.pt")

    elif args.mode == "train_manual":
        print("Training manual baseline config...")
        cfg = EvoConfig(
            d_model=256,
            n_heads=4,
            n_layers=4,
            d_ff=512,
            dropout=0.2,
            attention_type="full",
            chunk_size=32,
            batch_size=16,
        )
        print(f"Manual baseline config: {cfg}")
        best_val_loss, best_ppl, best_state = train_full(
            cfg,
            lm_datasets,
            tokenizer,
            vocab_size,
            block_size=args.block_size,
            epochs=args.epochs,
        )
        torch.save(
            {
                "config": cfg.__dict__,
                "state_dict": best_state,
            },
            "baseline_model.pt",
        )
        print(f"\nFinal baseline val_loss={best_val_loss:.4f}, ppl={best_ppl:.2f}")
        print("Saved baseline model to baseline_model.pt")

    elif args.mode == "full_eval":
        if not args.log_jsonl:
            raise ValueError("--log_jsonl is required for mode=full_eval")
        out_path = args.log_jsonl.replace(".jsonl", "_full_eval.jsonl")
        full_eval_from_log(
            args.log_jsonl,
            lm_datasets,
            tokenizer,
            vocab_size,
            block_size=args.block_size,
            epochs=args.epochs,
            output_path=out_path,
        )

    elif args.mode == "neighbors":
        if not args.log_jsonl:
            raise ValueError("--log_jsonl is required for mode=neighbors")
        out_path = args.log_jsonl.replace(".jsonl", "_neighbors.jsonl")
        sample_neighbors_from_log(
            args.log_jsonl,
            lm_datasets,
            tokenizer,
            vocab_size,
            block_size=args.block_size,
            max_train_steps=args.ea_train_steps,
            max_eval_batches=args.ea_eval_batches,
            num_neighbors=args.neighbors_k,
            output_path=out_path,
        )

    elif args.mode == "repeat_eval":
        if not args.log_jsonl:
            raise ValueError("--log_jsonl is required for mode=repeat_eval")
        out_path = args.log_jsonl.replace(".jsonl", "_repeats.jsonl")
        repeat_eval_from_log(
            args.log_jsonl,
            lm_datasets,
            tokenizer,
            vocab_size,
            block_size=args.block_size,
            max_train_steps=args.ea_train_steps,
            max_eval_batches=args.ea_eval_batches,
            target_arch_id=args.repeat_target_arch_id,
            n_runs=args.repeat_runs,
            output_path=out_path,
        )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")



if __name__ == "__main__":
    main()
