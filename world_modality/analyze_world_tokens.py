from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np

from .config import DataConfig
from .data_sr100 import build_cache_paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze cached world tokens for predictability and distribution.")
    p.add_argument("--dataset_name", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default="cache")
    p.add_argument("--split", type=str, default="train", help="Cache split prefix to load (usually 'train').")
    p.add_argument("--max_k", type=int, default=8, help="Max future step to analyze (k=1..max_k).")
    p.add_argument("--topn", type=int, default=20, help="How many most frequent tokens to print.")
    return p.parse_args()


def entropy_from_counts(counts: np.ndarray) -> float:
    p = counts.astype(np.float64)
    p = p / max(p.sum(), 1.0)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def transition_same_rate(tokens: np.ndarray, k: int) -> float:
    if len(tokens) <= k:
        return 0.0
    return float((tokens[:-k] == tokens[k:]).mean())


def copy_baseline_accuracy(tokens: np.ndarray, k: int) -> float:
    # Predict w_{t+k} = w_t. Equivalent to same-rate.
    return transition_same_rate(tokens, k)


def bigram_baseline_accuracy(tokens: np.ndarray) -> Tuple[float, Dict[int, int]]:
    """
    Baseline: predict w_{t+1} from w_t using empirical argmax transition.
    Returns accuracy and mapping w_t -> argmax next token.
    """
    if len(tokens) < 2:
        return 0.0, {}

    cur = tokens[:-1]
    nxt = tokens[1:]
    vocab = int(tokens.max()) + 1

    # Build counts per (cur, nxt) using a sparse approach.
    pairs = cur.astype(np.int64) * vocab + nxt.astype(np.int64)
    pair_counts = np.bincount(pairs, minlength=vocab * vocab)
    pair_counts = pair_counts.reshape(vocab, vocab)

    mapping = pair_counts.argmax(axis=1).astype(np.int64)
    pred = mapping[cur.astype(np.int64)]
    acc = float((pred == nxt).mean())
    return acc, {int(i): int(mapping[i]) for i in range(vocab)}


def main() -> None:
    args = parse_args()

    data_cfg = DataConfig(dataset_name=args.dataset_name, cache_dir=args.cache_dir)
    paths = build_cache_paths(data_cfg, args.split)

    if not os.path.exists(paths.tokens_path):
        raise FileNotFoundError(f"Missing tokens file: {paths.tokens_path}")

    tokens = np.load(paths.tokens_path, mmap_mode="r").astype(np.int64)
    T = int(tokens.shape[0])
    vocab_used = int(len(np.unique(tokens)))
    vocab_max = int(tokens.max()) + 1

    counts = np.bincount(tokens, minlength=vocab_max)
    ent = entropy_from_counts(counts)
    eff_vocab = float(np.exp(ent))

    top_ids = counts.argsort()[::-1][: args.topn]

    print("=== World Token Analysis ===")
    print(f"dataset_name: {args.dataset_name}")
    print(f"cache_dir:    {args.cache_dir}")
    print(f"split:        {args.split}")
    print(f"num_frames:   {T}")
    print(f"vocab_max:    {vocab_max}")
    print(f"vocab_used:   {vocab_used}")
    print(f"entropy:      {ent:.3f} nats")
    print(f"eff_vocab:    {eff_vocab:.1f} (exp(entropy))")
    print("")

    print(f"Top-{len(top_ids)} tokens by frequency:")
    for tid in top_ids:
        frac = counts[tid] / max(counts.sum(), 1)
        print(f"  token {int(tid):4d}: count={int(counts[tid]):8d}  frac={frac:7.4f}")
    print("")

    print("Temporal coherence / predictability:")
    for k in range(1, max(args.max_k, 1) + 1):
        same = transition_same_rate(tokens, k)
        print(f"  P(w[t]==w[t+{k}]) (copy baseline acc): {same:7.4f}")
    print("")

    bigram_acc, _ = bigram_baseline_accuracy(tokens)
    print(f"Bigram baseline acc (argmax p(w[t+1]|w[t])): {bigram_acc:7.4f}")

    # Guidance.
    print("\nInterpretation hints:")
    print("- If P(w[t]==w[t+1]) is ~0 and bigram acc is ~1/vocab, tokens are too noisy for prediction.")
    print("- If a few tokens dominate (high frac), try larger codebook or embedding normalization.")
    print("- If eff_vocab is much smaller than vocab_max, many codes are unused (k-means instability).")


if __name__ == "__main__":
    main()

