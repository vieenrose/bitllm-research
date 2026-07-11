"""
Data pipeline: pre-tokenized packed shards for efficient pretraining.

Two stages:
  1. prepare_data.py tokenizes + concatenates a weighted mix of zh/en corpora
     into flat uint16/uint32 token shards (one .bin per source split).
  2. PackedDataset (here) memory-maps those shards and yields fixed-length
     (seq_len+1) windows for next-token prediction, with a language-balanced
     sampler so zh and en stay mixed at the configured ratio every batch.

Keeping this format simple (raw token ids on disk) means the GPU box never
re-tokenizes and I/O is a sequential mmap read.
"""

from __future__ import annotations

import glob
import os
import numpy as np
import torch


def _dtype_for_vocab(vocab_size: int):
    return np.uint16 if vocab_size < 65536 else np.uint32


class PackedShard:
    """Memory-mapped flat token array with metadata sidecar."""

    def __init__(self, path: str):
        self.path = path
        meta_path = path + ".meta"
        self.vocab_dtype = np.uint16
        self.lang = "unk"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                for line in f:
                    k, _, v = line.strip().partition("=")
                    if k == "dtype":
                        self.vocab_dtype = np.uint16 if v == "uint16" else np.uint32
                    elif k == "lang":
                        self.lang = v
        self.data = np.memmap(path, dtype=self.vocab_dtype, mode="r")

    def __len__(self):
        return len(self.data)


class PackedDataset(torch.utils.data.Dataset):
    """Fixed-window views over one shard. Length = n_windows (non-overlapping)."""

    def __init__(self, shard: PackedShard, seq_len: int):
        self.shard = shard
        self.seq_len = seq_len
        self.n = (len(shard) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = i * self.seq_len
        chunk = np.asarray(self.shard.data[s : s + self.seq_len + 1]).astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class MixtureLoader:
    """Yields batches drawn from multiple shards at fixed language weights.

    weights are normalized; each micro-batch is filled by sampling a source
    according to the weights, so the zh/en ratio is honored in expectation on
    every step rather than only over an epoch.
    """

    def __init__(self, packed_dir: str, seq_len: int, micro_batch_size: int,
                 weights: dict | None = None, seed: int = 1234, device="cpu"):
        paths = sorted(glob.glob(os.path.join(packed_dir, "*.bin")))
        if not paths:
            raise FileNotFoundError(
                f"No .bin shards in {packed_dir}. Run scripts/prepare_data.py first."
            )
        self.datasets = []
        self.langs = []
        for p in paths:
            shard = PackedShard(p)
            ds = PackedDataset(shard, seq_len)
            if len(ds) == 0:
                continue
            self.datasets.append(ds)
            self.langs.append(shard.lang)
        self.seq_len = seq_len
        self.mbs = micro_batch_size
        self.device = device
        self.rng = np.random.default_rng(seed)

        # per-shard sampling probability
        w = np.array([
            (weights or {}).get(self.langs[i], 1.0) for i in range(len(self.datasets))
        ], dtype=np.float64)
        self.probs = w / w.sum()
        self._cursors = [self.rng.integers(0, len(d)) for d in self.datasets]

    def _next_from(self, di):
        d = self.datasets[di]
        idx = self._cursors[di] % len(d)
        self._cursors[di] = (self._cursors[di] + 1) % len(d)
        return d[idx]

    def batch(self):
        xs, ys = [], []
        for _ in range(self.mbs):
            di = int(self.rng.choice(len(self.datasets), p=self.probs))
            x, y = self._next_from(di)
            xs.append(x)
            ys.append(y)
        X = torch.stack(xs).to(self.device, non_blocking=True)
        Y = torch.stack(ys).to(self.device, non_blocking=True)
        return X, Y

    def iter_batches(self, n_batches):
        for _ in range(n_batches):
            yield self.batch()
