#!/usr/bin/env python3
"""
Evaluate a trained BitLM: held-out perplexity (separately for zh and en) plus a
ternary-sparsity report. Keeps eval cheap so it can run every ckpt on the GPU box.

Usage:
    python scripts/eval.py --config configs/small_60m.yaml --ckpt checkpoints/small_60m/ckpt_20000.pt
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bitllm.config import load_config  # noqa: E402
from bitllm.model import build_model  # noqa: E402
from bitllm.quant import ternarize_state_dict_report  # noqa: E402
from bitllm.data import PackedShard, PackedDataset  # noqa: E402


@torch.no_grad()
def perplexity(model, ds, device, max_batches=200, batch_size=8):
    model.eval()
    losses = []
    n = min(len(ds), max_batches * batch_size)
    for i in range(0, n, batch_size):
        xs, ys = [], []
        for j in range(i, min(i + batch_size, n)):
            x, y = ds[j]
            xs.append(x)
            ys.append(y)
        X = torch.stack(xs).to(device)
        Y = torch.stack(ys).to(device)
        _, loss = model(X, Y)
        losses.append(loss.item())
    mean = sum(losses) / max(len(losses), 1)
    return math.exp(mean), mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--max_batches", type=int, default=200)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(cfg.model).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.set_quant_alpha(1.0)  # eval at full ternary

    print("param report:", model.param_report())

    import glob
    for path in sorted(glob.glob(os.path.join(cfg.data.packed_dir, "*.bin"))):
        shard = PackedShard(path)
        ds = PackedDataset(shard, cfg.model.max_seq_len)
        if len(ds) == 0:
            continue
        ppl, nll = perplexity(model, ds, device, args.max_batches)
        print(f"[{shard.lang:>3}] {os.path.basename(path):40s} ppl={ppl:8.2f} nll={nll:.4f}")

    print("\nternary sparsity report:")
    rep = ternarize_state_dict_report(model)
    zeros = [v["zero_fraction"] for v in rep.values()]
    if zeros:
        print(f"  mean zero fraction across {len(zeros)} ternary layers: "
              f"{sum(zeros)/len(zeros):.3f}")


if __name__ == "__main__":
    main()
