#!/usr/bin/env python3
"""
Tokenize a weighted zh/en corpus mix into flat token shards for pretraining.

Reads sources from the experiment config (data.sources), streams each HF dataset,
tokenizes with the trained tokenizer, and writes one packed .bin per source plus
a .meta sidecar (dtype + lang). PackedDataset/MixtureLoader consume these.

Usage:
    python scripts/prepare_data.py --config configs/small_60m.yaml \
        --max_tokens_per_source 2_000_000_000

Each source in the YAML looks like:
    - name: HuggingFaceFW/fineweb-edu
      config: sample-10BT
      split: train
      text_field: text
      lang: en
      weight: 0.5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bitllm.config import load_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_tokens_per_source", type=int, default=2_000_000_000)
    ap.add_argument("--shard_flush", type=int, default=50_000_000)
    args = ap.parse_args()

    from datasets import load_dataset
    from tokenizers import Tokenizer

    cfg = load_config(args.config)
    tok = Tokenizer.from_file(cfg.data.tokenizer_path)
    vocab = tok.get_vocab_size()
    dtype = np.uint16 if vocab < 65536 else np.uint32
    eos_id = tok.token_to_id("</s>")

    os.makedirs(cfg.data.packed_dir, exist_ok=True)

    for src in cfg.data.sources:
        name = src["name"]
        lang = src.get("lang", "unk")
        text_field = src.get("text_field", "text")
        out_path = os.path.join(cfg.data.packed_dir, f"{lang}_{name.replace('/', '_')}.bin")
        meta_path = out_path + ".meta"
        print(f"\n=== {name} [{lang}] -> {out_path} ===")

        ds = load_dataset(
            name, src.get("config"), split=src.get("split", "train"), streaming=True
        )
        buf = []
        written = 0
        with open(out_path, "wb") as fout:
            for ex in ds:
                text = ex.get(text_field)
                if not text:
                    continue
                ids = tok.encode(text).ids
                ids.append(eos_id)
                buf.extend(ids)
                if len(buf) >= args.shard_flush:
                    arr = np.array(buf, dtype=dtype)
                    arr.tofile(fout)
                    written += len(buf)
                    buf = []
                    print(f"  {written/1e6:.1f}M tokens", end="\r")
                if written >= args.max_tokens_per_source:
                    break
            if buf:
                np.array(buf, dtype=dtype).tofile(fout)
                written += len(buf)
        with open(meta_path, "w") as f:
            f.write(f"dtype={'uint16' if dtype==np.uint16 else 'uint32'}\n")
            f.write(f"lang={lang}\n")
            f.write(f"tokens={written}\n")
            f.write(f"source={name}\n")
        print(f"\n  wrote {written/1e6:.1f}M tokens")


if __name__ == "__main__":
    main()
