#!/usr/bin/env python3
"""
Train a bilingual (zh/en) byte-level BPE tokenizer.

Why byte-level BPE for zh/en at small scale:
  * Byte fallback => zero UNK on any Unicode (rare hanzi, emoji) without a giant
    char vocab. Chinese chars become 1-3 byte-merges.
  * A single shared vocab for both languages keeps the (fp16) embedding table as
    small as possible — critical when embeddings are ~30-60% of a sub-100M budget.
  * We deliberately cap vocab size; sweep 32k/48k/64k and measure fertility
    (tokens per Chinese char and per English word) vs the embedding param cost.

Usage:
    python scripts/train_tokenizer.py \
        --zh_files data/raw/zh/*.txt --en_files data/raw/en/*.txt \
        --vocab_size 48000 --out tokenizer/bitllm-zhen.json
"""

from __future__ import annotations

import argparse
import glob

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


SPECIAL_TOKENS = ["<pad>", "<s>", "</s>", "<unk>"]


def iter_files(patterns):
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zh_files", nargs="*", default=[])
    ap.add_argument("--en_files", nargs="*", default=[])
    ap.add_argument("--vocab_size", type=int, default=48000)
    ap.add_argument("--min_frequency", type=int, default=2)
    ap.add_argument("--out", default="tokenizer/bitllm-zhen.json")
    args = ap.parse_args()

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    # Byte-level so any Unicode is representable; add_prefix_space for word boundaries.
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    def corpus():
        # Interleave zh and en so BPE merges see a balanced distribution.
        yield from iter_files(args.zh_files)
        yield from iter_files(args.en_files)

    tok.train_from_iterator(corpus(), trainer=trainer)

    tok.post_processor = processors.TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", tok.token_to_id("<s>")), ("</s>", tok.token_to_id("</s>"))],
    )

    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    print(f"saved tokenizer ({tok.get_vocab_size()} tokens) -> {args.out}")

    # quick fertility probe
    zh_sample = "人工智能正在改变世界，我们需要更小更高效的模型。"
    en_sample = "Artificial intelligence is changing the world; we need smaller models."
    print("zh tokens/char:",
          len(tok.encode(zh_sample).ids) / len(zh_sample))
    print("en tokens/word:",
          len(tok.encode(en_sample).ids) / len(en_sample.split()))


if __name__ == "__main__":
    main()
