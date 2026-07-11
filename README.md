# bitllm-research — 1-bit small language models at sub-100M (zh/en)

**Question:** Why do 1-bit LLM methods (BitNet / bitnet.cpp, and precision-scaling
work like PRISM) only pay off at **> 1B parameters**, and how do we build a *working*
1-bit / ternary **small** language model at **sub-100M** with **Chinese + English**
support?

This repo holds (1) a researched answer to the "why", and (2) a runnable QAT
training stack designed for the sub-100M regime, ready to drop onto a GPU box.

> Status: research + reference implementation. Training is CPU-free to *develop*
> (pure-python planner + torch-gated tests) and runs on the GPU box via
> `scripts/train.py`.

## TL;DR of the "why"

Extreme quantization is a **capacity tax that scales with the model, not with the
task**. The measured ternary-vs-FP16 gap **shrinks with scale and crosses over at
~1.3–3B** — which is exactly where the published wins (and PrismML's shipped
1.7B/4B/8B "Bonsai" models) live:

| Size | ternary vs FP16 |
|-----:|-----------------|
| 6–48M | **+50–84% perplexity** (tiny-decoder study) |
| 700M | +4% PPL / −1.2 acc pts (trails) |
| 1.3B | ~0 (parity) |
| 3B | **−0.13 PPL / +0.5 acc (ternary wins)** |

Five compounding mechanisms drive this (full cited analysis in
**[docs/WHY_1BIT_NEEDS_SCALE.md](docs/WHY_1BIT_NEEDS_SCALE.md)**); the three biggest:

1. **Effective-parameter penalty.** Low-precision training cuts the *effective*
   parameter count — Kumar et al. (*Scaling Laws for Precision*, ICLR 2025) fit
   `N_eff = N·(1 − e^(−P/γ))`; BitNet needs **~2× the hidden size** to match FP16. A
   3B model can spend that 2×; a 50M model that must double its width isn't sub-100M.
2. **Redundancy runs out.** Big models absorb ternary rounding error in spare
   capacity; sub-100M models are capacity-bound, so the same relative error bites —
   and pure binary sits below ParetoQ's **2→3-bit "reconstruction" cliff** (use
   ternary, not binary).
3. **The bilingual embedding tax inverts the compression story.** The full-precision
   token embedding + LM head are **~20–40% of a sub-100M zh/en model** (Chinese needs
   a big vocab). So only ~half the model is ternarized, the ~8–10× size win collapses
   to **~3.8–5.2× deployed** (`scripts/plan_budget.py`), and "1-bit" barely describes
   it. At >1B the transformer dwarfs the embeddings and 1-bit delivers fully.

**"PRISM" = PrismML** (HF `prism-ml`, Apache-2.0 1-bit/1.58-bit "Bonsai" LLMs,
Qwen3-derived GGUF `Q1_0` ≈1.125 bits/weight) — smallest text model **1.7B**,
nothing sub-1B, confirming the ">1B only" observation.

## The sub-100M plan

**Reframe the target: not FP16 parity (unrealistic sub-1B) but best capability-per-MB.**
The recommended model is **Bonsai-Nano-90M** (`configs/bonsai_nano_90m.yaml`):
~92M total (vocab 48k, d=512, 24 layers, GQA, tied int8 embeddings), ~48 MB deployed.
Counter-levers in order of measured impact (full recipe in
**[docs/DESIGN_SUB100M.md](docs/DESIGN_SUB100M.md)**):

1. **Distillation from a strong bilingual FP16 teacher** — the single biggest lever;
   injects capacity the ~92M ternary student (≈55–60M fp-equivalent) can't hold.
2. **Ternary {−1,0,+1}, not binary** — stay above the 2→3-bit cliff.
3. **Widen ~2× vs an FP16 design** at the same param target (budget around `N_eff`).
4. **QAT from scratch, never PTQ** (1-bit PTQ collapses to ~10²³ perplexity).
5. **W1.58A8 mixed-precision islands** — int8 embeddings/head/activations + fp
   boundary blocks; ternarize only the block matmuls (`fp_boundary_blocks`).
6. **Over-train** ~500–2000 tok/param — safe and *beneficial* under QAT.
7. **Compact ~48k bilingual byte-BPE** (a 32k vocab wrecks Chinese fertility; a
   128k+ LLM tokenizer blows the budget).
8. **Progressive fp→ternary annealing** + the BitNet stability recipe.
9. **Ship the bitnet.cpp I2_S kernel** — the efficiency is only realized in-kernel.

## Layout

```
src/bitllm/
  quant.py      # BitLinear: ternary weights + int8 acts, STE, anneal, mixed precision
  model.py      # Llama-style decoder (RoPE, GQA, SwiGLU) with fp islands
  data.py       # memory-mapped packed-token shards + language-balanced loader
  distill.py    # logit KD from an fp teacher
  config.py     # YAML -> typed dataclasses
scripts/
  plan_budget.py      # param-budget planner (NO torch needed) — run this first
  train_tokenizer.py  # bilingual byte-level BPE
  prepare_data.py     # tokenize+pack a weighted zh/en mixture
  train.py            # QAT pretraining (bf16, anneal, distill, ckpt)  [--smoke to test]
  eval.py             # per-language perplexity + ternary sparsity report
configs/          # experiment configs
docs/             # the research writeup + design
tests/            # torch-gated unit tests + pure-python budget test
```

## Quickstart

```bash
# 1. See how the param budget splits (no torch required)
python scripts/plan_budget.py

# 2. On the GPU box: install, then smoke-test the training graph on random data
pip install -r requirements.txt
python scripts/train.py --config configs/bonsai_nano_90m.yaml --smoke

# 3. Real run: tokenizer -> pack data -> train -> eval
python scripts/train_tokenizer.py --zh_files ... --en_files ... --vocab_size 48000
python scripts/prepare_data.py --config configs/bonsai_nano_90m.yaml
python scripts/train.py --config configs/bonsai_nano_90m.yaml
python scripts/eval.py --config configs/bonsai_nano_90m.yaml --ckpt checkpoints/bonsai_nano_90m/ckpt_20000.pt
```

## Key references

- Kumar et al., *Scaling Laws for Precision*, arXiv:2411.04330 (ICLR 2025)
- Ma et al., *The Era of 1-bit LLMs: BitNet b1.58*, arXiv:2402.17764
- *BitNet b1.58 2B4T Technical Report*, arXiv:2504.12285
- *1-bit AI Infra: BitNet b1.58 on CPUs (bitnet.cpp)*, arXiv:2410.16144
- *ParetoQ: Scaling Laws in Extremely Low-bit Quantization*, arXiv:2502.02631
- Nielsen & Schneider-Kamp, *BitNet b1.58 Reloaded* (tiny 100K–48M), arXiv:2407.09527
- Liu et al., *MobileLLM*, arXiv:2402.14905
- Kaushal et al., *Spectra / Spectra 1.1 (ternary TriLM)*, arXiv:2407.12327 / 2506.23025
- Cui et al., *Chinese LLaMA/Alpaca* (zh tokenizer), arXiv:2304.08177

Full, adversarially-verified reference list in **[docs/WHY_1BIT_NEEDS_SCALE.md](docs/WHY_1BIT_NEEDS_SCALE.md)**.

_(docs/ contains the fully-cited version, expanded from an adversarially-verified
literature sweep.)_
