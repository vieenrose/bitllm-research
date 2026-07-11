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
task**. Three forces make it hurt small models far more than large ones:

1. **Effective-parameter penalty.** Low-precision training reduces the *effective*
   parameter count — Kumar et al. (*Scaling Laws for Precision*, ICLR 2025) fit
   `N_eff = N·(1 − e^(−P/γ))`. BitNet's own results show ternary needs roughly
   **~2× the hidden size** to match FP16. A 3B model can spend that 2× and still be
   small; a 50M model that must double its width is no longer sub-100M.
2. **Redundancy runs out.** Large models are over-parameterized, so ternary
   rounding error is absorbed by spare capacity. Sub-100M models are already
   capacity-bound — every weight matters, so the same relative error hits harder.
3. **The embedding tax inverts the compression story.** In a *bilingual* sub-100M
   model the token embedding + LM head (which stay full precision) are **~45–60% of
   all parameters** (see `scripts/plan_budget.py`). So only ~half the model is
   actually ternarized, the ~8–10× size win collapses to **~1.5–2×**, and "1-bit"
   barely describes the model. At >1B the transformer dwarfs the embeddings, so
   ternarization delivers its full benefit.

The full, cited analysis is in **[docs/WHY_1BIT_NEEDS_SCALE.md](docs/WHY_1BIT_NEEDS_SCALE.md)**.

## The sub-100M plan

The counter-levers, in rough order of impact (full recipe in
**[docs/DESIGN_SUB100M.md](docs/DESIGN_SUB100M.md)**):

- **Mixed-precision islands** — keep embeddings, LM head, and the first/last blocks
  full precision; ternarize the rest (`fp_boundary_blocks` in the config).
- **Shrink the embedding tax** — a lean shared byte-level BPE vocab (32–48k) for
  zh/en; measure fertility vs param cost with the planner.
- **Distillation** from an fp teacher — soft targets buy back lost capacity.
- **Over-training** — spend far more than Chinchilla-optimal tokens/param (BitNet's
  2B4T used ~2000 tok/param); small models have the room to be over-trained cheaply.
- **Progressive quantization annealing** — start near fp, anneal to full ternary to
  tame straight-through-estimator noise at narrow width.

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
python scripts/train.py --config configs/small_60m.yaml --smoke

# 3. Real run: tokenizer -> pack data -> train -> eval
python scripts/train_tokenizer.py --zh_files ... --en_files ... --vocab_size 48000
python scripts/prepare_data.py --config configs/small_60m.yaml
python scripts/train.py --config configs/small_60m.yaml
python scripts/eval.py --config configs/small_60m.yaml --ckpt checkpoints/small_60m/ckpt_20000.pt
```

## Key references

- Kumar et al., *Scaling Laws for Precision*, arXiv:2411.04330 (ICLR 2025)
- Ma et al., *The Era of 1-bit LLMs: BitNet b1.58*, arXiv:2402.17764
- *BitNet b1.58 2B4T Technical Report*, arXiv:2504.12285
- *ParetoQ: ... extreme low-bit*, arXiv:2502.02631
- Liu et al., *MobileLLM*, arXiv:2402.14905
- *Spectra: ternary LLMs (TriLM)*, arXiv:2407.12327

_(docs/ contains the fully-cited version, expanded from an adversarially-verified
literature sweep.)_
