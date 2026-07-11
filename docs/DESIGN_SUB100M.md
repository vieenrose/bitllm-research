# Design: a sub-100M ternary zh/en SLM ("Bonsai-Nano-90M")

> The buildable plan. Reads on from [WHY_1BIT_NEEDS_SCALE.md](WHY_1BIT_NEEDS_SCALE.md):
> given that 1-bit is a capacity tax worst at small scale, this composes the ten
> counter-levers into one concrete, GPU-ready recipe. Code lives in `src/bitllm/`
> and `scripts/`; the reference config is `configs/bonsai_nano_90m.yaml`.

## 1. Design philosophy

**Do not target FP16 parity — it is unrealistic below ~1B** (WHY §1, §4). Target
the **best capability-per-MB** at ~27–39 MB deployed, CPU-runnable via bitnet.cpp.
Every decision below spends scarce params on capability, protects the parts that
resist quantization, and injects external capacity via distillation.

The honest budget: an ~92M nominal ternary model delivers only ~**55–60M
FP16-equivalent capacity**. Distillation is therefore **mandatory, not optional.**

## 2. Architecture — `BitLMConfig`

MobileLLM-style **deep-and-thin** dense decoder, deliberately **wide-for-its-depth**
vs a pure FP16 design because ternary needs ~2× hidden to recover effective width.

| Field | Value | Why |
|-------|-------|-----|
| `d_model` | 512 | wide-ish for the param class (ternary 2×-width rule) |
| `n_layers` | 24 | deep-and-thin (MobileLLM) |
| `n_heads` / `n_kv_heads` | 8 / 2 | GQA — cuts KV projection params |
| `ffn_hidden` | 1408 | ~2.75·d SwiGLU (≈8/3·d, rounded to 64) |
| `max_seq_len` | 2048 | |
| RoPE, RMSNorm+SubLN | — | RoPE saves a positional table; SubLN stabilizes QAT |
| `tie_embeddings` | **true** | avoids paying for the head twice — the #1 budget lever |
| `ffn_activation` | silu (opt. relu2) | ReLU² induces activation sparsity (BitNet 2B4T) |

## 3. Parameter budget (~92M total)

| Component | Precision | Params | Share |
|-----------|-----------|-------:|:-----:|
| Token embedding (tied w/ head) | **int8** deploy / fp train | 48,000 × 512 = **24.58M** | ~27% |
| 24 × transformer block | **ternary {−1,0,+1}** | 24 × 2.82M = **67.6M** | ~73% |
| — per block: attn (Wq 512², Wk/Wv 512×128, Wo 512²) | ternary | 0.66M | |
| — per block: SwiGLU (gate/up 512×1408, down 1408×512) | ternary | 2.16M | |
| RMSNorm / SubLN / RoPE / scales | fp16 | <0.1M | ~0% |
| **Total** | — | **~92.2M** | 100% |

**Deployed size** (from `scripts/plan_budget.py`, the authoritative arithmetic):
ternary blocks ~12 MB at 1.58 bit + the two fp16 boundary blocks ~11 MB + group
scales ~1 MB + embeddings **24.6 MB (int8)** → **~48 MB**, or **~36 MB** with int4
embeddings. (Quantizing the boundary blocks to int8 at deploy time instead of fp16
brings int8-emb down to ~43 MB.) Even so that is only a **~3.8–5.2× shrink** vs the
184 MB fp16 model — the embedding floor caps the win, exactly as WHY §2.5 predicts,
versus the ~8–10× you'd get at >1B. Verify any config with `plan_budget.py` first.

## 4. Quantization scheme — **W1.58A8**, mixed-precision islands

- **Ternarize {−1,0,+1}** via absmean RoundClip with an FP16 scale **per output
  channel** (or per 128-weight group, bitnet.cpp I2_S-compatible): **all** attention
  Q/K/V/O and **all** FFN gate/up/down. → `BitLinear(weight_bits="ternary")`.
- **Keep int8** (higher precision): the embedding table + tied LM head (per-row
  absmax; int4 optional to shave size), and **all activations** (dynamic per-token
  absmax int8 — the A8 path; **do not** push to A4 at this scale — FC2 outliers).
- **Keep FP16/FP32:** RMSNorm/SubLN weights, RoPE, scale factors.
- **First & last transformer block stay full precision** (`fp_boundary_blocks=1`) —
  layer-sensitivity evidence says the boundary blocks matter most.
- **Training time:** FP16/BF16 **latent master weights** + STE through the ternary
  forward. **Never PTQ an FP16 checkpoint to ternary** (→ PPL ~10²¹–10²³).

Embeddings are int8 at *deploy* time (int8 PTQ of a lookup table is ~lossless); at
*train* time they stay fp and the LM head is tied. `fp_boundary_blocks` and
`group_size` are config knobs; `act_bits=8`.

## 5. Tokenizer & data (zh/en)

**Tokenizer:** custom **byte-level BPE, vocab 48,000, byte fallback**, trained
jointly on zh+en. Rationale (WHY §2.5): 48K is large enough for adequate Chinese
coverage — avoiding the 3–4×-tokens-per-hanzi blowup a 32K Latin-centric vocab
causes (Chinese-LLaMA: adding ~20K Chinese tokens ~halves Chinese sequence length)
— yet bounded so the tied embedding stays ~27% of budget. **Do NOT reuse a
128K–151K LLM tokenizer** (Qwen/Llama-3): the embedding table alone would exceed the
whole param budget. → `scripts/train_tokenizer.py --vocab_size 48000`.

**Data:** ~**50/50 zh/en** by gradient steps (balance by *tokens-per-source*, not
raw docs — fertilities differ). English from FineWeb-Edu / DCLM-class; Chinese from
SkyPile / CCI / WanJuan-class; +~10–15% code & math for reasoning transfer. Total
budget **46B–180B tokens (~500–2000 tok/param)** — safe and beneficial under QAT.
→ `scripts/prepare_data.py` (weighted `MixtureLoader`).

## 6. Training recipe

**Phase 0 — teacher.** Pick a strong bilingual FP16 teacher (e.g. Qwen2.5-0.5B or
1.5B) for logit (+ optional hidden/attention) distillation. Tokenizer alignment is
the catch: either adopt a teacher whose tokenizer you mirror, or accept logit-only
KD with a shared-vocab teacher. (`src/bitllm/distill.py`.)

**Phase 1 — QAT pretrain from scratch** (`scripts/train.py`):
- FP16 latent weights + STE, absmean ternarization, **W1.58A8**.
- **Progressive fp→ternary anneal:** `quant_alpha` 0→1 over the first ~15% of steps
  (`quant_anneal_start/end`) — tames STE oscillation at narrow width.
- **BitNet stability recipe:** large peak LR (ternary tolerates LRs that diverge
  FP16), **two-stage LR** (high plateau → abrupt cooldown), **two-stage weight
  decay** (0.1 → 0), SubLN.
- **Loss** = next-token CE + KL distillation to teacher logits (T≈1–2) [+ optional
  feature/attention matching].
- **Over-train:** start 46B tokens, scale to 100–180B if compute allows.

**Phase 2 — SFT.** Sum-loss aggregation; more epochs / higher LR than an FP16 model
(per BitNet 2B4T). **Phase 3 (optional) — DPO:** ~2 epochs, LR ~2e-7, β 0.1.

## 7. Evaluation — controls first

The literature repeatedly conflates the ternary handicap with the token budget, so:
1. **Train an FP16 model of the *same* architecture on the *same* data/tokens** —
   this isolates the quantization gap from everything else.
2. **Perplexity:** held-out **zh and en splits reported separately**
   (`scripts/eval.py`).
3. **English zero-shot** (BitNet suite): ARC-e/c, HellaSwag, PIQA, WinoGrande,
   BoolQ, OpenBookQA. **Chinese:** C-Eval, CMMLU, CLUE.
4. **External anchors:** MobileLLM-125M, Qwen2.5-0.5B (FP16 sanity).
5. **Efficiency:** deployed MB, CPU tok/s and J/token via **bitnet.cpp I2_S** vs an
   fp16/llama.cpp baseline; report **score-per-MB** ("intelligence density").
6. **Ablations that test the thesis:** ternary vs binary (expect binary to lose);
   distilled vs from-scratch (expect distillation = largest lever); embeddings
   int8 vs int4 vs ternary; **width 512 vs 768 at matched ~92M** (tests the 2×-width
   rule); token budget 46B vs 100B vs 180B (tests overtraining-helps-under-QAT).

## 8. Levers, ranked by measured impact

1. **Distillation** from a strong bilingual FP16 teacher — biggest single lever.
2. **Ternary {−1,0,+1}**, not binary (stay above the 2→3-bit cliff).
3. **Widen ~2×** vs an FP16 design at the same param target (budget around N_eff).
4. **QAT from scratch, never PTQ.**
5. **W1.58A8** — int8 embeddings/head/activations, ternary only the block matmuls.
6. **Over-train** (~500–2000 tok/param).
7. **MobileLLM deep-and-thin + GQA + tied embeddings.**
8. **Compact ~48K bilingual vocab + byte fallback.**
9. **BitNet stability recipe + progressive anneal.**
10. **Ship the bitnet.cpp I2_S kernel** — the efficiency is only realized in-kernel.

## 9. Biggest risks (and mitigations)

| Risk | Mitigation |
|------|-----------|
| **Capacity floor** — ~92M ternary ≈ 55–60M fp-equiv, below the ~1.7–3B parity zone | Reframe target to capability-per-MB; lean hard on distillation; a visible gap may remain |
| **Embedding tax** — 48K vocab = ~27% of budget, un-ternarizable | int8/int4 embeddings; bound vocab; tie head; don't shrink vocab (wrecks zh fertility) |
| **STE instability / reconstruction regime** | full BitNet stability recipe + progressive anneal |
| **No native 1-bit hardware** | ship bitnet.cpp I2_S/TL kernels as part of the deliverable |
| **Teacher–student tokenizer mismatch** | pick a teacher you can vocab-align, or logit-only KD w/ shared vocab |

## 10. Repo mapping

| Step | Code |
|------|------|
| Sanity-check the budget (no torch) | `scripts/plan_budget.py` |
| Model + fp islands + anneal | `src/bitllm/model.py`, `src/bitllm/quant.py` |
| Tokenizer (48K byte-BPE) | `scripts/train_tokenizer.py` |
| Pack weighted zh/en mixture | `scripts/prepare_data.py`, `src/bitllm/data.py` |
| QAT pretrain (+ distill, anneal) | `scripts/train.py`, `src/bitllm/distill.py` |
| Per-language ppl + sparsity | `scripts/eval.py` |
| Reference config | `configs/bonsai_nano_90m.yaml` |
| Smoke-test the graph (CPU) | `python scripts/train.py --config configs/bonsai_nano_90m.yaml --smoke` |
