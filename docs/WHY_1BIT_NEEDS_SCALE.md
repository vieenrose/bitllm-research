# Why 1-bit LLMs (BitNet.cpp, PrismML) only pay off above ~1B — and what that means for sub-100M

> Research writeup. Every quantitative claim below was produced by a fan-out
> literature sweep and then adversarially fact-checked against primary sources
> (papers, official repos, model cards). Corrections from that verification pass
> are folded in and flagged. Sources are listed at the end.

## 0. TL;DR

Extreme quantization (1-bit binary / 1.58-bit ternary) is a **capacity tax that
scales with the model, not with the task.** The quality gap between a ternary model
and its full-precision twin **shrinks as the model grows** and disappears around
**~1.3–3B parameters** — so the published wins live there. At **sub-100M** the same
tax is devastating: measured perplexity penalties of **+50–84%** for tiny ternary
decoders, on top of a **bilingual embedding tax** that leaves only ~half the model
actually quantized. None of this is a hard wall — it is a set of five compounding
mechanisms, each with a concrete counter-lever (see
[DESIGN_SUB100M.md](DESIGN_SUB100M.md)).

## 1. The empirical fact: the gap closes with scale

The cleanest evidence is BitNet b1.58 (native ternary, trained from scratch) vs a
full-precision LLaMA of identical size and token budget:

| Size | Ternary PPL | FP16 PPL | Δ PPL | Ternary avg-acc | FP16 avg-acc | verdict |
|-----:|:-----------:|:--------:|:-----:|:---------------:|:------------:|---------|
| 700M | 12.87 | 12.33 | **+0.54 worse** | 44.3 | 45.5 | ternary trails (−1.2 pts) |
| 1.3B | 11.29 | 11.25 | +0.04 (tie) | 45.9 | 46.2 | gap essentially closed |
| 3B | 9.91 | 10.04 | **−0.13 better** | 50.2 | 49.7 | **crossover: ternary wins** |
| 3.9B | 9.62 | — | — | — | — | beats 3B FP16 at 3.3× less memory |

*(BitNet b1.58 paper Table 1; numbers corroborated by the independent 1bitLLM
reproduction on 100B RedPajama tokens: 3B repro 9.88 PPL / 49.6 avg.)*

Now push **below** 1B. "BitNet b1.58 Reloaded" trained tiny decoders from scratch
and measured the 1.58-bit vs 16-bit penalty directly:

| Size | 1.58-bit PPL (median) | 16-bit PPL | penalty |
|-----:|:---------------------:|:----------:|:-------:|
| 6M | 116.6 | 77.8 | **+50%** |
| 12M | 60.0 | 36.7 | **+63–84%** |
| 24M | 37.5 | 21.4 | **~+70%** |
| 48M | 26.8 | 16.6 | **~+62%** |

The trend is monotonic and unambiguous: **the ternary tax is enormous at 6–48M,
moderate at 700M, zero by 1.3B, and negative (a win) at 3B.** That single curve is
the answer to "why >1B." Binary (pure 1-bit) is worse still — it never closes the
gap even at 6.7B (17.07 vs 15.19 PPL).

A note on causality (verification caveat): the crossover is **scale-dependent and
strongest on downstream accuracy, not raw perplexity at every size** — at 700M
ternary is still slightly worse on PPL. The literature also repeatedly *conflates*
the ternary handicap with token budget, which is why our eval plan insists on an
FP16 control trained on identical data (see DESIGN §7).

## 2. Five mechanisms that make it worse at small scale

### (1) Effective-parameter saturation — the capacity headroom argument
Low-precision training multiplies usable capacity by a **saturating** factor.
Kumar et al. (*Scaling Laws for Precision*, ICLR 2025; fit on 465+ runs, R²≈0.97):

```
N_eff = N · (1 − e^(−P/γ))          # per component; ≈ N·(1−e^(−P/γ̄))³ over w/a/kv
L = A · N_eff^(−α) + B · D^(−β) + E  # slots into the Chinchilla loss
```

At ternary precision this is a real, quantified penalty: **"BitNet b1.58 Reloaded"
finds ternary needs ~2× the hidden size to match FP16 perplexity.** The *relative*
multiplier is roughly scale-invariant, but the **absolute capacity a model can
afford to surrender is tiny for a small net.** A 3B model has spare width to give
up; a sub-100M model that must double its width to compensate is no longer
sub-100M. This is precisely why parity arrives around ~1.3–3B.

### (2) Overtraining fragility — high tokens-per-parameter is a liability under *quantization*
PTQ degradation grows as **δ_PTQ ∝ D^γ_D / N^γ_N** (Kumar et al.): more training
tokens → *more* post-quant damage; bigger model → less. Corroborated by Ouyang et
al. (*Low-Bit Quantization Favors Undertrained LLMs*, 1500+ checkpoints) and a
QAT-specific law (Chen et al., 268 runs: error rises with tokens T, falls with size
N). A small model reaches the "fully-trained, sharp-minimum" regime at *far fewer*
tokens than a large one, so at any realistic budget a tiny model sits deep in the
fragile zone where a naive 1-bit perturbation ejects weights from their basin.

**The fix flips this into an asset:** use **QAT from scratch, never PTQ.** Under
QAT the latent weights adapt to the ternary grid, the D^γ_D fragility term never
applies, and *more* tokens help. Evidence of the collapse if you ignore this:
post-training-quantizing an FP16 model to 1-bit gives perplexity ~**3.5×10²³** vs
17.07 from-scratch. BitNet b1.58 2B4T deliberately trained **~2000 tokens/param
(~100× Chinchilla)** for exactly this reason.

### (3) A representational regime shift below 2 bits
ParetoQ (Meta) documents a **sharp learning transition between 2-bit and 3-bit**:
at ≥3-bit a model stays near its pretrained distribution ("compensation" regime,
where PTQ works); at ≤2-bit representations must be **relearned wholesale**
("reconstruction" regime). Wholesale relearning is capacity-hungry — a big model
has redundant width to spend on it, a sub-100M model must spend scarce capacity
relearning basics. **Ternary {−1,0,+1} sits just above the cliff; pure binary sits
below it and never closes the gap.** The single extra `0` state (only +0.58 bit,
enabling feature *filtering*) is the concrete difference between binary BitNet's
persistent gap and ternary b1.58's parity. → **Use ternary, not binary.**

### (4) STE gradient noise, latent-weight oscillation, and the ternary dead-zone
Ternary QAT trains FP16 latent weights through a straight-through estimator over a
non-differentiable RoundClip. Weights near the absmean thresholds **oscillate**
between bins across steps; the `0`-region is a **dead-zone** where small latent
weights get noisy sign assignments. Large models average this per-weight noise over
massive redundancy; in a small model each weight carries more of the function, so
the noise translates directly into loss variance and unstable convergence. → The
**BitNet stability recipe** (SubLN, two-stage LR with abrupt cooldown, two-stage
weight decay 0.1→0, large LR that FP16 can't tolerate, per-channel absmean scales)
plus a **progressive fp→ternary anneal** damp this cheaply.

### (5) The embedding & activation precision tax — the sub-100M-specific squeeze
Embeddings are a **lookup table, not a matmul** — ternarizing them destroys token
identity. Activations keep int8 (the "A8" in W1.58A8) because pushing to 4-bit hits
the FC2 activation-outlier failure mode (Chen et al.). So the components that
*resist* extreme quantization are exactly the embedding/head and the activation
path — and at small scale **those dominate the budget:**

| Model | Embedding fraction of params |
|-------|:----------------------------:|
| Pythia-70M | **73%** |
| Pythia-160M | 48% |
| GPT-2 small (124M) | 31% |
| SmolLM-135M | 21% |
| MobileLLM-125M (tied) | 15% |
| BitNet b1.58 2B4T | 39% |
| **an 8B model** | **a few %** |

For a **bilingual** model the squeeze is tighter: Chinese needs a large vocab
(Qwen uses 151,643; Llama-3/BitNet 128,256), and a 128k×256 table already *exceeds*
a 30M budget. So in a sub-100M zh/en model the un-ternarizable embedding+head is
**~20–40% of all params, fixed by vocab and independent of the ternary blocks.**
Only ~half the model is actually ternarized, the ~8–10× size win collapses to
**~1.5–3×**, and "1-bit" barely describes the model. At >1B the transformer dwarfs
the embeddings, so ternarization delivers its full benefit — the deepest reason the
technique "wants" to be large.

## 3. What "PRISM" is, and why it is a >1B story

"PRISM ML" is **PrismML** — HF org [`prism-ml`](https://huggingface.co/prism-ml),
GitHub `PrismML-Eng`, site prismml.com — a startup that emerged from stealth
~31 Mar 2026 (Caltech-derived research; coverage in The Register, Forbes, PR
Newswire). It ships open Apache-2.0 1-bit and 1.58-bit **"Bonsai"** LLMs.

- **Method:** Qwen3-dense checkpoints converted to a custom GGUF **"Q1_0"** format —
  one sign bit per weight (0→−scale, 1→+scale) with a single FP16 scale shared per
  128-weight group, ≈**1.125 effective bits/weight**, applied end-to-end (embeddings,
  attention/MLP projections, LM head), plus a ternary (1.58-bit, `q2_0`) variant.
  *Verification caveat:* "no higher-precision layers retained" overstates it —
  RMSNorm weights and the FP16 group scales stay higher precision. These are
  conversions/QAT of pretrained FP Qwen3, **not from-scratch 1-bit pretraining.**
- **Size regime:** the text family is **1.7B, 4B, 8B** (each in binary + ternary),
  plus a separate 4B text-to-image line. **Nothing sub-1B — 1.7B is the smallest.**
  That directly matches the observation that PRISM "works well only on >1B LLMs."

*Honesty caveat:* the model cards do **not** state an explicit hard sub-1B failure
threshold; the ">1B floor" is PrismML's shipped range **plus** the general scaling
literature in §1–2, not a claim PrismML publishes. Treat the exact floor as medium
confidence. (Four unrelated "PRISM" systems exist in ML inference — privacy-aware
routing, photonic KV-cache, GPU-sharing serving, an edge-orchestration project —
none concern 1-bit quantization.)

The same logic explains **bitnet.cpp**: its speed/energy wins (CPU 1.37–6.17×
faster, 55–82% less energy vs llama.cpp fp16; a 100B ternary model at 5–7 tok/s on
one CPU) come from ternary-optimized kernels (I2_S packs 1 weight→2 bits; TL1/TL2
use lookup tables). Those wins are largest where the ternary matmuls dominate —
i.e. at scale, where embeddings are a rounding error.

## 4. So is sub-100M hopeless? No — but the target must change

The honest framing: an ~92M ternary bilingual model delivers only ~**55–60M
FP16-equivalent capacity** (§2.1 + §2.5). FP16 *parity* is not a realistic target
below ~1B. The realistic — and useful — target is **best capability-per-MB** at
~27–39 MB deployed, runnable on CPU. The levers that make that achievable, in order
of measured impact:

1. **Distillation from a strong FP16 teacher** — the biggest lever; injects
   capacity the student can't hold. (FBI-LLM trains *binary* models from scratch by
   distillation and is "competitive" — though, per verification, **not** full
   perplexity parity. Token-scaled logit distillation reached **<1.0 PPL gap** on
   ternary generative LMs with no reasoning-accuracy loss.)
2. **Ternary, not binary** (stay above the 2→3-bit cliff).
3. **Widen ~2× vs an FP16 design** at the same param target (budget around N_eff).
4. **QAT from scratch, never PTQ.**
5. **Keep embeddings/head/activations at int8** (W1.58A8), ternarize only the block
   matmuls.
6. **Over-train (~500–2000 tok/param)** — safe and beneficial under QAT.
7. **MobileLLM deep-and-thin + GQA + tied embeddings.**
8. **Compact-but-Chinese-adequate vocab (~48K + byte fallback).** A Latin-centric
   tokenizer splits each Chinese char into 3–4 byte tokens; adding ~20K Chinese
   tokens (Chinese-LLaMA, →49,953 vocab) roughly *halves* Chinese sequence length.
9. **BitNet stability recipe + progressive anneal.**
10. **Ship the kernel** (bitnet.cpp I2_S) — efficiency is realized only in-kernel.

Encouraging existence proofs at small scale: **ParetoQ's 600M ternary beats the
prior SoTA ternary 3B** (5× fewer params) with a better QAT recipe; **TernaryBERT**
(~110M, ternary + distillation) loses only ~1 GLUE point at 14.9× compression; a
native **132M ternary decoder** reaches 58.42 val PPL on TinyStories. The recipe in
[DESIGN_SUB100M.md](DESIGN_SUB100M.md) composes all ten levers.

## 5. Corrections applied from adversarial verification

- The parametric Spectra ternary-vs-FP scaling law (A_TriLM=185 vs A_FloatLM=159,
  ε=1.76 vs 1.41) is from **Spectra 1.1** (arXiv:2506.23025, ACL 2025), **not** the
  original Spectra paper (2407.12327), which shows scaling only qualitatively.
- BitNet b1.58 "matches FP16" is a real existence proof but **scale-dependent** —
  at 700M it is slightly *worse* on perplexity; parity is strongest on downstream
  average accuracy and arrives ~1.3–3B.
- FBI-LLM distillation is "**competitive**," not full-parity — do not claim
  distillation "recovers essentially all" of the gap.
- PrismML Q1_0 retains FP16 scales + RMSNorm (not literally 100% 1-bit); text family
  is 1.7B/4B/8B (4B is also a text model, plus a separate image line).
- The 1bitLLM reproduction numbers (700M 12.78 PPL/44.5 avg; 3B 9.88/49.6) match the
  model cards verbatim (checked via the HF filesystem; WebFetch was 403-blocked).

## References

**Scaling / precision laws**
- Kumar, Ankner, Muennighoff et al. *Scaling Laws for Precision.* arXiv:2411.04330 (ICLR 2025).
- Ouyang et al. *Low-Bit Quantization Favors Undertrained LLMs (QiD).* arXiv:2411.17691.
- Chen et al. *Scaling Law for Quantization-Aware Training.* arXiv:2505.14302.
- *ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization.* arXiv:2502.02631 (+ PyTorch blog).

**BitNet lineage & inference**
- Wang et al. *BitNet: Scaling 1-bit Transformers.* arXiv:2310.11453.
- Ma et al. *The Era of 1-bit LLMs: BitNet b1.58.* arXiv:2402.17764.
- *BitNet b1.58 2B4T Technical Report.* arXiv:2504.12285.
- *1-bit AI Infra: BitNet b1.58 Inference on CPUs (bitnet.cpp).* arXiv:2410.16144.
- Nielsen & Schneider-Kamp. *BitNet b1.58 Reloaded* (tiny 100K–48M study, ~2×-hidden rule). arXiv:2407.09527.
- 1bitLLM. *BitNet b1.58 reproduction* (700M/1.3B/3B model cards, HF).

**Ternary at small scale / techniques**
- Kaushal et al. *Spectra: ternary LLMs (TriLM).* arXiv:2407.12327; *Spectra 1.1.* arXiv:2506.23025.
- *OneBit: 1-bit weight LLMs (SVID + distillation).* arXiv:2402.11295.
- Ma, Sun, Shen. *FBI-LLM: fully binary LLMs via autoregressive distillation.* arXiv:2407.07093.
- Zhang et al. *TernaryBERT.* EMNLP 2020 (aclanthology 2020.emnlp-main.37).
- *Token-Scaled Logit Distillation for ternary generative LMs.* arXiv:2308.06744.
- *BitNet Distillation* (attention/feature distillation). arXiv:2510.13998.

**Small-model design & bilingual tokenization**
- Liu et al. *MobileLLM.* arXiv:2402.14905.
- Cui et al. *Efficient and Effective Text Encoding for Chinese LLaMA and Alpaca.* arXiv:2304.08177.

**PrismML**
- PrismML *Bonsai-8B-gguf / Bonsai-1.7B* model cards, HF `prism-ml`.
- The Register, *PrismML emerges from stealth with 1-bit LLMs* (2026-04-04).

*(Some 2025–2026 sources post-date the author's training cutoff and were retrieved
live; the lowest-confidence ones — e.g. very recent native-ternary preprints — are
flagged inline and used only as corroboration, not as load-bearing evidence.)*
