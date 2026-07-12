# Results — ternary SlothLM-E, measured (2026-07-12)

Empirical follow-up to [`GUIDE_TERNARY_SLOTHLM_E.md`](GUIDE_TERNARY_SLOTHLM_E.md).
The guide was written *before* the runs and its targets (免選字 "84%", "32M→86%")
came from a **leaked benchmark** — corrected below. Bottom line: on honest data
a **25M ternary encoder beats the shipped 12M int8 model on quality *and* latency
*and* size**, and it does so at ~50× smaller scale than the ~1.3B threshold the
1-bit literature reports for decoder LLMs.

---

## 0. The load-bearing correction: the 免選字 benchmark was leaked

The 230-case `reference_mspy.tsv` used everywhere (incl. this guide's targets)
was built by **sampling sentences from the training corpus** and g2pW-labeling
them. Verified: **174/174** sampled reference lines are verbatim in training.
So the metric rewarded memorization and inflated whole-sentence scores by
~12–18 points. All prior 免選字 numbers (the shipped model's "84%", the guide's
"32M→86%", the capacity law "3.9M→76 … 32M→86") are memorization-inflated.

Rebuilt honest eval: **500 fresh c4 sentences (offset 600k), excluded against
the training corpus**, g2pW-labeled identically. Leaked vs held-out:

| model | 免選字 (leaked) | 免選字 (held-out, honest) |
|---|---|---|
| 4M int8 | 74% | **70%** |
| 12M int8 (shipped) | 84% | **72%** |
| 25M ternary, distilled | 78% | **76%** |
| 25M ternary, pure-CE | 96% | **71%** |

The homophone-hard (159) + toneless (159) sets were already clean (6/160 overlap)
and are the trustworthy metrics all along. The eval builder now takes
`--exclude <train_corpus>` to prevent recurrence.

---

## 1. Accuracy — honest held-out (the real result)

All same 25M shape (dim 352, depth 16, GQA 8/2, ffn 960), 16 epochs, DDP on 2×5090.

| model | 免選字 (held-out) | homophone-hard | toneless |
|---|---|---|---|
| **25M ternary, distilled (32M teacher)** | **76%** | **83%** | **81%** |
| 25M ternary, pure-CE (no teacher) | 71% | 84% | 75% |
| 12M int8 (shipped) | 72% | 82% | 79% |
| 4M int8 | 70% | 83% | 74% |

Findings:
- **The distilled ternary beats the shipped 12M int8 on all three honest metrics**
  (76/83/81 vs 72/82/79), at ~half the size. The leaked benchmark had this
  *inverted* (12M 84 > ternary 78) because the larger int8 model memorized more
  leaked common sentences (its 免選字 fell 84→72 on held-out; the ternary fell
  only 78→76).
- **Distillation is essential and it generalizes.** Pure CE collapses 96→71
  held-out (aggressive memorization); distillation → 76. This reverses a
  mid-experiment worry that distillation was hurting — the leak was hiding it.
  A strong teacher only helps a student with capacity to absorb it; here it did.
- **The honest capacity curve is nearly FLAT:** 4M 70 → 12M 72 → 25M 76. The
  steep leaked curve (74→84→86) was mostly memorization scaling with capacity.
  ⇒ *the "免選字 is capacity-bound, you need a bigger model" premise was largely
  a leakage artifact.* Diminishing honest returns beyond ~25M.

---

## 2. Speed — measured at our shapes on the real device

Faithful-shapes benchmark: a dummy at our **exact** dims (dim 352, ffn 960, 16L,
GQA 8/2 hd 44, vocab 8342; head = Q8_0 in both files so only the 16 block
matmuls differ), validated tensor-for-tensor against the real checkpoint. Run on
the **real ONYX BOOX — Snapdragon 662, baseline ARMv8.0, NO dotprod** (the stock
build hits "Illegal instruction"; you must force `DOTPROD/ARMV82_DOTPROD/
MATMUL_INT8/FP16_VECTOR` OFF + `GGML_NATIVE=OFF`). bitnet.cpp `llama-bench`,
4 threads (= the 4× A73 big cores; 6/8 regress on the little cores). Prompt
processing at batch=T = exactly our non-autoregressive forward.

| T (syllables) | I2_S ternary | Q8_0 int8 | ratio |
|---|---|---|---|
| 6 | **7.9 ms** | 14.1 ms | 1.79× |
| 12 | **12.4 ms** | 24.6 ms | 1.99× |
| 32 | 29.1 ms | 60.4 ms | 2.07× |

Linear fit (tight, <2% std): I2_S 0.85 ms/tok + 1.9 ms; Q8_0 1.80 ms/tok + 2.9 ms
→ **per-token block-matmul ternary speedup = 2.11×** (measured).

- **On this no-dotprod SD662, I2_S IS the fast kernel — TL1 is ~1.5× *slower*
  (measured, corrected premise).** We diagnosed + fixed TL1's M>1 segfault
  (`ggml_bitnet_can_mul_mat` gated to single-token gemv → M>1 fell through to a
  generic F32 matmul over 2-bit-packed data → overrun) and benchmarked it: on
  the d125m dummy TL1 pp6/pp12 = 143/170 t/s vs I2_S 211/261 vs Q8_0 117/128
  → **I2_S beats TL1 by ~1.5× at every size.** With dotprod OFF, this fork's
  I2_S (integer-widening batched gemm) outruns TL1's table-lookup gemv. TL1
  also *cannot tile our exact shapes* (codegen asserts bm∈{32,64}; our attn k/v
  weight is [88,352] and 88 divides neither; the packer hardcodes BM=256, our
  M∈{352,960,88} all fail). And I2_S has no per-shape tiling knob (generic ggml
  gemm; only thread count, and t=4 is already optimal). **⇒ I2_S is already the
  optimal ternary kernel on this device; 2.1× is the achievable kernel speedup,
  not a floor to beat with TL1.** The remaining speed lever is MODEL-SIDE (the
  legality-masked head, §3), not the kernel. (The "TL1 is faster" framing holds
  on dotprod-capable CPUs — a real, non-obvious device dependency.)
- **End-to-end is diluted by the head (labeled PROJECTED):** the real model is
  14 ternary + 2 fp blocks + a **per-position** int8 8342-way head that ternary
  can't accelerate (int8 in both). Corrected: ternary forward ≈ 1.22·T + 1.7 ms
  → ~9 ms @ T=6, ~16 ms @ T=12; **end-to-end speedup ≈ 1.5×**. The full
  end-to-end port (converter + standalone ggml graph w/ RoPE-NEOX, QK-norm,
  bidirectional attn, I2_S packing, logit-validation) is ~1 day and not yet done.

Deployed sizes (I2_S, our shapes): ternary **8.38 MB** vs int8-equivalent 24.75 MB
(3× smaller) and *smaller than the shipped 12M int8's 12.38 MB* despite 2× the
params. Effective **2.7 bits/param**.

**Cross-platform (x86, Intel Core Ultra 7 155H, AVX_VNNI, same 24M dims, I2_S vs
Q8_0):** pp12 3795 vs 2122 t/s (1.79×), pp32 1.53×, pp64 1.72× → **~1.7× ternary
win on x86 too** (noisy ±20-30% on this tiny model + turbo; TL2, x86's tuned LUT
kernel, untested). Absolute throughput ~15-20× the BOOX. **Key insight: the
ternary advantage is partly MEMORY-BANDWIDTH, not just compute** — 2-bit weights
move 4× fewer bytes than int8, and VNNI/dotprod accelerate int8 *compute* but not
its *memory* traffic. So ternary wins on BOTH a no-dotprod ARM (~2.0×) and a
VNNI x86 (~1.7×) — it is NOT a weak-CPU-only advantage. (This corrects an earlier
assumption that int8 accelerators would erase the ternary lead.)

---

## 3. The Pareto picture: 25M ternary dominates both int8 models

At the **4M's latency budget** (~9 ms measured), the 25M ternary projects to
~9 ms — same speed as the 4M but +6 免選字 (76 vs 70), and *faster than the 12M*
(13.3 ms) with +4 免選字. So it's Pareto-optimal on quality-vs-latency; the
shipped 12M int8 is dominated. Score-per-bit is a useful corrective, though:
because the honest curve is flat, the **4M is the most bit-efficient** (70%/4.64 MB
= 15.1 %/MB) and the 25M ternary (9.1) beats the 12M int8 (5.8) — i.e. at a given
*accuracy tier* ternary is the efficient choice, and the 12M is the worst of the
three on every axis.

**Biggest remaining lever = a legality-masked head.** Our IME's phonetic legality
means each position has only ~1–50 legal characters of 8342 — llama.cpp computes
the full vocab because a general LM must, but a custom forward can gather only the
legal rows. That collapses the head cost (~50–100×), which both **speeds up the
end-to-end forward toward the 2.1× block figure** and **shrinks the deployed size**
(the head dominates the 8.38 MB). Same optimization, both axes.

---

## 4. ⚑ The scale threshold is a *decoder* result — encoders break it

The 1-bit literature (BitNet b1.58; "When are 1.58 bits enough") reports ternary
matching fp only at **~1–3B params** — but those are all **causal decoder LMs
doing open-ended generation**. Our result reaches ternary competitiveness at
**~25M** (~50× smaller). Mechanistic reasons this should be architecture-specific,
not universal:

1. **No autoregressive error accumulation.** A decoder compounds per-step ternary
   noise across a generation; an encoder does **one bidirectional forward**, so
   weight noise stays local. This alone should slash the parity threshold.
2. **Constrained vs open-ended output.** Our task is aligned sequence labeling
   with a **hard legality mask** (~1–50 legal chars/position) + full bidirectional
   context — far more forgiving of noisy ternary weights than a 100k-way
   unconstrained softmax.
3. **~2× width** (dim 352 vs the 4M's 160) — matches the guide/BitNet-Reloaded
   width rule for small encoders.
4. **Task saturation** at low capacity (flat honest curve) — leaves little for the
   ternary penalty to bite on.

**Honesty caveat — suggestive, not proven.** We compared ternary-25M vs *int8*-12M
(not vs fp16, the literature's baseline); the task is saturated (which *masks*
quantization penalties); speed is partly projected; one task/dataset. The clean
test is an **fp control of the same 25M shape** (`--quant fp`, no teacher): if
ternary-25M ≈ fp-25M on honest held-out, the ternary penalty is genuinely gone at
25M for this architecture → the encoder-breaks-the-threshold claim is confirmed.
If fp-25M ≫ ternary-25M, the penalty is merely hidden by saturation. Pairing it
with a *less-saturated* yardstick (the harder homophone set) would make the claim
rigorous. **This fp control is the single most important next experiment.**

---

## 5. Reproduction & artifacts

- Trainer (now with hint/context/typo feature-parity + `--quant fp` control + DDP
  + `--resume`): `reference/slothe_ternary/train_slothe_ternary.py`. Gate:
  `gate_slothe_ternary.py` (pass `--mspy <held-out set>`).
- Honest eval builder: pass `--exclude <train_corpus>`; build from held-out data.
- Speed: dummy at our dims → I2_S/Q8_0 GGUF → `llama-bench -p 6,12 -n 0 -t 4 -r 5`
  on the device, baseline-ARMv8.0 build (dotprod/i8mm/fp16 OFF).
- Full end-to-end port (measured e2e latency) + legality-masked head: scoped, ~1 day.

## Open items
- [ ] **fp control @ 25M** — confirm/refute the encoder-breaks-the-threshold claim.
- [ ] Full ggml port with legality-masked head — measured e2e latency, targets ~6–7 ms.
- [x] TL1 vs I2_S settled: I2_S is faster on no-dotprod ARM; TL1 fixed (M>1
      segfault) but ~1.5× slower + untileable at our shapes. I2_S is optimal.
- [ ] Feature-parity ternary (hints/context/typo) held-out gate — drop-in confirmation.
