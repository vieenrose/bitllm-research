# Guide: train a 1.58-bit (ternary) SlothLM-E that beats the 12M int8 Zhuyin model

> **▶ RESULTS ARE IN — read [`RESULTS_SLOTHLM_E_TERNARY.md`](RESULTS_SLOTHLM_E_TERNARY.md) first.**
> Two corrections to this guide: (1) the 免選字 targets below ("84%", "32M→86%")
> came from a **LEAKED benchmark** (its sentences were sampled from the training
> corpus — 174/174 verbatim); honest held-out numbers are ~12-18 pts lower.
> (2) On honest data a **25M ternary already BEATS the 12M int8** (76 vs 72 免選字,
> at ~half the size and ~1.5-2× faster) — the recipe here works; the goalposts
> were just mismeasured. The results doc also reports the measured on-device
> speed and argues the ~1.3B ternary-parity threshold is a *decoder* result that
> encoders break (pending an fp control).

**Audience:** an engineer/agent with this repo cloned and an **RTX 5090** in a
separate project, who will train the model in the `slothing` / `slothlm-e`
pipeline. This guide is the master handoff: it distills the research in this repo
into one concrete, buildable recipe and a drop-in trainer.

**Read alongside:**
- [`docs/WHY_1BIT_NEEDS_SCALE.md`](WHY_1BIT_NEEDS_SCALE.md) — why 1-bit is a
  capacity tax worst at small scale (the reasoning behind every choice here).
- [`docs/DESIGN_SUB100M.md`](DESIGN_SUB100M.md) — the general sub-100M recipe.
- Drop-in code: [`reference/slothe_ternary/`](../reference/slothe_ternary/)
  (`train_slothe_ternary.py`, `gate_slothe_ternary.py`).

---

## 0. Goal & success criteria

Build a **ternary {−1,0,+1} bidirectional encoder** for Zhuyin→Traditional-Chinese
that **beats the deployed 12M int8 model** and is **deployable smaller/faster on ARM**.

Baseline to beat (`Luigi/slothlm-e-12m-zhuyin`, deployed int8):

| metric | 12M int8 (target) | goal |
|---|---|---|
| 免選字 (230, sentence-exact) | **84%** | **≥ 85%** |
| homophone-hard (159) | ~84% | **≥ 84%** (already saturated) |
| toneless (159) | 70% | **≥ 72%** |
| deployed size | ~13 MB int8 / 4.9 MB dl | **≤ ~7 MB** |
| ARM latency | int8 NEON baseline | **≤ baseline** (needs T-MAC — see §7) |

> **Honest framing (read this first).** Your own ledger says *"int8 勝 int4/QAT/
> ternary."* That is correct for a *naïve* ternary attempt run through
> onnxruntime. To flip it you must change **three** things together: **(1) go ~2×
> wider**, **(2) distill from the 32M ref**, **(3) ship a T-MAC/bitnet.cpp ARM
> kernel**. Miss any one and int8 wins again. This guide does all three. If the
> §7 T-MAC microbench doesn't beat int8 on your device, **stop and keep int8** —
> the accuracy win alone isn't worth a ternary model that isn't faster.

---

## 1. Why these choices (one paragraph each, cited)

- **Ternary, not binary.** Ternary sits just above ParetoQ's 2→3-bit
  "reconstruction" cliff; binary sits below and never closes the gap.
  ([ParetoQ 2502.02631](https://arxiv.org/abs/2502.02631),
  [BiBERT 2203.06390](https://arxiv.org/abs/2203.06390))
- **~2× width.** Ternary carries an effective-parameter penalty; small **encoders**
  specifically need ~double the hidden size to reach fp parity.
  ([BitNet b1.58 Reloaded 2407.09527](https://arxiv.org/abs/2407.09527),
  [When are 1.58 bits enough? 2411.05882](https://arxiv.org/abs/2411.05882))
- **absMEDIAN weight scale** (not absmean) — more robust for *small* models.
  ([2407.09527](https://arxiv.org/abs/2407.09527))
- **Distillation from the 32M ref** — the only lever that lets a small ternary
  student *beat* (not just match) the int8 12M; use logit KD (feature KD optional).
  ([TernaryBERT 2009.12812](https://arxiv.org/abs/2009.12812),
  [TernaryLLM 2406.07177](https://arxiv.org/abs/2406.07177))
- **Extra RMSNorm before every ternary linear (SubLN) + progressive fp→ternary
  anneal** — stabilizes the notoriously unstable ternary transition; matches
  distillation-only pipelines. Your model already has post-embed RMSNorm.
  ([An Extra RMSNorm is All You Need 2505.08823](https://arxiv.org/abs/2505.08823))
- **Keep embedding + char head + first/last block in higher precision (int8 at
  deploy).** They're the sensitive, un-redundant ~20% (the "embedding tax"); middle
  layers tolerate ternary best. ([TernaryLM 2602.07374](https://arxiv.org/abs/2602.07374))
- **W1.58A8** (int8 activations) — pushing activations to 4-bit hits the FC2
  outlier failure mode. ([QAT scaling law 2505.14302](https://arxiv.org/abs/2505.14302))
- **Retune LR/weight-decay** — small ternary models tolerate *higher* LR and behave
  differently from 7B BitNet. ([2407.09527](https://arxiv.org/abs/2407.09527))

---

## 2. The model — `SlothE_T` (≈20M ternary)

Same NAS-winning `SlothE` recipe (RoPE · GQA · QK-norm · RMSNorm · SwiGLU ·
post-embed RMSNorm · char-hint channel · phonetic-legality decode), **scaled ~2×
wide** and with block `Linear`s replaced by `BitLinear`:

| field | value | note |
|---|---|---|
| `dim` | **320** | ~2× the 4M's 160 (the width rule) |
| `depth` | **16** | deep-and-thin |
| `heads` / `kv` | 8 / 2 | GQA |
| `ffn` | 880 | ~2.75·dim SwiGLU |
| `n_syl` / `n_char` | 1539 / 8342 | from your vocab/tokenizer |
| `embed_norm` | true | ModernBERT post-embed norm |
| `weight_bits` | ternary | body only |
| `weight_quant` | median | absmedian scale |
| `fp_boundary` | 1 | first & last block stay fp |
| `pre_norm` | true | extra RMSNorm before each ternary linear |
| `act_bits` | 8 | W1.58A8 |

**Param / footprint** (from `scripts/plan_budget.py`-style math): ~**20.8M** total =
17.6M ternary body + 3.2M int8 head/embed → **~6.6 MB deployed** (ternary body @1.58b
+ int8 head/embed), vs the 12M int8's ~13 MB. Confirm with:
```bash
python3 - <<'EOF'
N_SYL,N_CHAR=1539,8342
def calc(dim,depth,heads,kv,ffn):
    hd=dim//heads; body=((2*dim*dim+2*dim*kv*hd)+3*dim*ffn)*depth
    head=N_CHAR*dim+N_SYL*dim; tot=body+head
    print(f"total {tot/1e6:.1f}M | ternary body {body/1e6:.1f}M + int8 head/emb {head/1e6:.1f}M "
          f"| deployed ~{(body*1.58+head*8)/8/1e6:.1f} MB")
calc(320,16,8,2,880)
EOF
```

If ~20M trains but still trails, step up to `dim352·depth16·ffn960` (~25M, ~7.7 MB).
If it beats the target comfortably, try `dim288·depth14·ffn768` (~15M, ~5.3 MB) — the
"smallest that still wins" — which also helps the speed goal.

---

## 3. Prerequisites

```bash
# in the training project (5090 box)
pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12.x
pip install transformers datasets tokenizers numpy g2pw onnx onnxruntime-gpu

# get the pipeline scripts + data builders
git clone https://github.com/vieenrose/sloth-zhuyin-linux   # 'model/' has the E pipeline
# copy the two drop-in scripts from THIS repo next to model/train_slothlm_e.py:
cp <this-repo>/reference/slothe_ternary/train_slothe_ternary.py  model/
cp <this-repo>/reference/slothe_ternary/gate_slothe_ternary.py   model/
```

You need these artifacts (all produced by the existing pipeline, see the 4M repo's
`REPRODUCE.md`): `train_e_g2pw.bin` (g2pW-labeled aligned data), `syl_vocab.json`,
`tokenizer/`, `phonetic_table.tsv`, and `eval/testset.tsv` + `eval/reference_mspy.tsv`.

---

## 4. Build the teacher (for distillation) — do this once

Distillation is the lever that lets a small ternary student *beat* the int8 12M.
The teacher must share the **same tokenizer + syllable vocab** (it does — same
pipeline) and output the same `n_char` head, so **logit KD** works directly.

Use your existing **32M** reference as the teacher (it hits 86% 免選字). If you
don't have its checkpoint, train one in fp with the original script:
```bash
python3 model/train_slothlm_e.py --data train_e_g2pw.bin \
    --vocab syl_vocab.json --tokenizer tokenizer --out slothe_32m \
    --dim 512 --depth 12 --heads 8 --kv-heads 2 --ffn 1408 --embed-norm \
    --batch 256 --epochs 8 --lr 1.5e-3       # ~1–2 h on a 5090
```
A stronger teacher → a stronger student; a 32–64M fp teacher is ideal. The teacher
stays fp and frozen; it's only used at train time.

---

## 5. Train the ternary student

```bash
python3 model/train_slothe_ternary.py \
    --data train_e_g2pw.bin --vocab syl_vocab.json --tokenizer tokenizer \
    --out slothe_t_20m \
    --dim 320 --depth 16 --heads 8 --kv-heads 2 --ffn 880 --embed-norm \
    --weight-quant median --fp-boundary 1 --pre-norm --act-bits 8 \
    --anneal-frac 0.15 \
    --teacher slothe_32m --distill-alpha 0.7 --distill-temp 2.0 \
    --batch 384 --epochs 8 --lr 2.5e-3 --weight-decay 0.1
```

What each knob does (and how to tune):

| flag | default | tuning |
|---|---|---|
| `--anneal-frac` | 0.15 | fp warmup then ramp to full ternary by 15% of steps. If loss spikes when ternary kicks in, raise to 0.25. |
| `--distill-alpha` | 0.7 | weight on KD vs CE. Higher = lean more on the teacher (good early). Try 0.5–0.8. |
| `--distill-temp` | 2.0 | KD softmax temperature. 1.5–3.0. |
| `--lr` | 2.5e-3 | ternary tolerates a higher peak than fp (fp used 1.5e-3). If unstable, drop to 2e-3. |
| `--weight-quant` | median | `median` for small models; `mean` = vanilla b1.58 (ablate). |
| `--fp-boundary` | 1 | keep first/last block fp. 0 = fully ternary body (smaller, riskier). |
| `--no-pre-norm` | — | disable the extra SubLN (ablation; expect worse stability). |
| `--epochs` | 8 | overtraining helps under QAT — try 12–16 if the curve is still improving. |

Training cost: the fp 4M took ~35 min; this ~20M ternary + KD run is ~**2–4 h** on a
5090. Watch that `kl` (distillation) and `ce` both fall and that loss doesn't jump at
`alpha→1`.

---

## 6. Evaluate — gate against the 12M

```bash
python3 model/gate_slothe_ternary.py --model slothe_t_20m \
    --tokenizer tokenizer --table phonetic_table.tsv \
    --testset eval/testset.tsv --mspy eval/reference_mspy.tsv
```
Prints homophone-hard / 免選字 / toneless at **full ternary** (`quant_alpha=1`),
next to the 12M int8 numbers. **Decision:**
- Beats 免選字 84% and holds homophone ≥84% → success; proceed to export + the speed gate.
- Trails → first raise `--epochs`/teacher strength and `--distill-alpha`; then step
  up to the 25M config (`dim352·ffn960`). If it still can't beat the int8 12M *and*
  the speed gate (§7) fails, your ledger stands — ship int8.

**Always also train an fp control of the SAME shape** (`--quant fp`) on the same data
so you can separate the ternary penalty from the size increase:
```bash
python3 model/train_slothe_ternary.py ... --out slothe_fp_20m --quant fp --teacher ""
```

---

## 7. The speed gate (do NOT skip) — settle "faster on ARM" *before* shipping

Ternary is only faster than int8 on ARM with a **lookup-table kernel**; through
onnxruntime it runs as unpacked int8 and is *slower*. So:

1. Export for functional/browser use (unchanged pipeline):
   ```bash
   python3 model/export_slothe_onnx.py --model slothe_t_20m --out slothe_t_20m_onnx
   ```
   (onnxruntime int8 here is for correctness/browser, not the speed claim.)
2. **Benchmark the ternary GEMM with a real LUT kernel** — Microsoft **T-MAC**
   (`github.com/microsoft/T-MAC`) or **bitnet.cpp** `TL1/TL2` — against your int8
   baseline at the `dim320·ffn880` shapes over your typical sentence length, on the
   actual ARM target (the phone/device you deploy to).
   - T-MAC ≥ ~2× over int8 → the 20M ternary lands at **parity-or-better latency**
     with higher accuracy: ship it.
   - T-MAC ~1.5× → you get accuracy at ~parity speed (still a win vs the *32M* int8,
     not the 12M): decide by product need.
   - No LUT kernel available → **keep int8.** The whole ternary case rests on this
     kernel; there is no ternary speedup without it.

See [`docs/WHY_1BIT_NEEDS_SCALE.md`](WHY_1BIT_NEEDS_SCALE.md) §"speed" and the
prior analysis: at batch-1/short-sentence NAR inference, the win comes from moving
fewer weight bytes + LUT reuse, and it must be measured, not assumed.

---

## 8. Ablation matrix (what to run if you have GPU budget)

| axis | settings | tests |
|---|---|---|
| size | 15M / 20M / 25M | the 2×-width rule; smallest that beats 84% |
| distill | on / off, α∈{0.5,0.7} | the #1 accuracy lever |
| teacher | 32M / 64M fp | stronger teacher → stronger student |
| weight-quant | median / mean | small-model median advantage |
| pre-norm | on / off | SubLN stability |
| fp-boundary | 0 / 1 / 2 | sensitivity of boundary blocks |
| anneal-frac | 0.0 / 0.15 / 0.25 | anneal vs hard-start |
| tokens | 8 / 12 / 16 epochs | overtraining-helps-under-QAT |

Report every run with the 3 gate numbers + deployed MB + the T-MAC latency.

---

## 9. What "done" looks like

A `slothe_t_20m/` checkpoint that: (a) gates **≥85% 免選字 / ≥84% homophone / ≥72%
toneless**, (b) exports to ~**6.6 MB**, and (c) **beats the 12M int8 latency under
T-MAC** on the target ARM device. Publish it as `slothlm-e-20m-ternary-zhuyin` with
the gate table + the T-MAC numbers, mirroring the 4M card.

---

## 10. Reading list

Scaling/precision: [2411.04330](https://arxiv.org/abs/2411.04330) ·
[2502.02631](https://arxiv.org/abs/2502.02631) ·
[2505.14302](https://arxiv.org/abs/2505.14302).
Small ternary encoders (core): [2407.09527](https://arxiv.org/abs/2407.09527) ·
[2411.05882](https://arxiv.org/abs/2411.05882) ·
[2505.08823](https://arxiv.org/abs/2505.08823).
Ternary + distillation: [2009.12812](https://arxiv.org/abs/2009.12812) ·
[2406.07177](https://arxiv.org/abs/2406.07177) ·
[2402.10631](https://arxiv.org/abs/2402.10631) ·
[2306.01841](https://arxiv.org/abs/2306.01841).
Binary BERT (why ternary): [2203.06390](https://arxiv.org/abs/2203.06390).
Gap recovery / packing / layers: [2602.05269](https://arxiv.org/abs/2602.05269) ·
[2601.07892](https://arxiv.org/abs/2601.07892) ·
[2602.07374](https://arxiv.org/abs/2602.07374).
BitNet lineage + CPU kernels: [2402.17764](https://arxiv.org/abs/2402.17764) ·
[2504.12285](https://arxiv.org/abs/2504.12285) ·
[2410.16144](https://arxiv.org/abs/2410.16144) (bitnet.cpp) · Microsoft T-MAC.
