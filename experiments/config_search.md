# Sub-100M-total bilingual config search (from scripts/plan_budget.py)

Target: **total** params < 100M, zh/en, ternary transformer + fp embed/head.

| vocab | d_model | L | heads/kv | tie | fp_bd | TOTAL | emb frac | packed (ternary) | eff-cap (fp-equiv) |
|------:|--------:|--:|:--------:|:---:|:-----:|------:|:--------:|:----------------:|:------------------:|
| 32000 | 512 | 24 | 8/2 | ✅ | 1 | **82.5M** | **19.9%** | 55.8 MB (3.0×) | 27.9M |
| 32000 | 576 | 26 | 9/3 | ✅ | 1 | 110.5M | 16.7% | 67.8 MB (3.3×) | 33.9M |
| 48000 | 512 | 22 | 8/2 | ✅ | 1 | 85.2M | 28.9% | 71.1 MB (2.4×) | 35.5M |
| 32000 | 448 | 28 | 7/1 | ❌ | 1 | 86.5M | 33.1% | 76.2 MB (2.3×) | 38.1M |
| 40000 | 512 | 24 | 8/2 | ❌ | 1 | 107.1M | 38.3% | 104.9 MB (2.0×) | 52.5M |
| 32000 | 512 | 24 | 8/2 | ❌ | 0 | 98.9M | 33.1% | 78.6 MB (2.5×) | 39.3M |

## Takeaways
- **Weight tying is the single biggest lever** for sub-100M bilingual: it halves the
  embedding tax (emb fraction 33% → 20%) and pushes compression 2.5× → 3.0×, freeing
  budget for transformer depth. Confirms MobileLLM's small-scale weight-sharing finding.
- **The embedding tax is real and large**: even tied, a 32k-vocab model spends ~20% of
  params on the fp embedding; at 48k it's ~29%. Keep the vocab lean.
- **Compression is only ~2.4–3.0×**, NOT the ~8–10× seen at >1B — because the fp embedding
  is a fixed floor that dominates at small scale. This is a core reason 1-bit "works"
  only at large scale.
- **Effective capacity is brutal**: an 82.5M nominal ternary model is ~28M fp-equivalent
  (33.8%). Over-training + distillation must buy this gap back.

## Recommended anchor
`vocab=32000, d_model=512, n_layers=24, n_heads=8, n_kv_heads=2, ffn=1365, tie=True,
 fp_boundary_blocks=1` → 82.5M total, 3.0× packed, ~56 MB on disk.
Deep-and-thin (24 layers @ 512) follows MobileLLM. GQA (kv=2) trims KV params.
