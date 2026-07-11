#!/usr/bin/env python3
"""
Param-budget planner for sub-100M 1-bit SLMs — NO torch required (pure arithmetic).

Shows how the bilingual embedding table competes with the transformer stack for a
fixed parameter budget, and how the "effective information budget" changes once
the transformer weights are ternary (~1.58 bits) while embeddings stay fp16.

Usage:
    python scripts/plan_budget.py
    python scripts/plan_budget.py --vocab 64000 --d_model 512 --n_layers 24
"""

import argparse


def transformer_params(d_model, n_layers, n_heads, n_kv_heads, ffn_hidden):
    head_dim = d_model // n_heads
    # attention: q(d*d) + k(d*kv) + v(d*kv) + o(d*d)
    q = d_model * (n_heads * head_dim)
    k = d_model * (n_kv_heads * head_dim)
    v = d_model * (n_kv_heads * head_dim)
    o = (n_heads * head_dim) * d_model
    attn = q + k + v + o
    # SwiGLU: gate + up + down
    ffn = 2 * d_model * ffn_hidden + ffn_hidden * d_model
    # norms (RMSNorm weights) are tiny; include for completeness
    norms = 2 * d_model + 2 * d_model  # attn_norm, ffn_norm + subln approx
    return (attn + ffn + norms) * n_layers


def report(vocab, d_model, n_layers, n_heads, n_kv_heads, ffn_hidden,
           tie_embeddings, fp_boundary_blocks, emb_bits=8):
    emb = vocab * d_model
    head = 0 if tie_embeddings else vocab * d_model
    tf_total = transformer_params(d_model, n_layers, n_heads, n_kv_heads, ffn_hidden)
    per_block = tf_total / n_layers
    fp_blocks = min(2 * fp_boundary_blocks, n_layers)
    fp_tf = per_block * fp_blocks
    ternary_tf = tf_total - fp_tf
    final_norm = d_model
    total = emb + head + tf_total + final_norm

    # Deployed bits: ternary weights ~1.58 bits; embedding+head at emb_bits (int8/int4
    # PTQ of the lookup table is ~lossless); remaining fp (boundary blocks, norms) 16 bits.
    emb_params = emb + head
    other_fp = total - ternary_tf - emb_params
    packed_bits = ternary_tf * 1.58 + emb_params * emb_bits + other_fp * 16
    packed_mb = packed_bits / 8 / 1e6
    fp16_mb = total * 16 / 8 / 1e6

    print(f"\n=== config: vocab={vocab} d_model={d_model} n_layers={n_layers} "
          f"heads={n_heads}/kv{n_kv_heads} ffn={ffn_hidden} "
          f"tie={tie_embeddings} fp_boundary={fp_boundary_blocks} emb_bits={emb_bits} ===")
    print(f"  embedding params      : {emb/1e6:7.2f}M  ({emb/total*100:5.1f}%)")
    print(f"  lm_head params        : {head/1e6:7.2f}M  ({head/total*100:5.1f}%)")
    print(f"  transformer (total)   : {tf_total/1e6:7.2f}M  ({tf_total/total*100:5.1f}%)")
    print(f"    - ternary blocks    : {ternary_tf/1e6:7.2f}M")
    print(f"    - fp boundary blocks: {fp_tf/1e6:7.2f}M  ({fp_blocks} blocks)")
    print(f"  ---------------------------------------------")
    print(f"  TOTAL params          : {total/1e6:7.2f}M")
    print(f"  embedding fraction    : {(emb+head)/total*100:5.1f}%")
    print(f"  deployed size         : {packed_mb:7.2f} MB   vs fp16 {fp16_mb:6.2f} MB "
          f"({fp16_mb/packed_mb:.1f}x smaller)  [emb@{emb_bits}b]")
    # QUALITY capacity (not memory!): the empirical rule is ternary ~= 0.5x fp
    # params in quality (BitNet needs ~2x hidden to match fp16, 2407.09527), NOT
    # the 1.58/16 memory ratio. fp/int8 params count ~fully.
    quality_eff = (total - ternary_tf) + ternary_tf * 0.5
    print(f"  ~quality-capacity     : {quality_eff/1e6:7.2f}M fp-equiv params "
          f"(ternary@0.5x; {quality_eff/total*100:4.1f}% of nominal)")
    return total


PRESETS = {
    # name: (vocab, d_model, n_layers, n_heads, n_kv, ffn, tie, fp_boundary)
    # >>> RECOMMENDED (docs/DESIGN_SUB100M.md): ~92M total, tied int8 embeddings <<<
    "bonsai-nano-90m": (48000, 512, 24, 8, 2, 1408, True, 1),
    # Smaller/leaner points
    "tiny-38m":  (32000, 384, 16, 6, 2, 1024, True, 1),
    "lean-82m":  (32000, 512, 24, 8, 2, 1365, True, 1),
    # Untied + big-vocab variants to SHOW the embedding tax (not recommended)
    "untied-99m":  (32000, 512, 24, 8, 2, 1365, False, 0),
    "zh-heavy-vocab": (100000, 512, 24, 8, 2, 1365, False, 1),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int)
    ap.add_argument("--d_model", type=int)
    ap.add_argument("--n_layers", type=int)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_kv_heads", type=int, default=2)
    ap.add_argument("--ffn_hidden", type=int)
    ap.add_argument("--tie", action="store_true")
    ap.add_argument("--fp_boundary", type=int, default=1)
    ap.add_argument("--emb_bits", type=int, default=8,
                    help="deployed embedding/head bit-width (8=int8, 4=int4, 16=fp16)")
    args = ap.parse_args()

    if args.vocab:
        ffn = args.ffn_hidden or int(args.d_model * 8 / 3)
        report(args.vocab, args.d_model, args.n_layers, args.n_heads,
               args.n_kv_heads, ffn, args.tie, args.fp_boundary, args.emb_bits)
    else:
        print("No config given — showing presets:")
        for name, p in PRESETS.items():
            print(f"\n### preset: {name}")
            report(*p)


if __name__ == "__main__":
    main()
