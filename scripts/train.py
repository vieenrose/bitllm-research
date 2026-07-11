#!/usr/bin/env python3
"""
Pretraining entrypoint for a 1-bit / ternary zh/en SLM.

Runs on the GPU box:
    python scripts/train.py --config configs/small_60m.yaml

Features:
  * QAT with progressive quantization annealing (alpha 0 -> 1).
  * Optional knowledge distillation from an fp teacher.
  * bf16 autocast, gradient accumulation to hit a target tokens/step.
  * Cosine LR with warmup, grad clipping, AdamW.
  * Checkpointing + resumption.

This file intentionally has no hard dependency at import time on a GPU; a tiny
smoke path (--smoke) builds a micro model and runs a few steps on random data so
CI / a laptop can validate the graph without the full corpus.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bitllm.config import load_config, dump_config, ExpConfig  # noqa: E402
from bitllm.model import build_model  # noqa: E402
from bitllm.distill import distillation_loss  # noqa: E402


def quant_alpha_at(step, max_steps, start_frac, end_frac):
    """Linear anneal of the ternary blend coefficient."""
    if end_frac <= start_frac:
        return 1.0
    s0 = start_frac * max_steps
    s1 = end_frac * max_steps
    if step <= s0:
        return 0.0
    if step >= s1:
        return 1.0
    return (step - s0) / (s1 - s0)


def cosine_lr(step, warmup, max_steps, lr, min_lr):
    if step < warmup:
        return lr * step / max(1, warmup)
    if step > max_steps:
        return min_lr
    r = (step - warmup) / max(1, max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * r))


def build_optimizer(model, cfg):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or n.endswith("norm.weight") or "embed" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.train.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.train.lr,
                             betas=(cfg.train.beta1, cfg.train.beta2))


def get_loader(cfg, device):
    from bitllm.data import MixtureLoader
    weights = {}
    for src in cfg.data.sources:
        # src is a dict with keys name/lang/weight
        weights[src.get("lang", "unk")] = src.get("weight", 1.0)
    return MixtureLoader(
        cfg.data.packed_dir, cfg.model.max_seq_len, cfg.train.micro_batch_size,
        weights=weights, seed=cfg.data.seed, device=device,
    )


def smoke_loader(cfg, device):
    """Random-token batches for the --smoke path (no corpus needed)."""
    V = cfg.model.vocab_size
    T = cfg.model.max_seq_len
    B = cfg.train.micro_batch_size
    g = torch.Generator(device="cpu").manual_seed(0)

    class _L:
        def batch(self):
            ids = torch.randint(0, V, (B, T + 1), generator=g)
            return ids[:, :-1].to(device), ids[:, 1:].to(device)
    return _L()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default="")
    ap.add_argument("--smoke", action="store_true",
                    help="run a few steps on random data (no corpus/tokenizer needed)")
    ap.add_argument("--smoke_steps", type=int, default=5)
    args = ap.parse_args()

    cfg: ExpConfig = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        # shrink to something that runs on CPU in seconds
        cfg.model.vocab_size = min(cfg.model.vocab_size, 2000)
        cfg.model.d_model = 128
        cfg.model.n_layers = 4
        cfg.model.n_heads = 4
        cfg.model.n_kv_heads = 2
        cfg.model.ffn_hidden = 256
        cfg.model.max_seq_len = 64
        cfg.train.micro_batch_size = 4
        cfg.train.max_steps = args.smoke_steps
        cfg.train.warmup_steps = 1

    model = build_model(cfg.model).to(device)
    print("param report:", model.param_report())

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[cfg.train.dtype]
    use_amp = device == "cuda" and dtype != torch.float32

    opt = build_optimizer(model, cfg)

    teacher = None
    if cfg.train.distill and not args.smoke:
        from transformers import AutoModelForCausalLM
        teacher = AutoModelForCausalLM.from_pretrained(
            cfg.train.teacher_model, torch_dtype=dtype
        ).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    loader = smoke_loader(cfg, device) if args.smoke else get_loader(cfg, device)

    # grad accumulation to hit target tokens/step
    tokens_per_micro = cfg.train.micro_batch_size * cfg.model.max_seq_len
    accum = cfg.train.grad_accum or max(1, cfg.train.batch_tokens // tokens_per_micro)

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
        print(f"resumed from {args.resume} at step {start_step}")

    os.makedirs(cfg.train.out_dir, exist_ok=True)
    dump_config(cfg, os.path.join(cfg.train.out_dir, "config.resolved.yaml"))

    model.train()
    t0 = time.time()
    for step in range(start_step, cfg.train.max_steps):
        lr = cosine_lr(step, cfg.train.warmup_steps, cfg.train.max_steps,
                       cfg.train.lr, cfg.train.min_lr)
        for g in opt.param_groups:
            g["lr"] = lr
        alpha = quant_alpha_at(step, cfg.train.max_steps,
                               cfg.train.quant_anneal_start, cfg.train.quant_anneal_end)
        model.set_quant_alpha(alpha)

        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(accum):
            X, Y = loader.batch()
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_amp):
                logits, ce = model(X, Y)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(X).logits
                    loss, _ = distillation_loss(
                        logits, t_logits, Y,
                        alpha=cfg.train.distill_alpha, temperature=cfg.train.distill_temp)
                else:
                    loss = ce
            (loss / accum).backward()
            loss_accum += loss.item() / accum

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()

        if step % cfg.train.log_every == 0:
            dt = time.time() - t0
            tok = (step - start_step + 1) * accum * tokens_per_micro
            print(f"step {step:>7} | loss {loss_accum:.4f} | lr {lr:.2e} | "
                  f"alpha {alpha:.2f} | {tok/max(dt,1e-9)/1e3:.1f}k tok/s")

        if step > 0 and step % cfg.train.ckpt_every == 0 and not args.smoke:
            path = os.path.join(cfg.train.out_dir, f"ckpt_{step}.pt")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "cfg": args.config}, path)
            print("saved", path)

    print(f"done in {time.time()-t0:.1f}s")
    if args.smoke:
        print("SMOKE OK")


if __name__ == "__main__":
    main()
