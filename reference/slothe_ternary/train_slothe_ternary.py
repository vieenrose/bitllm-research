#!/usr/bin/env python3
"""
Ternary (1.58-bit) SlothLM-E trainer — a drop-in replacement for
`reproduce/train_slothlm_e.py` that adds:

  * BitLinear (ternary {-1,0,+1} weights, absMEDIAN scale per output channel,
    int8 activations, straight-through estimator) in place of nn.Linear inside
    every transformer block.
  * Mixed-precision islands: the syllable embedding, the char head, and the
    first/last `--fp-boundary` blocks stay full precision (int8 at deploy).
  * Progressive fp->ternary annealing (`--anneal-frac`).
  * "Extra RMSNorm before every linear" (SubLN) for ternary stability.
  * Knowledge distillation from a full-precision SlothLM-E teacher (`--teacher`).

Module names MIRROR reproduce/train_slothlm_e.py exactly, so:
  - a full-precision teacher checkpoint saved by the original script loads into
    an all-fp SlothE_T for distillation, and
  - the saved ternary checkpoint carries a `config` dict that the companion
    gate/export scripts use to rebuild the model.

Design rationale + citations: see docs/GUIDE_TERNARY_SLOTHLM_E.md in this repo.

Example (on the RTX 5090):
    python3 train_slothe_ternary.py \
        --data train_e_g2pw.bin --vocab syl_vocab.json --tokenizer tokenizer \
        --out slothe_t_20m \
        --dim 320 --depth 16 --heads 8 --kv-heads 2 --ffn 880 --embed-norm \
        --teacher slothe_32m --distill-alpha 0.7 --distill-temp 2.0 \
        --anneal-frac 0.15 --weight-quant median --pre-norm \
        --batch 384 --epochs 8 --lr 2.5e-3
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ======================================================================
# Quantization primitives (ternary weights + int8 activations, STE)
# ======================================================================

def activation_quant(x, eps=1e-5):
    """Per-token int8 (absmax) fake-quant; returns a dequantized tensor."""
    scale = 127.0 / x.abs().amax(dim=-1, keepdim=True).clamp_(min=eps)
    return (x * scale).round().clamp_(-128, 127) / scale


def weight_quant_ternary(w, mode="median", eps=1e-5):
    """Ternarize to {-1,0,+1} * scale, per OUTPUT channel.

    mode="median" uses the absmedian scale (BitNet b1.58 Reloaded, arXiv:2407.09527
    — more robust than absmean for SMALL models). mode="mean" is vanilla b1.58.
    """
    if mode == "median":
        scale = w.abs().median(dim=1, keepdim=True).values.clamp_(min=eps)
    else:
        scale = w.abs().mean(dim=1, keepdim=True).clamp_(min=eps)
    return (w / scale).round().clamp_(-1, 1) * scale


def _ste(real, quant, alpha):
    """Straight-through estimator with anneal blend (alpha in [0,1])."""
    blended = real + alpha * (quant - real)
    return real + (blended - real).detach()


class RMSNorm(nn.Module):
    # param name `.w` matches reproduce/train_slothlm_e.py so teacher weights load
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.w


class BitLinear(nn.Linear):
    """nn.Linear drop-in. weight_bits='ternary' -> W1.58A8 QAT; 'fp' -> plain fp.

    pre_norm=True inserts an RMSNorm before the projection (the "extra RMSNorm"
    stabilizer, arXiv:2505.08823). For 'fp' islands we keep it off so the state
    dict matches the original SlothE exactly (teacher loading).
    """

    def __init__(self, in_f, out_f, bias=False, weight_bits="ternary",
                 act_bits=8, weight_quant="median", pre_norm=True):
        super().__init__(in_f, out_f, bias=bias)
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.wq_mode = weight_quant
        self.pre = RMSNorm(in_f) if (pre_norm and weight_bits == "ternary") else None
        self.register_buffer("quant_alpha", torch.tensor(1.0), persistent=True)

    def set_quant_alpha(self, a):
        self.quant_alpha.fill_(float(a))

    def forward(self, x):
        if self.pre is not None:
            x = self.pre(x)
        if self.weight_bits == "fp":
            return F.linear(x, self.weight, self.bias)
        a = float(self.quant_alpha)
        if self.act_bits:
            x = _ste(x, activation_quant(x), a)
        w = _ste(self.weight, weight_quant_ternary(self.weight, self.wq_mode), a)
        return F.linear(x, w, self.bias)


# ======================================================================
# Model — mirrors reproduce/train_slothlm_e.py, Linear -> BitLinear
# ======================================================================

def rope(x, pos, dim):
    half = dim // 2
    freq = 1.0 / (10000 ** (torch.arange(0, half, device=x.device) / half))
    ang = pos[:, None].float() * freq[None, :]
    cos = torch.cat([ang.cos(), ang.cos()], -1)[None, None]
    sin = torch.cat([ang.sin(), ang.sin()], -1)[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], -1)
    return x * cos + rot * sin


def _lin(in_f, out_f, fp, wq, pre_norm):
    return BitLinear(in_f, out_f, bias=False,
                     weight_bits="fp" if fp else "ternary",
                     weight_quant=wq, pre_norm=pre_norm)


class Attn(nn.Module):
    def __init__(self, dim, heads, kv, fp, wq, pre_norm):
        super().__init__()
        self.h, self.kv, self.dh = heads, kv, dim // heads
        self.q = _lin(dim, heads * self.dh, fp, wq, pre_norm)
        self.k = _lin(dim, kv * self.dh, fp, wq, pre_norm)
        self.v = _lin(dim, kv * self.dh, fp, wq, pre_norm)
        self.o = _lin(heads * self.dh, dim, fp, wq, pre_norm)
        self.qn = RMSNorm(self.dh)
        self.kn = RMSNorm(self.dh)

    def forward(self, x, pos, amask):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.kv, self.dh).transpose(1, 2)
        q, k = self.qn(q), self.kn(k)
        q, k = rope(q, pos, self.dh), rope(k, pos, self.dh)
        rep = self.h // self.kv
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=amask[:, None, None, :])
        o = o.transpose(1, 2).reshape(B, T, -1)
        return self.o(o)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden, fp, wq, pre_norm):
        super().__init__()
        self.w1 = _lin(dim, hidden, fp, wq, pre_norm)
        self.w3 = _lin(dim, hidden, fp, wq, pre_norm)
        self.w2 = _lin(hidden, dim, fp, wq, pre_norm)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, dim, heads, kv, ffn, fp, wq, pre_norm):
        super().__init__()
        self.n1 = RMSNorm(dim)
        self.attn = Attn(dim, heads, kv, fp, wq, pre_norm)
        self.n2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn, fp, wq, pre_norm)

    def forward(self, x, pos, amask):
        x = x + self.attn(self.n1(x), pos, amask)
        x = x + self.ffn(self.n2(x))
        return x


class SlothE_T(nn.Module):
    """Ternary SlothLM-E. embed + head + first/last `fp_boundary` blocks stay fp."""

    def __init__(self, n_syl, n_char, dim=320, depth=16, heads=8, kv=2, ffn=880,
                 embed_norm=True, weight_bits="ternary", weight_quant="median",
                 fp_boundary=1, pre_norm=True, act_bits=8):
        super().__init__()
        self.embed = nn.Embedding(n_syl, dim)                 # fp island
        self.embed_norm = RMSNorm(dim) if embed_norm else None
        blocks = []
        for i in range(depth):
            is_fp = (weight_bits == "fp") or i < fp_boundary or i >= depth - fp_boundary
            blocks.append(Block(dim, heads, kv, ffn, is_fp, weight_quant, pre_norm))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(dim)
        self.head = nn.Linear(dim, n_char, bias=False)        # fp island (int8 at deploy)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def set_quant_alpha(self, a):
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.set_quant_alpha(a)

    def forward(self, syl, amask):
        pos = torch.arange(syl.shape[1], device=syl.device)
        x = self.embed(syl)
        if self.embed_norm is not None:
            x = self.embed_norm(x)
        for b in self.blocks:
            x = b(x, pos, amask)
        return self.head(self.norm(x))


# ======================================================================
# Data (reuses the AlignedBin format from the original pipeline)
# ======================================================================

class AlignedBin(Dataset):
    def __init__(self, path):
        self.data = np.fromfile(path, dtype=np.uint16)
        self.idx = []
        i, d = 0, self.data
        while i < len(d):
            n = int(d[i]); self.idx.append((i + 1, n)); i += 1 + 2 * n
        print(f"{len(self.idx)} aligned pairs")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        s, n = self.idx[k]
        syl = self.data[s:s + n].astype(np.int64)
        chr_ = self.data[s + n:s + 2 * n].astype(np.int64)
        chr_[chr_ == 65535] = -100
        return torch.from_numpy(syl), torch.from_numpy(chr_)


def collate(batch):
    n = max(len(s) for s, _ in batch)
    B = len(batch)
    syl = torch.zeros(B, n, dtype=torch.long)
    chr_ = torch.full((B, n), -100, dtype=torch.long)
    mask = torch.zeros(B, n, dtype=torch.bool)
    for i, (s, c) in enumerate(batch):
        syl[i, :len(s)] = s; chr_[i, :len(c)] = c; mask[i, :len(s)] = True
    return syl, chr_, mask


# ======================================================================
# Teacher loading + distillation
# ======================================================================

def load_teacher(path, dev):
    """Load a full-precision SlothLM-E teacher (saved by either trainer)."""
    ck = torch.load(os.path.join(path, "slothe.pt"), map_location="cpu")
    c = ck["config"]
    t = SlothE_T(c["n_syl"], c["n_char"], c["dim"], c["depth"], c["heads"],
                 c["kv"], c["ffn"], embed_norm=c.get("embed_norm", False),
                 weight_bits="fp", fp_boundary=0, pre_norm=False)
    missing, unexpected = t.load_state_dict(ck["model"], strict=False)
    # only quant_alpha buffers should be "missing"; weights must all load
    bad = [k for k in missing if "quant_alpha" not in k]
    assert not bad, f"teacher weights failed to load: {bad[:5]}"
    t.to(dev).eval()
    for p in t.parameters():
        p.requires_grad_(False)
    print(f"teacher: {sum(p.numel() for p in t.parameters())/1e6:.1f}M (frozen)")
    return t


def distill_loss(student_logits, teacher_logits, targets, alpha, temp):
    """(1-alpha)*CE + alpha*T^2*KL, masked to valid (non -100) positions."""
    V = student_logits.shape[-1]
    ce = F.cross_entropy(student_logits.reshape(-1, V), targets.reshape(-1),
                         ignore_index=-100)
    mask = (targets != -100).reshape(-1, 1)
    s = F.log_softmax(student_logits.reshape(-1, V) / temp, dim=-1)
    with torch.no_grad():
        t = F.softmax(teacher_logits.reshape(-1, V) / temp, dim=-1)
    kl = (F.kl_div(s, t, reduction="none").sum(-1, keepdim=True) * mask).sum()
    kl = kl / mask.sum().clamp(min=1) * (temp * temp)
    return (1 - alpha) * ce + alpha * kl, ce.detach(), kl.detach()


# ======================================================================
# Train
# ======================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", default="syl_vocab.json")
    ap.add_argument("--tokenizer", default="tokenizer")
    ap.add_argument("--out", default="slothe_t")
    # architecture (recommended ~20M ternary: dim320 depth16 heads8 kv2 ffn880)
    ap.add_argument("--dim", type=int, default=320)
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--ffn", type=int, default=880)
    ap.add_argument("--embed-norm", action="store_true")
    # quantization
    ap.add_argument("--quant", choices=["ternary", "fp"], default="ternary")
    ap.add_argument("--weight-quant", choices=["median", "mean"], default="median")
    ap.add_argument("--fp-boundary", type=int, default=1)
    ap.add_argument("--pre-norm", action="store_true", default=True)
    ap.add_argument("--no-pre-norm", dest="pre_norm", action="store_false")
    ap.add_argument("--act-bits", type=int, default=8)
    ap.add_argument("--anneal-frac", type=float, default=0.15,
                    help="ramp quant_alpha 0->1 over this fraction of steps")
    # distillation
    ap.add_argument("--teacher", default="", help="dir with a fp teacher slothe.pt")
    ap.add_argument("--distill-alpha", type=float, default=0.7)
    ap.add_argument("--distill-temp", type=float, default=2.0)
    # optimization (retuned for small ternary: higher LR ok, see 2407.09527)
    ap.add_argument("--batch", type=int, default=384)
    ap.add_argument("--epochs", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--steps", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    syl_vocab = json.load(open(args.vocab, encoding="utf-8"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ds = AlignedBin(args.data)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=8,
                    collate_fn=collate, pin_memory=True, drop_last=True)
    model = SlothE_T(len(syl_vocab), len(tok), args.dim, args.depth, args.heads,
                     args.kv_heads, args.ffn, embed_norm=args.embed_norm,
                     weight_bits=args.quant, weight_quant=args.weight_quant,
                     fp_boundary=args.fp_boundary, pre_norm=args.pre_norm,
                     act_bits=args.act_bits).to(dev)
    np_ = sum(p.numel() for p in model.parameters())
    print(f"SlothLM-E-T {np_/1e6:.1f}M params | quant={args.quant} "
          f"wq={args.weight_quant} fp_boundary={args.fp_boundary} pre_norm={args.pre_norm}")

    teacher = load_teacher(args.teacher, dev) if args.teacher else None

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            weight_decay=args.weight_decay)
    total = args.steps or int(len(dl) * args.epochs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total,
                                                pct_start=0.03)
    anneal_steps = max(1, int(args.anneal_frac * total))

    model.train()
    step = 0
    for ep in range(math.ceil(args.epochs)):
        for syl, chr_, mask in dl:
            syl, chr_, mask = syl.to(dev), chr_.to(dev), mask.to(dev)
            model.set_quant_alpha(min(1.0, step / anneal_steps))
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                logits = model(syl, mask)
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = teacher(syl, mask)
                    loss, ce, kl = distill_loss(logits, t_logits, chr_,
                                                args.distill_alpha, args.distill_temp)
                else:
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                           chr_.reshape(-1), ignore_index=-100)
                    ce = kl = loss.detach()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            if step % 50 == 0:
                a = min(1.0, step / anneal_steps)
                print(f"step {step}/{total} loss {loss.item():.3f} "
                      f"ce {ce.item():.3f} kl {kl.item():.3f} alpha {a:.2f}", flush=True)
            if step >= total:
                break
        if step >= total:
            break

    model.set_quant_alpha(1.0)  # ensure saved weights reflect full ternary
    os.makedirs(args.out, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "config": {"n_syl": len(syl_vocab), "n_char": len(tok),
                           "dim": args.dim, "depth": args.depth, "heads": args.heads,
                           "kv": args.kv_heads, "ffn": args.ffn,
                           "embed_norm": args.embed_norm, "weight_bits": args.quant,
                           "weight_quant": args.weight_quant,
                           "fp_boundary": args.fp_boundary, "pre_norm": args.pre_norm,
                           "act_bits": args.act_bits}},
               os.path.join(args.out, "slothe.pt"))
    json.dump(syl_vocab, open(os.path.join(args.out, "syl_vocab.json"), "w",
                              encoding="utf-8"), ensure_ascii=False)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
