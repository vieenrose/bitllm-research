"""
A compact Llama-style decoder built on BitLinear, tuned for sub-100M 1-bit SLMs.

Design choices (justified in docs/WHY_1BIT_NEEDS_SCALE.md and docs/DESIGN_SUB100M.md):
  * Ternary (1.58-bit) weights inside every transformer block's linear layers.
  * FULL-PRECISION islands: token embedding + LM head stay fp/bf16, and the first
    and last transformer blocks can be kept full precision (``fp_boundary_blocks``).
    At small scale these carry outsized information and are the most
    quantization-sensitive layers.
  * RMSNorm (pre-norm) + an extra SubLN inside BitLinear before activation quant.
  * Rotary position embeddings (RoPE); no learned positional table (saves params).
  * Grouped-query attention (optional) to cut KV projection params.
  * Deep-and-thin bias (MobileLLM finding: for a fixed budget, more layers /
    smaller width beats fewer/wider at sub-1B).
  * Untied embeddings by default: with a large bilingual vocab, tying forces the
    fp head and fp embedding to share, which we found harmful; keep them separate
    but both fp. (Config flag ``tie_embeddings`` lets you A/B this.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import BitLinear


@dataclass
class BitLMConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 24
    n_heads: int = 8
    n_kv_heads: int = 4          # GQA; set == n_heads for full MHA
    ffn_hidden: int = 1365       # ~2.67x d_model (SwiGLU keeps 3 matrices ~= 2/3)
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = False
    # --- quantization controls ---
    weight_bits: str = "ternary"     # "ternary" | "fp"
    act_bits: int = 8
    group_size: int = 0              # 0 = per-output-channel scale
    fp_boundary_blocks: int = 1      # keep first & last N blocks full precision
    dropout: float = 0.0

    def head_dim(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(head_dim: int, seq_len: int, theta: float, device=None):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [B, H, T, D]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.shape[-2], :]
    sin = sin[None, None, : x.shape[-2], :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2] = rx1
    out[..., 1::2] = rx2
    return out


def _make_linear(cfg: BitLMConfig, in_f: int, out_f: int, full_precision: bool) -> nn.Module:
    """A BitLinear, or an fp BitLinear when this layer is a full-precision island."""
    return BitLinear(
        in_f,
        out_f,
        bias=False,
        weight_bits="fp" if (full_precision or cfg.weight_bits == "fp") else "ternary",
        act_bits=None if full_precision else cfg.act_bits,
        group_size=cfg.group_size,
        norm=RMSNorm(in_f, cfg.norm_eps),  # SubLN before quant
    )


class Attention(nn.Module):
    def __init__(self, cfg: BitLMConfig, full_precision: bool):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv = cfg.n_kv_heads
        self.hd = cfg.head_dim()
        self.q_proj = _make_linear(cfg, cfg.d_model, self.n_heads * self.hd, full_precision)
        self.k_proj = _make_linear(cfg, cfg.d_model, self.n_kv * self.hd, full_precision)
        self.v_proj = _make_linear(cfg, cfg.d_model, self.n_kv * self.hd, full_precision)
        self.o_proj = _make_linear(cfg, self.n_heads * self.hd, cfg.d_model, full_precision)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if self.n_kv != self.n_heads:
            rep = self.n_heads // self.n_kv
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: BitLMConfig, full_precision: bool):
        super().__init__()
        self.gate = _make_linear(cfg, cfg.d_model, cfg.ffn_hidden, full_precision)
        self.up = _make_linear(cfg, cfg.d_model, cfg.ffn_hidden, full_precision)
        self.down = _make_linear(cfg, cfg.ffn_hidden, cfg.d_model, full_precision)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: BitLMConfig, full_precision: bool):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg, full_precision)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg, full_precision)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class BitLM(nn.Module):
    def __init__(self, cfg: BitLMConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)  # fp island
        blocks = []
        for i in range(cfg.n_layers):
            is_boundary = i < cfg.fp_boundary_blocks or i >= cfg.n_layers - cfg.fp_boundary_blocks
            blocks.append(Block(cfg, full_precision=is_boundary))
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # fp island
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        cos, sin = precompute_rope(cfg.head_dim(), cfg.max_seq_len, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, BitLinear)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def set_quant_alpha(self, alpha: float) -> None:
        for m in self.modules():
            if isinstance(m, BitLinear):
                m.set_quant_alpha(alpha)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        x = self.embed(idx)
        cos = self.rope_cos[:T].to(x.device)
        sin = self.rope_sin[:T].to(x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    # ---- param accounting -------------------------------------------------
    def param_report(self) -> dict:
        emb = self.embed.weight.numel()
        head = 0 if self.cfg.tie_embeddings else self.lm_head.weight.numel()
        total = sum(p.numel() for p in self.parameters())
        ternary = sum(
            m.weight.numel()
            for m in self.modules()
            if isinstance(m, BitLinear) and m.weight_bits == "ternary"
        )
        fp_non_embed = total - emb - head - ternary
        return {
            "total_params": total,
            "embedding_params": emb,
            "lm_head_params": head,
            "ternary_params": ternary,
            "fp_transformer_params": fp_non_embed,
            "embedding_fraction": round((emb + head) / total, 4),
            "ternary_fraction": round(ternary / total, 4),
            # Effective footprint if ternary packs to 1.58 bits and fp is bf16(16b)
            "packed_MB_estimate": round(
                (ternary * 1.58 + (total - ternary) * 16) / 8 / 1e6, 2
            ),
        }


def build_model(cfg: BitLMConfig) -> BitLM:
    return BitLM(cfg)
