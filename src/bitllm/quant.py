"""
Quantization primitives for training 1-bit / ternary language models from scratch.

The core layer is ``BitLinear`` (BitNet b1.58 style): weights are ternarized to
{-1, 0, +1} with a single per-tensor absmean scale, activations are quantized to
int8 with per-token absmax, and gradients flow through a straight-through
estimator (STE). We extend the published recipe with two levers that our research
(see docs/WHY_1BIT_NEEDS_SCALE.md) identifies as decisive at sub-100M scale:

  1. Mixed-precision islands: individual BitLinear layers can be left in full
     precision (``weight_bits="fp"``). We keep the token embedding, the LM head,
     and (optionally) the first/last transformer blocks out of ternary space,
     because at small scale those carry a disproportionate share of the model's
     information and are the most quantization-sensitive.

  2. Progressive quantization annealing: ``quant_alpha`` blends the full-precision
     and quantized weight/activation, so training can start near full precision
     (alpha=0) and anneal to fully ternary (alpha=1). This stabilizes the STE at
     narrow width where gradient noise is largest.

References:
  - BitNet: Scaling 1-bit Transformers (arXiv:2310.11453)
  - The Era of 1-bit LLMs: BitNet b1.58 (arXiv:2402.17764)
  - BitNet b1.58 2B4T Technical Report (arXiv:2504.12285)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Fake-quant functions (return dequantized tensors; caller wraps them in STE)
# ---------------------------------------------------------------------------

def activation_quant(x: torch.Tensor, num_bits: int = 8, eps: float = 1e-5) -> torch.Tensor:
    """Per-token absmax symmetric quantization of activations.

    Default int8 (num_bits=8) matches BitNet b1.58. The scale is computed over
    the last dim (per token), which is robust to the per-position magnitude
    variation you see in language models.
    """
    qmax = 2 ** (num_bits - 1) - 1  # 127 for int8
    scale = qmax / x.abs().amax(dim=-1, keepdim=True).clamp_(min=eps)
    return (x * scale).round().clamp_(-qmax - 1, qmax) / scale


def weight_quant_ternary(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Ternarize weights to {-1, 0, +1} * scale using a per-tensor absmean scale.

    This is the BitNet b1.58 "1.58-bit" scheme: scale = mean(|W|); each weight is
    rounded to the nearest of {-1, 0, 1} after dividing by the scale, then scaled
    back. Returns a dequantized (float) tensor of the same shape.
    """
    scale = w.abs().mean().clamp_(min=eps)
    return (w / scale).round().clamp_(-1, 1) * scale


def weight_quant_ternary_grouped(w: torch.Tensor, group_size: int = 0, eps: float = 1e-5) -> torch.Tensor:
    """Ternarize with per-output-channel (or per-group) absmean scales.

    group_size=0 -> per output row (channel). group_size>0 -> per contiguous
    group of ``group_size`` input features within each row. Finer scales recover
    accuracy at small scale at the cost of storing more fp16 scale values, but
    those are negligible vs the weight matrix. Set group_size=-1 to fall back to
    a single per-tensor scale (identical to ``weight_quant_ternary``).
    """
    if group_size == -1:
        return weight_quant_ternary(w, eps)
    out_features, in_features = w.shape
    if group_size and group_size > 0 and in_features % group_size == 0:
        wg = w.view(out_features, in_features // group_size, group_size)
        scale = wg.abs().mean(dim=-1, keepdim=True).clamp_(min=eps)
        q = (wg / scale).round().clamp_(-1, 1) * scale
        return q.view(out_features, in_features)
    # per-output-channel
    scale = w.abs().mean(dim=-1, keepdim=True).clamp_(min=eps)
    return (w / scale).round().clamp_(-1, 1) * scale


def _ste(real: torch.Tensor, quantized: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Straight-through estimator with optional annealing.

    Forward value is ``(1-alpha)*real + alpha*quantized``; gradient flows to
    ``real`` as if the quantizer were the identity. alpha=1 -> fully quantized
    forward (standard STE). alpha=0 -> pure full precision.
    """
    blended = real + alpha * (quantized - real)
    return real + (blended - real).detach()


# ---------------------------------------------------------------------------
# BitLinear
# ---------------------------------------------------------------------------

class BitLinear(nn.Linear):
    """Drop-in ``nn.Linear`` with ternary weights + int8 activations (QAT).

    Args (beyond nn.Linear):
      weight_bits: "ternary" (1.58-bit) or "fp" (leave full precision — used for
                   mixed-precision islands like embed-adjacent projections).
      act_bits:    activation quantization bit-width (8 recommended; set 0/None
                   to disable activation quant, e.g. paired with an fp layer).
      group_size:  weight scale granularity (see weight_quant_ternary_grouped).
      norm:        optional pre-quant normalization module (SubLN). BitNet
                   normalizes activations before quantizing; pass an RMSNorm.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        weight_bits: str = "ternary",
        act_bits: int | None = 8,
        group_size: int = 0,
        norm: nn.Module | None = None,
        **kwargs,
    ) -> None:
        super().__init__(in_features, out_features, bias=bias, **kwargs)
        assert weight_bits in ("ternary", "fp")
        self.weight_bits = weight_bits
        self.act_bits = act_bits
        self.group_size = group_size
        self.norm = norm
        # Annealing coefficient in [0, 1]; the training loop can set this via
        # ``set_quant_alpha``. Registered as a buffer so it moves with .to(device)
        # and is saved in checkpoints, but it is NOT a learnable parameter.
        self.register_buffer("quant_alpha", torch.tensor(1.0), persistent=True)

    def set_quant_alpha(self, alpha: float) -> None:
        self.quant_alpha.fill_(float(alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm is not None:
            x = self.norm(x)

        alpha = float(self.quant_alpha)

        if self.act_bits and self.weight_bits == "ternary":
            x = _ste(x, activation_quant(x, self.act_bits), alpha)

        if self.weight_bits == "ternary":
            w_q = weight_quant_ternary_grouped(self.weight, self.group_size)
            w = _ste(self.weight, w_q, alpha)
        else:
            w = self.weight

        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, weight_bits={self.weight_bits}, "
            f"act_bits={self.act_bits}, group_size={self.group_size}"
        )


# ---------------------------------------------------------------------------
# Deployment-time weight packing (for reference / kernel bring-up on the GPU box)
# ---------------------------------------------------------------------------

@torch.no_grad()
def ternarize_state_dict_report(model: nn.Module) -> dict:
    """Diagnostics: fraction of zeros and effective bits-per-weight per BitLinear.

    Useful to confirm the ternary sparsity level (bitnet.cpp's I2_S/TL kernels
    exploit the {-1,0,1} distribution) and to estimate the packed model size.
    """
    report = {}
    for name, m in model.named_modules():
        if isinstance(m, BitLinear) and m.weight_bits == "ternary":
            w = m.weight.detach()
            scale = w.abs().mean().clamp(min=1e-5)
            q = (w / scale).round().clamp(-1, 1)
            zeros = (q == 0).float().mean().item()
            report[name] = {
                "shape": tuple(w.shape),
                "zero_fraction": round(zeros, 4),
                # ternary packs to ~1.58 bits/weight (log2(3)); I2_S uses 2 bits.
                "bits_per_weight_packed": 1.58,
                "params": w.numel(),
            }
    return report
