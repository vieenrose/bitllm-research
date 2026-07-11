"""
Tests that run WITHOUT a GPU. The torch-dependent tests skip gracefully if torch
is not installed (this repo is developed on a CPU box; torch lives on the GPU box).
"""

import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

HAS_TORCH = importlib.util.find_spec("torch") is not None

import pytest  # noqa: E402


# ---- pure-python: budget planner arithmetic ------------------------------

def test_budget_planner_embedding_fraction():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    import plan_budget as pb
    # A bilingual sub-100M-core config should have a large embedding fraction.
    total = pb.report(48000, 512, 20, 8, 2, 1365, tie_embeddings=False,
                      fp_boundary_blocks=1)
    assert total > 0
    emb = 48000 * 512
    head = 48000 * 512
    assert (emb + head) / total > 0.4  # embeddings dominate — the core thesis


# ---- torch-gated: quantizer + model ---------------------------------------

@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (CPU dev box)")
def test_ternary_values():
    import torch
    from bitllm.quant import weight_quant_ternary
    w = torch.randn(64, 64)
    q = weight_quant_ternary(w)
    scale = w.abs().mean().clamp(min=1e-5)
    levels = torch.unique((q / scale).round())
    assert set(levels.tolist()).issubset({-1.0, 0.0, 1.0})


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (CPU dev box)")
def test_ste_gradient_flows():
    import torch
    from bitllm.quant import BitLinear
    lin = BitLinear(32, 16, weight_bits="ternary", act_bits=8)
    x = torch.randn(4, 32, requires_grad=True)
    y = lin(x).sum()
    y.backward()
    # STE must pass gradient to the full-precision weight and input
    assert lin.weight.grad is not None
    assert lin.weight.grad.abs().sum() > 0
    assert x.grad is not None


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (CPU dev box)")
def test_quant_alpha_blend():
    import torch
    from bitllm.quant import BitLinear
    lin = BitLinear(32, 16, weight_bits="ternary", act_bits=8)
    x = torch.randn(2, 32)
    lin.set_quant_alpha(0.0)
    y_fp = lin(x)
    lin.set_quant_alpha(1.0)
    y_q = lin(x)
    # alpha=0 should be closer to a plain linear than alpha=1 (which is ternary)
    import torch.nn.functional as F
    y_ref = F.linear(x, lin.weight)  # NOTE: BitLinear has SubLN=None here
    assert (y_fp - y_ref).abs().mean() < (y_q - y_ref).abs().mean()


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed (CPU dev box)")
def test_model_forward_and_report():
    import torch
    from bitllm.model import BitLMConfig, build_model
    cfg = BitLMConfig(vocab_size=256, d_model=64, n_layers=4, n_heads=4,
                      n_kv_heads=2, ffn_hidden=128, max_seq_len=32,
                      fp_boundary_blocks=1)
    model = build_model(cfg)
    idx = torch.randint(0, 256, (2, 32))
    logits, loss = model(idx, idx)
    assert logits.shape == (2, 32, 256)
    assert loss.item() > 0
    rep = model.param_report()
    assert rep["ternary_params"] > 0
    assert rep["embedding_params"] > 0
    # boundary blocks are fp, so not everything is ternary
    assert rep["fp_transformer_params"] > 0
