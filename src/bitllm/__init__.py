"""bitllm: training 1-bit / ternary small language models (sub-100M) from scratch."""

from .quant import (
    BitLinear,
    activation_quant,
    weight_quant_ternary,
    weight_quant_ternary_grouped,
    ternarize_state_dict_report,
)
from .model import BitLM, BitLMConfig, build_model

__all__ = [
    "BitLinear",
    "activation_quant",
    "weight_quant_ternary",
    "weight_quant_ternary_grouped",
    "ternarize_state_dict_report",
    "BitLM",
    "BitLMConfig",
    "build_model",
]

__version__ = "0.1.0"
