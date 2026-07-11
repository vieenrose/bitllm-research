"""Config loading: merge YAML files into typed dataclasses."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

import yaml

from .model import BitLMConfig


@dataclass
class DataConfig:
    tokenizer_path: str = "tokenizer/bitllm-zhen.json"
    # HuggingFace dataset specs as (name, config, split, weight, text_field, lang)
    sources: list = field(default_factory=list)
    seq_len: int = 2048
    seed: int = 1234
    num_workers: int = 4
    # Directory of pre-tokenized/packed .bin shards (created by prepare_data.py)
    packed_dir: str = "data/packed"


@dataclass
class TrainConfig:
    out_dir: str = "checkpoints/run"
    # optimization
    lr: float = 1.5e-3
    min_lr: float = 1.5e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 2000
    max_steps: int = 200000
    batch_tokens: int = 500000          # global tokens per optimizer step
    micro_batch_size: int = 16
    grad_accum: int = 0                 # 0 = derive from batch_tokens/seq_len/micro
    # precision
    dtype: str = "bfloat16"
    # QAT annealing: keep alpha=0 (fp) for the first `anneal_start` fraction, then
    # linearly ramp to 1.0 (full ternary) by `anneal_end` fraction of max_steps.
    quant_anneal_start: float = 0.0
    quant_anneal_end: float = 0.15
    # distillation
    distill: bool = False
    teacher_model: str = ""             # HF id of an fp teacher (e.g. Qwen2.5-0.5B)
    distill_alpha: float = 0.5          # weight on KL(student||teacher) vs CE
    distill_temp: float = 2.0
    # logging / ckpt
    log_every: int = 20
    eval_every: int = 2000
    ckpt_every: int = 2000
    wandb_project: str = ""


@dataclass
class ExpConfig:
    model: BitLMConfig = field(default_factory=BitLMConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _fill(dc_type, d: dict):
    """Instantiate a dataclass from a dict, ignoring unknown keys."""
    if d is None:
        return dc_type()
    fields = {f.name for f in dataclasses.fields(dc_type)}
    kwargs = {k: v for k, v in d.items() if k in fields}
    return dc_type(**kwargs)


def load_config(path: str) -> ExpConfig:
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return ExpConfig(
        model=_fill(BitLMConfig, raw.get("model")),
        data=_fill(DataConfig, raw.get("data")),
        train=_fill(TrainConfig, raw.get("train")),
    )


def dump_config(cfg: ExpConfig, path: str) -> None:
    out = {
        "model": dataclasses.asdict(cfg.model),
        "data": dataclasses.asdict(cfg.data),
        "train": dataclasses.asdict(cfg.train),
    }
    with open(path, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False, allow_unicode=True)
