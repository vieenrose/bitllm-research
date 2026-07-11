"""
Knowledge distillation from a full-precision teacher into the ternary student.

Distillation is one of the highest-leverage tools for sub-100M 1-bit models: the
teacher's soft targets inject far more information per token than the one-hot
label, partly buying back the capacity the student loses to ternarization. We
support standard logit KD (temperature-scaled KL). The teacher must share the
student's tokenizer/vocab — train the tokenizer first, then pick (or continue-
pretrain) a teacher on it, OR distill on-policy from a teacher with the same
tokenizer family (e.g. a small Qwen2.5 if you adopt its 151k vocab).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(student_logits, teacher_logits, targets,
                      alpha: float = 0.5, temperature: float = 2.0,
                      ignore_index: int = -100):
    """Combine hard-label CE with temperature-scaled KL to the teacher.

    loss = (1-alpha) * CE(student, targets) + alpha * T^2 * KL(soft_student || soft_teacher)
    """
    ce = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        targets.view(-1),
        ignore_index=ignore_index,
    )
    T = temperature
    s = F.log_softmax(student_logits / T, dim=-1)
    with torch.no_grad():
        t = F.softmax(teacher_logits / T, dim=-1)
    # mask padded positions
    mask = (targets != ignore_index).unsqueeze(-1)
    kl = F.kl_div(s, t, reduction="none").sum(-1, keepdim=True)
    kl = (kl * mask).sum() / mask.sum().clamp(min=1)
    kl = kl * (T * T)
    return (1 - alpha) * ce + alpha * kl, {"ce": ce.detach(), "kl": kl.detach()}


@torch.no_grad()
def teacher_forward(teacher, idx):
    """Run an HF causal-LM teacher and return logits aligned to next-token targets."""
    out = teacher(idx)
    return out.logits
