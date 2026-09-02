"""Causal language-model knowledge-distillation objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate(temperature: float, vocabulary_size: int, top_k: int | None = None) -> None:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if vocabulary_size <= 0:
        raise ValueError("aligned vocabulary size must be positive")
    if top_k is not None and not 0 < top_k <= vocabulary_size:
        raise ValueError(
            f"top_k ({top_k}) must be between 1 and aligned vocabulary size ({vocabulary_size})"
        )


def full_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Full-vocabulary KL loss aligned to next-token causal labels."""

    vocabulary_size = min(student_logits.size(-1), teacher_logits.size(-1))
    _validate(temperature, vocabulary_size)
    student = student_logits[:, :-1, :vocabulary_size] / temperature
    teacher = teacher_logits[:, :-1, :vocabulary_size] / temperature
    mask = labels[:, 1:] != -100
    if not torch.any(mask):
        return student_logits.new_zeros(())

    student = student[mask]
    teacher = teacher[mask]
    teacher_probability = F.softmax(teacher, dim=-1)
    student_log_probability = F.log_softmax(student, dim=-1)
    return (
        F.kl_div(student_log_probability, teacher_probability, reduction="batchmean")
        * temperature**2
    )


def topk_kl_divergence(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    """Teacher top-k KL loss aligned to next-token causal labels."""

    vocabulary_size = min(student_logits.size(-1), teacher_logits.size(-1))
    _validate(temperature, vocabulary_size, top_k)
    student = student_logits[:, :-1, :vocabulary_size] / temperature
    teacher = teacher_logits[:, :-1, :vocabulary_size] / temperature
    mask = labels[:, 1:] != -100
    if not torch.any(mask):
        return student_logits.new_zeros(())

    student = student[mask]
    teacher = teacher[mask]
    teacher_topk, indices = torch.topk(teacher, k=top_k, dim=-1)
    student_topk = torch.gather(student, dim=-1, index=indices)
    teacher_probability = F.softmax(teacher_topk, dim=-1)
    student_log_probability = F.log_softmax(student_topk, dim=-1)
    per_token_kl = torch.sum(
        teacher_probability * (torch.log(teacher_probability + 1e-8) - student_log_probability),
        dim=-1,
    )
    return per_token_kl.mean() * temperature**2
