import torch
import torch.nn.functional as F


def compute_full_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Full-logit KD using the teacher distribution over the full vocabulary.
    Only positions with labels != -100 contribute to the loss.
    """
    # student/teacher logits: [B, T, V]
    # shift to align with CE loss (next-token prediction)
    student_logits = student_logits[:, :-1, :]
    teacher_logits = teacher_logits[:, :-1, :]
    labels = labels[:, 1:]

    mask = (labels != -100).float()
    if mask.sum() == 0:
        return torch.zeros((), device=student_logits.device)

    t = max(float(temperature), 1e-6)
    student_t = student_logits / t
    teacher_t = teacher_logits / t

    teacher_probs = F.softmax(teacher_t, dim=-1)
    student_log_probs = F.log_softmax(student_t, dim=-1)

    kl_per_pos = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-8) - student_log_probs), dim=-1)
    kl_per_pos = kl_per_pos * mask

    loss = kl_per_pos.sum() / mask.sum()
    return loss * (t ** 2)
