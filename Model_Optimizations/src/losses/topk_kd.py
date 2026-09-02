import torch
import torch.nn.functional as F


def compute_topk_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    top_k: int = 50,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Top-K KD using teacher's top-k distribution per position.
    Only positions with labels != -100 contribute to the loss.
    """
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    # student/teacher logits: [B, T, V]
    # shift to align with CE loss (next-token prediction)
    student_logits = student_logits[:, :-1, :]
    teacher_logits = teacher_logits[:, :-1, :]
    labels = labels[:, 1:]

    # mask out prompt tokens
    mask = (labels != -100).float()
    if mask.sum() == 0:
        return torch.zeros((), device=student_logits.device)

    t = max(float(temperature), 1e-6)
    student_t = student_logits / t
    teacher_t = teacher_logits / t

    # top-k on teacher
    teacher_topk_vals, teacher_topk_idx = torch.topk(teacher_t, k=top_k, dim=-1)
    student_topk_vals = torch.gather(student_t, dim=-1, index=teacher_topk_idx)

    teacher_probs = F.softmax(teacher_topk_vals, dim=-1)
    student_log_probs = F.log_softmax(student_topk_vals, dim=-1)

    # KL(teacher || student) over top-k
    kl_per_pos = torch.sum(teacher_probs * (torch.log(teacher_probs + 1e-8) - student_log_probs), dim=-1)
    kl_per_pos = kl_per_pos * mask

    loss = kl_per_pos.sum() / mask.sum()
    return loss * (t ** 2)
