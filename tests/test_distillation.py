from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
distillation = pytest.importorskip("llm_optimization.distillation")
full_kl_divergence = distillation.full_kl_divergence
topk_kl_divergence = distillation.topk_kl_divergence


@pytest.mark.torch
@pytest.mark.parametrize("loss_function", [full_kl_divergence, topk_kl_divergence])
def test_causal_shift_excludes_final_logit(loss_function) -> None:
    teacher = torch.zeros((1, 3, 4), dtype=torch.float32)
    labels = torch.tensor([[-100, -100, 1]])
    student = teacher.clone()
    student[:, -1, 0] = 100.0

    args = (student, teacher, labels, 2.0)
    loss = loss_function(*args, 2) if loss_function is topk_kl_divergence else loss_function(*args)
    assert loss.item() == pytest.approx(0.0)


@pytest.mark.torch
@pytest.mark.parametrize("loss_function", [full_kl_divergence, topk_kl_divergence])
def test_causal_shift_scores_preceding_logit(loss_function) -> None:
    teacher = torch.zeros((1, 3, 4), dtype=torch.float32)
    teacher[:, 1, 0] = 3.0
    teacher[:, 1, 1] = 2.0
    labels = torch.tensor([[-100, -100, 1]])
    student = teacher.clone()
    student[:, 1, 0] = 10.0

    args = (student, teacher, labels, 2.0)
    loss = loss_function(*args, 2) if loss_function is topk_kl_divergence else loss_function(*args)
    assert loss.item() > 0


@pytest.mark.torch
def test_all_masked_batch_returns_scalar_zero() -> None:
    logits = torch.randn((2, 3, 5))
    labels = torch.full((2, 3), -100)
    loss = full_kl_divergence(logits, logits, labels, 1.0)
    assert loss.shape == ()
    assert loss.item() == 0.0


@pytest.mark.torch
def test_invalid_hyperparameters() -> None:
    logits = torch.zeros((1, 2, 3))
    labels = torch.zeros((1, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="temperature"):
        full_kl_divergence(logits, logits, labels, 0.0)
    with pytest.raises(ValueError, match="top_k"):
        topk_kl_divergence(logits, logits, labels, 1.0, 4)
