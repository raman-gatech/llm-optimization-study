import argparse
import json
import os
import random
import shutil
from dataclasses import dataclass
from typing import Dict, Any

import yaml
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

from src.data.gsm8k_dataset import GSM8KDataset
from src.data.collator import DataCollator
from src.losses.ce_loss import compute_ce_loss
from src.losses.topk_kd import compute_topk_kd_loss
from src.losses.full_kd import compute_full_kd_loss


@dataclass
class TrainState:
    global_step: int = 0
    best_val_loss: float = float("inf")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def build_optimizer(model, lr: float, weight_decay: float):
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight", "norm.weight"]
    decay_params = []
    nodecay_params = []

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(nd in n for nd in no_decay):
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    optimizer_grouped_parameters = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)


def split_train_val_indices(n: int, val_ratio: float, seed: int):
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(n * val_ratio))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def evaluate(model, dataloader, device, use_amp: bool, amp_dtype) -> float:
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = compute_ce_loss(outputs.logits, batch["labels"])

            total_loss += loss.item()
            total_batches += 1

    return total_loss / max(total_batches, 1)


def build_models(config: Dict[str, Any], device: str, amp_dtype):
    method = config.get("method", "method0_ce")
    if method in {"method2_topk_kd", "method1_full_kd"}:
        student_name = config["student_model_name"]
        teacher_name = config["teacher_model_name"]
        student = AutoModelForCausalLM.from_pretrained(
            student_name,
            dtype=amp_dtype if device == "cuda" else torch.float32,
        ).to(device)
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            dtype=amp_dtype if device == "cuda" else torch.float32,
        ).to(device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        return student, teacher

    model_name = config["model_name"]
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=amp_dtype if device == "cuda" else torch.float32,
    ).to(device)
    return model, None


def save_checkpoint(
    model,
    tokenizer,
    optimizer,
    scheduler,
    state: TrainState,
    save_dir: str,
    checkpoint_name: str,
) -> None:
    ckpt_dir = os.path.join(save_dir, checkpoint_name)
    ensure_dir(ckpt_dir)

    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    trainer_state = {
        "global_step": state.global_step,
        "best_val_loss": state.best_val_loss,
        "lr": scheduler.get_last_lr()[0] if scheduler is not None else None,
    }

    with open(os.path.join(ckpt_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
        json.dump(trainer_state, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/method0_ce.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    seed = config.get("seed", 42)
    set_seed(seed)

    device = get_device()
    amp_dtype = get_dtype()
    use_amp = device == "cuda"

    method = config.get("method", "method0_ce")
    model_name = config.get("model_name", config.get("student_model_name"))
    output_dir = config["output_dir"]
    log_dir = os.path.join(output_dir, "logs")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    ensure_dir(output_dir)
    ensure_dir(log_dir)
    ensure_dir(ckpt_dir)

    print(f"device: {device}")
    print(f"amp dtype: {amp_dtype}")
    print(f"method: {method}")
    print(f"model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, teacher_model = build_models(config, device, amp_dtype)

    train_dataset_full = GSM8KDataset(
        tokenizer=tokenizer,
        split="train",
        max_length=config["max_length"],
    )

    train_indices, val_indices = split_train_val_indices(
        n=len(train_dataset_full),
        val_ratio=config.get("val_ratio", 0.05),
        seed=seed,
    )

    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(train_dataset_full, val_indices)

    collator = DataCollator(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config.get("num_workers", 2),
        pin_memory=(device == "cuda"),
        collate_fn=collator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["eval_batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        pin_memory=(device == "cuda"),
        collate_fn=collator,
    )

    optimizer = build_optimizer(
        model=model,
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.01)),
    )

    max_steps = int(config["max_steps"])
    grad_accum_steps = int(config.get("gradient_accumulation_steps", 1))
    warmup_ratio = float(config.get("warmup_ratio", 0.03))
    warmup_steps = max(1, int(max_steps * warmup_ratio))

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )

    state = TrainState()
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda" and amp_dtype == torch.float16))

    train_log_path = os.path.join(log_dir, "train_log.jsonl")
    val_log_path = os.path.join(log_dir, "val_log.jsonl")

    log_every = int(config.get("log_every", 10))
    eval_every = int(config.get("eval_every", 100))
    save_every = int(config.get("save_every", 250))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))

    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss_since_log = 0.0
    micro_batches_since_log = 0
    micro_step = 0

    while state.global_step < max_steps:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                ce_loss = compute_ce_loss(outputs.logits, batch["labels"])
                if method in {"method2_topk_kd", "method1_full_kd"}:
                    kd_labels = batch["labels"]
                    if config.get("kd_on_final_answer_only", False) and "kd_labels" in batch:
                        kd_labels = batch["kd_labels"]
                    with torch.no_grad():
                        teacher_outputs = teacher_model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                    if method == "method2_topk_kd":
                        kd_loss = compute_topk_kd_loss(
                            student_logits=outputs.logits,
                            teacher_logits=teacher_outputs.logits,
                            labels=kd_labels,
                            top_k=int(config.get("top_k", 50)),
                            temperature=float(config.get("kd_temperature", 1.0)),
                        )
                    else:
                        kd_loss = compute_full_kd_loss(
                            student_logits=outputs.logits,
                            teacher_logits=teacher_outputs.logits,
                            labels=kd_labels,
                            temperature=float(config.get("kd_temperature", 1.0)),
                        )

                    kd_alpha = float(config.get("kd_alpha", 0.5))
                    loss = (1.0 - kd_alpha) * ce_loss + kd_alpha * kd_loss
                else:
                    loss = ce_loss

                loss_for_backward = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

            loss_since_log += loss.item()
            micro_batches_since_log += 1
            micro_step += 1

            if micro_step % grad_accum_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1

                if state.global_step % log_every == 0:
                    avg_train_loss = loss_since_log / max(micro_batches_since_log, 1)
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"step {state.global_step} | "
                        f"train_loss {avg_train_loss:.4f} | "
                        f"lr {lr_now:.6e}"
                    )
                    save_jsonl(
                        train_log_path,
                        {
                            "step": state.global_step,
                            "train_loss": avg_train_loss,
                            "lr": lr_now,
                        },
                    )
                    loss_since_log = 0.0
                    micro_batches_since_log = 0

                if state.global_step % save_every == 0:
                    save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        state=state,
                        save_dir=ckpt_dir,
                        checkpoint_name=f"step-{state.global_step}",
                    )

                if state.global_step % eval_every == 0:
                    val_loss = evaluate(model, val_loader, device, use_amp, amp_dtype)
                    print(f"step {state.global_step} | val_loss {val_loss:.4f}")
                    save_jsonl(
                        val_log_path,
                        {
                            "step": state.global_step,
                            "val_loss": val_loss,
                        },
                    )

                    if val_loss < state.best_val_loss:
                        state.best_val_loss = val_loss

                        best_dir = os.path.join(ckpt_dir, "best")
                        if os.path.exists(best_dir):
                            shutil.rmtree(best_dir)

                        save_checkpoint(
                            model=model,
                            tokenizer=tokenizer,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            state=state,
                            save_dir=ckpt_dir,
                            checkpoint_name="best",
                        )
                        print(f"saved new best checkpoint at step {state.global_step}")

                    model.train()

                if state.global_step >= max_steps:
                    break

        if state.global_step >= max_steps:
            break

    final_val_loss = evaluate(model, val_loader, device, use_amp, amp_dtype)
    print(f"final val_loss {final_val_loss:.4f}")
    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        save_dir=ckpt_dir,
        checkpoint_name="final",
    )


if __name__ == "__main__":
    main()
