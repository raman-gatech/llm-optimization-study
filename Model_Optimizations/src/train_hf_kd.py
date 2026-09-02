"""
HF Trainer-based KD fine-tuning for GSM8K.

Supports:
  - full-logit KD (method=full)
  - top-k KD (method=topk)

Prompt format matches teammate:
  question + "\\n\\nLet's think step by step.\\n"
Labels mask prompt tokens so loss is computed on answer tokens only.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

from llm_optimization.gsm8k import (
    extract_numeric_answer,
    format_reasoning_prompt,
    numeric_answers_match,
)
from llm_optimization.distillation import full_kl_divergence, topk_kl_divergence


def format_prompt(question: str) -> str:
    return format_reasoning_prompt(question)


def format_example(question: str, answer: str) -> str:
    return format_prompt(question) + answer


def extract_last_number(text: str) -> Optional[float]:
    """Backward-compatible alias for the shared numeric normalizer."""

    return extract_numeric_answer(text)


def build_kd_labels(question: str, answer: str, labels: List[int], tokenizer, max_seq_length: int) -> List[int]:
    answer_text = answer.strip()
    kd_labels = [-100] * len(labels)

    if "####" not in answer_text:
        return labels.copy()

    marker_idx = answer_text.find("####")
    pre_answer = answer_text[: marker_idx + 4]
    if len(answer_text) > marker_idx + 4 and answer_text[marker_idx + 4] == " ":
        pre_answer += " "

    pre_text = format_prompt(question) + pre_answer
    tokenized_pre = tokenizer(
        pre_text,
        truncation=True,
        max_length=max_seq_length,
        padding=False,
    )
    pre_len = len(tokenized_pre["input_ids"])
    for i in range(pre_len, len(labels)):
        kd_labels[i] = labels[i]
    return kd_labels


def build_datasets(tokenizer, max_seq_length: int, val_ratio: float, seed: int):
    ds = load_dataset("gsm8k", "main")
    train_val_split = ds["train"].train_test_split(
        test_size=val_ratio,
        seed=seed,
        shuffle=True,
    )
    train_ds = train_val_split["train"]
    val_ds = train_val_split["test"]

    def tokenize(example: Dict):
        full_text = format_example(example["question"], example["answer"])
        prompt_text = format_prompt(example["question"])

        full = tokenizer(
            full_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )
        prompt = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

        input_ids = full["input_ids"]
        labels = list(input_ids)
        prompt_len = len(prompt["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        full["labels"] = labels
        full["kd_labels"] = build_kd_labels(
            question=example["question"],
            answer=example["answer"],
            labels=labels,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )
        return full

    train_tokenized = train_ds.map(
        tokenize,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    val_tokenized = val_ds.map(
        tokenize,
        remove_columns=val_ds.column_names,
        desc="Tokenizing val",
    )

    return train_tokenized, val_tokenized, val_ds


def validate_tokenizer_alignment(student_tokenizer, teacher_tokenizer) -> None:
    if student_tokenizer.get_vocab() == teacher_tokenizer.get_vocab():
        return

    mismatches = []
    common_vocab = min(student_tokenizer.vocab_size, teacher_tokenizer.vocab_size)
    for idx in range(common_vocab):
        s_tok = student_tokenizer.convert_ids_to_tokens(idx)
        t_tok = teacher_tokenizer.convert_ids_to_tokens(idx)
        if s_tok != t_tok:
            mismatches.append((idx, s_tok, t_tok))
        if len(mismatches) >= 5:
            break

    mismatch_text = ", ".join(
        f"id {idx}: student={s_tok!r}, teacher={t_tok!r}"
        for idx, s_tok, t_tok in mismatches
    )
    raise ValueError(
        "Student and teacher tokenizers are not id-aligned, so logit distillation is unsafe. "
        f"Example mismatches: {mismatch_text}"
    )


class KDDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        model_features = [
            {
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
            }
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_features,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        max_len = batch["input_ids"].shape[1]
        label_keys = ["labels"]
        if "kd_labels" in features[0]:
            label_keys.append("kd_labels")

        for key in label_keys:
            padded_values = []
            for feature in features:
                values = list(feature[key])
                padded_values.append(values + ([-100] * (max_len - len(values))))
            batch[key] = torch.tensor(padded_values, dtype=torch.long)

        return batch


def kd_loss_full(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    return full_kl_divergence(student_logits, teacher_logits, labels, temperature)


def kd_loss_topk(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    return topk_kl_divergence(
        student_logits,
        teacher_logits,
        labels,
        temperature,
        top_k,
    )


class KDTrainer(Trainer):
    def __init__(
        self,
        teacher_model,
        alpha: float,
        temperature: float,
        kd_method: str,
        top_k: int,
        hidden_match: bool,
        hidden_match_weight: float,
        hidden_match_layer: int,
        hidden_match_on_answer_only: bool,
        metric_tokenizer,
        metric_eval_dataset,
        metric_max_new_tokens: int,
        metric_eval_examples: Optional[int],
        kd_on_answer_only: bool,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha  # weight for CE
        self.temperature = temperature
        self.kd_method = kd_method
        self.top_k = top_k
        self.hidden_match = hidden_match
        self.hidden_match_weight = hidden_match_weight
        self.hidden_match_layer = hidden_match_layer
        self.hidden_match_on_answer_only = hidden_match_on_answer_only
        self.metric_tokenizer = metric_tokenizer
        self.metric_eval_dataset = metric_eval_dataset
        self.metric_max_new_tokens = metric_max_new_tokens
        self.metric_eval_examples = metric_eval_examples
        self.kd_on_answer_only = kd_on_answer_only

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        kd_labels = inputs.get("kd_labels", labels)
        model_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }

        outputs = model(**model_inputs, output_hidden_states=self.hidden_match)
        student_logits = outputs.logits
        ce_loss = outputs.loss

        if self.teacher_model is None or self.alpha >= 1.0:
            return (ce_loss, outputs) if return_outputs else ce_loss

        with torch.no_grad():
            teacher_inputs = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
            teacher_outputs = self.teacher_model(**teacher_inputs, output_hidden_states=self.hidden_match)
            teacher_logits = teacher_outputs.logits

        kd_target_labels = kd_labels if self.kd_on_answer_only else labels
        if self.kd_method == "topk":
            kd = kd_loss_topk(student_logits, teacher_logits, kd_target_labels, self.temperature, self.top_k)
        else:
            kd = kd_loss_full(student_logits, teacher_logits, kd_target_labels, self.temperature)

        hidden_loss = torch.tensor(0.0, device=student_logits.device)
        if self.hidden_match:
            student_hidden_states = outputs.hidden_states
            teacher_hidden_states = teacher_outputs.hidden_states
            if student_hidden_states is None or teacher_hidden_states is None:
                raise ValueError("Hidden states requested but not returned by model.")

            student_hidden = student_hidden_states[self.hidden_match_layer]
            teacher_hidden = teacher_hidden_states[self.hidden_match_layer]

            if hasattr(model, "hidden_proj") and model.hidden_proj is not None:
                student_hidden = model.hidden_proj(student_hidden)

            match_labels = kd_labels if self.hidden_match_on_answer_only else labels
            mask = match_labels != -100
            if mask.sum() > 0:
                student_masked = student_hidden[mask].float()
                teacher_masked = teacher_hidden[mask].float()
                hidden_loss = F.mse_loss(student_masked, teacher_masked)

        loss = self.alpha * ce_loss + (1.0 - self.alpha) * kd + self.hidden_match_weight * hidden_loss

        if self.state.global_step % self.args.logging_steps == 0:
            self.log({
                "loss_ce": ce_loss.item(),
                "loss_kd": kd.item(),
                "loss_hidden": hidden_loss.item(),
                "loss_total": loss.item(),
            })

        return (loss, outputs) if return_outputs else loss

    def _compute_relaxed_metrics(self, model) -> Dict[str, float]:
        eval_dataset = self.metric_eval_dataset
        if eval_dataset is None:
            return {}

        if self.metric_eval_examples is not None:
            eval_dataset = eval_dataset.select(range(min(self.metric_eval_examples, len(eval_dataset))))

        was_training = model.training
        model.eval()

        total = 0
        correct = 0
        generated_tokens = []

        for example in eval_dataset:
            prompt = format_prompt(example["question"])
            inputs = self.metric_tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=self.metric_max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.metric_tokenizer.eos_token_id,
                    eos_token_id=self.metric_tokenizer.eos_token_id,
                    use_cache=True,
                )

            prompt_len = inputs["input_ids"].shape[1]
            generated_only_ids = generated[0][prompt_len:]
            generated_tokens.append(int(generated_only_ids.shape[0]))

            completion = self.metric_tokenizer.decode(
                generated_only_ids,
                skip_special_tokens=True,
            )
            pred = extract_last_number(completion)
            gt = extract_last_number(example["answer"])
            is_correct = numeric_answers_match(pred, gt)
            total += 1
            correct += int(is_correct)

        if was_training:
            model.train()

        accuracy = correct / total if total else 0.0
        avg_tokens = sum(generated_tokens) / total if total else 0.0
        return {
            "eval_relaxed_accuracy": accuracy,
            "eval_relaxed_correct": float(correct),
            "eval_relaxed_examples": float(total),
            "eval_avg_generated_tokens": avg_tokens,
        }

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        relaxed_metrics = self._compute_relaxed_metrics(self.model)
        metrics.update(relaxed_metrics)
        if relaxed_metrics:
            self.log(relaxed_metrics)
        return metrics


def cleanup_checkpoint_dirs(output_dir: str) -> None:
    root = Path(output_dir)
    for ckpt_dir in root.glob("checkpoint-*"):
        if ckpt_dir.is_dir():
            shutil.rmtree(ckpt_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--teacher_model", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--output_dir", default="./checkpoints/llama3b-gsm8k-ce-kd")
    parser.add_argument("--num_train_epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--save_strategy", choices=["steps", "epoch", "no"], default="steps")
    parser.add_argument("--eval_strategy", choices=["steps", "epoch", "no"], default="steps")
    parser.add_argument(
        "--load_best_model_at_end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--no_save_only_model", action="store_true")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--metric_max_new_tokens", type=int, default=256)
    parser.add_argument("--metric_eval_examples", type=int, default=None)
    parser.add_argument("--disable_kd", action="store_true")
    parser.add_argument("--kd_on_answer_only", action="store_true")
    parser.add_argument("--hidden_match", action="store_true")
    parser.add_argument("--hidden_match_weight", type=float, default=0.05)
    parser.add_argument("--hidden_match_layer", type=int, default=-1)
    parser.add_argument("--hidden_match_on_answer_only", action="store_true")
    # KD-specific
    parser.add_argument("--alpha", type=float, default=0.9,
                        help="Weight for CE loss. KD loss weight = 1 - alpha.")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Distillation temperature for softening distributions.")
    parser.add_argument("--kd_method", choices=["full", "topk"], default="full")
    parser.add_argument("--top_k", type=int, default=64)
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.top_k <= 0:
        parser.error("--top_k must be positive")
    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val_ratio must be between 0 and 1")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    teacher_model = None
    if not args.disable_kd and args.alpha < 1.0:
        teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
        validate_tokenizer_alignment(tokenizer, teacher_tokenizer)
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.teacher_model,
            dtype=torch.bfloat16,
            device_map="auto",
        )
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False

    student_model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto",
    )
    student_model.config.use_cache = False

    if args.hidden_match:
        if teacher_model is None:
            raise ValueError("Hidden-state matching requires a teacher model.")
        student_hidden = student_model.config.hidden_size
        teacher_hidden = teacher_model.config.hidden_size
        if student_hidden != teacher_hidden:
            student_model.hidden_proj = nn.Linear(student_hidden, teacher_hidden, bias=False).to(student_model.device)
        else:
            student_model.hidden_proj = None

    train_dataset, eval_dataset, metric_eval_dataset = build_datasets(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    data_collator = KDDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8)

    load_best_model_at_end = args.load_best_model_at_end and args.eval_strategy != "no"
    save_only_model = not args.no_save_only_model

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_only_model=save_only_model,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="eval_relaxed_accuracy",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
    )

    trainer = KDTrainer(
        teacher_model=teacher_model,
        alpha=args.alpha,
        temperature=args.temperature,
        kd_method=args.kd_method,
        top_k=args.top_k,
        hidden_match=args.hidden_match,
        hidden_match_weight=args.hidden_match_weight,
        hidden_match_layer=args.hidden_match_layer,
        hidden_match_on_answer_only=args.hidden_match_on_answer_only,
        metric_tokenizer=tokenizer,
        metric_eval_dataset=metric_eval_dataset,
        metric_max_new_tokens=args.metric_max_new_tokens,
        metric_eval_examples=args.metric_eval_examples,
        kd_on_answer_only=args.kd_on_answer_only,
        model=student_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    cleanup_checkpoint_dirs(args.output_dir)


if __name__ == "__main__":
    main()
