import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from src.train_hf_kd import KDDataCollator, extract_last_number, format_prompt
from llm_optimization.gsm8k import numeric_answers_match


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--distill_data_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_train_epochs", type=int, default=6)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_total_limit", type=int, default=2)
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
    parser.add_argument("--mix_gold_data", action="store_true")
    parser.add_argument("--require_correct", action="store_true")
    return parser.parse_args()


def split_train_val_indices(n: int, val_ratio: float, seed: int):
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(n * val_ratio))
    val_indices = set(indices[:val_size])
    train_indices = set(indices[val_size:])
    return train_indices, val_indices


def tokenize_examples(tokenizer, examples: Dataset, max_seq_length: int):
    def tokenize(example: Dict):
        prompt_text = format_prompt(example["question"])
        full_text = prompt_text + example["answer"]

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

        labels = list(full["input_ids"])
        prompt_len = len(prompt["input_ids"])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        full["labels"] = labels
        return full

    return examples.map(
        tokenize,
        remove_columns=examples.column_names,
        desc="Tokenizing",
    )


def load_distilled_rows(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SequenceDistillTrainer(Trainer):
    def __init__(self, metric_tokenizer, metric_eval_dataset, metric_max_new_tokens, metric_eval_examples, **kwargs):
        super().__init__(**kwargs)
        self.metric_tokenizer = metric_tokenizer
        self.metric_eval_dataset = metric_eval_dataset
        self.metric_max_new_tokens = metric_max_new_tokens
        self.metric_eval_examples = metric_eval_examples

    def _compute_relaxed_metrics(self, model) -> Dict[str, float]:
        eval_dataset = self.metric_eval_dataset
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

        return {
            "eval_relaxed_accuracy": correct / total if total else 0.0,
            "eval_relaxed_correct": float(correct),
            "eval_relaxed_examples": float(total),
            "eval_avg_generated_tokens": sum(generated_tokens) / total if total else 0.0,
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
    args = parse_args()
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("--val_ratio must be between 0 and 1")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_train = load_dataset("gsm8k", "main", split="train")
    train_indices, val_indices = split_train_val_indices(len(raw_train), args.val_ratio, args.seed)

    distilled_rows = load_distilled_rows(args.distill_data_path)
    filtered_rows = []
    for row in distilled_rows:
        source_idx = int(row["source_idx"])
        if source_idx not in train_indices:
            continue
        if args.require_correct and not bool(row.get("correct", False)):
            continue
        teacher_answer = row.get("teacher_completion", "").strip()
        if not teacher_answer:
            continue
        filtered_rows.append(
            {
                "question": row["question"],
                "answer": teacher_answer,
                "source": "teacher",
            }
        )

    if args.mix_gold_data:
        for idx in sorted(train_indices):
            ex = raw_train[int(idx)]
            filtered_rows.append(
                {
                    "question": ex["question"],
                    "answer": ex["answer"],
                    "source": "gold",
                }
            )

    if not filtered_rows:
        raise ValueError("No training rows available after filtering the sequence distillation dataset.")

    val_examples = [raw_train[int(idx)] for idx in sorted(val_indices)]
    train_dataset = Dataset.from_list(filtered_rows)
    val_dataset = Dataset.from_list(val_examples)

    tokenized_train = tokenize_examples(tokenizer, train_dataset, args.max_seq_length)
    tokenized_val = tokenize_examples(tokenizer, val_dataset, args.max_seq_length)

    model = AutoModelForCausalLM.from_pretrained(
        args.student_model,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.config.use_cache = False

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
        eval_strategy="steps",
        save_strategy="best",
        save_only_model=True,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_relaxed_accuracy",
        greater_is_better=True,
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
    )

    trainer = SequenceDistillTrainer(
        metric_tokenizer=tokenizer,
        metric_eval_dataset=val_dataset,
        metric_max_new_tokens=args.metric_max_new_tokens,
        metric_eval_examples=args.metric_eval_examples,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=KDDataCollator(tokenizer=tokenizer, pad_to_multiple_of=8),
    )

    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")
    cleanup_checkpoint_dirs(args.output_dir)


if __name__ == "__main__":
    main()
