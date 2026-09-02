"""
HF Trainer-based CE fine-tuning for GSM8K (teacher or student).

Prompt format:
  question + "\\n\\nLet's think step by step.\\n"
Labels mask prompt tokens so loss is computed on answer tokens only.
"""

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

from llm_optimization.gsm8k import format_reasoning_prompt


def format_prompt(question: str) -> str:
    return format_reasoning_prompt(question)


def format_example(question: str, answer: str) -> str:
    return format_prompt(question) + answer


def build_dataset(tokenizer, max_seq_length: int, val_ratio: float, seed: int):
    ds = load_dataset("gsm8k", "main")
    split = ds["train"].train_test_split(
        test_size=val_ratio,
        seed=seed,
        shuffle=True,
    )
    train_ds = split["train"]
    val_ds = split["test"]

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
        return full

    train_tokenized = train_ds.map(
        tokenize,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train",
    )
    val_tokenized = val_ds.map(
        tokenize,
        remove_columns=val_ds.column_names,
        desc="Tokenizing validation",
    )

    return train_tokenized, val_tokenized


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="meta-llama/Llama-3.2-3B")
    parser.add_argument("--output_dir", default="./checkpoints/llama-ce-gsm8k")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--save_strategy", default="steps", choices=["steps", "epoch", "no"])
    parser.add_argument("--eval_strategy", default="steps", choices=["steps", "epoch", "no"])
    parser.add_argument(
        "--load_best_model_at_end",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--val_ratio", type=float, default=0.05)
    args = parser.parse_args()

    if not 0.0 < args.val_ratio < 1.0:
        parser.error("--val_ratio must be between 0 and 1")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(args.output_dir) / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    model.config.use_cache = False

    train_dataset, eval_dataset = build_dataset(
        tokenizer,
        args.max_seq_length,
        args.val_ratio,
        args.seed,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
    )

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
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=(
            args.load_best_model_at_end
            and args.eval_strategy != "no"
            and args.save_strategy != "no"
        ),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to="none",
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(f"{args.output_dir}/final")
    tokenizer.save_pretrained(f"{args.output_dir}/final")


if __name__ == "__main__":
    main()
