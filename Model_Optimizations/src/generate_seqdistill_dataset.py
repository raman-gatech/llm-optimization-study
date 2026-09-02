import argparse
import json
import os
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.train_hf_kd import extract_last_number, format_prompt
from llm_optimization.gsm8k import numeric_answers_match


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--filtered_output_jsonl", required=True)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_generation_kwargs(args, tokenizer):
    kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
    }
    if args.do_sample:
        kwargs["temperature"] = args.temperature
        kwargs["top_p"] = args.top_p
    else:
        kwargs["temperature"] = None
        kwargs["top_p"] = None
    return kwargs


def maybe_slice_dataset(dataset, start_idx: int, limit: Optional[int]):
    if start_idx > 0:
        dataset = dataset.select(range(start_idx, len(dataset)))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def main():
    args = parse_args()

    ensure_parent(args.output_jsonl)
    ensure_parent(args.filtered_output_jsonl)
    summary_json = args.summary_json or str(Path(args.filtered_output_jsonl).with_suffix(".summary.json"))
    ensure_parent(summary_json)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        dtype=amp_dtype if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    dataset = load_dataset("gsm8k", "main", split=args.split)
    original_size = len(dataset)
    dataset = maybe_slice_dataset(dataset, args.start_idx, args.limit)

    gen_kwargs = build_generation_kwargs(args, tokenizer)

    total = 0
    correct = 0
    generated_tokens_total = 0

    if os.path.exists(args.output_jsonl):
        os.remove(args.output_jsonl)
    if os.path.exists(args.filtered_output_jsonl):
        os.remove(args.filtered_output_jsonl)

    with open(args.output_jsonl, "w", encoding="utf-8") as all_f, open(
        args.filtered_output_jsonl, "w", encoding="utf-8"
    ) as filtered_f:
        for local_idx, ex in enumerate(tqdm(dataset, desc="Generating teacher traces")):
            source_idx = args.start_idx + local_idx
            question = ex["question"]
            gold_answer = ex["answer"]
            gold_final = extract_last_number(gold_answer)

            prompt = format_prompt(question)
            inputs = tokenizer(prompt, return_tensors="pt")
            if device == "cuda":
                inputs = inputs.to(model.device)
            else:
                inputs = inputs.to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    **gen_kwargs,
                )

            prompt_len = inputs["input_ids"].shape[1]
            generated_only_ids = generated[0][prompt_len:]
            generated_tokens = int(generated_only_ids.shape[0])
            generated_tokens_total += generated_tokens

            teacher_completion = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
            teacher_full = tokenizer.decode(generated[0], skip_special_tokens=True)
            teacher_pred = extract_last_number(teacher_completion)
            is_correct = numeric_answers_match(
                teacher_pred,
                gold_final,
                tolerance=args.tolerance,
            )

            row = {
                "source_idx": source_idx,
                "question": question,
                "prompt": prompt,
                "gold_answer": gold_answer,
                "gold_final": gold_final,
                "teacher_completion": teacher_completion,
                "teacher_full_output": teacher_full,
                "teacher_prediction": teacher_pred,
                "correct": is_correct,
                "generated_tokens": generated_tokens,
                "teacher_model": args.teacher_model,
            }
            all_f.write(json.dumps(row) + "\n")
            if is_correct:
                filtered_f.write(json.dumps(row) + "\n")
                correct += 1
            total += 1

    summary = {
        "teacher_model": args.teacher_model,
        "split": args.split,
        "original_split_size": original_size,
        "start_idx": args.start_idx,
        "generated_examples": total,
        "filtered_correct_examples": correct,
        "teacher_relaxed_accuracy": correct / total if total else 0.0,
        "avg_generated_tokens": generated_tokens_total / total if total else 0.0,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": args.top_p if args.do_sample else None,
        "output_jsonl": args.output_jsonl,
        "filtered_output_jsonl": args.filtered_output_jsonl,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
