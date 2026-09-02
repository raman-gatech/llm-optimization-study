"""
Relaxed GSM8K evaluation to mirror teammate's script:
  - Prompt: question + "\\n\\nLet's think step by step.\\n"
  - Final answer: last number extracted
  - Numeric match with tolerance
  - Generation via model.generate (no vLLM dependency)
"""

import argparse
import json
import os
from typing import Optional, Dict, List

import torch
import evaluate
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_optimization.gsm8k import (
    extract_numeric_answer,
    format_reasoning_prompt,
    numeric_answers_match,
)


def extract_last_number(text: str) -> Optional[float]:
    """Backward-compatible alias for the shared numeric normalizer."""

    return extract_numeric_answer(text)


def format_prompt(question: str) -> str:
    return format_reasoning_prompt(question)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", required=True, help="Path to trained checkpoint (e.g., output_dir/checkpoint-xxxx or /final)")
    parser.add_argument(
        "--student_model",
        default=None,
        help="Tokenizer source; defaults to checkpoint_dir",
    )
    parser.add_argument("--output_dir", required=True, help="Where to write eval outputs")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative")
    os.makedirs(args.output_dir, exist_ok=True)

    pred_path = os.path.join(args.output_dir, "test_predictions.jsonl")
    summary_path = os.path.join(args.output_dir, "test_summary.json")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(args.student_model or args.checkpoint_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        dtype=amp_dtype if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    ds = load_dataset("gsm8k", "main", split="test")
    if args.limit is not None:
        ds = ds.select(range(min(args.limit, len(ds))))

    total = 0
    correct = 0
    generated_tokens_list: List[int] = []

    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")
    predictions_for_bleu: List[str] = []
    references_for_bleu: List[List[str]] = []
    predictions_for_rouge: List[str] = []
    references_for_rouge: List[str] = []

    if os.path.exists(pred_path):
        os.remove(pred_path)

    with open(pred_path, "w", encoding="utf-8") as pred_f:
        for i, ex in enumerate(tqdm(ds, desc="Evaluating (relaxed)")):
            question = ex["question"]
            gt_text = ex["answer"]
            gt = extract_last_number(gt_text)
            if gt is None:
                continue

            prompt = format_prompt(question)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            prompt_len = inputs["input_ids"].shape[1]
            generated_only_ids = generated[0][prompt_len:]
            gen_tokens = int(generated_only_ids.shape[0])

            completion = tokenizer.decode(generated_only_ids, skip_special_tokens=True)
            pred = extract_numeric_answer(completion)
            is_correct = numeric_answers_match(pred, gt, tolerance=args.tolerance)

            total += 1
            if is_correct:
                correct += 1
            generated_tokens_list.append(gen_tokens)

            predictions_for_bleu.append(completion)
            references_for_bleu.append([gt_text])
            predictions_for_rouge.append(completion)
            references_for_rouge.append(gt_text)

            row = {
                "idx": i,
                "question": question,
                "ground_truth": gt,
                "prediction": pred,
                "correct": is_correct,
                "prompt": prompt,
                "raw_output": completion,
                "gold_full": gt_text,
                "pred_full": completion,
                "pred_final": pred,
                "exact_match": int(is_correct),
                "generated_tokens": gen_tokens,
            }
            pred_f.write(json.dumps(row) + "\n")

    acc = correct / total if total else 0.0
    avg_generated_tokens = sum(generated_tokens_list) / total if total else 0.0

    bleu = bleu_metric.compute(
        predictions=predictions_for_bleu,
        references=references_for_bleu,
    )

    rouge = rouge_metric.compute(
        predictions=predictions_for_rouge,
        references=references_for_rouge,
        use_stemmer=True,
    )

    summary = {
        "checkpoint_dir": args.checkpoint_dir,
        "examples": total,
        "correct": correct,
        "exact_match_accuracy": acc,
        "avg_generated_tokens": avg_generated_tokens,
        "bleu": bleu,
        "rouge": rouge,
        "prompt_style": "cot_step_by_step",
        "numeric_tolerance": args.tolerance,
        "scored_text": "generated_completion_only",
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
