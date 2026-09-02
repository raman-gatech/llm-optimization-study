import argparse
import json
import os
from typing import Optional, Dict, List

import yaml
import torch
import evaluate
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from llm_optimization.gsm8k import extract_numeric_answer, numeric_answers_match


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_text(s: str) -> str:
    return " ".join(s.strip().split())


def extract_final_answer(text: str) -> Optional[float]:
    return extract_numeric_answer(text)


def safe_strip_prompt(decoded_text: str, prompt: str) -> str:
    if decoded_text.startswith(prompt):
        return decoded_text[len(prompt):].strip()
    return decoded_text.strip()


def build_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def compute_exact_match(pred: Optional[float], gold: Optional[float]) -> int:
    return int(numeric_answers_match(pred, gold))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/method0_ce.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # ---- paths ----
    output_dir = config["output_dir"]
    ckpt_dir = os.path.join(output_dir, "checkpoints", "best")
    eval_dir = os.path.join(output_dir, "eval")
    ensure_dir(eval_dir)

    pred_path = os.path.join(eval_dir, "test_predictions.jsonl")
    summary_path = os.path.join(eval_dir, "test_summary.json")

    # ---- model/tokenizer ----
    base_model = config.get("model_name", config.get("student_model_name"))
    if base_model is None:
        raise ValueError("config must include model_name or student_model_name")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        ckpt_dir,
        dtype=amp_dtype if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()

    # ---- generation config ----
    max_new_tokens = int(config.get("eval_max_new_tokens", 192))
    do_sample = bool(config.get("eval_do_sample", False))
    temperature = float(config.get("eval_temperature", 1.0))
    top_p = float(config.get("eval_top_p", 1.0))
    test_limit = config.get("eval_test_limit", None)

    # ---- dataset ----
    test_ds = load_dataset("gsm8k", "main")["test"]
    if test_limit is not None:
        test_ds = test_ds.select(range(int(test_limit)))

    # ---- metrics ----
    bleu_metric = evaluate.load("bleu")
    rouge_metric = evaluate.load("rouge")

    predictions_for_bleu: List[str] = []
    references_for_bleu: List[List[str]] = []
    predictions_for_rouge: List[str] = []
    references_for_rouge: List[str] = []

    total = 0
    exact_match_correct = 0
    extraction_failures = 0
    total_generated_tokens = 0

    if os.path.exists(pred_path):
        os.remove(pred_path)

    with open(pred_path, "a", encoding="utf-8") as fout:
        for idx, sample in enumerate(tqdm(test_ds, desc="Evaluating")):
            question = sample["question"]
            gold_full = normalize_text(sample["answer"])
            gold_final = extract_final_answer(gold_full)

            prompt = build_prompt(question)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )

            prompt_len = inputs["input_ids"].shape[1]
            generated_only_ids = generated[0][prompt_len:]
            total_generated_tokens += int(generated_only_ids.shape[0])

            decoded_full = tokenizer.decode(generated[0], skip_special_tokens=True)
            pred_full = safe_strip_prompt(decoded_full, prompt)
            pred_full = normalize_text(pred_full)
            pred_final = extract_final_answer(pred_full)

            em = compute_exact_match(pred_final, gold_final)
            exact_match_correct += em
            if pred_final is None:
                extraction_failures += 1

            predictions_for_bleu.append(pred_full)
            references_for_bleu.append([gold_full])

            predictions_for_rouge.append(pred_full)
            references_for_rouge.append(gold_full)

            row = {
                "idx": idx,
                "question": question,
                "prompt": prompt,
                "gold_full": gold_full,
                "gold_final": gold_final,
                "pred_full": pred_full,
                "pred_final": pred_final,
                "exact_match": em,
                "generated_tokens": int(generated_only_ids.shape[0]),
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            total += 1

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
        "checkpoint_path": ckpt_dir,
        "num_examples": total,
        "exact_match_accuracy": exact_match_correct / total if total > 0 else 0.0,
        "exact_match_correct": exact_match_correct,
        "extraction_failures": extraction_failures,
        "extraction_success_rate": (total - extraction_failures) / total if total > 0 else 0.0,
        "avg_generated_tokens": total_generated_tokens / total if total > 0 else 0.0,
        "bleu": bleu,
        "rouge": rouge,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== EVAL SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
