"""Run a small interactive generation against a local checkpoint."""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--tokenizer",
        help="Tokenizer name/path; defaults to the checkpoint.",
    )
    parser.add_argument(
        "--prompt",
        default="Question: If you have 10 apples and eat 3, how many are left?\nAnswer:",
    )
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint).to(args.device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    completion_ids = outputs[0, inputs["input_ids"].shape[1] :]
    completion = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
    print(completion)


if __name__ == "__main__":
    main()
