# download_and_check_models.py

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODELS = [
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
]


def get_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def load_one_model(model_name: str) -> None:
    print("=" * 80)
    print(f"Loading model: {model_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = get_dtype()

    print(f"Device: {device}")
    print(f"Dtype:  {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizer loaded")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Pad token: {tokenizer.pad_token}")
    print(f"EOS token: {tokenizer.eos_token}")

    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
        ).to(device)

    model.eval()

    print("Model loaded")
    print(f"config.vocab_size:       {model.config.vocab_size}")
    print(f"config.hidden_size:      {model.config.hidden_size}")
    print(f"config.num_hidden_layers:{model.config.num_hidden_layers}")

    text = "Q: What is 12 plus 7?\nA:"
    batch = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    if device == "cuda":
        # For device_map='auto', keep tensors on cuda so generation/forward works
        batch = {k: v.to("cuda") for k, v in batch.items()}
    else:
        batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        outputs = model(**batch, output_hidden_states=True)

    print(f"logits shape: {outputs.logits.shape}")
    print(f"num hidden state tensors: {len(outputs.hidden_states)}")
    print(f"last hidden state shape: {outputs.hidden_states[-1].shape}")
    print("Forward pass OK")


def main() -> None:
    print("Python:", sys.version)
    print("Torch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
        print("BF16 supported:", torch.cuda.is_bf16_supported())

    for model_name in MODELS:
        try:
            load_one_model(model_name)
        except Exception as exc:
            print(f"\nFAILED for model: {model_name}")
            print(f"Error type: {type(exc).__name__}")
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()