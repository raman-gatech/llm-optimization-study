# load_teacher_student_pair.py

import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


TEACHER_NAME = "meta-llama/Llama-3.1-8B"
STUDENT_NAME = "meta-llama/Llama-3.2-3B"


def pick_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def print_model_summary(tag: str, model, tokenizer) -> None:
    print(f"\n=== {tag} SUMMARY ===")
    print(f"model name: {model.name_or_path}")
    print(f"tokenizer vocab size: {len(tokenizer)}")
    print(f"config vocab size: {model.config.vocab_size}")
    print(f"hidden size: {model.config.hidden_size}")
    print(f"num hidden layers: {model.config.num_hidden_layers}")


def main() -> None:
    print("Python:", sys.version)
    print("Torch:", torch.__version__)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = pick_dtype()

    print("Device:", device)
    print("Dtype:", dtype)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("BF16 supported:", torch.cuda.is_bf16_supported())

    print("\n=== LOADING TOKENIZERS ===")
    teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_NAME)
    student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_NAME)

    if teacher_tokenizer.pad_token is None:
        teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
    if student_tokenizer.pad_token is None:
        student_tokenizer.pad_token = student_tokenizer.eos_token

    print("\n=== LOADING TEACHER ===")
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_NAME,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device != "cuda":
        teacher = teacher.to(device)
    teacher.eval()

    print("\n=== LOADING STUDENT ===")
    student = AutoModelForCausalLM.from_pretrained(
        STUDENT_NAME,
        torch_dtype=dtype,
    ).to(device)
    student.train()

    print_model_summary("TEACHER", teacher, teacher_tokenizer)
    print_model_summary("STUDENT", student, student_tokenizer)

    print("\n=== BASIC COMPATIBILITY CHECK ===")
    print("same tokenizer vocab size:", len(teacher_tokenizer) == len(student_tokenizer))
    print("same config vocab size:", teacher.config.vocab_size == student.config.vocab_size)
    print("same hidden size:", teacher.config.hidden_size == student.config.hidden_size)

    text = "Q: If John has 3 apples and buys 2 more, how many apples does he have?\nA:"
    batch = student_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    print("\n=== FORWARD PASS ===")
    with torch.no_grad():
        teacher_outputs = teacher(**batch, output_hidden_states=True)

    student_outputs = student(**batch, output_hidden_states=True)

    print("teacher logits shape:", teacher_outputs.logits.shape)
    print("student logits shape:", student_outputs.logits.shape)
    print("teacher last hidden shape:", teacher_outputs.hidden_states[-1].shape)
    print("student last hidden shape:", student_outputs.hidden_states[-1].shape)

    print("\n=== KD READINESS ===")
    if teacher_outputs.logits.shape[-1] == student_outputs.logits.shape[-1]:
        print("Full-logit KD is structurally possible.")
    else:
        print("Full-logit KD is NOT directly possible due to vocab/logit mismatch.")

    if teacher_outputs.hidden_states[-1].shape[-1] == student_outputs.hidden_states[-1].shape[-1]:
        print("Hidden-state matching can be done directly.")
    else:
        print("Hidden-state matching will need a projection layer.")

    print("\nAll checks completed.")


if __name__ == "__main__":
    main()