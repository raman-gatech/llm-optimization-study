# just_download_models.py

from transformers import AutoTokenizer, AutoModelForCausalLM

MODELS = [
    "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-3.1-8B",
]

for model_name in MODELS:
    print(f"Downloading tokenizer for {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Downloading model for {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"Done: {model_name}")