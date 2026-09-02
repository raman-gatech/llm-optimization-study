from datasets import load_dataset
from torch.utils.data import Dataset


class GSM8KDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_length=512):
        self.dataset = load_dataset("gsm8k", "main")[split]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        question = sample["question"]
        answer = sample["answer"]

        # split reasoning + final answer
        # format: "... #### 42"
        answer_text = answer.strip()
        # build prompt
        prompt = f"Question: {question}\nAnswer:"
        full_text = prompt + " " + answer_text

        # tokenize
        tokenized_full = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
        )

        tokenized_prompt = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = tokenized_full["input_ids"]
        attention_mask = tokenized_full["attention_mask"]

        # create labels
        labels = input_ids.copy()

        prompt_len = len(tokenized_prompt["input_ids"])

        # mask prompt tokens
        for i in range(prompt_len):
            if i < len(labels):
                labels[i] = -100

        # build KD labels that focus on the final answer span (after "####")
        kd_labels = [-100] * len(labels)
        if "####" in answer_text:
            marker_idx = answer_text.find("####")
            pre_answer = answer_text[: marker_idx + 4]
            if len(answer_text) > marker_idx + 4 and answer_text[marker_idx + 4] == " ":
                pre_answer += " "

            pre_text = prompt + " " + pre_answer
            tokenized_pre = self.tokenizer(
                pre_text,
                truncation=True,
                max_length=self.max_length,
            )
            pre_len = len(tokenized_pre["input_ids"])
            for i in range(pre_len, len(labels)):
                kd_labels[i] = labels[i]
        else:
            kd_labels = labels.copy()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "kd_labels": kd_labels,
        }
