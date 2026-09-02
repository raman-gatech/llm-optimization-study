import torch


class DataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        labels = [item["labels"] for item in batch]
        kd_labels = [item.get("kd_labels") for item in batch]

        batch_inputs = self.tokenizer.pad(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
            padding=True,
            return_tensors="pt",
        )

        max_len = batch_inputs["input_ids"].shape[1]

        padded_labels = []
        for l in labels:
            padded = l + [-100] * (max_len - len(l))
            padded_labels.append(padded)

        batch_inputs["labels"] = torch.tensor(padded_labels)

        if all(k is not None for k in kd_labels):
            padded_kd = []
            for k in kd_labels:
                padded = k + [-100] * (max_len - len(k))
                padded_kd.append(padded)
            batch_inputs["kd_labels"] = torch.tensor(padded_kd)

        return batch_inputs
