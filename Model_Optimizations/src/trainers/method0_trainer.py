import torch
from src.losses.ce_loss import compute_ce_loss


class Method0Trainer:
    def __init__(self, model, dataloader, lr=2e-5):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_step(self, batch):
        self.model.train()

        batch = {k: v.to("cuda") for k, v in batch.items()}

        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )

        loss = compute_ce_loss(outputs.logits, batch["labels"])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        self.model.save_pretrained(path)
