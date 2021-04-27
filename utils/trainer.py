import torch

import pytorch_lightning as pl
from transformers import AdamW
from sklearn.metrics import accuracy_score


class ModelTrainer(pl.LightningModule):
    def __init__(self,
                 model,
                 criterion,
                 lr=1e-4):
        super().__init__()

        self.model = model
        self.criterion = criterion

        # other
        self.lr = lr

    def forward(self, ids, mask):
        logits_start, logits_end = self.model.forward(ids, mask)

        return logits_start, logits_end

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=1e-8)
        return optimizer

    def _custom_step(self, batch):
        text = batch['input_ids']
        mask = batch['attention_mask']
        start = batch['start_positions']
        end = batch['end_positions']

        logits_start, logits_end = self(text, mask)

        loss_start = self.criterion(logits_start, start.view(-1, 1))
        loss_end = self.criterion(logits_end, end.view(-1, 1))

        loss = loss_start + loss_end

        start_pred = torch.argmax(logits_start, dim=1).squeeze(-1).cpu().detach().numpy()
        start_gold = start.cpu().detach().numpy()
        accuracy_start = accuracy_score(start_pred, start_gold)

        end_pred = torch.argmax(logits_end, dim=1).squeeze(-1).cpu().detach().numpy()
        end_gold = end.cpu().detach().numpy()
        accuracy_end = accuracy_score(end_pred, end_gold)

        return loss, loss_start, loss_end, loss_end, accuracy_start, accuracy_end

    def training_step(self, batch, batch_idx):
        loss, loss_start, loss_end, loss_end, accuracy_start, accuracy_end = self._custom_step(batch)

        values = {'train_loss_sum': loss,
                  'train_start_loss': loss_start,
                  'train_end_loss': loss_end,
                  'train_accuracy_start': accuracy_start,
                  'train_accuracy_end': accuracy_end}

        self.log_dict(values)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, loss_start, loss_end, loss_end, accuracy_start, accuracy_end = self._custom_step(batch)

        values = {'val_loss_sum': loss,
                  'val_start_loss': loss_start,
                  'val_end_loss': loss_end,
                  'val_accuracy_start': accuracy_start,
                  'val_accuracy_end': accuracy_end}

        self.log_dict(values)

        return loss


if __name__ == '__main__':
    pass
