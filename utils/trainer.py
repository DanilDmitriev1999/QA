import torch

import pytorch_lightning as pl
from transformers import AdamW
from metrics import ExactMatch


class ModelTrainer(pl.LightningModule):
    def __init__(self, model, criterion, tokenizer, lr=1e-4):
        super().__init__()

        self.model = model
        self.criterion = criterion

        self.em = ExactMatch()
        self.tokenizer = tokenizer

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

        end_pred = torch.argmax(logits_end, dim=1).squeeze(-1).cpu().detach().numpy()
        end_gold = end.cpu().detach().numpy()

        text = text.cpu().detach().numpy()

        for element_idx in range(len(text)):
            pred = self.tokenizer.decode(text[element_idx][start_pred[element_idx]: end_pred[element_idx]])
            gold = self.tokenizer.decode(text[element_idx][start_gold[element_idx]: end_gold[element_idx]])

            em_step = self.em(pred, gold)

        return loss, loss_start, loss_end, em_step

    def training_step(self, batch, batch_idx):
        loss, loss_start, loss_end, em = self._custom_step(batch)

        values = {'train_loss_sum': loss,
                  'train_start_loss': loss_start,
                  'train_end_loss': loss_end,
                  'train_exact_match': em}

        self.log_dict(values)

        return loss

    def training_epoch_end(self, outputs):
        values = {'Train Epoch Exact-Match': self.em.compute()}

        self.log_dict(values)

    def validation_step(self, batch, batch_idx):
        loss, loss_start, loss_end, em = self._custom_step(batch)

        values = {'val_loss_sum': loss,
                  'val_start_loss': loss_start,
                  'val_end_loss': loss_end,
                  'val_exact_match': em}

        self.log_dict(values)

        return loss

    def validation_epoch_end(self, outputs):
        values = {'Validation Epoch Exact-Match': self.em.compute()}

        self.log_dict(values)