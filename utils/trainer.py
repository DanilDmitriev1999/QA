import pytorch_lightning as pl
from transformers import AdamW
from torchmetrics import Metric
import re
import string
import torch


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class F1Score(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("score", default=torch.tensor(0))
        self.add_state("recall", default=torch.tensor(0))
        self.add_state("precision", default=torch.tensor(0))

    def update(self, pred, gold):
        pred_tokens = normalize_answer(pred).split()
        truth_tokens = normalize_answer(gold).split()

        # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = set(pred_tokens) & set(truth_tokens)

        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0

        self.precision = torch.tensor(len(common_tokens) / len(pred_tokens))
        self.recall = torch.tensor(len(common_tokens) / len(truth_tokens))

    def compute(self):
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


class ExactMatch(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0))
        self.add_state("total", default=torch.tensor(0))
        self.add_state("exact_match", default=torch.tensor(0))

    def update(self, pred, gold):
        result = int(normalize_answer(gold) == normalize_answer(pred))

        self.total += torch.tensor(1)

        if result == 1:
            self.correct += torch.tensor(1)

    def compute(self):
        return self.correct.float() / self.total.float()


class ModelTrainer(pl.LightningModule):
    def __init__(self, model, criterion, tokenizer, lr=1e-4):
        super().__init__()

        self.model = model
        self.criterion = criterion

        self.em = ExactMatch()
        self.f1 = F1Score()
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
            f1_step = self.f1(pred, gold)

        return loss, loss_start, loss_end, em_step, f1_step

    def training_step(self, batch, batch_idx):
        loss, loss_start, loss_end, em, f1 = self._custom_step(batch)

        values = {'train_loss_sum': loss,
                  'train_start_loss': loss_start,
                  'train_end_loss': loss_end,
                  'train_exact_match': em,
                  'train_f1': f1}

        self.log_dict(values)

        return loss

    def training_epoch_end(self, outputs):
        values = {'Train Epoch Exact-Match': self.em.compute(),
                  'Train Epoch F1': self.f1.compute()}

        self.log_dict(values)

    def validation_step(self, batch, batch_idx):
        loss, loss_start, loss_end, em, f1 = self._custom_step(batch)

        values = {'val_loss_sum': loss,
                  'val_start_loss': loss_start,
                  'val_end_loss': loss_end,
                  'val_exact_match': em,
                  'val_f1': f1}

        self.log_dict(values)

        return loss

    def validation_epoch_end(self, outputs):
        values = {'Validation Epoch Exact-Match': self.em.compute(),
                  'Validation Epoch F1': self.f1.compute()}

        self.log_dict(values)