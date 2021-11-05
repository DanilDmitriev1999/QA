import pytorch_lightning as pl
from transformers import AdamW
import torch.nn as nn
from utils.metrics import *
import torch



class ClassificationTrainer(pl.LightningModule):
    def __init__(self, model, tokenizer, params: dict):
        super().__init__()

        self.model = model
        self.params = params
        self.tokenizer = tokenizer

        # other
        self.lr = self.params['lr']
        self.check_train_metrics = self.params['check_train_metrics']
        self.counter_train = 0

        # self.save_hyperparameters()
        self.dv = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.val_result = {
            'pred': [],
            'gold': [],
        }
        self.test_result = {
            'pred': [],
            'gold': [],
        }
        self.flatten = lambda l: [item for sublist in l for item in sublist]

    def forward(self, ids, mask):
        logits = self.model.forward(ids, mask)

        return logits

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.params['weight_decay'],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.params['lr'], eps=self.params['adam_epsilon'])
        return optimizer
        
    def _custom_step(self, batch, batch_idx, mode=None):
        text = batch['input_ids']
        mask = batch['attention_mask']
        start = batch['start_positions']
        end = batch['end_positions']

        gold = []
        pred = []

        logits = self(text, mask)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if len(start.size()) > 1:
            start = start.squeeze(-1)
        if len(end.size()) > 1:
            end = end.squeeze(-1)

        ignored_index = start_logits.size(1)
        start.clamp_(0, ignored_index)
        end.clamp_(0, ignored_index)

        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)

        loss_start = loss_fct(start_logits, start.long().to(self.dv))
        loss_end = loss_fct(end_logits, end.long().to(self.dv))

        loss = (loss_start + loss_end) / 2

        if mode:
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)

            start_pred = torch.argmax(start_logits, dim=1).squeeze(-1).cpu().detach().numpy()
            start_gold = start.cpu().detach().numpy()

            end_pred = torch.argmax(end_logits, dim=1).squeeze(-1).cpu().detach().numpy()
            end_gold = end.cpu().detach().numpy()

            for i in range(text.shape[0]):
                all_tokens = self.tokenizer.convert_ids_to_tokens(text[i])

                answer_pred = ' '.join(all_tokens[start_pred[i] : end_pred[i]])
                ans_ids_pred = self.tokenizer.convert_tokens_to_ids(answer_pred.split())
                answer_pred = self.tokenizer.decode(ans_ids_pred)

                answer_gold = ' '.join(all_tokens[start_gold[i] : end_gold[i]])
                ans_ids_gold = self.tokenizer.convert_tokens_to_ids(answer_gold.split())
                answer_gold = self.tokenizer.decode(ans_ids_gold)

                pred.append(answer_pred)
                gold.append(answer_gold)

            return loss, pred, gold
        
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._custom_step(batch, batch_idx)

        values = {'train_loss': loss}

        self.log_dict(values)

        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, pred, gold = self._custom_step(batch, batch_idx, 'val')

        values = {'loss_val': loss}

        self.val_result['pred'].append(pred)
        self.val_result['gold'].append(gold)

        self.log_dict(values)

        return loss

    def validation_epoch_end(self, outputs):
        predict_result = self.flatten(self.val_result['pred'])
        gold_result = self.flatten(self.val_result['gold'])

        metrics = evaluate(gold_result, predict_result)

        f1 = metrics['f1']
        EM = metrics['exact_match']
        
        values = {
            'f1_val': f1,
            'EM_val': EM, 
        }
        self.val_result['pred'] = []
        self.val_result['gold'] = []

        self.log_dict(values)
    
    def test_step(self, batch, batch_idx):
        loss, pred, gold = self._custom_step(batch, batch_idx, 'test')

        values = {'loss_test': loss}

        self.test_result['pred'].append(pred)
        self.test_result['gold'].append(gold)

        self.log_dict(values)

        return loss
    
    def test_epoch_end(self, outputs):
        predict_result = self.flatten(self.test_result['pred'])
        gold_result = self.flatten(self.test_result['gold'])

        metrics = evaluate(gold_result, predict_result)

        f1 = metrics['f1']
        EM = metrics['exact_match']
        
        values = {
            'f1_test': f1,
            'EM_test': EM, 
        }
        self.test_result['pred'] = []
        self.test_result['gold'] = []

        self.log_dict(values)

if __name__ == '__main__':
    pass