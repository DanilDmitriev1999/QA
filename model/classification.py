import torch
import torch.nn as nn
from transformers import AutoModel


class QALinear(nn.Module):
    def __init__(self,
                 model: str,
                 emb_size: int = 768,
                 output_attentions: bool = False):
        super().__init__()
        self.output_attentions = output_attentions
        self.transformer = model
        self.drop = nn.Dropout(0.1)
        self.fc = nn.Linear(emb_size, 2)

    def forward(self, ids: torch.tensor, mask: torch.tensor):
        output = self.transformer(ids, attention_mask=mask)
        transformer_output = self.drop(output['last_hidden_state'])
        logits = self.fc(transformer_output)

        return logits


if __name__ == '__main__':
    pass
