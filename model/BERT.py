import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class QA2Linear(nn.Module):
    def __init__(self, transformer_name, output_attentions):
        super().__init__()
        self.output_attentions = output_attentions
        self.transformer = AutoModel.from_pretrained(transformer_name, output_hidden_states=True,
                                                     output_attentions=self.output_attentions)
        self.drop = nn.Dropout(0.1)
        self.fc_start = nn.Linear(768, 1)
        self.fc_end = nn.Linear(768, 1)

    def forward(self, ids, mask):
        output = self.transformer(ids, attention_mask=mask)
        transformer_output = self.drop(output['last_hidden_state'])
        logits_start = self.fc_start(transformer_output)
        logits_end = self.fc_end(transformer_output)

        return logits_start, logits_end


if __name__ == '__main__':
    pass
