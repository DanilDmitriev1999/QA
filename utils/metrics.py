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