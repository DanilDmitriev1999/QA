from collections import Counter
import string


def normalize_answer(s):
    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))

def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def evaluate(gold_answers, predictions):
    f1 = exact_match = total = 0

    for ground_truths, prediction in zip(gold_answers, predictions):
        total += 1
        exact_match += exact_match_score(prediction, ground_truths)
        f1 += f1_score(prediction, ground_truths)
    
    exact_match = 100.0 * exact_match / total
    f1 = f1 / total

    return {'exact_match': exact_match, 'f1': f1}

if __name__ == '__main__':
    pass
