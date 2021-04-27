from typing import List

class QADataset:
    def __init__(self, dataset: List[dict], tokenizer) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> dict:
        context = self.dataset[idx]['context']
        question = self.dataset[idx]['qas'][0]['question']
        encodings = self.tokenizer(context, question, truncation=True)

        answer = self.dataset[idx]['qas'][0]['answers'][0]['text']
        start_idx = self.dataset[idx]['qas'][0]['answers'][0]['answer_start']
        end_idx = start_idx + len(answer)
        start_positions = encodings.char_to_token(start_idx)
        end_positions = encodings.char_to_token(end_idx)

        if end_positions is None:
            end_positions = encodings.char_to_token(end_idx - 1)
        if start_positions is None:
            start_positions = encodings.char_to_token(start_idx + 1)

        if start_positions is None or end_positions is None:
            start_positions = 1
            end_positions = 2
            encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
            return encodings
        else:
            if start_positions != end_positions:
                encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
                return encodings
            else:
                encodings.update({'start_positions': start_positions, 'end_positions': end_positions + 1})
                return encodings


if __name__ == '__main__':
    pass
