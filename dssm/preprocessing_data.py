import json
from typing import List
from razdel import sentenize
from pprint import pprint
from tqdm import tqdm
import collections


class PrepareData:
    def __init__(self, filepath: str,
                 save_result: bool = True,
                 path_save: str = '../dssm/data/dssm_train.json') -> None:
        self.data = self.read_json(filepath)
        self.dt = self.convert_data(self.data)

        if save_result:
            self.to_json(self.dt, path_save)

    # @staticmethod
    def convert_data(self, data: List[dict]) -> List[dict]:
        prep_data = []
        for i in tqdm(data, total=len(data)):
            result_prepare = {
                'id_wrong_answer': [],
                'id_correct_answer': [],
                              }
            split_context = [t.text for t in list(sentenize(i['context']))]
            word_index = 0
            question = i['qas'][0]['question']
            answer = i['qas'][0]['answers'][0]['text']
            answer_start = i['qas'][0]['answers'][0]['answer_start']
            result_prepare['question'] = question
            result_prepare['context'] = split_context
            result_prepare['short_answer'] = answer
            result_prepare['answer_start'] = answer_start
            for idx, sentences in enumerate(split_context):
                for _ in sentences:
                    if word_index == answer_start:
                        result_prepare['id_correct_answer'].append(idx)
                    word_index += 1
                # word_index += 1
                if idx not in result_prepare['id_correct_answer']:
                    result_prepare['id_wrong_answer'].append(idx)

            if not result_prepare['id_correct_answer'] or not result_prepare['id_wrong_answer']:
                continue

            prep_data.append(result_prepare)

        return prep_data

    @staticmethod
    def read_json(filepath: str) -> List[dict]:
        with open(filepath, "r") as read_file:
            data = json.load(read_file)
        return data['data'][0]['paragraphs']

    @staticmethod
    def to_json(json_result: List[dict], path='../dssm/data/dssm_train.json') -> None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(json_result, file, ensure_ascii=False)


if __name__ == '__main__':
    train = PrepareData('../data/sber_squad/train-v1.1.json', True)
    dev = PrepareData('../data/sber_squad/dev-v1.1.json', True, '../dssm/data/dssm_dev.json')

