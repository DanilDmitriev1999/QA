import json
from typing import List
from razdel import sentenize
from pprint import pprint


def read_json(filepath: str) -> List[dict]:
    with open(filepath, "r") as read_file:
        data = json.load(read_file)
    return data['data'][0]['paragraphs']


def to_json(json_result: List[dict], path='../dssm/data/dssm_train.json') -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(json_result, file, ensure_ascii=False)


def prepare_data(data: List[dict]) -> List[dict]:
    prep_data = []
    for i in data:
        result_prepare = {}
        split_context = [t.text for t in list(sentenize(i['context']))]
        question = i['qas'][0]['question']
        answer = i['qas'][0]['answers'][0]['text']
        result_prepare['question'] = question
        result_prepare['context'] = split_context
        result_prepare['short_answer'] = answer
        for idx, s in enumerate(split_context):
            if answer in s:
                result_prepare['answer'] = s
                result_prepare['id_sentence'] = idx
                break

        prep_data.append(result_prepare)

    return prep_data


if __name__ == '__main__':
    train = read_json('../data/sber_squad/train-v1.1.json')
    dev = read_json('../data/sber_squad/dev-v1.1.json')
    dssm_train = prepare_data(train)
    dssm_dev = prepare_data(dev)
    to_json(dssm_train, '../dssm/data/dssm_train.json')
    to_json(dssm_dev, '../dssm/data/dssm_dev.json')
