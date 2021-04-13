import json
from typing import List
from pprint import pprint


class ReadData:
    def __init__(self, file_path: str) -> None:
        self.data = self.read_json(file_path)
        self.flatten = lambda l: [item for sublist in l for item in sublist]

    @staticmethod
    def read_json(file_path: str) -> List[dict]:
        with open(file_path, "r") as read_file:
            data = json.load(read_file)
        return data

    def print_example(self, n: int) -> None:
        pprint(self.data[n], width=150)


if __name__ == '__main__':
    pass
