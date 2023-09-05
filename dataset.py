import json
from pathlib import Path
from typing import Union

from torch.utils.data import Dataset


class AHPDataset(Dataset):

    def __init__(self, dataset: Union[str, Path]):
        super().__init__()
        fin = open(dataset, 'r')
        self.data = json.load(fin)
        print(self.data)

    def __getitem__(self, index):
        return self.data[index]
