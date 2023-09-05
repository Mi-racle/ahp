import json
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset


class AHPDataset(Dataset):

    def __init__(self, dataset: Union[str, Path]):
        super().__init__()
        fin = open(dataset, 'r')
        self.data = json.load(fin)
        fin.close()

    def __getitem__(self, index):
        datum = self.data[index]
        return torch.tensor(datum['scores'], requires_grad=True), torch.tensor(datum['label'], requires_grad=True)

    def __len__(self):
        return len(self.data)
