import os
from pathlib import Path
from typing import Union

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AHPDataset
from loss import Loss
from net import Net
from utils import increment_path


def detect(
        model: nn.Module,
        loaded_set: DataLoader,
        loss_computer: Loss,
        output_dir: Path
):
    total_acc = 0.
    for i, (inputs, target) in tqdm(enumerate(loaded_set), total=len(loaded_set)):
        pred = model(inputs)
        loss = loss_computer(pred, target)
        acc = 1. - loss.item()
        total_acc += acc

    average_acc = total_acc / len(loaded_set)

    fout = open(output_dir / 'result.txt', 'w')
    fout.write(str(average_acc))
    fout.close()

    return average_acc


def run():
    weights = 'logs/train/best.pt'
    dataset = 'data/test.json'
    batch_size = 1

    model = Net(7)
    model.load_state_dict(torch.load(weights))
    data = AHPDataset(dataset)
    loaded_set = DataLoader(dataset=data, batch_size=batch_size)
    loss_computer = Loss()

    if not os.path.exists('logs'):
        os.mkdir('logs')
    output_dir = increment_path('logs/detect')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    accuracy = detect(model, loaded_set, loss_computer, output_dir)
    print(f'Average accuracy: {accuracy}')

    print(f'\033[92mResults have been saved to {output_dir}\033[0m')


if __name__ == '__main__':
    run()
