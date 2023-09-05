import os

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import AHPDataset
from logger import Logger
from loss import Loss
from net import Net
from utils import increment_path, log_epoch


def train(
        model: nn.Module,
        loaded_set: DataLoader,
        loss_computer: Loss,
        optimizer: Optimizer
):
    total_loss = 0

    for i, (inputs, target) in tqdm(enumerate(loaded_set), total=len(loaded_set)):
        pred = model(inputs)
        loss = loss_computer(pred, target)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(loaded_set)

    return average_loss


def run():
    dataset = 'data/train.json'
    batch_size = 1
    epochs = 300
    lr = 1e-3

    model = Net(7)
    data = AHPDataset(dataset)
    loaded_set = DataLoader(dataset=data, batch_size=batch_size)
    loss_computer = Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    if not os.path.exists('logs'):
        os.mkdir('logs')
    output_dir = increment_path('logs/train')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    logger = Logger(output_dir)

    best_loss = float('inf')
    for epoch in range(0, epochs):
        print(f'Epoch {epoch}:')
        model.train()
        loss = train(model, loaded_set, loss_computer, optimizer)
        log_epoch(logger, epoch, model, loss, best_loss)
        best_loss = min(loss, best_loss)

    print(f'\033[92mResults have been saved to {output_dir}\033[0m')


if __name__ == '__main__':
    run()
