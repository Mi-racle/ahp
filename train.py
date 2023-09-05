import torch
from torch.utils.data import DataLoader

from dataset import AHPDataset
from net import Net


def run():
    dataset = 'data/train.json'
    batch_size = 1
    epochs = 100
    lr = 1e-3

    model = Net(7)
    # model.load_state_dict(torch.load('best.pt'))
    data = AHPDataset(dataset)
    loaded_set = DataLoader(dataset=data, batch_size=batch_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99))

    for epoch in range(0, epochs):
        print(f'Epoch {epoch}:')
        model.train()

    print(f'\033[92mResults have saved to \033[0m')


def train():
    pass


if __name__ == '__main__':
    run()
