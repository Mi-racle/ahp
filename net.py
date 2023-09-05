import torch.nn as nn


class Net(nn.Module):

    def __init__(self, ch):
        super().__init__()
        self.linear = nn.Linear(ch, 32)
        self.fc = nn.Linear(32, 3)

    def forward(self, x):
        x = self.linear(x)
        x = self.fc(x)
        return x
