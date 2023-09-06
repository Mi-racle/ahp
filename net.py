import torch.nn as nn


class Net(nn.Module):

    def __init__(self, ch):
        super().__init__()
        neu_num0 = 128
        neu_num1 = 64
        self.linear0 = nn.Linear(ch, neu_num0)
        self.linear1 = nn.Linear(neu_num0, neu_num1)
        self.fc = nn.Linear(neu_num1, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear0(x)
        x = self.linear1(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
