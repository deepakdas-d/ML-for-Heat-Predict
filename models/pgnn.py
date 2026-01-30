import torch
import torch.nn as nn

class HeatSinkPGNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(7, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

        self.scale = 20.0  # hard physical constraint

    def forward(self, x):
        return self.scale * self.net(x)
