import torch.nn as nn
from src.augmentation import config_aug as C

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C.LATENT_DIM, C.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(C.HIDDEN_DIM, C.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(C.HIDDEN_DIM, C.FEATURE_DIM),
            # no activation
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C.FEATURE_DIM, C.HIDDEN_DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(C.HIDDEN_DIM, C.HIDDEN_DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(C.HIDDEN_DIM, 1),
            # raw score (no sigmoid)
        )

    def forward(self, x):
        return self.net(x).view(-1)
