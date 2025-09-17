# src/augmentation/model.py
# last edit: 2025-09-17
# author: Callum Musselwhite

import torch.nn as nn
from src.augmentation import config_aug as C

class Generator(nn.Module):
    """Simple MLP: z (LATENT_DIM) → x (FEATURE_DIM)."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C.LATENT_DIM, C.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(C.HIDDEN_DIM, C.HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(C.HIDDEN_DIM, C.FEATURE_DIM),  # no output activation
        )

    def forward(self, z):
        # z: [batch, LATENT_DIM] → [batch, FEATURE_DIM]
        return self.net(z)

class Discriminator(nn.Module):
    """Score real/fake on feature vectors; raw (WGAN-GP) score out."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(C.FEATURE_DIM, C.HIDDEN_DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(C.HIDDEN_DIM, C.HIDDEN_DIM),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(C.HIDDEN_DIM, 1),  # raw score (no sigmoid)
        )

    def forward(self, x):
        # x: [batch, FEATURE_DIM] → [batch]
        return self.net(x).view(-1)
