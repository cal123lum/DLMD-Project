# src/augmentation/config_aug.py
# author: Callum Musselwhite
# last edit: 2025-09-17

from src.paths import BODMAS_NPZ

# dims
LATENT_DIM = 100          # noise z size
FEATURE_DIM = 2381        # must match X.shape[1]
HIDDEN_DIM = 512          # MLP width for G/D

# training
BATCH_SIZE = 128
LR_G = 2e-4               # Adam LR (G)
LR_D = 2e-4               # Adam LR (D)
BETAS = (0.5, 0.999)
EPOCHS = 100

# paths
NPZ_PATH = str(BODMAS_NPZ)        # NPZ with X,y
MODEL_OUT_DIR = "models/augmentation"
