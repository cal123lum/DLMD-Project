# src/augmentation/config_aug.py
from src.paths import BODMAS_NPZ
# dimensions
LATENT_DIM     = 100            # size of Z vector
FEATURE_DIM    = 2381           # must match X.shape[1]
HIDDEN_DIM     = 512            # size of MLP hidden layers

# training params
BATCH_SIZE     = 128
LR_G = 2e-4           # learning rate, generator
LR_D           = 2e-4           # learning rate, discriminator
BETAS          = (0.5, 0.999)    # Adam betas
EPOCHS         = 100           

# file paths
NPZ_PATH       = str(BODMAS_NPZ)
MODEL_OUT_DIR  = 'models/augmentation'
