# src/augmentation/verify_generator.py

import torch
import numpy as np
from src.augmentation.model import Generator
from src.augmentation import config_aug as C

def main():
    # 1) Instantiate the generator and load weights
    G = Generator().to('cpu')
    state = torch.load(f"{C.MODEL_OUT_DIR}/generator.pth", map_location='cpu')
    G.load_state_dict(state)
    G.eval()
    print("✔ Loaded generator.pth successfully")

    # 2) Sample a small batch of latent vectors
    z = torch.randn(5, C.LATENT_DIM)           # 5 random noise vectors
    with torch.no_grad():
        fake = G(z).numpy()                    # shape (5, FEATURE_DIM)

    # 3) Quick checks
    print("Generated synthetic batch shape:", fake.shape)
    print(f"Feature-wise stats over all 5 samples:")
    print("  mean =", np.mean(fake))
    print("   std =", np.std(fake))
    print("   min =", np.min(fake))
    print("   max =", np.max(fake))

    # 4) Ensure no NaNs or infinities
    print("Any NaNs? ", np.isnan(fake).any())
    print("Any infinities? ", np.isinf(fake).any())

if __name__ == '__main__':
    main()
