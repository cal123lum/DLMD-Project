import os, sys
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from src.augmentation import config_aug as C
from src.augmentation.model import Generator, Discriminator
from src.paths import GAN_GENERATOR_PTH



def gradient_penalty(D, real, fake):
    bs = real.size(0)
    eps = torch.rand(bs, 1)
    eps = eps.expand_as(real)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    d_interp = D(interp)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
    )[0]
    gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
    return gp

def load_data():
    arr = np.load(C.NPZ_PATH)['X'].astype(np.float32)
    scaled = StandardScaler().fit_transform(arr)
    return torch.from_numpy(scaled)

def main():
    os.makedirs(C.MODEL_OUT_DIR, exist_ok=True)
    print("Starting WGAN-GP training…")

    # Data
    X = load_data()
    loader = DataLoader(TensorDataset(X), batch_size=C.BATCH_SIZE, shuffle=True)
    print(f"  → {len(loader)} batches of size {C.BATCH_SIZE}")

    # Models
    G = Generator().to('cpu')
    D = Discriminator().to('cpu')

    # Optimizers (standard WGAN-GP settings)
    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
    n_critic = 5
    λ_gp = 10.0

    for epoch in range(1, C.EPOCHS + 1):
        tot_d, tot_g = 0.0, 0.0
        for i, (real_batch,) in enumerate(loader, 1):
            bs = real_batch.size(0)
            real = real_batch

            # —— Train Discriminator n_critic times ——
            for _ in range(n_critic):
                z = torch.randn(bs, C.LATENT_DIM)
                fake = G(z).detach()
                d_real = D(real).mean()
                d_fake = D(fake).mean()
                gp = gradient_penalty(D, real, fake)
                loss_D = d_fake - d_real + λ_gp * gp

                opt_D.zero_grad()
                loss_D.backward()
                opt_D.step()
                tot_d += loss_D.item()

            # —— Train Generator once ——
            z2 = torch.randn(bs, C.LATENT_DIM)
            loss_G = -D(G(z2)).mean()
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            tot_g += loss_G.item()

        avg_d = tot_d / (len(loader) * n_critic)
        avg_g = tot_g / len(loader)
        print(f"Epoch {epoch}/{C.EPOCHS}  D_loss={avg_d:.4f}  G_loss={avg_g:.4f}")

    # Save only the generator
    torch.save(G.state_dict(), str(GAN_GENERATOR_PTH))
    print("✅ WGAN-GP training complete – generator saved.")

if __name__ == "__main__":
    main()
