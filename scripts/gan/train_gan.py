import os
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.augmentation import config_aug as C
from src.augmentation.model import Generator, Discriminator
from src.paths import GAN_GENERATOR_PTH
from src.holdouts import SplitIndices
from pathlib import Path


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


def load_xy():
    z = np.load(C.NPZ_PATH, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.int32) if "y" in z else None
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices-json", type=str, default=None,
                    help="Use only TRAIN indices from this SplitIndices JSON.")
    ap.add_argument("--malware-only", action="store_true", default=False,
                    help="If set, train GAN on y==1 only.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output path for the generator .pth (overrides src.paths.GAN_GENERATOR_PTH).")
    args = ap.parse_args()

    os.makedirs(C.MODEL_OUT_DIR, exist_ok=True)
    print("Starting WGAN-GP training…")

    # Data (raw)
    X_all, y_all = load_xy()

    # Restrict to split TRAIN if provided
    if args.indices_json:
        split = SplitIndices.from_json(Path(args.indices_json))
        X_all = X_all[split.train]
        y_all = y_all[split.train] if y_all is not None else None
        print(f"[gan] restricted to TRAIN: {X_all.shape[0]} rows")

    # Malware-only, if requested
    if args.malware_only:
        if y_all is None:
            raise ValueError("--malware-only requires labels y in the NPZ")
        keep = (y_all == 1)
        X_all = X_all[keep]
        y_all = y_all[keep]
        print(f"[gan] malware-only: {X_all.shape[0]} rows")

    # Scale (fit only on the selected training data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all.astype(np.float32))
    X_tensor = torch.from_numpy(X_scaled)

    # DataLoader
    loader = DataLoader(TensorDataset(X_tensor), batch_size=C.BATCH_SIZE, shuffle=True)
    print(f"  → {len(loader)} batches of size {C.BATCH_SIZE}")

    # Models
    G = Generator().to("cpu")
    D = Discriminator().to("cpu")

    # Optimizers (standard WGAN-GP settings)
    opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))
    n_critic = 5
    lambda_gp = 10.0

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
                loss_D = d_fake - d_real + lambda_gp * gp

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

    # Save only the generator (to specified path or default)
    save_path = Path(args.out) if args.out else GAN_GENERATOR_PTH
    torch.save(G.state_dict(), str(save_path))
    print(f"✅ WGAN-GP training complete – generator saved to: {save_path}")


if __name__ == "__main__":
    main()
