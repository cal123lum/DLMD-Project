#!/usr/bin/env python
import os
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

from src.augmentation import config_aug as C
from src.augmentation.model import Generator, Discriminator
from src.paths import GAN_GENERATOR_PTH
from src.holdouts import SplitIndices


def gradient_penalty(D, real, fake, lambda_gp: float):
    """
    WGAN-GP gradient penalty on the interpolation between real and fake.
    Assumes D returns a (batch, 1) or (batch,) tensor.
    """
    bs = real.size(0)
    device = real.device
    eps = torch.rand(bs, 1, device=device, dtype=real.dtype)
    eps = eps.expand_as(real)
    interp = (eps * real + (1.0 - eps) * fake).requires_grad_(True)
    d_interp = D(interp)
    if d_interp.dim() > 1:
        d_interp = d_interp.squeeze(-1)
    grad_outputs = torch.ones_like(d_interp, device=device, dtype=real.dtype)
    grads = torch.autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gp = ((grads.norm(2, dim=1) - 1.0) ** 2).mean()
    return lambda_gp * gp


def load_xy():
    z = np.load(C.NPZ_PATH, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.int32) if "y" in z else None
    return X, y


def pick_device(pref: str) -> torch.device:
    pref = (pref or "auto").lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(pref)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices-json", type=str, default=None,
                    help="Use only TRAIN indices from this SplitIndices JSON.")
    ap.add_argument("--malware-only", action="store_true", default=False,
                    help="If set, train GAN on y==1 only.")
    ap.add_argument("--out", type=str, default=None,
                    help="Output path for the generator .pth (overrides src.paths.GAN_GENERATOR_PTH).")

    # NEW: training knobs (with sensible defaults; fall back to C if present)
    ap.add_argument("--epochs", type=int, default=getattr(C, "EPOCHS", 100))
    ap.add_argument("--batch-size", type=int, default=getattr(C, "BATCH_SIZE", 128))
    ap.add_argument("--lr", type=float, default=1e-4,
                    help="Adam LR for both G and D (default 1e-4)")
    ap.add_argument("--n-critic", type=int, default=5,
                    help="Number of D steps per G step (default 5)")
    ap.add_argument("--lambda-gp", type=float, default=10.0,
                    help="Gradient penalty coefficient (default 10)")
    ap.add_argument("--device", type=str, default="auto",
                    help="auto|cpu|cuda|mps (default auto)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Repro-ish
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device choice
    device = pick_device(args.device)

    os.makedirs(C.MODEL_OUT_DIR, exist_ok=True)
    print(f"Starting WGAN-GP training… "
          f"epochs={args.epochs} batch_size={args.batch_size} lr={args.lr} "
          f"n_critic={args.n_critic} lambda_gp={args.lambda_gp} device={device}")

    # Data (raw)
    X_all, y_all = load_xy()

    # Restrict to split TRAIN if provided
    if args.indices_json:
        p = Path(args.indices_json)
        try:
            d = json.loads(p.read_text())
            # handle capped-subset JSONs we write
            if "train_only" in d:
                idx = list(map(int, d["train_only"]))
            elif "train" in d and "test" not in d:
                idx = list(map(int, d["train"]))
            elif "indices" in d:
                idx = list(map(int, d["indices"]))
            else:
                raise KeyError("not a simple indices json")

            X_all = X_all[idx]
            y_all = y_all[idx] if y_all is not None else None
            print(f"[gan] restricted to custom TRAIN indices: {len(idx)} rows")
        except Exception:
            from src.holdouts import SplitIndices
            split = SplitIndices.from_json(p)
            X_all = X_all[split.train]
            y_all = y_all[split.train] if y_all is not None else None
            print(f"[gan] restricted to SplitIndices TRAIN: {len(split.train)} rows")

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
    X_tensor = torch.from_numpy(X_scaled)  # keep on CPU; move per-batch to device

    # DataLoader
    loader = DataLoader(
        TensorDataset(X_tensor),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,          # helps batch-norm-ish stability even if not using BN
        num_workers=0,           # 0 is usually best on macOS; bump on Linux if you like
        pin_memory=(device.type in {"cuda", "mps"}),
    )
    print(f"  → {len(loader)} batches of size {args.batch_size} "
          f"(dataset={len(X_tensor):,} rows)")

    # Models
    latent_dim = getattr(C, "LATENT_DIM", 128)
    G = Generator().to(device)
    D = Discriminator().to(device)

    # Optimizers (standard WGAN-GP settings)
    opt_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.0, 0.9))

    for epoch in range(1, args.epochs + 1):
        tot_d, tot_g, n_batches = 0.0, 0.0, 0
        for (real_batch,) in loader:
            n_batches += 1
            bs = real_batch.size(0)
            real = real_batch.to(device, non_blocking=True)

            # —— Train Discriminator n_critic times ——
            for _ in range(args.n_critic):
                z = torch.randn(bs, latent_dim, device=device)
                with torch.no_grad():
                    fake = G(z)
                d_real = D(real).mean()
                d_fake = D(fake).mean()
                gp = gradient_penalty(D, real, fake, lambda_gp=args.lambda_gp)
                loss_D = d_fake - d_real + gp

                opt_D.zero_grad(set_to_none=True)
                loss_D.backward()
                opt_D.step()
                tot_d += loss_D.item()

            # —— Train Generator once ——
            z2 = torch.randn(bs, latent_dim, device=device)
            loss_G = -D(G(z2)).mean()
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            opt_G.step()
            tot_g += loss_G.item()

        avg_d = tot_d / max(1, n_batches * args.n_critic)
        avg_g = tot_g / max(1, n_batches)
        print(f"Epoch {epoch}/{args.epochs}  D_loss={avg_d:.4f}  G_loss={avg_g:.4f}")

    # Save only the generator (to specified path or default)
    save_path = Path(args.out) if args.out else GAN_GENERATOR_PTH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(G.state_dict(), str(save_path))
    print(f"✅ WGAN-GP training complete – generator saved to: {save_path}")


if __name__ == "__main__":
    main()
