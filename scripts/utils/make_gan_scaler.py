#!/usr/bin/env python3
# scripts/utils/make_gan_scaler.py
# Fit and save a StandardScaler for GAN sampling, using the same rows as GAN training.

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.holdouts import SplitIndices


def load_x(npz_path: str) -> np.ndarray:
    z = np.load(npz_path, allow_pickle=True)
    return z["X"].astype(np.float32)


def read_train_indices(indices_json: Path) -> list[int]:
    d = json.loads(indices_json.read_text())

    # Simple forms we write in experiments:
    #  - {"train": [...]}  (no test)
    #  - {"train_only": [...]}
    #  - {"indices": [...]}
    if "train_only" in d:
        return list(map(int, d["train_only"]))
    if "train" in d and "test" not in d:
        return list(map(int, d["train"]))
    if "indices" in d:
        return list(map(int, d["indices"]))

    # Fallback: SplitIndices format (train/test keys etc.)
    split = SplitIndices.from_json(indices_json)
    return list(map(int, split.train))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices-json", required=True, help="JSON containing TRAIN indices")
    ap.add_argument("--out", required=True, help="where to write scaler.npz")
    ap.add_argument("--npz", required=True, help="dataset NPZ with X (and optionally y)")
    args = ap.parse_args()

    idx_path = Path(args.indices_json)
    out_path = Path(args.out)

    X = load_x(args.npz)
    train_idx = read_train_indices(idx_path)

    X_sub = X[train_idx]
    scaler = StandardScaler().fit(X_sub)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
    )

    print(f"[ok] wrote {out_path} (rows={len(train_idx)}, d={X.shape[1]})")


if __name__ == "__main__":
    main()
