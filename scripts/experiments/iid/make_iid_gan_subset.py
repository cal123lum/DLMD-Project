#!/usr/bin/env python3
# scripts/experiments/iid/make_iid_gan_subset.py
# Subset maker for IID GAN training (malware-only)
# Author: Callum Musselwhite
# Last edit: 2025-09-17
# Purpose: take an IID split JSON and output a capped list of malware train indices for GAN training

import argparse, json
from pathlib import Path
import numpy as np
from src.paths import ROOT, BODMAS_NPZ  # ROOT kept for consistency

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-json", required=True, help="path to iid_split_seed*.json")
    ap.add_argument("--max-train-rows", type=int, default=20000, help="cap malware train rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True, help="where to write gan_subset_indices.json")
    ap.add_argument("--npz", type=str, default=str(BODMAS_NPZ))
    args = ap.parse_args()

    # load labels
    z = np.load(args.npz, allow_pickle=True)
    y = z["y"].astype(int)

    # read IID split and collect malware indices from TRAIN
    split = json.loads(Path(args.split_json).read_text())
    tr_idx = np.array(split["train"], dtype=int)
    pos_tr = tr_idx[y[tr_idx] == 1]

    # optionally cap with a reproducible sample
    rng = np.random.default_rng(args.seed)
    if len(pos_tr) > args.max_train_rows:
        pos_tr = rng.choice(pos_tr, size=args.max_train_rows, replace=False)

    # write subset as {"train": [...]} JSON
    out = {"train": pos_tr.tolist()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    print(f"[ok] wrote subset {out_path} (malware train rows={len(pos_tr)})")

if __name__ == "__main__":
    main()
