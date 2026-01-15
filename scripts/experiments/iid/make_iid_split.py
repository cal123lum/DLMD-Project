#!/usr/bin/env python3
# scripts/experiments/iid/make_iid_split.py
# IID train/test split creator
# Author: Callum Musselwhite
# Last edit: 2025-09-17
# Purpose: stratified IID split over BODMAS labels and write indices to JSON

import argparse, json, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

# repo root helpers
from src.paths import ROOT, BODMAS_NPZ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--npz", type=str, default=str(BODMAS_NPZ))
    args = ap.parse_args()

    # load labels and make an IID stratified split
    z = np.load(args.npz, allow_pickle=True)
    y = z["y"].astype(int)
    n = y.shape[0]

    idx = np.arange(n, dtype=int)
    tr, te = train_test_split(idx, test_size=args.test_frac, stratify=y, random_state=args.seed)

    # default output path mirrors prior convention if --out is not provided
    out_path = Path(args.out) if args.out else (ROOT / "data" / "holdouts" / f"iid_split_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"train": tr.tolist(), "test": te.tolist()}))
    print(f"[ok] wrote {out_path}  (train={len(tr)}, test={len(te)})")


if __name__ == "__main__":
    main()
