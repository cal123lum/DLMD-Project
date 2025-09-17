#!/usr/bin/env python3
# scripts/utils/make_gan_scaler.py
# Author: Callum Musselwhite
# Last edit: 2025-09-17
# Putpose: Fit a StandardScaler on malware rows of TRAIN and save to .npz (for GAN sampling/eval).

import argparse, json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.paths import ROOT, TEMPORAL_SPLIT
from src.holdouts import SplitIndices

def load_train_indices(json_path: Path):
    """
    Accept:
      {"train_only":[...]}  or  {"indices":[...]} / {"train":[...]}
      or a full SplitIndices JSON with train/test.
    """
    d = json.loads(json_path.read_text())
    if "train_only" in d: return list(map(int, d["train_only"]))
    if "indices" in d:    return list(map(int, d["indices"]))
    if "train" in d:      return list(map(int, d["train"]))
    # fallback: full SplitIndices structure
    s = SplitIndices.from_json(json_path)
    return list(map(int, s.train))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Path to write scaler .npz")
    ap.add_argument("--indices-json", type=str, default=None,
                    help="TRAIN selection JSON (subset or SplitIndices).")
    args = ap.parse_args()

    # load full dataset once
    z = np.load(ROOT / "data" / "raw" / "bodmas.npz", allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(int)

    # choose TRAIN rows: explicit indices json or default temporal split
    if args.indices_json:
        idx = load_train_indices(Path(args.indices_json))
    else:
        s = SplitIndices.from_json(TEMPORAL_SPLIT)
        idx = list(map(int, s.train))

    # restrict to TRAIN and then malware only (y==1)
    Xtr, ytr = X[idx], y[idx]
    Xmal = Xtr[ytr == 1]
    if Xmal.shape[0] == 0:
        raise SystemExit("[gan-scaler] No malware rows in TRAIN; cannot fit scaler.")

    # fit scaler on malware distribution
    sc = StandardScaler().fit(Xmal)

    # save light-weight params for fast reuse
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, mean_=sc.mean_.astype(np.float32), scale_=sc.scale_.astype(np.float32))
    print(f"[gan-scaler] wrote {outp} using {Xmal.shape[0]} malware rows")

if __name__ == "__main__":
    main()
