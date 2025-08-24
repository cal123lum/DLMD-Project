#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.paths import ROOT, TEMPORAL_SPLIT
from src.holdouts import SplitIndices

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Path to write scaler .npz")
    args = ap.parse_args()

    z = np.load(ROOT/'data'/'raw'/'bodmas.npz', allow_pickle=True)
    X = z['X'].astype(np.float32); y = z['y'].astype(int)

    s = SplitIndices.from_json(TEMPORAL_SPLIT)
    Xtr, ytr = X[s.train], y[s.train]
    Xmal = Xtr[ytr==1]
    if Xmal.shape[0] == 0:
        raise SystemExit("[gan-scaler] No malware rows in train split; cannot fit scaler.")

    sc = StandardScaler().fit(Xmal)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(outp), mean_=sc.mean_, scale_=sc.scale_)
    print(f"[gan-scaler] wrote {outp} using {Xmal.shape[0]} malware rows")

if __name__ == "__main__":
    main()
