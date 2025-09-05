#!/usr/bin/env python3
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
    args = ap.parse_args()

    z = np.load(str(BODMAS_NPZ), allow_pickle=True)
    y = z["y"].astype(int)
    n = y.shape[0]

    idx = np.arange(n, dtype=int)
    tr, te = train_test_split(idx, test_size=args.test_frac, stratify=y, random_state=args.seed)

    out_path = Path(args.out) if args.out else (ROOT / "data" / "holdouts" / f"iid_split_seed{args.seed}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"train": tr.tolist(), "test": te.tolist()}))
    print(f"[ok] wrote {out_path}  (train={len(tr)}, test={len(te)})")

if __name__ == "__main__":
    main()
