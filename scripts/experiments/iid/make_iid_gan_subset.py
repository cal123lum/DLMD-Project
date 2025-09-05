#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from src.paths import ROOT, BODMAS_NPZ

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-json", required=True, help="Path to iid_split_seed*.json")
    ap.add_argument("--max-train-rows", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(str(BODMAS_NPZ), allow_pickle=True)
    y = z["y"].astype(int)

    split = json.loads(Path(args.split_json).read_text())
    tr_idx = np.array(split["train"], dtype=int)

    pos_tr = tr_idx[y[tr_idx] == 1]
    rng = np.random.default_rng(args.seed)
    if len(pos_tr) > args.max_train_rows:
        pos_tr = rng.choice(pos_tr, size=args.max_train_rows, replace=False)

    out = {"train": pos_tr.tolist()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out))
    print(f"[ok] wrote subset {out_path} (malware train rows={len(pos_tr)})")

if __name__ == "__main__":
    main()
