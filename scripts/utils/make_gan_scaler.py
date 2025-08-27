#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.paths import ROOT, TEMPORAL_SPLIT
from src.holdouts import SplitIndices

def load_train_indices(json_path: Path):
    """
    Accept either:
      - our capped-subset JSON: {"train_only":[...]}
      - a simple indices JSON:  {"indices":[...]} or {"train":[...]}
      - a full SplitIndices JSON: {"train":[...], "test":[...]}
    """
    d = json.loads(json_path.read_text())
    if "train_only" in d:
        return list(map(int, d["train_only"]))
    if "indices" in d:
        return list(map(int, d["indices"]))
    if "train" in d:
        return list(map(int, d["train"]))
    # Fallback: try full SplitIndices object
    s = SplitIndices.from_json(json_path)
    return list(map(int, s.train))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Path to write scaler .npz")
    ap.add_argument("--indices-json", type=str, default=None,
                    help="JSON deciding which TRAIN rows to fit the scaler on.")
    args = ap.parse_args()

    z = np.load(ROOT / "data" / "raw" / "bodmas.npz", allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(int)

    if args.indices_json:
        idx = load_train_indices(Path(args.indices_json))
    else:
        # backwards compat: use the default temporal TRAIN split
        s = SplitIndices.from_json(TEMPORAL_SPLIT)
        idx = list(map(int, s.train))

    Xtr, ytr = X[idx], y[idx]
    Xmal = Xtr[ytr == 1]
    if Xmal.shape[0] == 0:
        raise SystemExit("[gan-scaler] No malware rows in selected TRAIN; cannot fit scaler.")

    sc = StandardScaler().fit(Xmal)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    np.savez(outp, mean_=sc.mean_.astype(np.float32), scale_=sc.scale_.astype(np.float32))
    print(f"[gan-scaler] wrote {outp} using {Xmal.shape[0]} malware rows")

if __name__ == "__main__":
    main()
