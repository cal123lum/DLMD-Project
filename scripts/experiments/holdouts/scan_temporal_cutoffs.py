#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

from src.data.metadata import load_metadata
from src.paths import ROOT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default="2016-01-01")
    ap.add_argument("--end",   type=str, default="2020-01-01")
    ap.add_argument("--freq",  type=str, default="MS", help="pandas freq (e.g., MS=monthly, QS=quarterly)")
    ap.add_argument("--min-train-pos", type=int, default=500, help="minimum malware (y=1) required in train")
    ap.add_argument("--min-train-pos-rate", type=float, default=0.05, help="min positive rate in train")
    ap.add_argument("--out", type=str, default=str(ROOT / "data" / "processed" / "metrics" / "temporal_scan.csv"))
    args = ap.parse_args()

    meta = load_metadata()
    y = np.load(ROOT / "data" / "raw" / "bodmas.npz", allow_pickle=True)["y"]

    dates = pd.date_range(args.start, args.end, freq=args.freq, tz="UTC")
    rows = []
    for c in dates:
        mask_train = (meta["timestamp"] <= c)
        ytr = y[mask_train.values]
        yte = y[~mask_train.values]

        tr_u, tr_c = np.unique(ytr, return_counts=True) if len(ytr) else (np.array([]), np.array([]))
        te_u, te_c = np.unique(yte, return_counts=True) if len(yte) else (np.array([]), np.array([]))
        tr = dict(zip(tr_u.astype(int).tolist(), tr_c.tolist()))
        te = dict(zip(te_u.astype(int).tolist(), te_c.tolist()))
        tr_n = int(tr.get(0,0) + tr.get(1,0))
        te_n = int(te.get(0,0) + te.get(1,0))
        tr_pos = int(tr.get(1,0)); te_pos = int(te.get(1,0))
        tr_rate = (tr_pos / tr_n) if tr_n else 0.0
        te_rate = (te_pos / te_n) if te_n else 0.0

        rows.append({
            "cutoff": c.isoformat(),
            "train_n": tr_n, "train_pos": tr_pos, "train_pos_rate": round(tr_rate, 4),
            "test_n": te_n,  "test_pos": te_pos,  "test_pos_rate":  round(te_rate, 4),
        })

    df = pd.DataFrame(rows)
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)
    print(df.to_string(index=False))

    # Suggest the first cutoff meeting thresholds
    ok = df[(df.train_pos >= args.min_train_pos) & (df.train_pos_rate >= args.min_train_pos_rate)]
    if len(ok):
        print("\n[scan] Suggested cutoff (first meeting thresholds):", ok.iloc[0]["cutoff"])
    else:
        print("\n[scan] No cutoff met thresholds. Consider lowering --min-train-pos or --min-train-pos-rate.")

if __name__ == "__main__":
    main()
