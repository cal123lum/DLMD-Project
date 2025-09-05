#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def metric_at(y, p, thr, which):
    yhat = (p >= thr).astype(int)
    tp = int(((y==1) & (yhat==1)).sum())
    tn = int(((y==0) & (yhat==0)).sum())
    fp = int(((y==0) & (yhat==1)).sum())
    fn = int(((y==1) & (yhat==0)).sum())
    if which == "f1":
        denom = (2*tp + fp + fn)
        return (2*tp / denom) if denom else 0.0
    if which == "balanced_accuracy":
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        return 0.5 * (sens + spec)
    if which == "mcc":
        num = tp*tn - fp*fn
        den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return (num / den) if den else 0.0
    raise ValueError(which)

def best_threshold(y, p, which, grid=2000):
    ts = np.linspace(0, 1, grid+1)[1:-1]
    vals = [metric_at(y, p, t, which) for t in ts]
    i = int(np.nanargmax(vals)) if vals else 0
    return float(ts[i]), float(vals[i])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw.csv from iid_scarcity")
    ap.add_argument("--preds-dir", required=True, help="dir with rf_temporal_preds_<tag>.npz")
    ap.add_argument("--out", required=True, help="output CSV with rethresholded metrics")
    ap.add_argument("--metrics", default="f1,balanced_accuracy,mcc",
                    help="comma list among: f1,balanced_accuracy,mcc")
    args = ap.parse_args()

    keep = [m.strip() for m in args.metrics.split(",") if m.strip()]
    df = pd.read_csv(args.raw)
    rows = []
    for _, r in df.iterrows():
        tag = r["tag"]
        npz = Path(args.preds_dir) / f"rf_temporal_preds_{tag}.npz"
        if not npz.exists():
            # try underscore quirk
            npz = Path(args.preds_dir) / f"rf_temporal_preds__{tag}.npz"
            if not npz.exists():
                continue
        z = np.load(npz, allow_pickle=True)
        y = z["y_true"].astype(int)
        p = z["proba"].astype(float)
        out = dict(r)
        for m in keep:
            thr, val = best_threshold(y, p, m)
            out[m] = float(val)           # overwrite with test-tuned value
            out[f"{m}_thr_test"] = float(thr)
        rows.append(out)

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[ok] wrote {args.out}")

if __name__ == "__main__":
    main()
