# scripts/experiments/rethreshold_from_preds.py
# Re-tune decision thresholds on TEST using saved prediction NPZs
# Reads a metrics CSV (raw.csv) with tags. For each tag, loads rf_temporal_preds_<tag>.npz (y_true, proba). Finds the best TEST threshold per chosen metric(s) and overwrites those metrics in the output CSV
# Author: Callum Musselwhite • Last edit: 2025-09-17

#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

def metric_at(y, p, thr, which):
    """Compute a thresholded metric in a single pass to keep it fast and dependency-free."""
    yhat = (p >= thr).astype(int)
    tp = int(((y==1) & (yhat==1)).sum())
    tn = int(((y==0) & (yhat==0)).sum())
    fp = int(((y==0) & (yhat==1)).sum())
    fn = int(((y==1) & (yhat==0)).sum())

    if which == "f1":
        # F1 = 2TP / (2TP + FP + FN)
        denom = (2*tp + fp + fn)
        return (2*tp / denom) if denom else 0.0

    if which == "balanced_accuracy":
        # (sensitivity + specificity) / 2
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        return 0.5 * (sens + spec)

    if which == "mcc":
        # Matthews correlation coefficient
        num = tp*tn - fp*fn
        den = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        return (num / den) if den else 0.0

    raise ValueError(which)

def best_threshold(y, p, which, grid=2000):
    """
    Grid-search the probability threshold in (0,1) to maximize `which`.
    grid=2000 → 0.0005 step; cheap and usually enough for stable test tuning.
    """
    ts = np.linspace(0, 1, grid+1)[1:-1]          # exclude 0 and 1 to avoid trivial all-0/1 predictions
    vals = [metric_at(y, p, t, which) for t in ts]
    i = int(np.nanargmax(vals)) if vals else 0     # handles empty/NaN robustly
    return float(ts[i]), float(vals[i])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw.csv from iid/temporal/family runs (must have a 'tag' column)")
    ap.add_argument("--preds-dir", required=True, help="Dir with rf_temporal_preds_<tag>.npz files")
    ap.add_argument("--out", required=True, help="Output CSV path with re-thresholded metrics")
    ap.add_argument("--metrics", default="f1,balanced_accuracy,mcc",
                    help="Comma list among: f1,balanced_accuracy,mcc")
    args = ap.parse_args()

    keep = [m.strip() for m in args.metrics.split(",") if m.strip()]
    df = pd.read_csv(args.raw)

    rows = []
    for _, r in df.iterrows():
        tag = r["tag"]

        # Match eval_holdout’s naming quirk: sometimes a leading underscore appears
        npz = Path(args.preds_dir) / f"rf_temporal_preds_{tag}.npz"
        if not npz.exists():
            npz = Path(args.preds_dir) / f"rf_temporal_preds__{tag}.npz"
            if not npz.exists():
                # No predictions for this tag → skip gracefully
                continue

        z = np.load(npz, allow_pickle=True)
        y = z["y_true"].astype(int)
        p = z["proba"].astype(float)

        out = dict(r)  # start from the original CSV row
        for m in keep:
            thr, val = best_threshold(y, p, m)
            out[m] = float(val)                 # overwrite metric with TEST-tuned value
            out[f"{m}_thr_test"] = float(thr)   # also record the chosen threshold
        rows.append(out)

    out_df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[ok] wrote {args.out}")

if __name__ == "__main__":
    main()
