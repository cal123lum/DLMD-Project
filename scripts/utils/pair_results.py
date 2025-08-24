#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="e.g. cut2019_09 or cut2019_09_cv60d")
    ap.add_argument("--metrics-root", default="data/processed/metrics/temporal")
    ap.add_argument("--raw", default=None, help="optional path to raw.csv")
    ap.add_argument("--paired-out", default=None, help="optional path to paired.csv")
    ap.add_argument("--paired-cv-out", default=None, help="optional path to paired_cv.csv")
    args = ap.parse_args()

    root   = Path(args.metrics_root)
    outdir = root / args.prefix
    raw_p  = Path(args.raw) if args.raw else (outdir / "raw.csv")
    paired_p    = Path(args.paired_out) if args.paired_out else (outdir / "paired.csv")
    paired_cv_p = Path(args.paired_cv_out) if args.paired_cv_out else (outdir / "paired_cv.csv")

    if not raw_p.exists():
        raise SystemExit(f"[pair] missing raw csv: {raw_p}")

    df = pd.read_csv(raw_p)

    # ---- normalize dtypes
    if "used_gan" in df.columns and df["used_gan"].dtype != bool:
        df["used_gan"] = df["used_gan"].astype(str).str.lower().isin(["true","1","yes"])
    for c in ("frac","const_train_size"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "seed" not in df.columns:
        df["seed"] = 0

    # Optional window columns – include them if present
    window_cols = [c for c in ("test_start","test_end","test_window_days") if c in df.columns]

    keys_base = ["prefix","kind","frac","seed"] + window_cols
    keys_aug  = keys_base + ["const_train_size"]

    core  = ["auc","pr_auc","f1","balanced_accuracy","mcc"]
    other = ["precision","recall","accuracy","specificity",
             "n_train_real","n_train_synth","n_train_total","tn","fp","fn","tp","threshold"]

    df["run_idx"] = np.arange(len(df))

    base = (df[(~df["used_gan"])]
              .sort_values("run_idx")
              .groupby(keys_base, as_index=False).tail(1)
              .rename(columns={c:f"{c}_real" for c in core+other}))

    aug  = (df[(df["used_gan"])]
              .sort_values("run_idx")
              .groupby(keys_aug, as_index=False).tail(1)
              .rename(columns={c:f"{c}_aug" for c in core+other}))

    keep   = keys_base + [f"{c}_real" for c in core+other]
    paired = aug.merge(base[keep], on=keys_base, how="left")

    for m in core:
        paired[f"delta_{m}"] = paired[f"{m}_aug"] - paired[f"{m}_real"]

    cols = (keys_aug
            + [f"{m}_real" for m in core]
            + [f"{m}_aug"  for m in core]
            + [f"delta_{m}" for m in core])

    paired = paired[cols].sort_values(["frac","const_train_size","seed"] + window_cols)
    outdir.mkdir(parents=True, exist_ok=True)
    paired.to_csv(paired_p, index=False)
    print(f"[pair] wrote {paired_p} ({len(paired)} rows)")

    # ---- aggregate across seeds (and window repeats if present)
    group_keys = [k for k in keys_aug if k != "seed"]
    metrics = [c for c in paired.columns if c.endswith("_real") or c.endswith("_aug") or c.startswith("delta_")]
    g = paired.groupby(group_keys)
    mean = g[metrics].mean().add_suffix("_mean")
    std  = g[metrics].std(ddof=1).add_suffix("_std")
    cv = pd.concat([mean, std], axis=1).reset_index()
    cv.to_csv(paired_cv_p, index=False)
    print(f"[pair] wrote {paired_cv_p} (groups={len(cv)})")

if __name__ == "__main__":
    main()
