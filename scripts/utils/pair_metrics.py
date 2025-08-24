#!/usr/bin/env python
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="in_csv",  required=True, help="raw.csv from sweep")
    ap.add_argument("--out", dest="out_csv", required=True, help="paired.csv to write")
    args = ap.parse_args()

    src = Path(args.in_csv)
    if not src.exists():
        raise SystemExit(f"[pair] missing raw csv: {src}")

    df = pd.read_csv(src)

    # robust typing
    if df["used_gan"].dtype != bool:
        df["used_gan"] = df["used_gan"].astype(str).str.strip().str.lower().isin(["true","1","yes"])
    df["frac"] = pd.to_numeric(df["frac"], errors="coerce")
    df["const_train_size"] = pd.to_numeric(df["const_train_size"], errors="coerce")
    df["run_idx"] = np.arange(len(df))

    metrics = ["auc","pr_auc","f1","balanced_accuracy","mcc","precision","recall","accuracy","specificity"]
    other   = ["n_train_real","n_train_synth","n_train_total","tn","fp","fn","tp"]

    base = (df[df["used_gan"]==False].sort_values("run_idx")
            .groupby(["prefix","kind","frac"], as_index=False).tail(1)
            .rename(columns={c:f"{c}_real" for c in metrics+other}))

    aug  = (df[df["used_gan"]==True].sort_values("run_idx")
            .groupby(["prefix","kind","frac","const_train_size"], as_index=False).tail(1)
            .rename(columns={c:f"{c}_aug" for c in metrics+other}))

    keep = ["prefix","kind","frac"]+[f"{c}_real" for c in metrics+other]
    paired = aug.merge(base[keep], on=["prefix","kind","frac"], how="left")

    for m in ["auc","pr_auc","f1","balanced_accuracy","mcc"]:
        paired[f"delta_{m}"] = paired[f"{m}_aug"] - paired[f"{m}_real"]

    cols = ["prefix","kind","frac","const_train_size"] \
         + [f"{m}_real" for m in ["auc","pr_auc","f1","balanced_accuracy","mcc"]] \
         + [f"{m}_aug"  for m in ["auc","pr_auc","f1","balanced_accuracy","mcc"]] \
         + [f"delta_{m}" for m in ["auc","pr_auc","f1","balanced_accuracy","mcc"]]

    paired = paired[cols].sort_values(["frac","const_train_size"])
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(args.out_csv, index=False)
    print(f"[paired] wrote {args.out_csv} ({len(paired)} rows)")

if __name__ == "__main__":
    main()
