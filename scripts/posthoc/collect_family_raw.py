#!/usr/bin/env python3
# scripts/posthoc/collect_fmaily_raw.py
# Author: Callum Musselwhite
# Last edit: 2025-09-17
# Purpose: sweep family-holdout metrics JSONs and materialize per-family CSVs
#          (one raw.csv per family under data/processed/metrics/family/<family>/).
# Notes:
#   - Expects files named like: data/processed/metrics/rf_family_metrics_<TAG>.json
#   - <TAG> is parsed with a strict regex that matches the LOFO tags you used
#     (e.g., "wacatac_msmw_full_w0.01000_s42_r20000_gan").
#   - This script is intentionally minimal and *only* post-processes results.

import json, re
from pathlib import Path
import pandas as pd

# Where eval_holdout writes metrics JSON files
METRICS = Path("data/processed/metrics")

# Output root: per-family subdirs will be created here (…/family/<family>/raw.csv)
OUTROOT = Path("data/processed/metrics/family")
OUTROOT.mkdir(parents=True, exist_ok=True)

rows = []

# Iterate over all family metrics files produced by eval_holdout
for p in METRICS.glob("rf_family_metrics_*.json"):
    d = json.loads(p.read_text())

    # Recover the original tag suffix from the filename (drop the leading prefix)
    # Some runs may have a leading underscore; we strip it out below.
    tag = p.stem.replace("rf_family_metrics", "").strip("_")

    # Tag pattern:
    #   <name>_msmw_full_w<frac>_s<seed>_r<const>_<variant>
    # where:
    #   <name>    = family name (e.g., wacatac)
    #   <frac>    = real fraction used (e.g., 0.01000)
    #   <seed>    = RNG seed (int)
    #   <const>   = const_train_size (int, e.g., 20000)
    #   <variant> = one of {real, gan, oversample, smote}
    m = re.search(
        r"^(?P<name>.+?)_msmw_full_w(?P<frac>0?\.\d+)_s(?P<seed>\d+)_r(?P<const>\d+)_(?P<variant>real|gan|oversample|smote)$",
        tag,
    )
    if not m:
        # If the tag doesn't match the exact scheme above, skip it silently.
        # (Keeps this script aligned with the runs that used this naming.)
        continue

    # Build one tidy row per JSON → will later split per-family to raw.csv
    rows.append({
        "prefix": f"{m.group('name')}_msmw_full",   # stable prefix used across plots
        "family": m.group("name"),
        "frac": float(m.group("frac")),
        "seed": int(m.group("seed")),
        "const_train_size": int(m.group("const")),
        "variant": m.group("variant"),
        # Core metrics (copied verbatim from the JSON payload)
        "auc": d.get("auc"),
        "pr_auc": d.get("pr_auc"),
        "f1": d.get("f1"),
        "balanced_accuracy": d.get("balanced_accuracy"),
        "mcc": d.get("mcc"),
        "threshold": d.get("threshold"),
        # Useful counts for analysis/auditing
        "n_train_real": d.get("n_train_real"),
        "n_train_total": d.get("n_train_total"),
    })

# Materialize one CSV per family for easy plotting/aggregation
df = pd.DataFrame(rows)
if df.empty:
    print("[warn] no rf_family_metrics_*.json found")
else:
    # Each family gets data/processed/metrics/family/<family>/raw.csv
    for fam, sub in df.groupby("family", dropna=False):
        outdir = OUTROOT / str(fam)
        outdir.mkdir(parents=True, exist_ok=True)

        # Sort for reproducibility: by fraction, then variant, then seed
        sub.sort_values(["frac", "variant", "seed"]).to_csv(outdir / "raw.csv", index=False)
        print("[ok] wrote", outdir / "raw.csv")
