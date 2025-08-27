#!/usr/bin/env python3
import json, re
from pathlib import Path
import pandas as pd

METRICS = Path("data/processed/metrics")            # where eval writes JSON
OUTROOT = Path("data/processed/metrics/family")     # per-family dirs
OUTROOT.mkdir(parents=True, exist_ok=True)

rows=[]
for p in METRICS.glob("rf_family_metrics_*.json"):
    d=json.loads(p.read_text())
    tag=p.stem.replace("rf_family_metrics","").strip("_")

    # e.g. "wacatac_msmw_full_w0.01000_s42_r20000_gan"
    m=re.search(r"^(?P<name>.+?)_msmw_full_w(?P<frac>0?\.\d+)_s(?P<seed>\d+)_r(?P<const>\d+)_(?P<variant>real|gan|oversample|smote)$", tag)
    if not m:
        continue

    rows.append({
        "prefix": f"{m.group('name')}_msmw_full",
        "family": m.group("name"),
        "frac": float(m.group("frac")),
        "seed": int(m.group("seed")),
        "const_train_size": int(m.group("const")),
        "variant": m.group("variant"),
        "auc": d.get("auc"),
        "pr_auc": d.get("pr_auc"),
        "f1": d.get("f1"),
        "balanced_accuracy": d.get("balanced_accuracy"),
        "mcc": d.get("mcc"),
        "threshold": d.get("threshold"),
        "n_train_real": d.get("n_train_real"),
        "n_train_total": d.get("n_train_total"),
    })

df=pd.DataFrame(rows)
if df.empty:
    print("[warn] no rf_family_metrics_*.json found")
else:
    for fam, sub in df.groupby("family"):
        outdir=OUTROOT/fam
        outdir.mkdir(parents=True, exist_ok=True)
        sub.sort_values(["frac","variant","seed"]).to_csv(outdir/"raw.csv", index=False)
        print("[ok] wrote", outdir/"raw.csv")
