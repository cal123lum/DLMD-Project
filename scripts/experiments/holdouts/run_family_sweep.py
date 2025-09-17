#!/usr/bin/env python3
# scripts/experiments/holdouts/run_family_sweep.py
# Batch runner for family LOFO GAN training and evaluations
# Author: Callum Musselwhite
# Last edit: 2025-09-17
# Purpose: for each family holdout, build a capped malware subset, train or reuse a GAN, fit a scaler, then evaluate across scarcity × seeds × methods

import argparse, json, subprocess, sys, shlex
from pathlib import Path
import pandas as pd
import numpy as np
from src.paths import ROOT

# directories
SPLITS_DIR = ROOT / "data" / "processed" / "splits" / "family_lofo"   # where holdout_*.json live
GAN_DIR = ROOT / "models" / "gan" / "family_final"                    # per-family generator + scaler
MET_FAM = ROOT / "data" / "processed" / "metrics" / "family_final"    # per-family metrics

# scarcity levels and robustness seeds
FRACS = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005,
         0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01]
SEEDS = [42, 1337, 2025]

# train-time constraints
CONST_TRAIN_SIZE = 20000   # target size after augmentation
MIN_TRAIN_POS = 50         # ensure some positives before augment

def run(cmd):
    # echo and execute a subprocess command
    print("[run]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)

def load_metric_json(tag: str):
    # eval_holdout writes rf_{kind}_metrics_{tag}.json but older runs sometimes had a double underscore
    base = MET_FAM
    p1 = base / f"rf_family_metrics_{tag}.json"
    p2 = base / f"rf_family_metrics__{tag}.json"  # legacy leading underscore
    if p1.exists():
        return json.loads(p1.read_text())
    if p2.exists():
        return json.loads(p2.read_text())
    return None

def append_raw_row(raw_csv: Path, row: dict):
    # append a row to raw.csv creating it if missing
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(raw_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=str, default=None,
                    help="comma-separated list to filter, e.g., picsys,autoit")
    ap.add_argument("--max-gan-malware", type=int, default=20000,
                    help="cap malware rows used to train the GAN subset")
    ap.add_argument("--epochs", type=int, default=30,
                    help="GAN training epochs for the capped subset")
    ap.add_argument("--skip-gan-train", action="store_true",
                    help="reuse existing generator if present")

    # GAN hyperparameters forwarded to train_gan
    ap.add_argument("--n-critic", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-gp", type=float, default=10.0)
    ap.add_argument("--gan-extra", type=str, default="", help="extra args for train_gan quoted as a single string")

    args = ap.parse_args()

    # discover split files and optionally filter by family
    splits = sorted(SPLITS_DIR.glob("holdout_*.json"))
    if args.families:
        keep = {x.strip().lower() for x in args.families.split(",") if x.strip()}
        splits = [p for p in splits if p.stem.replace("holdout_", "").lower() in keep]

    if not splits:
        print("[err] no LOFO splits found", file=sys.stderr)
        sys.exit(1)

    for sp in splits:
        family = sp.stem.replace("holdout_", "")
        print(f"\n=== [{family}] ===")

        fam_gan_dir = GAN_DIR / family
        fam_gan_dir.mkdir(parents=True, exist_ok=True)
        subset_json = fam_gan_dir / "gan_subset_indices.json"
        gen_path = fam_gan_dir / "generator.pth"
        scaler_path = fam_gan_dir / "scaler.npz"

        # 1) build a capped malware-only subset for this family's LOFO train
        run([
            sys.executable, "-m", "scripts.experiments.holdouts.make_family_gan_subset",
            "--split-json", str(sp),
            "--max-train-rows", str(args.max_gan_malware),
            "--stratify-by-family",
            "--out", str(subset_json),
        ])

        # 2) train GAN unless a generator already exists and skipping is requested
        if gen_path.exists() and args.skip_gan_train:
            print(f"[skip] generator exists: {gen_path}")
        else:
            gan_cmd = [
                sys.executable, "-m", "scripts.gan.train_gan",
                "--indices-json", str(subset_json),
                "--malware-only",
                "--out", str(gen_path),
                "--epochs", str(args.epochs),
                "--n-critic", str(args.n_critic),
                "--batch-size", str(args.batch_size),
                "--device", args.device,
                "--lr", str(args.lr),
                "--lambda-gp", str(args.lambda_gp),
            ]
            # allow extra flags via a quoted string
            if args.gan_extra:
                gan_cmd.extend(shlex.split(args.gan_extra))
            run(gan_cmd)

        # 3) fit GAN scaler on the same subset if not already present
        if not scaler_path.exists():
            run([
                sys.executable, "-m", "scripts.utils.make_gan_scaler",
                "--indices-json", str(subset_json),
                "--out", str(scaler_path),
            ])

        # 4) evaluate across scarcity × seeds × methods and append rows to raw.csv
        outdir = MET_FAM / family
        raw_csv = outdir / "raw.csv"

        # common eval flags for eval_holdout
        common = [
            "--val-threshold", "balacc",
            "--rf-class-weight", "none",
            "--rf-max-depth", "20",
            "--rf-n-est", "400",
            "--rf-n-jobs", "-1",
            "--metrics-subdir", "family_final",
        ]

        # method variants and their extra flags
        methods = [
            ("real",       []),
            ("gan",        [
                "--use-gan",
                "--gan-generator", str(gen_path),
                "--gan-scaler",    str(scaler_path),
                "--gan-like", "full",
                "--gan-synth-per-real", "40",
                "--gan-quality", "nn",
                "--gan-qmult", "5",
            ]),
            ("evogan",     [
                "--use-gan",
                "--gan-evo-refine",
                "--gan-generator", str(gen_path),
                "--gan-scaler",    str(scaler_path),
                "--gan-like", "full",
                "--gan-synth-per-real", "40",
                "--gan-quality", "nn",
                "--gan-qmult", "5",
                "--evo-parent-source", "gan",
                "--evo-mutate-sigma", "0.15",
                "--evo-cx-alpha", "2.0",
                "--evo-qlow", "0.01", "--evo-qhigh", "0.99",
                "--evo-boundary-low", "0.15", "--evo-boundary-high", "0.70",
                "--evo-boundary-k", "5",
            ]),
            ("evo",        [
                "--use-evo",
                "--evo-like", "full",
                "--evo-synth-per-real", "40",
                "--evo-quality", "nn",
                "--evo-qmult", "5",
                "--evo-mutate-sigma", "0.10",
                "--evo-cx-alpha", "2.0",
                "--evo-qlow", "0.01", "--evo-qhigh", "0.99",
                "--evo-boundary-low", "0.20", "--evo-boundary-high", "0.60",
                "--evo-boundary-k", "5",
            ]),
            ("oversample", ["--oversample"]),
            ("smote",      ["--smote"]),
        ]

        for frac in FRACS:
            for seed in SEEDS:
                for variant, extra in methods:
                    tag = f"{family}_f{frac}_s{seed}_{variant}"
                    cmd = [
                        sys.executable, "-m", "scripts.experiments.holdouts.eval_holdout",
                        "--use-family",
                        "--split-json", str(sp),
                        "--scarce-real-frac", str(frac),
                        "--min-train-pos", str(MIN_TRAIN_POS),
                        "--min-train-neg", "50",
                        "--const-train-size", str(CONST_TRAIN_SIZE),
                        "--rf-n-est", "400",
                        "--seed", str(seed),
                        "--tag", tag,
                    ]
                    # extend so each flag is a separate argument
                    cmd += common
                    cmd += extra

                    try:
                        run(cmd)
                    except subprocess.CalledProcessError:
                        print(f"[warn] eval failed for {tag} – continuing")
                        continue

                    m = load_metric_json(tag)
                    if not m:
                        print(f"[warn] missing metrics for {tag}")
                        continue

                    row = dict(
                        prefix=f"fam_{family}",
                        kind="family",
                        frac=float(frac),
                        const_train_size=CONST_TRAIN_SIZE,
                        variant=m.get("variant", "real"),
                        used_gan=bool(m.get("used_gan", False)),
                        tag=tag,
                        auc=m.get("auc"),
                        pr_auc=m.get("pr_auc"),
                        accuracy=m.get("accuracy"),
                        precision=m.get("precision"),
                        recall=m.get("recall"),
                        f1=m.get("f1"),
                        specificity=m.get("specificity"),
                        balanced_accuracy=m.get("balanced_accuracy"),
                        mcc=m.get("mcc"),
                        n_train_real=m.get("n_train_real"),
                        n_train_synth=m.get("n_train_synth"),
                        n_train_total=m.get("n_train_total"),
                        tn=m.get("tn"), fp=m.get("fp"), fn=m.get("fn"), tp=m.get("tp"),
                        threshold=m.get("threshold"),
                        seed=int(seed),
                    )
                    append_raw_row(raw_csv, row)

        print(f"[ok] wrote/updated {raw_csv}")

if __name__ == "__main__":
    main()
