#!/usr/bin/env python3
# scripts/experiments/holdouts/run_temporal_sweep.py
# Temporal sweep for GAN training and evaluation with optional rolling windows
# Author: Callum Musselwhite
# Last edit: 2025-09-17
# Purpose: for each temporal cutoff, ensure split and GAN assets exist, then evaluate methods across scarcity × seeds, optionally per rolling window

import subprocess, sys, json
from pathlib import Path
import pandas as pd
from src.paths import ROOT
import argparse

# cutoffs, scarcity fractions, and robustness seeds
CUTS = ["2019-10-01", "2019-12-01", "2020-03-01", "2020-06-01", "2020-09-01"]
FRACS = [0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005,
         0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01]
SEEDS = [42, 1337, 2025]

# model and split locations
GAN_ROOT = ROOT / "models" / "gan" / "temporal_final"          # per-cutoff generator + scaler
SPLIT_JSON = ROOT / "data" / "holdouts" / "temporal_indices.json"  # produced by make_holdouts

def run(cmd):
    # echo then run a subprocess command
    print("[run]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)

def append_raw_row(pth: Path, row: dict):
    # append one row to a CSV, creating the file if it does not exist
    pth.parent.mkdir(parents=True, exist_ok=True)
    if pth.exists():
        df = pd.read_csv(pth)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(pth, index=False)

def load_metric_json(tag: str, *, subdir: str):
    # metrics files may have a legacy double-underscore tag
    base = ROOT / "data" / "processed" / "metrics" / subdir
    p1 = base / f"rf_temporal_metrics_{tag}.json"
    p2 = base / f"rf_temporal_metrics__{tag}.json"
    if p1.exists():
        return json.loads(p1.read_text())
    if p2.exists():
        return json.loads(p2.read_text())
    return None

def read_test_indices():
    # load test indices from the canonical split json supporting a few historical shapes
    j = json.loads(SPLIT_JSON.read_text())
    if "test" in j:
        return j["test"]
    if len(j) == 1:
        _, v = next(iter(j.items()))
        return v.get("test") or v.get("test_idx") or v.get("test_indices")
    raise RuntimeError(f"Unrecognized split format in {SPLIT_JSON}")

def infer_tmin_from_meta(meta_path: Path) -> pd.Timestamp | None:
    # anchor rolling windows at the earliest test timestamp if available
    try:
        if meta_path.suffix.lower() in (".parquet", ".pq"):
            meta = pd.read_parquet(meta_path)
        else:
            meta = pd.read_csv(meta_path)
        assert "timestamp" in meta.columns
        test_idx = read_test_indices()
        ts = pd.to_datetime(meta.loc[test_idx, "timestamp"], utc=True)
        return ts.min()
    except Exception:
        return None

def windows_for_cutoff(cutoff_iso: str, W_days=60, S_days=30, N=6, anchor_ts: pd.Timestamp | None = None):
    """
    rolling windows [start, end) in UTC
    if anchor_ts is given use it, else anchor at the cutoff date
    """
    anchor = pd.Timestamp(cutoff_iso).tz_localize("UTC") if anchor_ts is None else pd.Timestamp(anchor_ts).tz_convert("UTC")
    out = []
    for k in range(N):
        s = anchor + pd.Timedelta(days=k * S_days)
        e = s + pd.Timedelta(days=W_days)
        out.append((f"w{k}", s.date().isoformat(), e.date().isoformat()))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-gan-train", action="store_true", help="reuse existing generator if present")
    ap.add_argument("--methods", type=str, default="real,gan,evogan,oversample,smote,evo",
                    help="subset to run, e.g., 'evogan' or 'real,gan'")
    # rolling windows controls
    ap.add_argument("--rolling-windows", action="store_true", help="enable rolling window evaluation")
    ap.add_argument("--win-days", type=int, default=60, help="window width W in days")
    ap.add_argument("--win-stride", type=int, default=30, help="stride S in days")
    ap.add_argument("--win-n", type=int, default=6, help="number of windows per cutoff")
    ap.add_argument("--meta-path", type=str, default=None,
                    help="optional meta with timestamp column to anchor at earliest test time")
    ap.add_argument("--cutoffs", type=str, default=None, help="comma list to override CUTS")
    ap.add_argument("--rf-n-jobs", type=int, default=1, help="RF jobs, keep low if running alongside family sweeps")
    ap.add_argument("--fractions", type=str, default=None,
                    help="comma list to override FRACS")
    ap.add_argument("--seeds-override", type=str, default=None,
                    help="comma list to override SEEDS")
    ap.add_argument("--only-w", type=str, default=None,
                    help="run only a specific window id like w5")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip a tag if its metrics JSON already exists")
    ap.add_argument("--rf-n-est", type=int, default=400,
                    help="RandomForest n_estimators override")
    args = ap.parse_args()

    # resolve run sets from CLI or defaults
    fracs = FRACS if not args.fractions else [float(x) for x in args.fractions.split(",") if x.strip()]
    seeds = SEEDS if not args.seeds_override else [int(x) for x in args.seeds_override.split(",") if x.strip()]
    cuts = CUTS if not args.cutoffs else [x.strip() for x in args.cutoffs.split(",") if x.strip()]
    want = {m.strip().lower() for m in args.methods.split(",") if m.strip()}

    # choose metrics subdir based on windows flag
    metrics_subdir = "temporal_final_window" if args.rolling_windows else "temporal_final"
    MET_ROOT = ROOT / "data" / "processed" / "metrics" / metrics_subdir

    # define method variants and common flags
    methods = [
        ("real", []),
        ("gan", [
            "--use-gan", "--gan-like", "full", "--gan-synth-per-real", "40", "--gan-quality", "nn", "--gan-qmult", "5",
        ]),
        ("oversample", ["--oversample"]),
        ("smote",      ["--smote"]),
        ("evo", [
            "--use-evo", "--evo-like", "full", "--evo-synth-per-real", "40", "--evo-quality", "nn", "--evo-qmult", "5",
            "--evo-mutate-sigma", "0.10", "--evo-cx-alpha", "2.0",
            "--evo-qlow", "0.01", "--evo-qhigh", "0.99",
            "--evo-boundary-low", "0.20", "--evo-boundary-high", "0.60", "--evo-boundary-k", "5",
        ]),
        ("evogan", [
            "--use-gan", "--gan-evo-refine", "--gan-like", "full", "--gan-synth-per-real", "40", "--gan-quality", "nn", "--gan-qmult", "5",
            "--evo-parent-source", "gan",
            "--evo-mutate-sigma", "0.10", "--evo-cx-alpha", "2.0",
            "--evo-qlow", "0.01", "--evo-qhigh", "0.99",
            "--evo-boundary-low", "0.20", "--evo-boundary-high", "0.60", "--evo-boundary-k", "5",
        ]),
    ]
    methods = [(n, e) for (n, e) in methods if n in want]

    for cutoff in cuts:
        # ensure the temporal split exists and is valid
        run([sys.executable, "-m", "scripts.experiments.holdouts.make_holdouts", "--temporal-cutoff", cutoff])
        run([sys.executable, "-m", "scripts.experiments.holdouts.verify_holdouts"])

        prefix = f"cut{cutoff[:4]}_{cutoff[5:7]}"
        out_csv = MET_ROOT / prefix / "raw.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # generator and scaler per cutoff
        gendir = GAN_ROOT / prefix
        gendir.mkdir(parents=True, exist_ok=True)
        gen = gendir / "generator.pth"
        scal = gendir / "scaler.npz"

        # optionally reuse an existing generator
        if args.skip_gan_train and gen.exists():
            print(f"[skip] generator exists: {gen}")
        else:
            if args.skip_gan_train and not gen.exists():  # keep informative message if reuse requested but missing
                print(f"[warn] --skip-gan-train set but no generator found; training now: {gen}")
            run([sys.executable, "-m", "scripts.gan.train_gan",
                 "--indices-json", str(SPLIT_JSON),
                 "--malware-only", "--out", str(gen),
                 "--epochs", "80", "--n-critic", "4", "--batch-size", "128",
                 "--device", "auto", "--lr", "1e-4", "--lambda-gp", "10.0"])

        # fit scaler once
        if not scal.exists():
            run([sys.executable, "-m", "scripts.utils.make_gan_scaler",
                 "--indices-json", str(SPLIT_JSON), "--out", str(scal)])

        # build optional rolling windows anchored at earliest test time if provided
        tmin = None
        if args.meta_path:
            tmin = infer_tmin_from_meta(Path(args.meta_path))
            if tmin is None:
                print(f"[warn] could not infer t_c^min from meta; using cutoff {cutoff} as anchor")
        windows = [("none", None, None)]
        if args.rolling_windows:
            windows = windows_for_cutoff(cutoff, W_days=args.win_days, S_days=args.win_stride, N=args.win_n, anchor_ts=tmin)
        if args.only_w:
            windows = [w for w in windows if w[0] == args.only_w]

        # flags common to all eval runs
        common = [
            "--val-threshold", "balacc",
            "--rf-class-weight", "none",
            "--rf-max-depth", "20",
            "--rf-n-est", str(args.rf_n_est),
            "--rf-n-jobs", str(args.rf_n_jobs),
            "--metrics-subdir", metrics_subdir,
            "--split-json", str(SPLIT_JSON),
        ]

        for wid, wstart, wend in windows:
            for frac in fracs:
                for seed in seeds:
                    for variant, extra in methods:
                        tag = f"{prefix}_f{frac}_s{seed}_{variant}" if wid == "none" else f"{prefix}_{wid}_{wstart}_{wend}_f{frac}_s{seed}_{variant}"
                        cmd = [sys.executable, "-m", "scripts.experiments.holdouts.eval_holdout",
                               "--use-temporal",
                               "--scarce-real-frac", str(frac),
                               "--min-train-pos", "50", "--min-train-neg", "50",
                               "--const-train-size", "20000",
                               "--seed", str(seed), "--tag", tag]
                        if wid != "none":
                            cmd += ["--test-start", wstart, "--test-end", wend]
                        cmd += common
                        if variant in ("gan", "evogan"):
                            cmd += ["--gan-generator", str(gen), "--gan-scaler", str(scal)]
                        cmd += extra

                        try:
                            if args.skip_existing and load_metric_json(tag, subdir=metrics_subdir):
                                print(f"[skip] existing metrics: {tag}")
                                continue
                            run(cmd)
                        except subprocess.CalledProcessError:
                            print(f"[warn] failed: {tag}")
                            continue

                        m = load_metric_json(tag, subdir=metrics_subdir)
                        if not m:
                            print(f"[warn] missing metrics for {tag}")
                            continue

                        row = dict(
                            prefix=prefix, kind="temporal", frac=float(frac),
                            const_train_size=20000,
                            variant=m.get("variant", "real"),
                            used_gan=bool(m.get("used_gan", False)), tag=tag,
                            auc=m.get("auc"), pr_auc=m.get("pr_auc"),
                            accuracy=m.get("accuracy"), precision=m.get("precision"),
                            recall=m.get("recall"), f1=m.get("f1"),
                            specificity=m.get("specificity"),
                            balanced_accuracy=m.get("balanced_accuracy"),
                            mcc=m.get("mcc"),
                            n_train_real=m.get("n_train_real"),
                            n_train_synth=m.get("n_train_synth"),
                            n_train_total=m.get("n_train_total"),
                            tn=m.get("tn"), fp=m.get("fp"), fn=m.get("fn"), tp=m.get("tp"),
                            threshold=m.get("threshold"), seed=int(seed),
                            test_start=m.get("test_start"), test_end=m.get("test_end"),
                            test_window_days=m.get("test_window_days"),
                        )
                        append_raw_row(out_csv, row)

        print(f"[ok] wrote/updated {out_csv}")

if __name__ == "__main__":
    main()
