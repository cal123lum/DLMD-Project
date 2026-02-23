#!/usr/bin/env python3
# scripts/experiments/holdouts/run_temporal_sweep.py
# Temporal sweep runner (dataset-agnostic)
# - builds per-cutoff temporal split from meta timestamps
# - trains/reuses GAN per cutoff (optional)
# - fits scaler per cutoff
# - evaluates across fractions × seeds × methods via eval_holdout
#
# Works for bodmas/ember/sorel as long as:
# - NPZ has X and y
# - meta-csv has a timestamp column (default "timestamp", SOREL uses "timestamp_utc")
#
# Author: Callum Musselwhite
# Last edit: 2026-01-14 (dataset-agnostic fix)

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.paths import ROOT

CUTS_DEFAULT = {
    "bodmas": ["2019-10-01", "2019-12-01", "2020-03-01", "2020-06-01", "2020-09-01"],
    "ember":  ["2018-07-01", "2018-08-01", "2018-09-01", "2018-10-01", "2018-11-01"],
    "sorel":  ["2018-02-01", "2018-03-01", "2018-04-01", "2018-05-01", "2018-06-01"],
}

FRACS_DEFAULT = [
    0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005,
    0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01
]
SEEDS_DEFAULT = [42, 1337, 2025]


def run(cmd):
    print("[run]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def load_metric_json(tag: str, *, metrics_subdir: str):
    base = ROOT / "data" / "processed" / "metrics" / metrics_subdir
    p1 = base / f"rf_temporal_metrics_{tag}.json"
    p2 = base / f"rf_temporal_metrics__{tag}.json"  # legacy
    if p1.exists():
        return json.loads(p1.read_text())
    if p2.exists():
        return json.loads(p2.read_text())
    return None


def append_raw_row(pth: Path, row: dict):
    pth.parent.mkdir(parents=True, exist_ok=True)
    if pth.exists():
        df = pd.read_csv(pth)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(pth, index=False)


def build_temporal_split(*, meta_csv: str, time_col: str, cutoff: str, out_json: Path):
    meta = pd.read_csv(meta_csv).fillna("")
    if time_col not in meta.columns:
        raise RuntimeError(f"meta-csv missing '{time_col}' column: {meta_csv}")

    ts = pd.to_datetime(meta[time_col], utc=True, errors="coerce")
    if ts.isna().all():
        raise RuntimeError(f"could not parse any timestamps from '{time_col}' in {meta_csv}")

    cutoff_ts = pd.Timestamp(cutoff).tz_localize("UTC")
    train_idx = meta.index[ts < cutoff_ts].to_numpy()
    test_idx = meta.index[ts >= cutoff_ts].to_numpy()

    out = {"train": train_idx.tolist(), "test": test_idx.tolist()}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2))
    print(f"[ok] wrote {out_json} (train={len(train_idx)} test={len(test_idx)})")

def load_hparams(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"hparams not found: {p}")
    return json.loads(p.read_text())

def apply_hparams_temporal(args, hp: dict):
    sh = hp.get("shared", {})
    gan = hp.get("gan", {})
    evo = hp.get("evo", {})

    # shared RF + selection + budget
    args.const_train_size = int(sh.get("const_train_size", args.const_train_size))
    args.min_train_pos = int(sh.get("min_train_pos", args.min_train_pos))
    args.min_train_neg = int(sh.get("min_train_neg", args.min_train_neg))
    args.val_threshold = str(sh.get("val_threshold", args.val_threshold))

    args.rf_n_est = int(sh.get("rf_n_est", args.rf_n_est))
    args.rf_max_depth = int(sh.get("rf_max_depth", args.rf_max_depth))
    args.rf_n_jobs = int(sh.get("rf_n_jobs", args.rf_n_jobs))
    args.rf_class_weight = str(sh.get("rf_class_weight", args.rf_class_weight))

    # GAN training hyperparams (temporal-specific)
    args.epochs = int(gan.get("epochs_temporal", args.epochs))
    args.n_critic = int(gan.get("n_critic_temporal", args.n_critic))
    args.batch_size = int(gan.get("batch_size", args.batch_size))
    args.lr = float(gan.get("lr", args.lr))
    args.lambda_gp = float(gan.get("lambda_gp", args.lambda_gp))

    # stash the “method-budget” params we’ll use when building method flags
    args._hp_alpha = int(sh.get("alpha_synth_per_real", 40))
    args._hp_qmult = int(sh.get("qmult", 5))
    args._hp_like = str(sh.get("like", "full"))
    args._hp_gate = str(sh.get("quality_gate", "nn"))

    # evo params (temporal uses the base evo settings)
    args._hp_evo = {
        "cx_alpha": float(evo.get("cx_alpha", 2.0)),
        "mutate_sigma": float(evo.get("mutate_sigma", 0.10)),
        "qlow": float(evo.get("qlow", 0.01)),
        "qhigh": float(evo.get("qhigh", 0.99)),
        "boundary_low": float(evo.get("boundary_low", 0.20)),
        "boundary_high": float(evo.get("boundary_high", 0.60)),
        "boundary_k": int(evo.get("boundary_k", 5)),
    }


def main():
    ap = argparse.ArgumentParser()

    # dataset inputs
    ap.add_argument("--dataset", type=str, default="bodmas")
    ap.add_argument("--npz", type=str, default=str(ROOT / "data" / "raw" / "bodmas.npz"))
    ap.add_argument("--meta-csv", type=str, default=str(ROOT / "data" / "raw" / "bodmas_metadata.csv"))
    ap.add_argument("--time-col", type=str, default="timestamp",
                    help="timestamp column in meta-csv (e.g. timestamp, timestamp_utc)")

    # sweep controls
    ap.add_argument("--cutoffs", type=str, default="")
    ap.add_argument("--fractions", type=str, default=",".join(map(str, FRACS_DEFAULT)))
    ap.add_argument("--seeds-override", type=str, default=",".join(map(str, SEEDS_DEFAULT)))

    ap.add_argument("--methods", type=str, default="real,gan,evogan,oversample,smote,evo")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--skip-gan-train", action="store_true")

    # RF
    ap.add_argument("--rf-n-est", type=int, default=400)
    ap.add_argument("--rf-max-depth", type=int, default=20)
    ap.add_argument("--rf-n-jobs", type=int, default=-1)
    ap.add_argument("--rf-class-weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--val-threshold", choices=["balacc", "f1", "mcc", "none"], default="balacc")
    ap.add_argument("--min-train-pos", type=int, default=50)
    ap.add_argument("--min-train-neg", type=int, default=50)
    ap.add_argument("--const-train-size", type=int, default=20000)

    # GAN options (only used if gan/evogan requested)
    ap.add_argument("--max-gan-malware", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--n-critic", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-gp", type=float, default=10.0)
    ap.add_argument("--gan-extra", type=str, default="", help="extra args for train_gan as one quoted string")
    ap.add_argument("--gan-seed", type=int, default=42, help="seed used for GAN training per cutoff")
    ap.add_argument("--hparams", type=str, default="",
                help="path to JSON with shared hyperparams (Table 1/2). If set, overrides runner defaults.")


    args = ap.parse_args()

    hp = None
    if args.hparams:
        hp = load_hparams(args.hparams)
        apply_hparams_temporal(args, hp)
    
    print(
        "[hparams]",
        "alpha=", getattr(args, "_hp_alpha", None),
        "qmult=", getattr(args, "_hp_qmult", None),
        "like=", getattr(args, "_hp_like", None),
        "gate=", getattr(args, "_hp_gate", None),
        "epochs=", getattr(args, "epochs", None),
        "n_critic=", getattr(args, "n_critic", None),
        "rf_n_est=", getattr(args, "rf_n_est", None),
        "rf_max_depth=", getattr(args, "rf_max_depth", None),
        "const_train_size=", getattr(args, "const_train_size", None),
    )
    print("[hparams] evo_default=", getattr(args, "_hp_evo", None))
    print("[hparams] evogan_lofo=", getattr(args, "_hp_evogan_lofo", None))


    if args.cutoffs.strip():
        cuts = [x.strip() for x in args.cutoffs.split(",") if x.strip()]
    else:
        cuts = CUTS_DEFAULT.get(args.dataset, CUTS_DEFAULT["bodmas"])
    fracs = [float(x) for x in args.fractions.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds_override.split(",") if x.strip()]
    want = {m.strip().lower() for m in args.methods.split(",") if m.strip()}

    metrics_subdir = f"temporal_{args.dataset}"
    MET_ROOT = ROOT / "data" / "processed" / "metrics" / metrics_subdir

    # method variants
    alpha = getattr(args, "_hp_alpha", 40)
    qmult = getattr(args, "_hp_qmult", 5)
    like = getattr(args, "_hp_like", "full")
    gate = getattr(args, "_hp_gate", "nn")
    evo_hp_default = getattr(args, "_hp_evo", {
        "cx_alpha": 2.0, "mutate_sigma": 0.10,
        "qlow": 0.01, "qhigh": 0.99,
        "boundary_low": 0.20, "boundary_high": 0.60, "boundary_k": 5,
    })

    evo_hp = evo_hp_default
    evo_gan_hp = getattr(args, "_hp_evogan_lofo", {
        **evo_hp_default,
        "mutate_sigma": 0.15,
        "boundary_low": 0.15,
        "boundary_high": 0.70,
    })


    methods = [
        ("real", []),
        ("oversample", ["--oversample"]),
        ("smote", ["--smote"]),
        ("evo", [
            "--use-evo",
            "--evo-like", like,
            "--evo-synth-per-real", str(alpha),
            "--evo-quality", gate,
            "--evo-qmult", str(qmult),
            "--evo-mutate-sigma", str(evo_hp["mutate_sigma"]),
            "--evo-cx-alpha", str(evo_hp["cx_alpha"]),
            "--evo-qlow", str(evo_hp["qlow"]), "--evo-qhigh", str(evo_hp["qhigh"]),
            "--evo-boundary-low", str(evo_hp["boundary_low"]),
            "--evo-boundary-high", str(evo_hp["boundary_high"]),
            "--evo-boundary-k", str(evo_hp["boundary_k"]),
        ]),
        ("gan", [
            "--use-gan",
            "--gan-like", like,
            "--gan-synth-per-real", str(alpha),
            "--gan-quality", gate,
            "--gan-qmult", str(qmult),
        ]),
        ("evogan", [
            "--use-gan",
            "--gan-evo-refine",
            "--gan-like", like,
            "--gan-synth-per-real", str(alpha),
            "--gan-quality", gate,
            "--gan-qmult", str(qmult),
            "--evo-parent-source", "gan",
            "--evo-mutate-sigma", str(evo_gan_hp["mutate_sigma"]),
            "--evo-cx-alpha", str(evo_gan_hp["cx_alpha"]),
            "--evo-qlow", str(evo_gan_hp["qlow"]), "--evo-qhigh", str(evo_gan_hp["qhigh"]),
            "--evo-boundary-low", str(evo_gan_hp["boundary_low"]),
            "--evo-boundary-high", str(evo_gan_hp["boundary_high"]),
            "--evo-boundary-k", str(evo_gan_hp["boundary_k"]),
        ]),
    ]

    methods = [(n, extra) for (n, extra) in methods if n in want]

    # directories for GAN assets per cutoff
    GAN_ROOT = ROOT / "models" / "gan" / f"temporal_{args.dataset}"
    GAN_ROOT.mkdir(parents=True, exist_ok=True)

    for cutoff in cuts:
        prefix = f"cut{cutoff[:4]}_{cutoff[5:7]}"
        split_json = ROOT / "data" / "holdouts" / f"{args.dataset}_temporal_{prefix}.json"

        build_temporal_split(meta_csv=args.meta_csv, time_col=args.time_col, cutoff=cutoff, out_json=split_json)

        # per-cutoff gan assets
        gendir = GAN_ROOT / prefix
        gendir.mkdir(parents=True, exist_ok=True)
        subset_json = gendir / "gan_subset_indices.json"
        gen = gendir / "generator.pth"
        scal = gendir / "scaler.npz"

        need_gan = any(v in want for v in ("gan", "evogan"))
        if need_gan:
            # build malware-only subset indices for GAN training using the same helper as IID
            run([
                sys.executable, "-m", "scripts.experiments.iid.make_iid_gan_subset",
                "--split-json", str(split_json),
                "--max-train-rows", str(args.max_gan_malware),
                "--seed", str(args.gan_seed),
                "--out", str(subset_json),
                "--npz", args.npz,
            ])

            if not (args.skip_gan_train and gen.exists()):
                gan_cmd = [
                    sys.executable, "-m", "scripts.gan.train_gan",
                    "--indices-json", str(subset_json),
                    "--malware-only",
                    "--out", str(gen),
                    "--epochs", str(args.epochs),
                    "--n-critic", str(args.n_critic),
                    "--batch-size", str(args.batch_size),
                    "--device", args.device,
                    "--lr", str(args.lr),
                    "--lambda-gp", str(args.lambda_gp),
                    "--npz", args.npz,
                    "--seed", str(args.gan_seed),
                ]
                if args.gan_extra:
                    gan_cmd.extend(shlex.split(args.gan_extra))
                run(gan_cmd)
            else:
                print(f"[skip] generator exists: {gen}")

            if not scal.exists():
                run([
                    sys.executable, "-m", "scripts.utils.make_gan_scaler",
                    "--indices-json", str(subset_json),
                    "--out", str(scal),
                    "--npz", args.npz,
                ])

        # per-cutoff raw.csv output (kept grouped)
        out_csv = MET_ROOT / prefix / "raw.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # common eval flags
        common = [
            "--val-threshold", args.val_threshold,
            "--rf-class-weight", args.rf_class_weight,
            "--rf-max-depth", str(args.rf_max_depth),
            "--rf-n-est", str(args.rf_n_est),
            "--rf-n-jobs", str(args.rf_n_jobs),
            "--metrics-subdir", metrics_subdir,
            "--split-json", str(split_json),
            "--npz", args.npz,
            "--meta-csv", args.meta_csv,
        ]

        for frac in fracs:
            for seed in seeds:
                for variant, extra in methods:
                    tag = f"{args.dataset}_{prefix}_f{frac}_s{seed}_{variant}"

                    if args.skip_existing and load_metric_json(tag, metrics_subdir=metrics_subdir):
                        print(f"[skip] existing metrics: {tag}")
                        continue

                    cmd = [
                        sys.executable, "-m", "scripts.experiments.holdouts.eval_holdout",
                        "--use-temporal",
                        "--scarce-real-frac", str(frac),
                        "--min-train-pos", str(args.min_train_pos),
                        "--min-train-neg", str(args.min_train_neg),
                        "--const-train-size", str(args.const_train_size),
                        "--seed", str(seed),
                        "--tag", tag,
                    ]
                    cmd += common

                    if variant in ("gan", "evogan"):
                        cmd += ["--gan-generator", str(gen), "--gan-scaler", str(scal)]
                    cmd += extra

                    try:
                        run(cmd)
                    except subprocess.CalledProcessError:
                        print(f"[warn] failed: {tag}")
                        continue

                    m = load_metric_json(tag, metrics_subdir=metrics_subdir)
                    if not m:
                        print(f"[warn] missing metrics for {tag}")
                        continue

                    row = dict(
                        prefix=prefix,
                        kind="temporal",
                        frac=float(frac),
                        const_train_size=int(args.const_train_size),
                        variant=m.get("variant", variant),
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
                        cutoff=cutoff,
                    )
                    append_raw_row(out_csv, row)

        print(f"[ok] wrote/updated {out_csv}")


if __name__ == "__main__":
    main()
