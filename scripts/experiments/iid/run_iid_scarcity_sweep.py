#!/usr/bin/env python3
# scripts/experiments/iid/run_iid_sweep.py
# IID sweep: build splits, train or reuse GAN per seed, fit scaler, then evaluate across fractions × methods
# Author: Callum Musselwhite
# Last edit: 2026-01-14
# Purpose: orchestrate IID experiments to mirror holdout sweeps while supporting multiple datasets (bodmas/ember/...)

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd 
from src.paths import ROOT


FRACS_DEFAULT = [
    0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005,
    0.0055, 0.006, 0.0065, 0.007, 0.0075, 0.008, 0.0085, 0.009, 0.0095, 0.01
]
SEEDS_DEFAULT = [42, 1337, 2025]


def run(cmd):
    print("[run]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def load_metric_json(met_dir: Path, tag: str):
    # eval_holdout writes rf_temporal_metrics_* when using --use-temporal (we use it for IID custom split)
    p1 = met_dir / f"rf_temporal_metrics_{tag}.json"
    p2 = met_dir / f"rf_temporal_metrics__{tag}.json"  # legacy
    if p1.exists():
        return json.loads(p1.read_text())
    if p2.exists():
        return json.loads(p2.read_text())
    return None


def append_raw_row(raw_csv: Path, row: dict):
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(raw_csv, index=False)


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
    args.rf_n_jobs = int(sh.get("rf_n_jobs", getattr(args, "rf_n_jobs", -1)))
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

    # evo params
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

    ap.add_argument("--fractions", type=str, default=",".join(map(str, FRACS_DEFAULT)))
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS_DEFAULT)))
    ap.add_argument(
        "--methods",
        type=str,
        default="real,gan,oversample,smote,evo,evogan",
        help="comma list subset to run, e.g., 'evo,evogan'"
    )

    # dataset controls
    ap.add_argument("--dataset", type=str, default="bodmas")
    ap.add_argument("--npz", type=str, default=str(ROOT / "data" / "raw" / "bodmas.npz"))
    ap.add_argument("--meta-csv", type=str, default=str(ROOT / "data" / "raw" / "bodmas_metadata.csv"))

    # train-time constraints
    ap.add_argument("--const-train-size", type=int, default=20000)
    ap.add_argument("--min-train-pos", type=int, default=50)
    ap.add_argument("--min-train-neg", type=int, default=50)

    # GAN training options
    ap.add_argument("--max-gan-malware", type=int, default=20000, help="cap malware rows for GAN training subset")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-critic", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-gp", type=float, default=10.0)
    ap.add_argument("--skip-gan-train", action="store_true", help="reuse existing generator if present")
    ap.add_argument("--gan-extra", type=str, default="", help="extra args for train_gan quoted as one string")

    # eval parity with holdouts
    ap.add_argument("--rf-n-est", type=int, default=400)
    ap.add_argument("--rf-max-depth", type=int, default=20)
    ap.add_argument("--rf-class-weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--val-threshold", choices=["balacc", "f1", "mcc", "none"], default="balacc")
    ap.add_argument("--rf-n-jobs", type=int, default=-1)

    ap.add_argument(
        "--hparams",
        type=str,
        default="",
        help="path to JSON with shared hyperparams (Table 1/2). If set, overrides runner defaults."
    )

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
        "rf_n_jobs=", getattr(args, "rf_n_jobs", None),
    )
    print("[hparams] evo_default=", getattr(args, "_hp_evo", None))

    fracs = [float(x) for x in args.fractions.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    want = {m.strip().lower() for m in args.methods.split(",") if m.strip()}

    met_dir = ROOT / "data" / "processed" / "metrics" / f"iid_{args.dataset}"
    gan_dir = ROOT / "models" / "gan" / f"iid_{args.dataset}"
    met_dir.mkdir(parents=True, exist_ok=True)
    gan_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = met_dir / "raw.csv"

    for seed in seeds:
        split_json = ROOT / "data" / "holdouts" / f"{args.dataset}_iid_split_seed{seed}.json"
        run([
            sys.executable, "-m", "scripts.experiments.iid.make_iid_split",
            "--seed", str(seed),
            "--out", str(split_json),
            "--npz", str(args.npz),
        ])

        gan_seed_dir = gan_dir / f"seed{seed}"
        gan_seed_dir.mkdir(parents=True, exist_ok=True)
        subset_json = gan_seed_dir / "gan_subset_indices.json"
        gen_path = gan_seed_dir / "generator.pth"
        scaler_path = gan_seed_dir / "scaler.npz"

        run([
            sys.executable, "-m", "scripts.experiments.iid.make_iid_gan_subset",
            "--split-json", str(split_json),
            "--max-train-rows", str(args.max_gan_malware),
            "--seed", str(seed),
            "--out", str(subset_json),
            "--npz", str(args.npz),
        ])

        # train GAN (ONLY if gan/evogan requested)
        if gen_path.exists() and args.skip_gan_train:
            print(f"[skip] generator exists: {gen_path}")
        elif ("gan" in want) or ("evogan" in want):
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
                "--npz", str(args.npz),     # FIX: ensure train_gan loads the correct dataset
                "--seed", str(seed),        # FIX: deterministic GAN init per seed
            ]
            if args.gan_extra:
                gan_cmd.extend(shlex.split(args.gan_extra))
            run(gan_cmd)

        # fit GAN scaler (ONLY if gan/evogan requested)
        if ("gan" in want or "evogan" in want) and (not scaler_path.exists()):
            run([
                sys.executable, "-m", "scripts.utils.make_gan_scaler",
                "--indices-json", str(subset_json),
                "--out", str(scaler_path),
                "--npz", str(args.npz),     # FIX: ensure scaler uses the correct dataset
            ])

        common = [
            "--val-threshold", args.val_threshold,
            "--rf-class-weight", args.rf_class_weight,
            "--rf-max-depth", str(args.rf_max_depth),
            "--rf-n-est", str(args.rf_n_est),
            "--rf-n-jobs", str(args.rf_n_jobs),
            "--seed", str(seed),
            "--split-json", str(split_json),
            "--metrics-subdir", f"iid_{args.dataset}",
            "--npz", str(args.npz),
            "--meta-csv", str(args.meta_csv),
        ]

        alpha = getattr(args, "_hp_alpha", 40)
        qmult = getattr(args, "_hp_qmult", 5)
        like = getattr(args, "_hp_like", "full")
        gate = getattr(args, "_hp_gate", "nn")
        evo_hp = getattr(args, "_hp_evo", {
            "cx_alpha": 2.0, "mutate_sigma": 0.10,
            "qlow": 0.01, "qhigh": 0.99,
            "boundary_low": 0.20, "boundary_high": 0.60, "boundary_k": 5,
        })

        methods = [
            ("real", []),

            ("gan", [
                "--use-gan",
                "--gan-generator", str(gen_path),
                "--gan-scaler", str(scaler_path),
                "--gan-like", like,
                "--gan-synth-per-real", str(alpha),
                "--gan-quality", gate,
                "--gan-qmult", str(qmult),
            ]),

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

            ("evogan", [
                "--use-gan",
                "--gan-evo-refine",
                "--gan-generator", str(gen_path),
                "--gan-scaler", str(scaler_path),
                "--gan-like", like,
                "--gan-synth-per-real", str(alpha),
                "--gan-quality", gate,
                "--gan-qmult", str(qmult),
                "--evo-parent-source", "gan",
                "--evo-mutate-sigma", str(evo_hp["mutate_sigma"]),
                "--evo-cx-alpha", str(evo_hp["cx_alpha"]),
                "--evo-qlow", str(evo_hp["qlow"]), "--evo-qhigh", str(evo_hp["qhigh"]),
                "--evo-boundary-low", str(evo_hp["boundary_low"]),
                "--evo-boundary-high", str(evo_hp["boundary_high"]),
                "--evo-boundary-k", str(evo_hp["boundary_k"]),
            ]),
        ]
        methods = [(name, extra) for (name, extra) in methods if name in want]

        for frac in fracs:
            for variant, extra in methods:
                tag = f"{args.dataset}_iid_f{frac}_s{seed}_{variant}"

                cmd = [
                    sys.executable, "-m", "scripts.experiments.holdouts.eval_holdout",
                    "--use-temporal",  # custom split via --split-json; this flag satisfies eval_holdout group
                    "--scarce-real-frac", str(frac),
                    "--min-train-pos", str(args.min_train_pos),
                    "--min-train-neg", str(args.min_train_neg),
                    "--const-train-size", str(args.const_train_size),
                    "--tag", tag,
                ]
                cmd += common
                cmd += extra

                try:
                    run(cmd)
                except subprocess.CalledProcessError:
                    print(f"[warn] eval failed for {tag} – continuing")
                    continue

                m = load_metric_json(met_dir, tag)
                if not m:
                    print(f"[warn] missing metrics for {tag}")
                    continue

                row = dict(
                    dataset=args.dataset,
                    prefix="iid",
                    kind="iid",
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
                )
                append_raw_row(raw_csv, row)

    print(f"[ok] wrote/updated {raw_csv}")


if __name__ == "__main__":
    main()
