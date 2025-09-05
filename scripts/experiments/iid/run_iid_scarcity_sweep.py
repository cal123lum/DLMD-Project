#!/usr/bin/env python3
import argparse, json, subprocess, sys, shlex
from pathlib import Path
import pandas as pd
from src.paths import ROOT

# FRACS and seeds default to your requested list
FRACS_DEFAULT = [0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.0045,0.005,
                 0.0055,0.006,0.0065,0.007,0.0075,0.008,0.0085,0.009,0.0095,0.01]
SEEDS_DEFAULT = [42, 1337, 2025]

MET_IID = ROOT / "data" / "processed" / "metrics" / "iid_scarcity"
GAN_DIR = ROOT / "models" / "gan" / "iid"

def run(cmd):
    print("[run]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)

def load_metric_json(tag: str):
    base = MET_IID
    p1 = base / f"rf_temporal_metrics_{tag}.json"
    p2 = base / f"rf_temporal_metrics__{tag}.json"  # fallback underscore quirk
    if p1.exists(): return json.loads(p1.read_text())
    if p2.exists(): return json.loads(p2.read_text())
    return None

def append_raw_row(raw_csv: Path, row: dict):
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(raw_csv, index=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fractions", type=str, default=",".join(map(str, FRACS_DEFAULT)))
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS_DEFAULT)))

    # train-time constraints
    ap.add_argument("--const-train-size", type=int, default=20000)
    ap.add_argument("--min-train-pos", type=int, default=50)
    ap.add_argument("--min-train-neg", type=int, default=50)

    # GAN training options
    ap.add_argument("--max-gan-malware", type=int, default=20000,
                    help="Cap malware rows for GAN training subset.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--n-critic", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-gp", type=float, default=10.0)
    ap.add_argument("--skip-gan-train", action="store_true",
                    help="Reuse existing generator if present.")
    ap.add_argument("--gan-extra", type=str, default="", help="Extra args for train_gan (quoted)")

    # Eval parity with your holdouts
    ap.add_argument("--rf-n-est", type=int, default=400)
    ap.add_argument("--rf-max-depth", type=int, default=20)
    ap.add_argument("--rf-class-weight", choices=["none","balanced"], default="none")
    ap.add_argument("--val-threshold", choices=["balacc","f1","mcc","none"], default="balacc")

    args = ap.parse_args()

    fracs = [float(x) for x in args.fractions.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    MET_IID.mkdir(parents=True, exist_ok=True)
    raw_csv = MET_IID / "raw.csv"

    for seed in seeds:
        # --- 1) Build IID split for this seed
        split_json = ROOT / "data" / "holdouts" / f"iid_split_seed{seed}.json"
        run([sys.executable, "-m", "scripts.experiments.iid.make_iid_split",
             "--seed", str(seed), "--out", str(split_json)])

        # --- 2) Build malware-only capped subset for GAN training
        gan_seed_dir = GAN_DIR / f"seed{seed}"
        gan_seed_dir.mkdir(parents=True, exist_ok=True)
        subset_json = gan_seed_dir / "gan_subset_indices.json"
        gen_path    = gan_seed_dir / "generator.pth"
        scaler_path = gan_seed_dir / "scaler.npz"

        run([sys.executable, "-m", "scripts.experiments.iid.make_iid_gan_subset",
             "--split-json", str(split_json),
             "--max-train-rows", str(args.max_gan_malware),
             "--seed", str(seed),
             "--out", str(subset_json)])

        # --- 3) Train GAN (unless skipping)
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
            if args.gan_extra:
                gan_cmd.extend(shlex.split(args.gan_extra))
            run(gan_cmd)

        # --- 4) Fit GAN scaler on the same subset
        if not scaler_path.exists():
            run([sys.executable, "-m", "scripts.utils.make_gan_scaler",
                 "--indices-json", str(subset_json),
                 "--out", str(scaler_path)])

        # --- 5) Evaluate across FRACS × methods
        common = [
            "--val-threshold", args.val_threshold,
            "--rf-class-weight", args.rf_class_weight,
            "--rf-max-depth", str(args.rf_max_depth),
            "--rf-n-est", str(args.rf_n_est),
            "--seed", str(seed),
            "--split-json", str(split_json),
            "--metrics-subdir", "iid_scarcity",  # write under MET_IID
        ]

        methods = [
            ("real",       []),
            ("gan",        [
                "--use-gan",
                "--gan-generator", str(gen_path),
                "--gan-scaler",    str(scaler_path),
                "--gan-like", "full",
                "--gan-synth-per-real", "80",
                "--gan-quality", "nn_boundary",
                "--gan-qmult", "5",
            ]),
            ("oversample", ["--oversample"]),
            ("smote",      ["--smote"]),
        ]

        for frac in fracs:
            for variant, extra in methods:
                tag = f"iid_f{frac}_s{seed}_{variant}"
                cmd = [
                    sys.executable, "-m", "scripts.experiments.holdouts.eval_holdout",
                    # we pass --use-temporal solely to satisfy the required arg group;
                    # actual split comes from --split-json, so this is "IID via custom split".
                    "--use-temporal",
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

                m = load_metric_json(tag)
                if not m:
                    print(f"[warn] missing metrics for {tag}")
                    continue

                row = dict(
                    prefix="iid",
                    kind="iid",
                    frac=float(frac),
                    const_train_size=int(args.const_train_size),
                    variant=m.get("variant","real"),
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
                append_raw_row(MET_IID / "raw.csv", row)

    print(f"[ok] wrote/updated {MET_IID / 'raw.csv'}")

if __name__ == "__main__":
    main()
