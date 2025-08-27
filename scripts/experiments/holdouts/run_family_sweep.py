#!/usr/bin/env python3
import argparse, json, subprocess, sys, shlex
from pathlib import Path
import pandas as pd
import json
from src.paths import ROOT

SPLITS_DIR = ROOT / "data" / "processed" / "splits" / "family_lofo"
GAN_DIR    = ROOT / "models" / "gan" / "family"
MET_FAM    = ROOT / "data" / "processed" / "metrics" / "family"

# scarcity levels & robustness seeds
FRACS = [0.0005, 0.001, 0.002, 0.005, 0.01]
SEEDS = [42, 1337, 2025]

# train-time constraints
CONST_TRAIN_SIZE = 20000         # augment-to size (like temporal)
MIN_TRAIN_POS    = 50            # ensure some positives before augment

def run(cmd):
    print("[run]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)

def load_metric_json(tag: str):
    """eval_holdout writes 'rf_{kind}_metrics_{tag}.json' but it internally
    prefixes the tag with an underscore; try both to be safe."""
    base = ROOT / "data" / "processed" / "metrics"
    p1 = base / f"rf_family_metrics_{tag}.json"
    p2 = base / f"rf_family_metrics__{tag}.json"  # leading underscore
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", type=str, default=None,
                    help="Comma list to filter (e.g., picsys,autoit). Default = all split files.")
    ap.add_argument("--max-gan-malware", type=int, default=20000,
                    help="Cap malware rows for GAN training subset.")
    ap.add_argument("--epochs", type=int, default=30,
                    help="GAN epochs (capped subset keeps this fast).")
    ap.add_argument("--skip-gan-train", action="store_true",
                    help="Reuse existing generator if present.")
    # GAN hyper-params passthrough
    ap.add_argument("--n-critic", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", choices=["auto","cpu","cuda","mps"], default="auto")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lambda-gp", type=float, default=10.0)
    ap.add_argument("--gan-extra", type=str, default="", help="extra args for train_gan (quoted)")
    args = ap.parse_args()

    splits = sorted(SPLITS_DIR.glob("holdout_*.json"))
    if args.families:
        keep = {x.strip().lower() for x in args.families.split(",") if x.strip()}
        splits = [p for p in splits if p.stem.replace("holdout_","").lower() in keep]

    if not splits:
        print("[err] no LOFO splits found", file=sys.stderr)
        sys.exit(1)

    for sp in splits:
        family = sp.stem.replace("holdout_","")
        print(f"\n=== [{family}] ===")
        fam_gan_dir = GAN_DIR / family
        fam_gan_dir.mkdir(parents=True, exist_ok=True)
        subset_json = fam_gan_dir / "gan_subset_indices.json"
        gen_path    = fam_gan_dir / "generator.pth"
        scaler_path = fam_gan_dir / "scaler.npz"

        # -----------------------------
        # 1) Build capped malware-only subset for this family's LOFO train
        # -----------------------------
        run([
            sys.executable, "-m", "scripts.experiments.holdouts.make_family_gan_subset",
            "--split-json", str(sp),
            "--max-train-rows", str(args.max_gan_malware),
            "--stratify-by-family",
            "--out", str(subset_json),
        ])

        # -----------------------------
        # 2) Train GAN (skip if exists and user asked to skip)
        # -----------------------------
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

        # -----------------------------
        # 3) Fit GAN scaler on the same subset (malware rows in LOFO-train)
        # -----------------------------
        if not scaler_path.exists():
            run([
                sys.executable, "-m", "scripts.utils.make_gan_scaler",
                "--indices-json", str(subset_json),
                "--out", str(scaler_path),
            ])

        # -----------------------------
        # 4) Evaluate across scarcity × seeds × methods
        # -----------------------------
        outdir  = MET_FAM / family
        raw_csv = outdir / "raw.csv"

        # common eval flags (do NOT duplicate these in per-method extras)
        common = [
            "--val-threshold", "balacc",
            "--balance-after-augment",
            "--rf-class-weight", "none",
            "--rf-max-depth", "20",
        ]

        methods = [
            ("real",       []),
            ("gan",        [
                "--use-gan",
                "--gan-generator", str(gen_path),
                "--gan-scaler",    str(scaler_path),
                "--gan-like", "full",                 # or "scarce" if you prefer
                "--gan-synth-per-real", "2",
                "--gan-quality", "nn_boundary",
                "--gan-qmult", "5",
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
                    # IMPORTANT: extend (flatten) — do NOT append the list as one element
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
                        const_train_size=int(CONST_TRAIN_SIZE),
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
                    append_raw_row(raw_csv, row)

        print(f"[ok] wrote/updated {raw_csv}")


if __name__ == "__main__":
    main()
