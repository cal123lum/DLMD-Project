#!/usr/bin/env python3
import subprocess, sys, json
from pathlib import Path
import pandas as pd
from src.paths import ROOT

CUTS = [
    "2019-10-01","2019-12-01","2020-03-01","2020-06-01","2020-09-01"
]
FRACS = [0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.0045,0.005,0.0055,0.006,0.0065,0.007,0.0075,0.008,0.0085,0.009,0.0095,0.01]
SEEDS = [42,1337,2025]

MET_ROOT = ROOT / "data" / "processed" / "metrics" / "temporal_final"
GAN_ROOT = ROOT / "models" / "gan" / "temporal_final"

def run(cmd):
    print("[run]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)

def append_raw_row(pth: Path, row: dict):
    pth.parent.mkdir(parents=True, exist_ok=True)
    if pth.exists():
        df = pd.read_csv(pth)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(pth, index=False)

def load_metric_json(tag: str):
    b = MET_ROOT
    p1 = b / f"rf_temporal_metrics_{tag}.json"
    p2 = b / f"rf_temporal_metrics__{tag}.json"
    if p1.exists(): return json.loads(p1.read_text())
    if p2.exists(): return json.loads(p2.read_text())
    return None

def main():
    for cutoff in CUTS:
        # 1) make/verify split for this cutoff
        run([sys.executable, "-m", "scripts.experiments.holdouts.make_holdouts", "--temporal-cutoff", cutoff])
        run([sys.executable, "-m", "scripts.experiments.holdouts.verify_holdouts"])

        prefix = f"cut{cutoff[:4]}_{cutoff[5:7]}"
        out_csv = MET_ROOT / prefix / "raw.csv"
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        # 2) GAN paths & train fresh
        gendir = GAN_ROOT / prefix
        gendir.mkdir(parents=True, exist_ok=True)
        gen = gendir / "generator.pth"
        scal = gendir / "scaler.npz"

        run([sys.executable, "-m", "scripts.gan.train_gan",
             "--indices-json", str(ROOT/"data/holdouts/temporal_indices.json"),
             "--malware-only", "--out", str(gen),
             "--epochs", "30", "--n-critic", "5", "--batch-size", "128",
             "--device", "auto", "--lr", "1e-4", "--lambda-gp", "10.0"])

        if not scal.exists():
            run([sys.executable, "-m", "scripts.utils.make_gan_scaler",
                 "--indices-json", str(ROOT/"data/holdouts/temporal_indices.json"),
                 "--out", str(scal)])

        # 3) evaluate across fracs × seeds × methods
        common = [
            "--val-threshold","balacc",
            "--rf-class-weight","none",
            "--rf-max-depth","20",
            "--rf-n-est","400",
            "--metrics-subdir","temporal_final",
        ]
        methods = [
            ("real", []),
            ("gan", [
                "--use-gan",
                "--gan-generator", str(gen),
                "--gan-scaler",    str(scal),
                "--gan-like","full",
                "--gan-synth-per-real","2",
                "--gan-quality","nn_boundary",
                "--gan-qmult","5",
            ]),
            ("oversample", ["--oversample"]),
            ("smote",      ["--smote"]),
            ("evo", [
                "--use-evo",
                "--evo-like","full",
                "--evo-synth-per-real","2",
                "--evo-quality","nn_boundary",
                "--evo-qmult","5",
                "--evo-mutate-sigma","0.10",
                "--evo-cx-alpha","2.0",
                "--evo-qlow","0.01","--evo-qhigh","0.99",
                "--evo-boundary-low","0.20","--evo-boundary-high","0.60",
                "--evo-boundary-k","5",
            ]),
            ("evogan", [
                "--use-gan","--gan-evo-refine",
                "--gan-generator", str(gen),
                "--gan-scaler",    str(scal),
                "--gan-like","full",
                "--gan-synth-per-real","2",
                "--gan-quality","nn_boundary",
                "--gan-qmult","5",
                "--evo-parent-source","gan",
                "--evo-mutate-sigma","0.10",
                "--evo-cx-alpha","2.0",
                "--evo-qlow","0.01","--evo-qhigh","0.99",
                "--evo-boundary-low","0.20","--evo-boundary-high","0.60",
                "--evo-boundary-k","5",
            ]),
        ]

        for frac in FRACS:
            for seed in SEEDS:
                for variant, extra in methods:
                    tag = f"{prefix}_f{frac}_s{seed}_{variant}"
                    cmd = [sys.executable, "-m", "scripts.experiments.holdouts.eval_holdout",
                           "--use-temporal",
                           "--scarce-real-frac", str(frac),
                           "--min-train-pos", "50", "--min-train-neg","50",
                           "--const-train-size","20000",
                           "--split-json", str(ROOT/"data/holdouts/temporal_indices.json"),
                           "--seed", str(seed), "--tag", tag]
                    cmd += common; cmd += extra
                    try:
                        run(cmd)
                    except subprocess.CalledProcessError:
                        print(f"[warn] failed: {tag}")
                        continue

                    m = load_metric_json(tag)
                    if not m: 
                        print(f"[warn] missing metrics for {tag}")
                        continue

                    row = dict(
                        prefix=prefix, kind="temporal", frac=float(frac),
                        const_train_size=20000, variant=m.get("variant","real"),
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
                    )
                    append_raw_row(out_csv, row)
        print(f"[ok] wrote/updated {out_csv}")

if __name__ == "__main__":
    main()
