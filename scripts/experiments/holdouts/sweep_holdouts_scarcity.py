#!/usr/bin/env python
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path

from src.paths import ROOT


def run_eval(kind_flag: str, tag: str, extra_args: list[str]) -> Path:
    """Run eval_holdout once and return the path to the metrics JSON it wrote."""
    kind = "temporal" if kind_flag == "--use-temporal" else "family"
    cmd = [sys.executable, "-m", "scripts.experiments.holdouts.eval_holdout",
           kind_flag, "--tag", tag] + extra_args
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    return ROOT / "data" / "processed" / "metrics" / f"rf_{kind}_metrics_{tag}.json"


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--use-temporal", action="store_true")
    g.add_argument("--use-family", action="store_true")

    ap.add_argument("--fractions", type=str, required=True,
                    help="Comma-separated real fractions (e.g., 0.0005,0.001,0.002)")
    ap.add_argument("--const-train-size", type=int, required=True,
                    help="Total train size target for augmented runs (baseline ignores this)")
    ap.add_argument("--gan-generator", type=str, required=True,
                    help="Path to leakage-safe generator.pth trained on the split's TRAIN")
    ap.add_argument("--min-train-pos", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag-prefix", type=str, required=True,
                    help="Prefix to identify this split, e.g., cut2019_09 or fam_sfone")
    ap.add_argument("--out-csv", type=str, default=None,
                    help="CSV to append results to (default: data/processed/metrics/holdout_scarcity_<prefix>.csv)")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--grid", choices=["light", "medium", "heavy"], default="light")

    args = ap.parse_args()
    kind_flag = "--use-temporal" if args.use_temporal else "--use-family"
    prefix = args.tag_prefix
    out_csv = Path(args.out_csv) if args.out_csv else (ROOT / "data" / "processed" / "metrics" / f"holdout_scarcity_{prefix}.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fracs = [float(x) for x in args.fractions.split(",") if x.strip()]

    # Header
    if not out_csv.exists():
        out_csv.write_text(",".join([
            "prefix","kind","frac","const_train_size","used_gan","tag",
            "auc","pr_auc","accuracy","precision","recall","f1",
            "specificity","balanced_accuracy","mcc",
            "n_train_real","n_train_synth","n_train_total","tn","fp","fn","tp"
        ]) + "\n")

    def append_row(tag: str, used_gan: bool, frac: float, const_size: str | int, jd: dict):
        with out_csv.open("a") as fcsv:
            fcsv.write(",".join(map(str, [
                prefix, ('temporal' if args.use_temporal else 'family'), frac, const_size, used_gan, tag,
                jd.get("auc", "nan"), jd.get("pr_auc", "nan"), jd.get("accuracy", "nan"),
                jd.get("precision", "nan"), jd.get("recall", "nan"), jd.get("f1", "nan"),
                jd.get("specificity", "nan"), jd.get("balanced_accuracy", "nan"), jd.get("mcc", "nan"),
                jd.get("n_train_real", "nan"), jd.get("n_train_synth", "nan"), jd.get("n_train_total", "nan"),
                jd.get("tn", "nan"), jd.get("fp", "nan"), jd.get("fn", "nan"), jd.get("tp", "nan"),
            ])) + "\n")

    # For each fraction: run baseline (real-only) and augmented (const size)
    for f in fracs:
        frac_str = str(f).rstrip("0").rstrip(".")
        frac_tag = frac_str.replace(".", "")  # e.g., 0.002 -> 0002

        # --- baseline ---
        tag_real = f"{prefix}_real{frac_tag}"    
        extra = ["--scarce-real-frac", str(f), "--min-train-pos", str(args.min_train_pos), "--seed", str(args.seed)]
        if args.tune:
            extra += ["--tune", "--grid", args.grid]
        mpath = run_eval(kind_flag, tag_real, extra)
        jd = json.loads(mpath.read_text())
        print(f"frac={100*f:.3f}% → AUC real={float(jd['auc']):.4f}")
        append_row(tag_real, False, f, "", jd)

        # --- augmented (GAN) ---
        tag_gan = f"{prefix}_r{frac_tag}_gan{args.const_train_size}"
        extra_aug = extra + [
            "--use-gan",
            "--gan-generator", args.gan_generator,
            "--gan-like", "scarce",
            "--const-train-size", str(args.const_train_size),
        ]
        mpath2 = run_eval(kind_flag, tag_gan, extra_aug)
        jd2 = json.loads(mpath2.read_text())
        print(f"frac={100*f:.3f}% → AUC aug ={float(jd2['auc']):.4f}")
        append_row(tag_gan, True, f, args.const_train_size, jd2)

    print(f"[done] Results appended to {out_csv}")


if __name__ == "__main__":
    main()
