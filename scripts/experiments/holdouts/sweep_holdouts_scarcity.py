#!/usr/bin/env python
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from src.paths import ROOT, TEMPORAL_SPLIT, FAMILY_SPLIT
from src.holdouts import SplitIndices
from src.data.metadata import load_metadata
import pandas as pd
import numpy as np

# ---------------- helpers ----------------

def run_eval(kind_flag: str, tag: str, extra_args: list[str]) -> None:
    """Call eval_holdout with given args."""
    cmd = [sys.executable, "-m", "scripts.experiments.holdouts.eval_holdout",
           kind_flag, "--tag", tag] + list(map(str, extra_args))
    print("[run]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

def load_metric_json(kind: str, tag: str, metrics_subdir: str = "") -> dict | None:
    """
    eval_holdout writes rf_{kind}_metrics_{tag}.json, sometimes with a leading underscore.
    We look under data/processed/metrics[/<metrics_subdir>] for both forms.
    """
    base = ROOT / "data" / "processed" / "metrics"
    if metrics_subdir:
        base = base / metrics_subdir
    p1 = base / f"rf_{kind}_metrics_{tag}.json"
    p2 = base / f"rf_{kind}_metrics__{tag}.json"
    if p1.exists():
        return json.loads(p1.read_text())
    if p2.exists():
        return json.loads(p2.read_text())
    return None

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--use-temporal", action="store_true")
    g.add_argument("--use-family", action="store_true")

    ap.add_argument("--fractions", type=str, required=True,
                    help="Comma list, e.g. 0.001,0.002")
    ap.add_argument("--const-train-size", type=int, required=True)
    ap.add_argument("--gan-generator", type=str)
    ap.add_argument("--gan-scaler", type=str, default=None)

    ap.add_argument("--min-train-pos", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma list (overrides --seed), e.g. 42,1337,2025")

    ap.add_argument("--tag-prefix", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default=None)

    # model / eval knobs
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--grid", choices=["light","medium","heavy"], default="light")
    ap.add_argument("--val-threshold", choices=["none","f1","mcc","balacc"], default="balacc")
    ap.add_argument("--rf-class-weight", choices=["balanced","none"], default="none")
    ap.add_argument("--rf-max-depth", type=int, default=20)
    ap.add_argument("--rf-n-est", type=int, default=400)
    ap.add_argument("--balance-after-augment", action="store_true")

    # GAN quality knobs
    ap.add_argument("--gan-like", choices=["scarce","full"], default="full")
    ap.add_argument("--gan-synth-per-real", type=float, default=2.0)
    ap.add_argument("--gan-quality", choices=["none","nn","nn_boundary"], default="nn_boundary")
    ap.add_argument("--gan-qmult", type=float, default=5.0)

    # which variants to compare
    ap.add_argument("--compare", type=str, default="gan,oversample,smote",
                    help="Comma list among {gan,oversample,smote}")

    # rolling windows (temporal only)
    ap.add_argument("--test-window-days", type=int, default=None)
    ap.add_argument("--test-window-step-days", type=int, default=None)
    ap.add_argument("--n-test-windows", type=int, default=None)

    # where eval_holdout writes JSON/NPZ
    ap.add_argument("--metrics-subdir", type=str, default="",
                    help="Place JSON/NPZ under data/processed/metrics/<subdir>/")

    args = ap.parse_args()

    # seeds
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]

    # build windows
    windows = [(None, None)]  # full test by default
    kind_flag = "--use-temporal" if args.use_temporal else "--use-family"
    kind_name = "temporal" if args.use_temporal else "family"

    if args.use_temporal and args.test_window_days:
        split_path = TEMPORAL_SPLIT
        split = SplitIndices.from_json(split_path)
        meta = load_metadata()
        te_ts = meta.loc[split.test, "timestamp"].sort_values()
        if len(te_ts) == 0:
            raise SystemExit("[sweep] empty temporal test set")

        start = te_ts.min()    # keep tz-awareness if present
        end_max = te_ts.max()
        step_days = int(args.test_window_step_days or args.test_window_days)

        # how many windows at most
        nmax = args.n_test_windows
        if not nmax:
            total_days = (end_max - start) / pd.Timedelta(days=1)
            nmax = int(np.ceil(total_days / step_days))

        windows = []
        for k in range(nmax):
            wstart = start + pd.Timedelta(days=k * step_days)
            wend   = wstart + pd.Timedelta(days=args.test_window_days)
            if wstart >= end_max:
                break
            # preserve tz info when stringifying
            wstart_s = wstart.isoformat()
            wend_s   = wend.isoformat()
            windows.append((wstart_s, wend_s))

    # outputs
    prefix = args.tag_prefix
    out_csv = Path(args.out_csv) if args.out_csv else (
        ROOT / "data" / "processed" / "metrics" / f"holdout_scarcity_{prefix}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # parse fraction list and methods
    fracs = [float(x) for x in args.fractions.split(",") if x.strip()]
    methods = [m.strip() for m in args.compare.split(",") if m.strip()]

    # create CSV header if missing
    if not out_csv.exists():
        out_csv.write_text(",".join([
            "prefix","kind","frac","const_train_size","variant","used_gan","tag",
            "auc","pr_auc","accuracy","precision","recall","f1",
            "specificity","balanced_accuracy","mcc",
            "n_train_real","n_train_synth","n_train_total","tn","fp","fn","tp","threshold",
            "seed","test_start","test_end","test_window_days"
        ]) + "\n")

    def append_row(tag, variant, used_gan, frac, const_size, jd, seed, wstart, wend):
        with out_csv.open("a") as fcsv:
            fcsv.write(",".join(map(str, [
                prefix, kind_name, frac, const_size, variant, used_gan, tag,
                jd.get("auc","nan"), jd.get("pr_auc","nan"), jd.get("accuracy","nan"),
                jd.get("precision","nan"), jd.get("recall","nan"), jd.get("f1","nan"),
                jd.get("specificity","nan"), jd.get("balanced_accuracy","nan"), jd.get("mcc","nan"),
                jd.get("n_train_real","nan"), jd.get("n_train_synth","nan"), jd.get("n_train_total","nan"),
                jd.get("tn","nan"), jd.get("fp","nan"), jd.get("fn","nan"), jd.get("tp","nan"),
                jd.get("threshold","nan"),
                seed,
                (wstart or ""), (wend or ""),
                jd.get("test_window_days","")
            ])) + "\n")

    # flags common to every eval
    common = []
    if args.metrics_subdir:
        common += ["--metrics-subdir", args.metrics_subdir]
    common += ["--rf-n-est", str(args.rf_n_est)]
    common += ["--rf-class-weight", args.rf_class_weight]
    common += ["--rf-max-depth", str(args.rf_max_depth)]
    common += ["--val-threshold", args.val_threshold]
    if args.balance_after_augment:
        common += ["--balance-after-augment"]

    # sweep
    for w_i, (wstart, wend) in enumerate(windows):
        for seed in seeds:
            for f in fracs:
                frac_str = str(f).rstrip("0").rstrip(".")
                frac_tag = frac_str.replace(".", "")  # e.g. 0.002 -> 0002

                # tag suffix (window + seed)
                suff = ""
                if wstart is not None:
                    suff += f"_w{w_i:02d}"
                if args.seeds:
                    suff += f"_s{seed}"

                # -------- baseline (real only) --------
                tag_real = f"{prefix}{suff}_real{frac_tag}"
                extra = [
                    "--scarce-real-frac", str(f),
                    "--min-train-pos", str(args.min_train_pos),
                    "--seed", str(seed),
                ] + common
                if args.use_temporal and wstart is not None:
                    extra += ["--test-start", wstart, "--test-end", wend]
                if args.tune:
                    extra += ["--tune", "--grid", args.grid]

                run_eval(kind_flag, tag_real, extra)
                jd = load_metric_json(kind_name, tag_real, args.metrics_subdir)
                if not jd:
                    print(f"[warn] missing metrics for {tag_real}")
                else:
                    print(f"[{tag_real}] frac={100*f:.3f}% seed={seed}"
                          + (f" window={w_i}" if wstart else "")
                          + f" → AUC real={float(jd.get('auc', float('nan'))):.4f}")
                    append_row(tag_real, "real", False, f, "", jd, seed, wstart, wend)

                # -------- GAN --------
                if "gan" in methods:
                    if not args.gan_generator:
                        print("[warn] --compare includes 'gan' but no --gan-generator provided; skipping GAN")
                    else:
                        tag_gan = f"{prefix}{suff}_r{frac_tag}_gan{args.const_train_size}"
                        extra_g = extra + [
                            "--use-gan",
                            "--gan-generator", args.gan_generator,
                            "--const-train-size", str(args.const_train_size),
                            "--gan-like", args.gan_like,
                            "--gan-synth-per-real", str(args.gan_synth_per_real),
                            "--gan-quality", args.gan_quality,
                            "--gan-qmult", str(args.gan_qmult),
                        ]
                        if args.gan_scaler:
                            extra_g += ["--gan-scaler", args.gan_scaler]
                        run_eval(kind_flag, tag_gan, extra_g)
                        jd2 = load_metric_json(kind_name, tag_gan, args.metrics_subdir)
                        if not jd2:
                            print(f"[warn] missing metrics for {tag_gan}")
                        else:
                            print(f"[{tag_gan}] frac={100*f:.3f}% seed={seed}"
                                  + (f" window={w_i}" if wstart else "")
                                  + f" → AUC gan={float(jd2.get('auc', float('nan'))):.4f}")
                            append_row(tag_gan, "gan", True, f, args.const_train_size, jd2, seed, wstart, wend)

                # -------- oversample --------
                if "oversample" in methods:
                    tag_os = f"{prefix}{suff}_r{frac_tag}_os{args.const_train_size}"
                    extra_os = extra + ["--oversample", "--const-train-size", str(args.const_train_size)]
                    run_eval(kind_flag, tag_os, extra_os)
                    jd3 = load_metric_json(kind_name, tag_os, args.metrics_subdir)
                    if not jd3:
                        print(f"[warn] missing metrics for {tag_os}")
                    else:
                        print(f"[{tag_os}] frac={100*f:.3f}% seed={seed}"
                              + (f" window={w_i}" if wstart else "")
                              + f" → AUC oversample={float(jd3.get('auc', float('nan'))):.4f}")
                        append_row(tag_os, "oversample", False, f, args.const_train_size, jd3, seed, wstart, wend)

                # -------- smote --------
                if "smote" in methods:
                    tag_sm = f"{prefix}{suff}_r{frac_tag}_sm{args.const_train_size}"
                    extra_sm = extra + ["--smote", "--const-train-size", str(args.const_train_size)]
                    run_eval(kind_flag, tag_sm, extra_sm)
                    jd4 = load_metric_json(kind_name, tag_sm, args.metrics_subdir)
                    if not jd4:
                        print(f"[warn] missing metrics for {tag_sm}")
                    else:
                        print(f"[{tag_sm}] frac={100*f:.3f}% seed={seed}"
                              + (f" window={w_i}" if wstart else "")
                              + f" → AUC smote={float(jd4.get('auc', float('nan'))):.4f}")
                        append_row(tag_sm, "smote", False, f, args.const_train_size, jd4, seed, wstart, wend)

    print(f"[done] Results appended to {out_csv}")

if __name__ == "__main__":
    main()
