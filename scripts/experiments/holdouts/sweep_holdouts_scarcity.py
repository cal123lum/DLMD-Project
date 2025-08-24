#!/usr/bin/env python
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
from src.paths import ROOT
import pandas as pd, numpy as np
from src.holdouts import SplitIndices
from src.data.metadata import load_metadata
from src.paths import TEMPORAL_SPLIT, FAMILY_SPLIT

def run_eval(kind_flag, tag, extra_args):
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

    ap.add_argument("--fractions", type=str, required=True)
    ap.add_argument("--const-train-size", type=int, required=True)
    ap.add_argument("--gan-generator", type=str, required=False)
    ap.add_argument("--min-train-pos", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag-prefix", type=str, required=True)
    ap.add_argument("--out-csv", type=str, default=None)
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--grid", choices=["light","medium","heavy"], default="light")
    ap.add_argument("--val-threshold", choices=["none","f1"], default="f1",
                    help="Pick decision threshold on validation (default=f1)")
    ap.add_argument("--compare", type=str, default="gan,oversample,smote",
                    help="Comma list among {gan,oversample,smote}")
    ap.add_argument("--seeds", type=str, default=None,
                    help="Comma list of seeds, e.g. 42,1337,2025 (default: use --seed)")
    ap.add_argument("--test-window-days", type=int, default=None,
                    help="If set (temporal only), test on rolling fixed-length windows")
    ap.add_argument("--test-window-step-days", type=int, default=None,
                    help="Step between windows (defaults to window length)")
    ap.add_argument("--n-test-windows", type=int, default=None,
                    help="If set, limit number of windows from test start")


    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    # Build rolling windows (temporal only) or a single full-test window
    windows = [(None, None)]  # sentinel = full test
    if args.use_temporal and args.test_window_days:
        split = SplitIndices.from_json(TEMPORAL_SPLIT)
        meta = load_metadata()
        te_ts = meta.loc[split.test, "timestamp"].sort_values()
        if len(te_ts) == 0:
            raise SystemExit("[sweep] empty temporal test set")
        start = te_ts.min()
        end_max = te_ts.max()
        step_days = args.test_window_step_days or args.test_window_days
        nmax = args.n_test_windows or int(np.ceil((end_max - start) / pd.Timedelta(days=step_days)))
        windows = []
        for k in range(nmax):
            wstart = start + pd.Timedelta(days=k * step_days)
            wend   = wstart + pd.Timedelta(days=args.test_window_days)
            if wstart >= end_max:
                break
            windows.append((wstart.isoformat(), wend.isoformat()))

    kind_flag = "--use-temporal" if args.use_temporal else "--use-family"
    prefix = args.tag_prefix
    out_csv = Path(args.out_csv) if args.out_csv else (
        ROOT / "data" / "processed" / "metrics" / f"holdout_scarcity_{prefix}.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fracs = [float(x) for x in args.fractions.split(",") if x.strip()]
    methods = [m.strip() for m in args.compare.split(",") if m.strip()]

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
                prefix, ('temporal' if args.use_temporal else 'family'), frac, const_size, variant, used_gan, tag,
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

    # for each fraction: baseline + selected augmentation methods
    for w_i, (wstart, wend) in enumerate(windows):
        for seed in seeds:
            for f in fracs:
                frac_str = str(f).rstrip("0").rstrip(".")
                frac_tag = frac_str.replace(".", "")  # e.g., 0.002 -> 0002

                # window/seed suffix for tag
                suff = ""
                if wstart is not None:
                    suff += f"_w{w_i:02d}"
                if args.seeds:
                    suff += f"_s{seed}"

                # --- baseline (real only) ---
                tag_real = f"{prefix}{suff}_real{frac_tag}"
                extra = ["--scarce-real-frac", str(f),
                        "--min-train-pos", str(args.min_train_pos),
                        "--seed", str(seed),
                        "--val-threshold", args.val_threshold]
                if args.use_temporal and wstart is not None:
                    extra += ["--test-start", wstart, "--test-end", wend]
                if args.tune:
                    extra += ["--tune", "--grid", args.grid]

                mpath = run_eval(kind_flag, tag_real, extra)
                jd = json.loads(mpath.read_text())
                print(f"[{tag_real}] frac={100*f:.3f}% seed={seed}"
                    + (f" window={w_i}" if wstart else "")
                    + f" → AUC real={float(jd.get('auc', float('nan'))):.4f}")
                append_row(tag_real, "real", False, f, "", jd, seed, wstart, wend)

                # --- compare variants ---
                if "gan" in methods:
                    if not args.gan_generator:
                        print("[warn] --compare includes 'gan' but no --gan-generator provided; skipping GAN")
                    else:
                        tag_gan = f"{prefix}{suff}_r{frac_tag}_gan{args.const_train_size}"
                        extra_g = extra + ["--use-gan", "--gan-generator", args.gan_generator,
                                        "--gan-like", "scarce", "--const-train-size", str(args.const_train_size)]
                        mpath2 = run_eval(kind_flag, tag_gan, extra_g)
                        jd2 = json.loads(mpath2.read_text())
                        print(f"[{tag_gan}] frac={100*f:.3f}% seed={seed}"
                            + (f" window={w_i}" if wstart else "")
                            + f" → AUC gan={float(jd2.get('auc', float('nan'))):.4f}")
                        append_row(tag_gan, "gan", True, f, args.const_train_size, jd2, seed, wstart, wend)

                if "oversample" in methods:
                    tag_os = f"{prefix}{suff}_r{frac_tag}_os{args.const_train_size}"
                    extra_os = extra + ["--oversample", "--const-train-size", str(args.const_train_size)]
                    mpath3 = run_eval(kind_flag, tag_os, extra_os)
                    jd3 = json.loads(mpath3.read_text())
                    print(f"[{tag_os}] frac={100*f:.3f}% seed={seed}"
                        + (f" window={w_i}" if wstart else "")
                        + f" → AUC oversample={float(jd3.get('auc', float('nan'))):.4f}")
                    append_row(tag_os, "oversample", False, f, args.const_train_size, jd3, seed, wstart, wend)

                if "smote" in methods:
                    tag_sm = f"{prefix}{suff}_r{frac_tag}_sm{args.const_train_size}"
                    extra_sm = extra + ["--smote", "--const-train-size", str(args.const_train_size)]
                    mpath4 = run_eval(kind_flag, tag_sm, extra_sm)
                    jd4 = json.loads(mpath4.read_text())
                    print(f"[{tag_sm}] frac={100*f:.3f}% seed={seed}"
                        + (f" window={w_i}" if wstart else "")
                        + f" → AUC smote={float(jd4.get('auc', float('nan'))):.4f}")
                    append_row(tag_sm, "smote", False, f, args.const_train_size, jd4, seed, wstart, wend)


    print(f"[done] Results appended to {out_csv}")

if __name__ == "__main__":
    main()
