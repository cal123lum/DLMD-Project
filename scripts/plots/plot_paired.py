#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ---------------- formatting ----------------

def pct_fmt(x, pos):
    return f"{x*100:.3f}%"

def _plot_lines(ax, x, series, labels, title, ylabel, logx=False, bands=None):
    for i, (y, label) in enumerate(zip(series, labels)):
        ax.plot(x, y, marker='o', linewidth=2, label=label)
        if bands is not None and bands[i] is not None:
            s = bands[i]
            ax.fill_between(x, y - s, y + s, alpha=0.15)
    ax.set_title(title)
    ax.set_xlabel("Real fraction of TRAIN")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.5)
    if logx:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(pct_fmt))
    ax.legend()

def _plot_delta(ax, x, delta, title, ylabel="Δ (aug - real)", logx=False):
    ax.axhline(0.0, linewidth=1, linestyle="--")
    ax.plot(x, delta, marker='o', linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Real fraction of TRAIN")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.5)
    if logx:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(pct_fmt))

# ---------------- helpers ----------------

METRICS_RAW = dict(
    auc="auc",
    pr_auc="pr_auc",
    f1="f1",
    balanced_accuracy="balanced_accuracy",
    mcc="mcc",
)

def _agg_raw_by_variant(df_raw, metrics):
    """
    df_raw: raw.csv schema with columns:
      prefix, kind, frac, const_train_size, variant (real/gan/oversample/smote),
      auc, pr_auc, f1, balanced_accuracy, mcc, seed, test_start, test_end
    Returns aggregated per (prefix, const_train_size, frac, variant):
      mean & std for each metric.
    """
    keep_cols = ["prefix","const_train_size","frac","variant"] + list(METRICS_RAW[m] for m in metrics)
    missing = [c for c in keep_cols if c not in df_raw.columns]
    if missing:
        raise SystemExit(f"[raw] missing required columns: {missing}")

    df = df_raw[keep_cols].copy()
    # coerce numerics
    df["const_train_size"] = pd.to_numeric(df["const_train_size"], errors="coerce")
    df["frac"] = pd.to_numeric(df["frac"], errors="coerce")

    gp = df.groupby(["prefix","const_train_size","frac","variant"], dropna=False)
    agg = {}
    for m in metrics:
        col = METRICS_RAW[m]
        agg[col] = ["mean","std"]
    out = gp.agg(agg).reset_index()

    # flatten columns
    out.columns = ["_".join([c for c in tup if c]).rstrip("_") for tup in out.columns.to_flat_index()]
    # rename back to consistent names like m_mean/m_std
    ren = {}
    for m in metrics:
        col = METRICS_RAW[m]
        ren[f"{col}_mean"] = f"{m}_mean"
        ren[f"{col}_std"]  = f"{m}_std"
    out = out.rename(columns=ren)
    return out

def _shim_paired_cv(df, metrics):
    """
    Accept paired_cv.csv and expose columns as generic *_real/_aug + *_real_std/_aug_std
    so we can still draw 'real vs aug' (single augmented line). No method info here.
    """
    base = {}
    for side in ("real","aug"):
        for m in metrics:
            mean_col = f"{m}_{side}_mean"
            std_col  = f"{m}_{side}_std"
            if mean_col not in df.columns:
                raise SystemExit(f"[paired_cv] missing column: {mean_col}")
            base[f"{m}_{side}"] = df[mean_col]
            base[f"{m}_{side}_std"] = df[std_col] if std_col in df.columns else np.nan
    base_df = pd.DataFrame(base)
    base_df.insert(0, "frac", pd.to_numeric(df["frac"], errors="coerce"))
    base_df.insert(0, "const_train_size", pd.to_numeric(df["const_train_size"], errors="coerce"))
    base_df.insert(0, "prefix", df["prefix"])
    return base_df

def _collapse_means(df, keys, cols_mean_std_pairs):
    """
    Collapse duplicates by keys, averaging means and combining std via:
      sqrt( mean(std^2) + var(mean) ).
    """
    rows = []
    for k, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        for m_mean, m_std in cols_mean_std_pairs:
            means = pd.to_numeric(g[m_mean], errors="coerce").values
            mu = float(np.nanmean(means)) if means.size else np.nan
            var_between = float(np.nanvar(means, ddof=1)) if means.size > 1 else 0.0
            if m_std in g.columns:
                stds = pd.to_numeric(g[m_std], errors="coerce").values
                var_within = float(np.nanmean(stds**2))
            else:
                var_within = 0.0
            row[m_mean] = mu
            row[m_std]  = np.sqrt(max(var_within + var_between, 0.0))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(keys)

# ---------------- main plotting logic ----------------

def plot_from_raw(df_raw, out_dir, metrics, logx, draw_bands):
    agg = _agg_raw_by_variant(df_raw, metrics)

    # split variants
    real = agg[agg["variant"] == "real"].copy()
    methods = {m: agg[agg["variant"] == m].copy() for m in ("gan", "oversample", "smote")}

    # collapse REAL by (prefix, frac)  (const_train_size is NaN for real in raw.csv)
    cols = [(f"{m}_mean", f"{m}_std") for m in metrics]
    real_c = _collapse_means(real, ["prefix", "frac"], cols)

    # collapse METHODS by (prefix, frac) as well (we’ll ignore const for alignment)
    methods_c = {k: (_collapse_means(v, ["prefix", "frac"], cols) if not v.empty else None)
                 for k, v in methods.items()}

    # loop per prefix (ignore const in grouping)
    for prefix, real_g in real_c.groupby("prefix", dropna=False):
        real_g = real_g.sort_values("frac")
        x = real_g["frac"].values

        # get a display value for const_train_size from method runs for this prefix
        const_vals = []
        for v in methods.values():
            if v is not None:
                const_vals += list(pd.to_numeric(
                    v.loc[v["prefix"] == prefix, "const_train_size"], errors="coerce"
                ).dropna().unique())
        if len(const_vals) == 1:
            const_disp = int(const_vals[0])
        elif len(const_vals) == 0:
            const_disp = "NA"
        else:
            const_disp = "/".join(sorted({str(int(c)) for c in const_vals}))

        # align each method to real's x by fraction
        aligned = {}
        for name, dfm in methods_c.items():
            if dfm is None:
                aligned[name] = None
            else:
                sub = dfm[dfm["prefix"] == prefix]
                aligned[name] = pd.merge(
                    real_g[["frac"]],
                    sub[["frac"] + [f"{m}_mean" for m in metrics] +
                         [f"{m}_std" for m in metrics]],
                    on="frac", how="left"
                )

        out_dir.mkdir(parents=True, exist_ok=True)

        for m in metrics:
            # Real
            series = []
            bands  = []
            labels = []

            r = real_g[f"{m}_mean"].values
            r_std = real_g[f"{m}_std"].values if draw_bands else None
            series.append(r); bands.append(r_std); labels.append("real")

            # Methods
            deltas = {}
            for name in ("gan", "oversample", "smote"):
                dfm = aligned[name]
                if dfm is None or f"{m}_mean" not in (dfm.columns if dfm is not None else []):
                    y = np.full_like(x, np.nan, dtype=float)
                    y_std = None
                else:
                    y = dfm[f"{m}_mean"].values
                    y_std = dfm[f"{m}_std"].values if draw_bands else None
                series.append(y); bands.append(y_std); labels.append(name)
                deltas[name] = y - r

            # Main 4-line plot
            fig, ax = plt.subplots(figsize=(7.8, 4.6))
            title = f"{prefix}  |  const_train_size={const_disp}  |  {m.upper()} (real vs GAN/oversample/SMOTE)"
            _plot_lines(ax, x, series, labels, title, m.upper(), logx=logx, bands=bands)
            fig.tight_layout()
            fig.savefig(out_dir / f"{prefix}_{m}_real_vs_methods.png", dpi=150)
            plt.close(fig)

            # Δ(GAN - real) only
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            title = f"{prefix}  |  const_train_size={const_disp}  |  Δ{m.upper()} (gan - real)"
            _plot_delta(ax, x, deltas["gan"], title, ylabel=f"Δ{m.upper()}", logx=logx)
            fig.tight_layout()
            fig.savefig(out_dir / f"{prefix}_delta_{m}_gan.png", dpi=150)
            plt.close(fig)


def plot_from_paired_cv(df_cv, out_dir, metrics, logx, draw_bands):
    # Can only do "real vs aug" (no method info here)
    base = _shim_paired_cv(df_cv, metrics)
    for (prefix, const_size), g in base.groupby(["prefix","const_train_size"], dropna=False):
        g = g.sort_values("frac")
        x = g["frac"].values
        for m in metrics:
            real = g[f"{m}_real"].values
            aug  = g[f"{m}_aug"].values
            real_std = g[f"{m}_real_std"].values if draw_bands else None
            aug_std  = g[f"{m}_aug_std"].values  if draw_bands else None

            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            title = f"{prefix}  |  const_train_size={int(const_size) if not np.isnan(const_size) else 'NA'}  |  {m.upper()} (real vs aug)"
            _plot_lines(ax, x, [real, aug], ["real","augmented"], title, m.upper(), logx=logx,
                        bands=[real_std, aug_std] if draw_bands else None)
            fig.tight_layout()
            out_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_dir / f"{prefix}_{m}_real_vs_aug.png", dpi=150)
            plt.close(fig)

# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", help="raw.csv OR paired.csv / paired_cv.csv")
    ap.add_argument("--metrics", type=str, default="auc,f1",
                    help="Comma list from: auc,pr_auc,f1,balanced_accuracy,mcc")
    ap.add_argument("--logx", action="store_true")
    ap.add_argument("--no-bands", dest="no_bands", action="store_true",
                    help="Disable std shading bands")
    args = ap.parse_args()

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    allowed = {"auc","pr_auc","f1","balanced_accuracy","mcc"}
    bad = set(metrics) - allowed
    if bad:
        raise SystemExit(f"Unknown metrics: {bad}. Allowed: {sorted(allowed)}")

    for pth in args.csvs:
        p = Path(pth)
        if not p.exists():
            print(f"[warn] missing file: {p}")
            continue

        df = pd.read_csv(p)

        # Decide mode by columns
        if {"variant","auc","pr_auc","f1","balanced_accuracy","mcc"}.issubset(df.columns):
            # RAW MODE → can draw 4 lines (real+three methods)
            out_dir = p.parent / "plots"
            plot_from_raw(df, out_dir, metrics, args.logx, draw_bands=(not args.no_bands))
            print(f"[ok] wrote method plots from raw: {out_dir}")
        elif any(col.endswith("_real_mean") for col in df.columns):
            # PAIRED_CV MODE → only real vs aug
            out_dir = p.parent / "plots"
            plot_from_paired_cv(df, out_dir, metrics, args.logx, draw_bands=(not args.no_bands))
            print(f"[ok] wrote real vs aug plots from paired_cv: {out_dir}")
            print("[info] For per-method lines (gan/oversample/smote), pass raw.csv instead.")
        else:
            print(f"[warn] {p} has an unrecognized schema. Expect raw.csv or paired_cv.csv columns.")

if __name__ == "__main__":
    main()
