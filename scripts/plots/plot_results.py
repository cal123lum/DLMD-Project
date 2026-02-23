# scripts/plots/plot_results.py
# Results plotting CLI for IID, temporal, and family LOFO
# Author: Callum Musselwhite
# Last edit: 2025-09-17

import argparse
import os
import sys
import glob as _glob
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pathlib
import re
from matplotlib.ticker import MultipleLocator, AutoMinorLocator


# constants and palette
ALLOWED_METHODS = ["real", "gan", "evogan", "evo", "smote", "oversample"]
LEGEND_ORDER = ["real", "gan", "evogan", "evo", "smote", "oversample"]
COLOR = {
    "real": "#1f77b4",
    "gan": "#ff7f0e",
    "evogan": "#8c564b",
    "evo": "#9467bd",
    "oversample": "#2ca02c",
    "smote": "#d62728",
}
COLORS = COLOR
METRIC_COLUMNS = {"auc", "pr_auc", "f1", "balanced_accuracy", "mcc"}


# small helpers
def _sanitize_name(s: str) -> str:
    s = str(s)
    s = s.replace(os.sep, "_")
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    return s

def _parse_ylim(s):
    if not s:
        return None
    lo, hi = map(float, s.split(","))
    return (lo, hi)

def percent_axis(ax):
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))

def ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)

def guess_outdir_from_first_csv(csvs: List[str]) -> str:
    base = os.path.dirname(os.path.abspath(csvs[0]))
    return os.path.join(base, "plots")

def expand_csvs(paths: List[str]) -> List[str]:
    expanded = []
    for p in paths:
        matches = _glob.glob(p)
        if matches:
            expanded.extend(sorted(matches))
        else:
            expanded.append(p)  # let pandas raise if missing
    return expanded


def derive_variant_from_tag(df: pd.DataFrame) -> pd.DataFrame:
    # best effort: infer variant from the tag suffix when 'variant' column is missing
    df = df.copy()
    if "variant" not in df.columns and "tag" in df.columns:
        def derive(s: str) -> Optional[str]:
            if not isinstance(s, str):
                s = str(s)
            parts = s.split("_")
            cand = parts[-1].lower() if parts else None
            return cand
        df["variant"] = df["tag"].apply(derive)
    if "variant" in df.columns:
        df["variant"] = df["variant"].astype(str).str.strip().str.lower()
        alias = {"baseline": "real", "realonly": "real", "evogan_gp": "evogan", "evo_gan": "evogan"}
        df["variant"] = df["variant"].map(lambda v: alias.get(v, v))
    return df

def infer_family_from_path(df: pd.DataFrame, csv_path: str) -> pd.DataFrame:
    # infer family name from path segment if not provided in CSV
    df = df.copy()
    path_parts = pathlib.Path(csv_path).parts
    fam = None
    for i, part in enumerate(path_parts[:-1]):
        if part.lower() == "family_final" and i + 1 < len(path_parts):
            fam = path_parts[i + 1]
            break
    if "family" not in df.columns and fam is not None:
        df["family"] = fam
    return df

def add_regime_column(df: pd.DataFrame) -> pd.DataFrame:
    # tag each row as iid, temporal, or family for downstream grouping
    df = df.copy()
    cols = set(df.columns)
    if "test_start" in cols:
        df["regime"] = "temporal"
    elif "family" in cols:
        df["regime"] = "family"
    else:
        df["regime"] = "iid"
    return df

def _sd_lines(df, metric, regime):
    # compute SD bands in a regime-aware way
    d = df.copy()
    d["frac"] = pd.to_numeric(d["frac"], errors="coerce").round(6)
    d = d.dropna(subset=["frac", metric, "variant"])
    if regime == "iid":
        slice_cols = ["seed"]
    elif regime == "temporal":
        if "test_start" not in d.columns:
            return pd.DataFrame()
        d["window_idx"] = d.groupby("prefix")["test_start"].rank(method="dense").astype(int)
        slice_cols = ["prefix", "window_idx"]
    elif regime == "family":
        if "family" not in d.columns:
            return pd.DataFrame()
        slice_cols = ["family"]
    else:
        return pd.DataFrame()
    grp_cols = slice_cols + ["frac", "variant"]
    tidy = d.groupby(grp_cols, as_index=False)[metric].mean()
    out = tidy.groupby(["frac", "variant"], as_index=False).agg(sd=(metric, "std"), n=("variant", "size"))
    return out

def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    # make numeric and datetime columns reliable for plotting and grouping
    df = df.copy()
    for c in ["frac", "seed", "n_train_real", "n_train_synth", "n_train_total", "tn", "fp", "fn", "tp", "threshold"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if 'frac' in df.columns:
        df['frac'] = pd.to_numeric(df['frac'], errors='coerce').round(6)
    for c in ["test_start", "test_end"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "variant" in df.columns:
        df["variant"] = df["variant"].astype(str).str.lower().str.strip()
    if "prefix" in df.columns:
        df["prefix"] = df["prefix"].astype(str)
    for m in METRIC_COLUMNS:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")
    return df

def read_and_prepare(csv_paths: List[str]) -> pd.DataFrame:
    # load one or many CSVs and normalize columns for downstream plotting
    frames = []
    for p in csv_paths:
        df = pd.read_csv(p)
        df = normalize_headers(df)
        df = derive_variant_from_tag(df)
        df = infer_family_from_path(df, p)
        df = coerce_types(df)
        df = add_regime_column(df)
        if "prefix" not in df.columns:
            df["prefix"] = pathlib.Path(p).stem
        if "variant" in df.columns:
            df = df[df["variant"].isin(ALLOWED_METHODS)]
        frames.append(df)
    if not frames:
        raise ValueError("No CSVs loaded")
    all_df = pd.concat(frames, ignore_index=True)
    return all_df

def metric_label(metric: str) -> str:
    pretty = {
        "auc": "ROC–AUC",
        "pr_auc": "PR–AUC",
        "balanced_accuracy": "Balanced Accuracy",
        "mcc": "MCC",
        "f1": "F1",
    }
    return pretty.get(str(metric).lower(), str(metric).upper())

def title_for_lines(prefix: str, metric: str) -> str:
    return f"{prefix} | {metric.upper()} (real vs GAN/EvoGAN/EVO/oversample/SMOTE)"

def out_dir_default(csvs: List[str]) -> str:
    return guess_outdir_from_first_csv(csvs)


# light-weight stats helpers
def bootstrap_ci_mean(data: np.ndarray, n_boot: int = 2000, seed: int = 1337, alpha: float = 0.05) -> Tuple[float, float, float]:
    # paired bootstrap CI for the mean delta vector
    rng = np.random.default_rng(seed)
    data = np.asarray([d for d in data if pd.notna(d)], dtype=float)
    if data.size == 0:
        return (np.nan, np.nan, np.nan)
    n = len(data)
    boot_means = np.array([np.mean(data[rng.integers(0, n, size=n)]) for _ in range(n_boot)], dtype=float)
    mean = float(np.mean(data))
    low = float(np.percentile(boot_means, 100 * (alpha / 2)))
    high = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return mean, low, high

def sign_test_p_greater(deltas: np.ndarray) -> float:
    # exact one-sided sign test P(median>0) ignoring zeros
    x = np.asarray([d for d in deltas if d != 0 and pd.notna(d)], dtype=float)
    n = x.size
    if n == 0:
        return 1.0
    k = int(np.sum(x > 0))
    tail = sum(math.comb(n, i) * (0.5 ** n) for i in range(k, n + 1))
    return min(1.0, tail)

def holm_correction(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, float]:
    # simple Holm adjustment returning adjusted p-values
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    for rank, (k, p) in enumerate(items, start=1):
        adj[k] = min(1.0, (m - rank + 1) * p)
    return {k: adj[k] for k in pvals.keys()}

def p_to_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"

def star_from_ci(low: float, high: float) -> str:
    # star if CI entirely above zero
    if np.isfinite(low) and low > 0:
        return "*"
    return "ns"


# aggregation and pairing
def agg_by_seed_variant_mean(df: pd.DataFrame, metric: str, extra_group: Optional[List[str]] = None) -> pd.DataFrame:
    # average duplicates by (seed, variant, [extra...]) to a tidy frame
    if extra_group is None:
        extra_group = []
    group_cols = ["seed", "variant"] + extra_group
    out = df.groupby(group_cols, dropna=False, as_index=False)[metric].mean()
    return out

def paired_deltas_vs_real(df: pd.DataFrame, metric: str,
                          where: Optional[pd.Series] = None,
                          pair_by: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    # paired deltas per unit: metric(method) − metric(real) using a pivot
    if where is not None:
        df = df[where].copy()
    if "variant" not in df.columns:
        return {}
    if pair_by is None:
        pair_by = ["seed"]
    tidy = df.groupby(pair_by + ["variant"], dropna=False, as_index=False)[metric].mean()
    pivot = tidy.pivot_table(index=pair_by, columns="variant", values=metric, aggfunc="mean")
    out = {}
    if "real" not in pivot.columns:
        return out
    real_col = pivot["real"].dropna()
    for v in pivot.columns:
        if v == "real":
            continue
        deltas = (pivot[v] - real_col).dropna().to_numpy()
        if deltas.size > 0:
            out[v] = deltas
    return out


# plotting modes
def _rankdata_abs(x: np.ndarray) -> np.ndarray:
    # average ranks on |x| with ties
    a = np.abs(x)
    order = np.argsort(a)
    ranks = np.empty_like(a, dtype=float)
    i = 0; r = 1
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (r + (r + (j - i))) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        r += (j - i + 1)
        i = j + 1
    return ranks

def wilcoxon_p_greater(deltas: np.ndarray) -> float:
    # one-sided Wilcoxon signed-rank (normal approx with continuity correction)
    x = np.asarray([d for d in deltas if pd.notna(d) and d != 0], dtype=float)
    n = x.size
    if n == 0:
        return 1.0
    ranks = _rankdata_abs(x)
    Tplus = float(np.sum(ranks[x > 0]))
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (Tplus - mu - 0.5) / sigma
    p = 0.5 * math.erfc(z / math.sqrt(2.0))
    return max(0.0, min(1.0, p))

def plot_lines(all_df: pd.DataFrame, metric: str, outdir: str, logx: bool, show_bands: bool):
    # mean ± SD across seeds for each method and fraction, per prefix
    required = {"frac", "variant", "seed", "prefix", metric}
    missing = required - set(all_df.columns)
    if missing:
        print(f"[warn] lines: missing columns {missing}; skipping")
        return
    prefixes = list(pd.unique(all_df["prefix"]))
    for pref in prefixes:
        sub = all_df[all_df["prefix"] == pref]
        grp = sub.groupby(["frac", "variant"], as_index=False).agg(mean=(metric, "mean"), sd=(metric, "std"))
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        for v in LEGEND_ORDER:
            g = grp[grp["variant"] == v].sort_values("frac")
            if g.empty:
                continue
            ax.plot(g["frac"], g["mean"], label=v, color=COLOR.get(v, None), linewidth=2.0, marker="o", markersize=3)
            if show_bands:
                ax.fill_between(g["frac"], g["mean"] - g["sd"], g["mean"] + g["sd"], alpha=0.20, color=COLOR.get(v, None))
        ticks = sorted(sub["frac"].dropna().astype(float).unique())
        if ticks:
            ax.set_xticks(ticks)
        if logx:
            ax.set_xscale("log")
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
        ax.set_xlabel("Real fraction of TRAIN")
        ax.set_ylabel(metric_label(metric))
        ax.grid(True, axis="y", linestyle=":", alpha=0.7)
        ax.set_title(title_for_lines(pref, metric))
        ax.legend(ncols=2, frameon=False)
        fname = os.path.join(outdir, f"{_sanitize_name(pref)}_{metric}_lines.png")
        fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
        print(f"[ok] wrote {fname}")

def plot_delta_bars(
    all_df: pd.DataFrame,
    metric: str,
    outdir: str,
    fractions: List[float],
    seed: int,
    *,
    alpha: float = 0.05,
    p_adjust: str = "holm",
):
    # delta bars vs real with paired bootstrap CIs + adjusted p-value stars
    required = {"frac", "variant", "seed", metric}
    missing = required - set(all_df.columns)
    if missing:
        print(f"[warn] delta-bars: missing columns {missing}; skipping")
        return

    # Choose pairing units without collapsing replicates:
    # - IID: pair on seed
    # - Temporal: pair on (cutoff/prefix, window, seed)
    # - Family: pair on (family, seed)
    if "test_start" in all_df.columns:
        pair_by = ["prefix", "test_start", "seed"]
    elif "family" in all_df.columns:
        pair_by = ["family", "seed"]
    else:
        pair_by = ["seed"]

    # Prefer SciPy exact wilcoxon if available; fallback to local approx.
    _scipy_wilcoxon = None
    try:
        from scipy.stats import wilcoxon as _scipy_wilcoxon  # type: ignore
    except Exception:
        _scipy_wilcoxon = None

    def one_sided_p_greater(deltas: np.ndarray) -> float:
        x = np.asarray([d for d in deltas if pd.notna(d) and d != 0], dtype=float)
        if x.size == 0:
            return 1.0
        if _scipy_wilcoxon is not None:
            # SciPy handles ties/zeros more robustly than our normal approx
            try:
                res = _scipy_wilcoxon(x, alternative="greater", zero_method="wilcox", correction=True, mode="auto")
                return float(res.pvalue)
            except Exception:
                pass
        return wilcoxon_p_greater(x)

    for f in fractions:
        mask = np.isclose(all_df["frac"].astype(float), float(f))
        deltas = paired_deltas_vs_real(all_df, metric, where=mask, pair_by=pair_by)

        methods = [m for m in LEGEND_ORDER if m != "real" and m in deltas and len(deltas[m]) > 0]
        if not methods:
            print(f"[warn] delta-bars: no methods with paired deltas at f={f}")
            continue

        means, lows, highs = [], [], []
        raw_p: Dict[str, float] = {}

        for m in methods:
            v = np.asarray(deltas[m], dtype=float)
            mean, lo, hi = bootstrap_ci_mean(v, n_boot=2000, seed=seed, alpha=0.05)
            means.append(mean)
            lows.append(lo)
            highs.append(hi)
            raw_p[m] = one_sided_p_greater(v)

        p_adjust = (p_adjust or "holm").lower().strip()
        if p_adjust == "none":
            adj_p = raw_p
        elif p_adjust == "bh":
            adj_p = bh_correction(raw_p)
        else:
            adj_p = holm_correction(raw_p, alpha=alpha)

        stars = [p_to_stars(adj_p.get(m, 1.0)) for m in methods]

        x = np.arange(len(methods))
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        ax.bar(x, means, color=[COLOR.get(m, "#777777") for m in methods], alpha=0.95)

        ci_low = np.array(means) - np.array(lows)
        ci_high = np.array(highs) - np.array(means)
        ax.errorbar(x, means, yerr=[ci_low, ci_high], fmt="none", ecolor="black", elinewidth=1.2, capsize=4)

        ymax = max((m + (h if np.isfinite(h) else 0.0)) for m, h in zip(means, ci_high))
        ymin = min((m - (l if np.isfinite(l) else 0.0)) for m, l in zip(means, ci_low))
        span = max(ymax - ymin, 1e-6)
        ax.set_ylim(ymin - 0.06 * span, ymax + 0.12 * span)

        for i, star in enumerate(stars):
            y_text = means[i] + (ci_high[i] if np.isfinite(ci_high[i]) else 0.0) + 0.03 * span
            y_text = min(y_text, ax.get_ylim()[1] - 0.02 * span)
            ax.text(x[i], y_text, star, ha="center", va="bottom", fontsize=11)

        ax.set_xticks(x, methods)
        ax.set_ylabel(f"Δ{metric.upper()}")
        ax.set_title(f"Δ{metric.upper()} vs Real @ f={f} (95% paired bootstrap CI; {p_adjust}-adjusted p)")
        ax.axhline(0, color="black", linewidth=1.0)
        ax.grid(True, axis="y", linestyle=":", alpha=0.7)

        fname = os.path.join(outdir, f"delta_{metric}_f{_sanitize_name(str(f))}.png")
        fig.tight_layout()
        fig.savefig(fname, dpi=200)
        plt.close(fig)
        print(f"[ok] wrote {fname}")

# keep both corrections for compatibility with existing calls elsewhere in the repo
def holm_correction(pvals: Dict[str, float], alpha: float = 0.05) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted = [min(1.0, (m - i) * p) for i, (_, p) in enumerate(items)]
    for i in range(1, m):
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    return {k: adjusted[i] for i, (k, _) in enumerate(items)}

def bh_correction(pvals: Dict[str, float]) -> Dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = []
    for i, (_, p) in enumerate(items, start=1):
        adj.append(min(1.0, p * m / i))
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    return {k: adj[i] for i, (k, _) in enumerate(items)}


def plot_windows(all_df: pd.DataFrame, metric: str, outdir: str, prefix: str, fraction: float, show_bands: bool):
    # temporal windows per prefix at a fixed fraction
    required = {"frac", "variant", "seed", "prefix", "test_start", metric}
    missing = required - set(all_df.columns)
    if missing:
        print(f"[warn] windows: missing columns {missing}; skipping")
        return
    df = all_df[(all_df["prefix"] == prefix) & (np.isclose(all_df["frac"].astype(float), float(fraction)))].copy()
    if df.empty:
        print(f"[warn] windows: no rows for prefix={prefix}, f={fraction}")
        return
    uniq = sorted(pd.unique(df["test_start"].dropna()))
    if not uniq:
        print("[warn] windows: no valid test_start values; skipping")
        return
    idx_map = {ts: i + 1 for i, ts in enumerate(uniq)}
    df["window_idx"] = df["test_start"].map(idx_map)
    grp = df.groupby(["window_idx", "variant"], as_index=False).agg(mean=(metric, "mean"), sd=(metric, "std"))
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for v in LEGEND_ORDER:
        g = grp[grp["variant"] == v].sort_values("window_idx")
        if g.empty:
            continue
        ax.plot(g["window_idx"], g["mean"], label=v, color=COLOR.get(v, None), linewidth=2.0)
        if show_bands:
            ax.fill_between(g["window_idx"], g["mean"] - g["sd"], g["mean"] + g["sd"], alpha=0.2, color=COLOR.get(v, None))
    ax.set_xlabel("Window index")
    ax.set_ylabel(metric_label(metric))
    ax.grid(True, axis="y", linestyle=":", alpha=0.7)
    ax.set_title(f"{prefix} | {metric.upper()} across windows @ f={fraction}")
    ax.legend(ncols=2, frameon=False)
    fname = os.path.join(outdir, f"{_sanitize_name(prefix)}_{metric}_windows_f{_sanitize_name(str(fraction))}.png")
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
    print(f"[ok] wrote {fname}")

def plot_heatmap(all_df: pd.DataFrame, metric: str, outdir: str, method: str, fraction: Optional[float] = None):
    # Δ(method − real) per window and cutoff as a heatmap
    required = {"frac", "variant", "seed", "prefix", "test_start", metric}
    missing = required - set(all_df.columns)
    if missing:
        print(f"[warn] heatmap: missing columns {missing}; skipping")
        return
    method = method.lower().strip()
    if method not in ALLOWED_METHODS or method == "real":
        print(f"[warn] heatmap: invalid method '{method}'. Choose one of: gan, evogan, evo, smote, oversample")
        return
    df = all_df.copy()
    if fraction is None:
        valid_fracs = sorted(pd.unique(df["frac"].dropna().astype(float)))
        if not valid_fracs:
            print("[warn] heatmap: no valid fractions found; skipping")
            return
        fraction = float(valid_fracs[0])
    else:
        fraction = float(fraction)
    df = df[np.isclose(df["frac"].astype(float), fraction)]
    if df.empty:
        print(f"[warn] heatmap: no rows at f={fraction}")
        return
    df = df.copy()
    df["test_start"] = pd.to_datetime(df["test_start"], errors="coerce")
    df = df.dropna(subset=["test_start"])
    df["window_idx"] = df.groupby("prefix", group_keys=False).apply(lambda g: g["test_start"].rank(method="dense").astype(int))
    deltas_list = []
    for pref, gpref in df.groupby("prefix"):
        tidy = gpref.groupby(["window_idx", "seed", "variant"], as_index=False)[metric].mean()
        piv = tidy.pivot_table(index=["window_idx", "seed"], columns="variant", values=metric, aggfunc="mean")
        if "real" not in piv.columns or method not in piv.columns:
            continue
        dd = (piv[method] - piv["real"]).groupby(level=0).mean()
        tmp = pd.DataFrame({"prefix": pref, "window_idx": dd.index.astype(int), "delta": dd.values})
        deltas_list.append(tmp)
    if not deltas_list:
        print("[warn] heatmap: no data after pairing; skipping")
        return
    H = pd.concat(deltas_list, ignore_index=True)
    pivot = H.pivot_table(index="prefix", columns="window_idx", values="delta", aggfunc="mean")
    vmax = np.nanmax(np.abs(pivot.values)) if pivot.size > 0 else 0.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1e-6
    fig, ax = plt.subplots(figsize=(max(6.0, 0.6 * pivot.shape[1] + 2.5), max(4.5, 0.35 * pivot.shape[0] + 1.5)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(pivot.shape[0]), pivot.index.tolist())
    ax.set_xticks(np.arange(pivot.shape[1]), [str(c) for c in pivot.columns.tolist()])
    ax.set_xlabel("Window index")
    ax.set_ylabel("Prefix")
    ax.set_title(f"Δ{metric.upper()} vs Real | {method} | f={fraction}")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"Δ{metric.upper()}")
    fname = os.path.join(outdir, f"heatmap_{method}_{metric}_f{_sanitize_name(str(fraction))}.png")
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
    print(f"[ok] wrote {fname}")

def plot_family_bars(all_df: pd.DataFrame, metric: str, outdir: str, fraction: float):
    # per-family Δ(method − real) with bootstrap CIs for a single fraction
    required = {"frac", "variant", "seed", "family", metric}
    missing = required - set(all_df.columns)
    if missing:
        print(f"[warn] family-bars: missing columns {missing}; skipping")
        return
    df = all_df[np.isclose(all_df["frac"].astype(float), float(fraction))].copy()
    if df.empty:
        print(f"[warn] family-bars: no rows at f={fraction}")
        return
    methods = ["evogan", "evo", "gan"]
    fams = sorted(pd.unique(df["family"].astype(str)))
    rows = []
    for fam in fams:
        sub = df[df["family"] == fam]
        deltas = paired_deltas_vs_real(sub, metric)
        for m in methods:
            vals = np.asarray(deltas.get(m, []), dtype=float)
            if vals.size == 0:
                mean, low, high = (np.nan, np.nan, np.nan)
            else:
                mean, low, high = bootstrap_ci_mean(vals, n_boot=2000, seed=1337)
            rows.append({"family": fam, "method": m, "mean": mean, "low": low, "high": high})
    T = pd.DataFrame(rows)
    if T.empty:
        print("[warn] family-bars: no paired data; skipping")
        return
    fams = sorted(pd.unique(T["family"]))
    y_base = np.arange(len(fams))
    bar_h = 0.22
    offsets = {"evogan": -bar_h, "evo": 0.0, "gan": bar_h}
    fig, ax = plt.subplots(figsize=(8.8, max(4.8, 0.35 * len(fams) + 1.5)))
    for m in methods:
        D = T[T["method"] == m].set_index("family").reindex(fams)
        ys = y_base + offsets[m]
        ax.barh(ys, D["mean"], height=bar_h * 0.9, color=COLOR.get(m, None), label=m)
        ci_low = D["mean"].values - D["low"].values
        ci_high = D["high"].values - D["mean"].values
        ax.errorbar(D["mean"], ys, xerr=[ci_low, ci_high], fmt="none", ecolor="black", elinewidth=1.0, capsize=3)
    ax.set_yticks(y_base, fams)
    ax.set_xlabel(f"Δ{metric.upper()} (method − real)")
    ax.set_title(f"Family LOFO Δ{metric.upper()} @ f={fraction}")
    ax.grid(True, axis="x", linestyle=":", alpha=0.7)
    ax.legend(frameon=False, ncols=3, loc="lower right")
    fname = os.path.join(outdir, f"family_delta_{metric}_f{_sanitize_name(str(fraction))}.png")
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
    print(f"[ok] wrote {fname}")

def _infer_regime_label(df: pd.DataFrame) -> str:
    # human-friendly regime label for titles when needed
    cols = set(df.columns)
    if "test_start" in cols:
        return "Temporal"
    if "family" in cols:
        return "Family"
    return "IID"

def plot_lines_agg(all_df: pd.DataFrame, metric: str, outdir: str,
                   logx: bool, show_bands: bool,
                   y_step=None, ylim=None,
                   ribbon: str = "sd", qlo: float = 0.25, qhi: float = 0.75,
                   min_groups: int = 1):
    # aggregated lines across slices (families or cutoffs) with SD or quantile ribbons
    required = {"frac", "variant", "seed", metric}
    missing = required - set(all_df.columns)
    if missing:
        print(f"[warn] lines-agg: missing columns {missing}; skipping")
        return

    df = all_df.copy()

    # infer regime for title + filename
    if "test_start" in df.columns:
        regime = "temporal"
        regime_title = "Temporal aggregation"
        regime_fname = "temporalALL"
        # temporal: average within each window/cutoff slice before aggregating
        df = df.groupby(["prefix", "frac", "variant", "seed"], as_index=False)[metric].mean()
    elif "family" in df.columns:
        regime = "family"
        regime_title = "Family aggregation"
        regime_fname = "familyALL"
        # family rows are already per-family; keep as-is
    else:
        regime = "iid"
        regime_title = "IID aggregation"
        regime_fname = "iidALL"

    grp = df.groupby(["frac", "variant"], as_index=False).agg(
        mean=(metric, "mean"),
        sd=(metric, "std"),
        n=(metric, "size"),
        qlo=(metric, lambda s: s.quantile(qlo)),
        qhi=(metric, lambda s: s.quantile(qhi)),
    )
    grp = grp[grp["n"] >= int(min_groups)]
    if grp.empty:
        print("[warn] lines-agg: empty after min_groups filter; nothing to plot")
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    if show_bands:
        for v in LEGEND_ORDER:
            g = grp[grp["variant"] == v].sort_values("frac")
            if g.empty:
                continue
            if ribbon == "q":
                lower = g["qlo"].astype(float); upper = g["qhi"].astype(float)
            else:
                lower = g["mean"].astype(float) - g["sd"].fillna(0.0)
                upper = g["mean"].astype(float) + g["sd"].fillna(0.0)
            ax.fill_between(g["frac"], lower, upper, alpha=0.14, color=COLOR.get(v, None), linewidth=0.0, zorder=1)

    for v in LEGEND_ORDER:
        g = grp[grp["variant"] == v].sort_values("frac")
        if g.empty:
            continue
        ax.plot(g["frac"], g["mean"], label=v, color=COLOR.get(v, None),
                linewidth=2.2, marker="o", markersize=4, zorder=6)

    if logx:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_xlabel("Real fraction of TRAIN")
    ax.set_ylabel(metric_label(metric))
    _tune_y_axis(ax, y_step=y_step, ylim=ylim)

    ax.set_title(f"{regime_title} | {metric_label(metric)} vs fraction")
    ax.legend(ncols=2, frameon=False)

    fname = os.path.join(outdir, f"{regime_fname}_{metric}_lines.png")
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close(fig)
    print(f"[ok] wrote {fname}")


def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    # relax common header variations across CSV exports
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {"prauc": "pr_auc", "pr-auc": "pr_auc", "pr auc": "pr_auc"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    return df

def plot_seed_violin(all_df: pd.DataFrame, metric: str, outdir: str, prefix: str, fraction: float, include_real: bool = True):
    # per-cutoff seed distribution violin at a fixed fraction
    required = {"frac", "variant", "seed", "prefix", metric}
    missing = required - set(all_df.columns)
    if missing:
        print(f"[warn] seed-violin: missing columns {missing}; skipping")
        return
    df = all_df[(all_df["prefix"] == prefix) & (np.isclose(all_df["frac"].astype(float), float(fraction)))].copy()
    if df.empty:
        print(f"[warn] seed-violin: no rows for prefix={prefix}, f={fraction}")
        return
    tidy = df.groupby(["seed", "variant"], as_index=False)[metric].mean()
    methods = [m for m in LEGEND_ORDER if (include_real or m != "real") and m in tidy["variant"].unique()]
    data = [tidy[tidy["variant"] == m][metric].to_numpy() for m in methods]
    if not methods:
        print("[warn] seed-violin: no methods present after filtering; skipping")
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for i, b in enumerate(parts["bodies"], start=0):
        m = methods[i]
        b.set_facecolor(COLOR.get(m, None)); b.set_alpha(0.65); b.set_edgecolor("black")
    if "cmeans" in parts:
        parts["cmeans"].set_linewidth(1.2)
    ax.set_xticks(np.arange(1, len(methods) + 1), methods)
    ax.set_ylabel(metric_label(metric))
    ax.set_title(f"{prefix} | Seed variance @ f={fraction}")
    ax.grid(True, axis="y", linestyle=":", alpha=0.7)
    fname = os.path.join(outdir, f"{_sanitize_name(prefix)}_{metric}_seed_violin_f{_sanitize_name(str(fraction))}.png")
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
    print(f"[ok] wrote {fname}")

def plot_seed_violin_agg(df, metric, outdir, fraction):
    # pooled seed variability across slices at a single fraction
    d = df[df["frac"] == float(fraction)].copy()
    methods = [m for m in LEGEND_ORDER if m in set(d["variant"])]
    data, labels = [], []
    for m in methods:
        vals = d.loc[d["variant"] == m, metric].dropna().values
        if len(vals) > 0:
            data.append(vals); labels.append(m)
    if not data:
        print("[warn] seed-violin-agg: no data"); return
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    parts = ax.violinplot(data, showmeans=True, showextrema=False)
    for body, lab in zip(parts["bodies"], labels):
        body.set_facecolor(COLOR[lab]); body.set_edgecolor("black"); body.set_alpha(0.35)
    if "cmeans" in parts:
        parts["cmeans"].set_linewidth(1.2)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
    ax.set_ylabel(metric.upper())
    ax.set_title(f"Aggregated variability across slices @ f={fraction}")
    ax.grid(True, axis="y", linestyle=":", alpha=0.7)
    fname = os.path.join(outdir, f"agg_{metric}_seed_violin_f{fraction}.png")
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
    print(f"[ok] wrote {fname}")

def plot_seed_box_agg(df, metric, outdir, fraction=None):
    # pooled boxplot across slices, optionally filtered to a fraction
    D = df.copy()
    if fraction is not None:
        D = D[np.isclose(D["frac"].astype(float), float(fraction))]
    if "test_start" in D.columns:
        D = D.groupby(["prefix", "test_start", "seed", "variant"], as_index=False)[metric].mean()
    if "family" in D.columns:
        D = D.groupby(["family", "seed", "variant"], as_index=False)[metric].mean()
    methods = [m for m in LEGEND_ORDER if m in set(D["variant"])]
    data, labels = [], []
    for m in methods:
        vals = D.loc[D["variant"] == m, metric].dropna().values
        if len(vals) > 0:
            data.append(vals); labels.append(m)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    bp = ax.boxplot(data, showmeans=True, meanprops=dict(marker="^", markersize=6, mec="black", mfc="black"), patch_artist=True)
    for box, m in zip(bp["boxes"], labels):
        box.set_facecolor(COLOR.get(m, "#cccccc")); box.set_alpha(0.35); box.set_edgecolor("black")
    rng = np.random.default_rng(1337)
    for i, vals in enumerate(data, start=1):
        xj = i + (rng.random(len(vals)) - 0.5) * 0.18
        ax.plot(xj, vals, "o", ms=2.0, alpha=0.35, color="black")
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels)
    ax.set_ylabel(metric_label(metric))
    title_suffix = f"@ f={fraction}" if fraction is not None else "(pooled over all fractions)"
    ax.set_title(f"Aggregated variability across slices {title_suffix}")
    ax.grid(True, axis="y", linestyle=":", alpha=0.7)
    fname = os.path.join(outdir, f"agg_{metric}_seed_box_{'all' if fraction is None else f'f{fraction}'}_.png")
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
    print(f"[ok] wrote {fname}")

def plot_sd_lines(all_df: pd.DataFrame, metric: str, outdir: str, *,
                  logx: bool, y_step: float | None = None, ylim: tuple | None = None,
                  min_groups: int = 3, title_prefix: str = "SD vs fraction"):
    # show mean SD across slices at each fraction to visualize variability vs scarcity
    required = {"frac", "variant", "seed", metric}
    missing = required - set(all_df.columns)
    if missing:
        print(f"[warn] sd-lines: missing columns {missing}; skipping")
        return
    df = all_df.copy()
    df["frac"] = pd.to_numeric(df["frac"], errors="coerce").round(6)
    df = df.dropna(subset=["frac", metric, "variant", "seed"])
    if "test_start" in df.columns:
        if not np.issubdtype(df["test_start"].dtype, np.datetime64):
            df["test_start"] = pd.to_datetime(df["test_start"], errors="coerce")
        df = df.dropna(subset=["test_start"])
        df["window_idx"] = df.groupby("prefix")["test_start"].rank(method="dense").astype(int)
        slice_cols = ["prefix", "window_idx"]; regime = "temporal"
    elif "family" in df.columns:
        slice_cols = ["family"]; regime = "family"
    else:
        if "prefix" not in df.columns:
            df["prefix"] = "iid"
        slice_cols = ["prefix"]; regime = "iid"
    group_cols = ["frac", "variant"] + slice_cols + ["seed"]
    per_seed = df.groupby(group_cols, as_index=False)[metric].mean()
    per_slice_sd = per_seed.groupby(["frac", "variant"] + slice_cols, as_index=False)[metric].std(ddof=1)
    per_slice_sd = per_slice_sd.rename(columns={metric: "sd"}).dropna(subset=["sd"])
    agg = per_slice_sd.groupby(["frac", "variant"], as_index=False).agg(n=("sd", "count"), mean_sd=("sd", "mean"))
    agg = agg[agg["n"] >= int(min_groups)]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    any_line = False
    for v in LEGEND_ORDER:
        g = agg[agg["variant"] == v].sort_values("frac")
        if g.empty:
            continue
        any_line = True
        ax.plot(g["frac"], g["mean_sd"], label=v, color=COLOR.get(v, None), linewidth=2.0, marker="o", markersize=3)
    if y_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_step))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if isinstance(ylim, tuple):
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", which="both", linestyle=":", alpha=0.7)
    if logx:
        ax.set_xscale("log")
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_xlabel("Real fraction of TRAIN")
    ax.set_ylabel(f"SD({metric.upper()})")
    ax.set_title(f"{title_prefix} | {metric.upper()} ({regime})")
    if any_line:
        ax.legend(ncols=2, frameon=False)
    fname = os.path.join(outdir, f"sd_lines_{regime}_{metric}.png")
    fig.tight_layout(); fig.savefig(fname, dpi=200); plt.close(fig)
    print(f"[ok] wrote {fname}")


# CLI
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Make Results figures from experiment CSVs (i.i.d., temporal, family LOFO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument("csv", nargs="+", help="One or more CSV paths (globs ok)")
    p.add_argument("--mode", required=True, choices=[
        "lines", "delta-bars", "windows", "heatmap",
        "family-bars", "seed-violin", "lines-agg",
        "seed-violin-agg", "seed-box-agg", "sd-lines"
    ], help="Plotting mode")
    p.add_argument("--metric", default="auc", choices=sorted(METRIC_COLUMNS), help="Metric to plot")
    p.add_argument("--out", default=None, help="Output directory for PNGs")
    p.add_argument("--logx", action="store_true", help="Log-scale x-axis for fraction plots")
    p.add_argument("--fractions", default=None, help="Comma list for delta-bars, e.g. 0.001,0.005,0.01")
    p.add_argument("--fraction", type=float, default=None, help="Single fraction for windows/family-bars/seed-violin/heatmap override")
    p.add_argument("--prefix", default=None, help="Prefix (e.g., cutoff) for windows/seed-violin")
    p.add_argument("--method", default=None, help="Method for heatmap, e.g. evogan")
    p.add_argument("--no-bands", action="store_true", help="Disable SD shading on lines/windows")
    p.add_argument("--seed", type=int, default=1337, help="Bootstrap RNG seed")
    p.add_argument("--y-step", type=float, default=None, help="Major y tick step, e.g. 0.01 for AUC")
    p.add_argument("--ylim", type=str, default=None, help="ymin,ymax, e.g. '0.88,1.00'")
    p.add_argument("--ribbon", choices=["sd", "q"], default="q", help="Uncertainty band type")
    p.add_argument("--qlo", type=float, default=0.25, help="Lower quantile when ribbon=q")
    p.add_argument("--qhi", type=float, default=0.75, help="Upper quantile when ribbon=q")
    p.add_argument("--min-groups", type=int, default=3, help="Min slices per (frac,variant) to plot SD")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    p.add_argument("--p-adjust", choices=["holm", "bh", "none"], default="holm", help="Multiple-comparison adjustment")
    return p.parse_args(argv)

def _tune_y_axis(ax, y_step=None, ylim=None):
    if y_step is not None:
        ax.yaxis.set_major_locator(MultipleLocator(y_step))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if isinstance(ylim, tuple):
        ax.set_ylim(*ylim)
    ax.grid(True, axis="y", which="major", linestyle=":", alpha=0.7)
    ax.grid(True, axis="y", which="minor", linestyle=":", alpha=0.25)

def main(argv=None):
    args = parse_args(argv)
    csvs = expand_csvs(args.csv)
    if not csvs:
        print("[error] no CSVs found"); return 2
    outdir = args.out if args.out else out_dir_default(csvs)
    ensure_outdir(outdir)
    df = read_and_prepare(csvs)
    mode = args.mode
    metric = args.metric

    if mode == "lines":
        plot_lines(df, metric, outdir, logx=args.logx, show_bands=not args.no_bands)

    elif mode == "delta-bars":
        if not args.fractions:
            print("[error] --fractions required for delta-bars"); return 2
        fracs = [float(x.strip()) for x in args.fractions.split(",") if x.strip()]
        plot_delta_bars(df, metric, outdir, fracs, seed=args.seed, alpha=args.alpha, p_adjust=args.p_adjust)

    elif mode == "windows":
        if args.prefix is None or args.fraction is None:
            print("[error] windows mode needs --prefix and --fraction"); return 2
        plot_windows(df, metric, outdir, prefix=args.prefix, fraction=float(args.fraction), show_bands=not args.no_bands)

    elif mode == "heatmap":
        if not args.method:
            print("[error] heatmap mode needs --method"); return 2
        plot_heatmap(df, metric, outdir, method=args.method, fraction=args.fraction)

    elif mode == "family-bars":
        if args.fraction is None:
            print("[error] family-bars needs --fraction"); return 2
        plot_family_bars(df, metric, outdir, fraction=float(args.fraction))

    elif mode == "seed-violin":
        if args.prefix is None or args.fraction is None:
            print("[error] seed-violin needs --prefix and --fraction"); return 2
        plot_seed_violin(df, metric, outdir, prefix=args.prefix, fraction=float(args.fraction), include_real=True)

    elif mode == "lines-agg":
        plot_lines_agg(
            df, metric, outdir,
            logx=args.logx, show_bands=not args.no_bands,
            y_step=args.y_step, ylim=_parse_ylim(args.ylim),
            ribbon=args.ribbon, qlo=args.qlo, qhi=args.qhi,
            min_groups=args.min_groups
        )

    elif mode == "seed-violin-agg":
        if args.fraction is None:
            print("[error] seed-violin-agg needs --fraction"); return 2
        plot_seed_violin_agg(df, metric, outdir, fraction=float(args.fraction))

    elif mode == "seed-box-agg":
        if args.fraction is None:
            print("[error] seed-box-agg needs --fraction"); return 2
        plot_seed_box_agg(df, metric, outdir, fraction=float(args.fraction))

    elif mode == "sd-lines":
        plot_sd_lines(
            df, metric, outdir,
            logx=args.logx,
            y_step=args.y_step,
            ylim=_parse_ylim(args.ylim),
            min_groups=args.min_groups,
            title_prefix="SD vs fraction"
        )

    else:
        print(f"[error] unknown mode {mode}")
        return 2

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
