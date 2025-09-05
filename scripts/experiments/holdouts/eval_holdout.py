#!/usr/bin/env python
import argparse, json, math, numpy as np
from collections import Counter
from pathlib import Path
from typing import Dict, Sequence, Optional

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV, StratifiedShuffleSplit
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler  # (import retained; may be used elsewhere)

from src.paths import ROOT, TEMPORAL_SPLIT, FAMILY_SPLIT
from src.holdouts import SplitIndices
import pandas as pd
from src.data.metadata import load_metadata
from src.augmentation import config_aug as C

# ----------------- constants -----------------
METRICS_DIR = ROOT / "data" / "processed" / "metrics"
PRED_DIR    = ROOT / "data" / "processed" / "preds"
METRICS_BASE = ROOT / "data" / "processed" / "metrics"

# ----------------- helpers -----------------
def load_metric_json(tag: str, *, subdir: str, kind: str = "temporal"):
    base = METRICS_BASE / subdir if subdir else METRICS_BASE
    p1 = base / f"rf_{kind}_metrics_{tag}.json"
    p2 = base / f"rf_{kind}_metrics__{tag}.json"  # historical underscore quirk
    if p1.exists(): return json.loads(p1.read_text())
    if p2.exists(): return json.loads(p2.read_text())
    return None

def month_tag(iso_start: str) -> str:
    t = pd.Timestamp(iso_start)
    if t.tz is None: t = t.tz_localize("UTC")
    else: t = t.tz_convert("UTC")
    return f"cut{t.year}_{t.month:02d}"

def append_raw_row(raw_csv: Path, row: dict):
    raw_csv.parent.mkdir(parents=True, exist_ok=True)
    if raw_csv.exists():
        df = pd.read_csv(raw_csv)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(raw_csv, index=False)

def load_xy():
    z = np.load(ROOT / "data" / "raw" / "bodmas.npz", allow_pickle=True)
    return z["X"], z["y"].astype(int)

def sub(X, idx): return X[idx] if hasattr(X, "__getitem__") else np.asarray(X)[idx]

def tune_rf(X, y, grid="light"):
    presets = {
        "light":  {"n_estimators":[200,400], "max_depth":[None,20]},
        "medium": {"n_estimators":[300,600], "max_depth":[None,16,24]},
        "heavy":  {"n_estimators":[400,800], "max_depth":[None,14,20,28]},
    }
    params = presets[grid]
    base = RandomForestClassifier(n_jobs=-1, random_state=42, oob_score=False)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(base, params, cv=cv, n_jobs=-1, scoring="roc_auc", verbose=0)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_

# ---------- guard-rails (new) ----------
def _ratio(n: int, d: int) -> float:
    return float(n) / float(d) if d else 0.0

def audit_splits_and_aug(
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    sha: Optional[Sequence[str]] = None,
    y_train_before_aug: Optional[np.ndarray] = None,
    y_train_after_aug: Optional[np.ndarray] = None,
    ros_dup_ratio: Optional[float] = None,
    smote_k: Optional[int] = None,
    smote_synth_count: Optional[int] = None,
    smote_med_nn_dist: Optional[float] = None,
    rf_params: Optional[Dict] = None,
) -> Dict:
    train_set, val_set, test_set = set(map(int, train_idx)), set(map(int, val_idx)), set(map(int, test_idx))
    # Validation must be subset of TRAIN (we pass only "real" val rows mapped to global indices)
    assert val_set.issubset(train_set), "Validation must be a subset of TRAIN."
    # No overlaps across partitions
    assert train_set.isdisjoint(test_set), "Leakage: TRAIN and TEST overlap!"
    assert val_set.isdisjoint(test_set), "Leakage: VAL and TEST overlap!"

    if sha is not None and len(sha):
        sha_train = {sha[i] for i in train_set}
        sha_test  = {sha[i] for i in test_set}
        assert sha_train.isdisjoint(sha_test), "SHA leakage across TRAIN/TEST!"

    audit = {}
    if y_train_before_aug is not None:
        n_pos_b = int((y_train_before_aug == 1).sum())
        n_neg_b = int((y_train_before_aug == 0).sum())
        audit.update({
            "train_pos_before_aug": n_pos_b,
            "train_neg_before_aug": n_neg_b,
            "train_total_before_aug": int(y_train_before_aug.shape[0]),
        })
    if y_train_after_aug is not None:
        n_pos_a = int((y_train_after_aug == 1).sum())
        n_neg_a = int((y_train_after_aug == 0).sum())
        audit.update({
            "train_pos_after_aug": n_pos_a,
            "train_neg_after_aug": n_neg_a,
            "train_total_after_aug": int(y_train_after_aug.shape[0]),
        })
        if y_train_before_aug is not None:
            audit["aug_added"] = int(y_train_after_aug.shape[0] - y_train_before_aug.shape[0])

    if ros_dup_ratio is not None:
        audit["ros_dup_ratio"] = float(ros_dup_ratio)
    if smote_k is not None:
        audit["smote_k_neighbors"] = int(smote_k)
    if smote_synth_count is not None:
        audit["smote_synth_count"] = int(smote_synth_count)
    if smote_med_nn_dist is not None:
        audit["smote_med_nn_dist"] = float(smote_med_nn_dist)

    if rf_params is not None:
        audit["rf_params"] = dict(rf_params)

    return audit

# ---------- augmentation helpers ----------
def _nn_min_dist(A, B, k=1):
    """Return distance to the nearest of B for each row in A."""
    if len(B) == 0 or len(A) == 0:
        return np.full(len(A), np.inf, dtype=np.float32)
    k_eff = min(int(k), len(B))
    nbrs = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nbrs.fit(B)
    dists = nbrs.kneighbors(A, return_distance=True)[0]
    return dists[:, 0].astype(np.float32)

def _as_utc(ts):
    if ts is None:
        return None
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")

def add_gan_synth(
    X_real, y_real, X_like_pos, n_synth, generator_path, *,
    seed=None, scaler_npz=None,
    quality="nn", qmult=3.0,
    X_neg_for_quality=None,
    boundary_low=0.20, boundary_high=0.60, boundary_k=5
):
    """
    Generate n_synth malware samples with G and append to (X_real, y_real).
    - quality='none'        : take first n_synth
    - quality='nn'          : keep n_synth closest-to-positive manifold
    - quality='nn_boundary' : keep near-negative boundary but still close to positive manifold
    """
    import torch, torch.nn as nn
    from src.augmentation.model import Generator

    if n_synth <= 0:
        return X_real, y_real

    if seed is not None:
        np.random.seed(int(seed))
        torch.manual_seed(int(seed))

    # load G
    G = Generator().cpu()
    sd = torch.load(generator_path, map_location="cpu")
    G.load_state_dict(sd)
    G.eval()

    # infer latent dim
    zdim = getattr(G, "latent_dim", None)
    if zdim is None:
        for m in G.modules():
            if isinstance(m, nn.Linear):
                zdim = m.in_features
                break
    zdim = int(zdim)

    # oversample then filter
    n_gen = int(max(n_synth, int(np.ceil(float(qmult) * n_synth))))
    with torch.no_grad():
        z = torch.randn(n_gen, zdim)
        X_syn = G(z).cpu().numpy().astype(np.float32)

    # inverse-transform if scaler provided
    if scaler_npz:
        sc = np.load(scaler_npz)
        mean, scale = sc["mean_"].astype(np.float32), sc["scale_"].astype(np.float32)
        X_syn = X_syn * scale + mean

    # quality filtering
    if quality == "none" or len(X_like_pos) == 0:
        keep_idx = np.arange(min(n_gen, n_synth))
    elif quality == "nn":
        dpos = _nn_min_dist(X_syn, X_like_pos, k=1)
        order = np.argsort(dpos)                 # closest to positive manifold
        keep_idx = order[:n_synth]
    elif quality == "nn_boundary":
        # need negatives to target the boundary
        if X_neg_for_quality is None or len(X_neg_for_quality) == 0:
            # fall back to 'nn'
            dpos = _nn_min_dist(X_syn, X_like_pos, k=1)
            order = np.argsort(dpos)
            keep_idx = order[:n_synth]
        else:
            dpos = _nn_min_dist(X_syn, X_like_pos, k=1)
            dneg = _nn_min_dist(X_syn, X_neg_for_quality, k=max(1, int(boundary_k)))

            # keep synth that are neither too far from positives nor too far/close to negatives
            low = np.quantile(dneg, float(boundary_low))
            high = np.quantile(dneg, float(boundary_high))
            # on-manifold wrt positives: below median positive distance
            pos_ok = dpos <= np.quantile(dpos, 0.50)
            # near boundary wrt negatives: distance within [low, high]
            neg_ok = (dneg >= low) & (dneg <= high)
            mask = pos_ok & neg_ok
            idx = np.where(mask)[0]

            if len(idx) >= n_synth:
                rng = np.random.default_rng(seed)
                keep_idx = rng.choice(idx, size=n_synth, replace=False)
            else:
                # take what we have, then fill with best-by-positive
                remaining = np.setdiff1d(np.arange(n_gen), idx, assume_unique=False)
                fill_order = remaining[np.argsort(dpos[remaining])]
                keep_idx = np.concatenate([idx, fill_order[:max(0, n_synth - len(idx))]])
    else:
        raise ValueError(f"unknown quality mode: {quality}")

    X_keep = X_syn[keep_idx]
    y_keep = np.ones(len(X_keep), dtype=y_real.dtype)
    return np.vstack([X_real, X_keep]), np.concatenate([y_real, y_keep])

def oversample_to_const(X, y, target_n, seed=42):
    """Duplicate malware (y=1) with replacement until len == target_n."""
    if target_n is None or target_n <= len(y): return X, y
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    if len(pos) == 0:  # nothing to oversample
        return X, y
    need = target_n - len(y)
    pick = rng.choice(pos, size=need, replace=True)
    X_syn = X[pick]
    y_syn = np.ones(len(pick), dtype=y.dtype)
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])

def smote_to_const(X, y, target_n, seed=42, k=5):
    """
    Tiny, robust SMOTE for numeric features only.
    Works even when the positive class is very small; falls back to oversampling when needed.
    """
    if target_n is None or target_n <= len(y):
        return X, y

    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    if len(pos_idx) < 2:
        # too few positives to interpolate
        return oversample_to_const(X, y, target_n, seed=seed)

    need = target_n - len(y)
    Xpos = X[pos_idx]

    # effective k must be < len(Xpos) to satisfy sklearn kneighbors
    k_eff = min(int(k), max(1, len(Xpos) - 1))
    if k_eff < 1:
        return oversample_to_const(X, y, target_n, seed=seed)

    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean")
    nn.fit(Xpos)

    # neighbors may include self; we'll remove self per-row later
    nbrs = nn.kneighbors(Xpos, return_distance=False)

    synth = []
    for _ in range(need):
        i = int(rng.integers(0, len(Xpos)))
        xi = Xpos[i]

        # exclude self from candidate neighbors
        cand = [int(j) for j in nbrs[i] if int(j) != i]
        if not cand:
            # pick any other positive at random
            j = int(rng.integers(0, len(Xpos) - 1))
            if j >= i:
                j += 1
        else:
            j = cand[int(rng.integers(0, len(cand)))]

        xj = Xpos[j]
        t = float(rng.random())
        synth.append(xi + t * (xj - xi))

    X_syn = np.vstack(synth)
    y_syn = np.ones(len(X_syn), dtype=y.dtype)
    return np.vstack([X, X_syn]), np.concatenate([y, y_syn])

def rebalance_after_augment(X, y, seed=42, ratio=1.0):
    """
    Downsample the majority class to at most `ratio` × minority size.
    ratio=1.0 → 50/50. Never upsamples; only trims the majority.
    """
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X, y  # nothing to rebalance

    n_min = min(len(pos_idx), len(neg_idx))
    n_max_keep = int(round(n_min * max(1.0, ratio)))

    if len(pos_idx) <= len(neg_idx):
        keep_pos = pos_idx
        keep_neg = rng.choice(neg_idx, size=min(len(neg_idx), n_max_keep), replace=False)
    else:
        keep_neg = neg_idx
        keep_pos = rng.choice(pos_idx, size=min(len(pos_idx), n_max_keep), replace=False)

    keep = np.sort(np.concatenate([keep_pos, keep_neg]))
    return X[keep], y[keep]

def pick_threshold(y_true, proba, which="balacc", grid=200):
    ts = np.linspace(0.0, 1.0, num=grid+1)[1:-1]
    if which == "f1":
        scorer = lambda yt, yp: f1_score(yt, yp, zero_division=0)
    elif which == "mcc":
        scorer = lambda yt, yp: matthews_corrcoef(yt, yp)
    elif which == "balacc":
        scorer = lambda yt, yp: balanced_accuracy_score(yt, yp)
    else:
        raise ValueError("unknown threshold metric: %s" % which)
    vals = [scorer(y_true, (proba >= t).astype(int)) for t in ts]
    best_i = int(np.nanargmax(vals)) if len(vals) else 0
    return float(ts[best_i]), float(vals[best_i])

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()

    # --- split selection (exactly one) ---
    gsplit = ap.add_mutually_exclusive_group(required=True)
    gsplit.add_argument("--use-temporal", action="store_true")
    gsplit.add_argument("--use-family", action="store_true")
    ap.add_argument("--smoke", action="store_true")

    # --- scarcity controls ---
    ap.add_argument("--scarce-real-frac", type=float, default=None)
    ap.add_argument("--min-train-pos", type=int, default=50)
    ap.add_argument("--min-train-neg", type=int, default=50)
    ap.add_argument("--const-train-size", type=int, default=None)

    # --- augmentation modes (exactly one of these three) ---
    gaug = ap.add_mutually_exclusive_group()
    gaug.add_argument("--use-gan", action="store_true")
    gaug.add_argument("--oversample", action="store_true")
    gaug.add_argument("--smote", action="store_true")

    # --- GAN options ---
    ap.add_argument("--gan-generator", type=str, default=None)
    ap.add_argument("--gan-scaler", type=str, default=None,
                    help="scaler.npz used during GAN training (for inverse transform)")
    ap.add_argument("--gan-like", choices=["scarce", "full"], default="scarce")
    ap.add_argument("--gan-synth-per-real", type=float, default=4.0,
                    help="max synthetic malware as a multiple of real positives")
    ap.add_argument("--gan-quality", choices=["none", "nn", "nn_boundary"], default="nn",
                    help="Quality filter for synth: nearest-neighbor only ('nn') or also boundary-targeted ('nn_boundary').")
    ap.add_argument("--gan-qmult", type=float, default=3.0,
                    help="oversample factor before quality-filtering; sample qmult× then keep best")
    ap.add_argument("--gan-boundary-low", type=float, default=0.20,
                    help="Lower quantile of synth->nearest-negative distance to keep (0..1).")
    ap.add_argument("--gan-boundary-high", type=float, default=0.60,
                    help="Upper quantile of synth->nearest-negative distance to keep (0..1).")
    ap.add_argument("--gan-boundary-k", type=int, default=5,
                    help="k for nearest-neighbor distance to negatives.")

    # --- validation thresholding ---
    ap.add_argument("--val-threshold", choices=["none","f1","mcc","balacc"], default="balacc",
                help="Pick decision threshold on a small validation split (default=balacc).")
    ap.add_argument("--test-start", type=str, default=None)
    ap.add_argument("--test-end",   type=str, default=None)

    # --- RF / tuning ---
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--grid", choices=["light","medium","heavy"], default="light")
    ap.add_argument("--rf-n-est", type=int, default=400)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--rf-class-weight", choices=["balanced","none"], default="balanced")

    # --- bookkeeping ---
    ap.add_argument("--model-out", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-json", type=str, default=None,
                    help="Override split file; JSON with fields 'train' and 'test' indices.")
    ap.add_argument("--balance-after-augment", action="store_true",
                    help="Downsample the majority class in TRAIN after augmentation.")
    ap.add_argument("--balance-ratio", type=float, default=1.0,
                    help="Cap majority/minority ratio after augmentation (1.0 = 50/50).")
    ap.add_argument("--metrics-subdir", type=str, default="",
                help="Write metrics/preds under data/processed/metrics/<subdir>/")

    args = ap.parse_args()

    # Apply metrics subdir early; from now on always use METRICS_DIR/PRED_DIR
    global METRICS_DIR, PRED_DIR
    if getattr(args, "metrics_subdir", ""):
        METRICS_DIR = METRICS_DIR / args.metrics_subdir
        PRED_DIR    = PRED_DIR    / args.metrics_subdir

    # resolve split path
    if args.split_json:
        split_path = Path(args.split_json)
    else:
        split_path = TEMPORAL_SPLIT if args.use_temporal else FAMILY_SPLIT

    # resolve variant string (for bookkeeping)
    variant = "real"
    if args.use_gan: variant = "gan"
    elif args.oversample: variant = "oversample"
    elif args.smote: variant = "smote"

    # load data + split
    X, y = load_xy()
    kind = "temporal" if args.use_temporal else ("family" if args.use_family else "custom")

    split = SplitIndices.from_json(split_path)

    # metadata for sha/timestamps
    meta = load_metadata()
    sha_all = None
    try:
        if "sha" in meta.columns:
            sha_all = meta["sha"].to_numpy()
    except Exception:
        sha_all = None

    # global train/test indices
    train_idx_full = np.array(split.train, dtype=int)
    test_idx_full  = np.array(split.test, dtype=int)

    Xtr_full, ytr_full = sub(X, train_idx_full), y[train_idx_full]
    Xte,      yte      = sub(X, test_idx_full),  y[test_idx_full]

    # optional: restrict test set to a time window
    test_mask = None
    if args.test_start or args.test_end:
        te_ts = meta.loc[test_idx_full, "timestamp"]  # tz-aware (UTC) series
        test_start = _as_utc(args.test_start) if args.test_start else None
        test_end   = _as_utc(args.test_end)   if args.test_end   else None
        mask = np.ones(len(te_ts), dtype=bool)
        if test_start is not None:
            mask &= (te_ts >= test_start)
        if test_end is not None:
            mask &= (te_ts <  test_end)
        test_mask = mask
        Xte, yte = Xte[mask], yte[mask]

    # Global TEST indices actually used
    test_idx_used = test_idx_full if test_mask is None else test_idx_full[test_mask]

    # scarce subset (ensure some positives AND some negatives)
    if args.scarce_real_frac is not None:
        n_total = max(1, int(round(args.scarce_real_frac * len(ytr_full))))
        rng = np.random.default_rng(args.seed)

        pos_idx = np.where(ytr_full == 1)[0]
        neg_idx = np.where(ytr_full == 0)[0]

        # start near the base rate but enforce floors
        pos_prior = len(pos_idx) / max(1, len(ytr_full))
        n_pos = min(len(pos_idx), max(args.min_train_pos, int(round(n_total * pos_prior))))
        n_neg = min(len(neg_idx), max(args.min_train_neg, n_total - n_pos))

        # top-up to n_total, alternating, while respecting available rows
        while (n_pos + n_neg) < n_total and (n_pos < len(pos_idx) or n_neg < len(neg_idx)):
            if n_pos < len(pos_idx):
                n_pos += 1
            if (n_pos + n_neg) < n_total and n_neg < len(neg_idx):
                n_neg += 1

        pos_sel = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos > 0 else np.array([], int)
        neg_sel = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([], int)
        scarce_idx = np.concatenate([pos_sel, neg_sel])
        rng.shuffle(scarce_idx)

        Xtr, ytr = sub(Xtr_full, scarce_idx), ytr_full[scarce_idx]
        like_src = ("scarce", Xtr[ytr == 1])
        # global TRAIN indices actually used (real rows)
        train_idx_used = train_idx_full[scarce_idx]
    else:
        Xtr, ytr = Xtr_full, ytr_full
        like_src = ("full", Xtr_full[ytr_full == 1])
        train_idx_used = train_idx_full

    print(f"[info] class_counts train={Counter(ytr)} test={Counter(yte)}")
    if len(set(ytr)) < 2:
        print("[warn] train has a single class; AUC undefined; model may be degenerate.")

    # capture pre-augmentation snapshot
    Xtr_before, ytr_before = Xtr.copy(), ytr.copy()
    n_train_real_before = len(ytr_before)
    n_real_pos = int((ytr_before == 1).sum())
    n_synth = 0

    # augment to constant size if requested
    if args.const_train_size is not None and args.const_train_size > len(ytr):
        if args.use_gan:
            assert args.gan_generator, "--gan-generator required with --use-gan"
            X_like = like_src[1] if args.gan_like == "scarce" else Xtr_full[ytr_full == 1]
            if X_like.size == 0:
                X_like = Xtr_full[ytr_full == 1]  # fall back to full positives
            n_synth = max(0, int(args.const_train_size) - len(ytr))

            # cap by per-real limit
            n_synth = min(n_synth, int(args.gan_synth_per_real * max(1, n_real_pos)))

            Xtr, ytr = add_gan_synth(
                Xtr, ytr, X_like, n_synth, args.gan_generator,
                seed=args.seed, scaler_npz=args.gan_scaler,
                quality=args.gan_quality, qmult=args.gan_qmult,
                X_neg_for_quality=Xtr_full[ytr_full == 0],
                boundary_low=args.gan_boundary_low,
                boundary_high=args.gan_boundary_high,
                boundary_k=args.gan_boundary_k,
            )
            n_synth = int((ytr == 1).sum() - n_real_pos)  # record actual synth count

        elif args.oversample:
            Xtr, ytr = oversample_to_const(Xtr, ytr, int(args.const_train_size), seed=args.seed)
            n_synth = int(args.const_train_size) - n_train_real_before

        elif args.smote:
            Xtr, ytr = smote_to_const(Xtr, ytr, int(args.const_train_size), seed=args.seed, k=5)
            n_synth = int(args.const_train_size) - n_train_real_before

        else:
            pass

    # --- optional: rebalance AFTER augmentation (single call; fixed) ---
    if args.balance_after_augment:
        Xtr, ytr = rebalance_after_augment(Xtr, ytr, seed=args.seed, ratio=args.balance_ratio)

    # --- diagnostics for ROS/SMOTE (after augmentation) ---
    ros_dup_ratio = None
    smote_k = None
    smote_synth_count = None
    smote_med_nn_dist = None

    if args.oversample and len(ytr) > 0:
        row_hash = np.apply_along_axis(lambda r: hash(tuple(np.asarray(r).tolist())), 1, Xtr)
        unique = len(np.unique(row_hash))
        ros_dup_ratio = 1.0 - (unique / float(Xtr.shape[0]))

    if args.smote and n_synth > 0:
        smote_k = 5  # matches smote_to_const default k
        smote_synth_count = int(n_synth)
        try:
            X_pos_real = Xtr_before[ytr_before == 1]
            X_pos_synth = Xtr[-n_synth:]  # appended at end in smote_to_const
            nn = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(X_pos_real)
            dists, _ = nn.kneighbors(X_pos_synth, n_neighbors=1, return_distance=True)
            smote_med_nn_dist = float(np.median(dists))
        except Exception:
            smote_med_nn_dist = None

    # fit classifier (with optional validation threshold selection)
    if args.tune:
        clf, best = tune_rf(Xtr, ytr, grid=args.grid)
        print(f"[tune] best={best}")
    else:
        clf = RandomForestClassifier(
            n_estimators=args.rf_n_est,
            max_depth=args.rf_max_depth,
            class_weight=(None if args.rf_class_weight == "none" else "balanced"),
            n_jobs=-1,
            random_state=args.seed,
            oob_score=False
        )

    threshold = 0.5
    # validation split built from the *augmented* TRAIN
    if args.val_threshold != "none":
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        counts = None
        try:
            counts = np.bincount(ytr.astype(int), minlength=2)
        except Exception:
            pass

        can_stratify = (
            ytr is not None
            and len(np.unique(ytr)) == 2
            and len(ytr) > 10
            and counts is not None
            and counts.min() >= 2
        )

        if can_stratify:
            tr_idx, val_idx_local = next(sss.split(Xtr, ytr))
        else:
            tr_idx = np.arange(len(ytr))
            val_idx_local = np.array([], dtype=int)

        # Map *real* part of val indices to global indices for audit (synthetic rows have no global index)
        if len(val_idx_local) > 0:
            # real rows are the first len(ytr_before) positions of Xtr
            real_mask_in_val = val_idx_local < len(ytr_before)
            val_idx_global = np.asarray(train_idx_used)[val_idx_local[real_mask_in_val]]
        else:
            val_idx_global = np.array([], dtype=int)

        # threshold selection (on val) then refit on full TRAIN
        if len(val_idx_local) > 0:
            clf.fit(Xtr[tr_idx], ytr[tr_idx])
            proba_val = clf.predict_proba(Xtr[val_idx_local])[:, 1]
            threshold, _ = pick_threshold(ytr[val_idx_local], proba_val, which=args.val_threshold, grid=200)
        clf.fit(Xtr, ytr)
    else:
        # no validation thresholding; just fit on full TRAIN
        val_idx_global = np.array([], dtype=int)
        clf.fit(Xtr, ytr)

    # --- guard-rail audit (after we know val indices & after augmentation) ---
    rf_params = {
        "n_estimators": args.rf_n_est,
        "max_depth": args.rf_max_depth,
        "class_weight": (None if args.rf_class_weight == "none" else "balanced"),
        "random_state": args.seed,
    }
    audit = audit_splits_and_aug(
        train_idx=np.asarray(train_idx_used),
        val_idx=np.asarray(val_idx_global),
        test_idx=np.asarray(test_idx_used),
        sha=sha_all,
        y_train_before_aug=ytr_before,
        y_train_after_aug=ytr,
        ros_dup_ratio=ros_dup_ratio,
        smote_k=smote_k,
        smote_synth_count=smote_synth_count,
        smote_med_nn_dist=smote_med_nn_dist,
        rf_params=rf_params,
    )

    # evaluate
    try:
        proba = clf.predict_proba(Xte)[:,1]
    except Exception:
        proba = None

    if proba is not None:
        ypred = (proba >= threshold).astype(int)
        try:
            roc_auc = roc_auc_score(yte, proba) if len(set(yte))==2 else float("nan")
            pr_auc  = average_precision_score(yte, proba) if len(set(yte))==2 else float("nan")
        except ValueError:
            roc_auc, pr_auc = float("nan"), float("nan")
    else:
        ypred = clf.predict(Xte)
        roc_auc, pr_auc = float("nan"), float("nan")

    acc = accuracy_score(yte, ypred)
    prec = precision_score(yte, ypred, zero_division=0)
    rec  = recall_score(yte, ypred, zero_division=0)
    f1   = f1_score(yte, ypred, zero_division=0)
    bal  = balanced_accuracy_score(yte, ypred)
    mcc  = matthews_corrcoef(yte, ypred) if len(set(yte)) == 2 else float("nan")

    tn = fp = fn = tp = 0
    if len(set(yte)) == 2:
        tn, fp, fn, tp = confusion_matrix(yte, ypred, labels=[0,1]).ravel()
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    # ensure dirs exist
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    tag = f"_{args.tag}" if args.tag else ""
    metrics_path = METRICS_DIR / f"rf_{kind}_metrics{tag}.json"
    preds_path   = PRED_DIR    / f"rf_{kind}_preds{tag}.npz"

    # save preds
    np.savez_compressed(preds_path, y_true=yte, y_pred=ypred, proba=proba, threshold=threshold)

    # assemble metrics (with audit)
    metrics = {
        "variant": variant,
        "threshold": float(threshold),
        "n_train_real": int(n_train_real_before),
        "n_train_synth": int(max(0, n_synth)),
        "n_train_total": int(len(ytr)),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "specificity": float(spec),
        "balanced_accuracy": float(bal),
        "mcc": float(mcc),
        "auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "scarce_real_frac": args.scarce_real_frac,
        "const_train_size": args.const_train_size,
        "used_gan": bool(args.use_gan),
        "audit": audit,
    }
    if args.test_start: metrics["test_start"] = args.test_start
    if args.test_end:   metrics["test_end"]   = args.test_end
    if args.test_start and args.test_end:
        try:
            d = (pd.Timestamp(args.test_end) - pd.Timestamp(args.test_start)).days
            metrics["test_window_days"] = int(d)
        except Exception:
            pass

    # save metrics
    metrics_path.write_text(json.dumps(metrics, indent=2))

    def _fmt(x): 
        return "nan" if isinstance(x, float) and math.isnan(x) else f"{x:.4f}"
    print(
        f"[eval] {kind} var={variant} thr={_fmt(threshold)} "
        f"AUC={_fmt(roc_auc)} AP={_fmt(pr_auc)} "
        f"Acc={acc:.4f} F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} "
        f"Spec={_fmt(spec)} BalAcc={bal:.4f} MCC={_fmt(mcc)}"
    )

if __name__ == "__main__":
    main()
