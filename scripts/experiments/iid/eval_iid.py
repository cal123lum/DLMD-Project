#!/usr/bin/env python
import argparse, json, math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

# repo paths
ROOT_DIR = os.path.abspath(os.path.join(__file__, "..", "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.paths import ROOT, BODMAS_NPZ
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix,
)
from sklearn.neighbors import NearestNeighbors

# ---------------- GAN bits (simple, malware-only) ----------------
from src.augmentation.model import Generator
from src.augmentation import config_aug as C

@torch.no_grad()
def gan_generate(G: Generator, n_samples: int, seed: int) -> np.ndarray:
    if n_samples <= 0:
        return np.empty((0, C.FEATURE_DIM), dtype=np.float32)
    torch.manual_seed(seed)
    z = torch.randn(n_samples, C.LATENT_DIM)
    return G(z).cpu().numpy().astype(np.float32)

def load_generator(gen_path: str) -> Generator:
    G = Generator().to("cpu")
    state = torch.load(gen_path, map_location="cpu")
    G.load_state_dict(state)
    G.eval()
    return G

# ---------------- simple augmenters ----------------
def oversample_to_const(X, y, target_n, seed=42):
    if target_n is None or target_n <= len(y):
        return X, y, 0
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    need = target_n - len(y)
    if len(pos) == 0 or need <= 0:
        return X, y, 0
    pick = rng.choice(pos, size=need, replace=True)
    X_syn = X[pick]
    y_syn = np.ones(len(pick), dtype=y.dtype)
    X2 = np.vstack([X, X_syn])
    y2 = np.concatenate([y, y_syn])
    return X2, y2, int(len(pick))

def smote_to_const(X, y, target_n, seed=42, k=5):
    if target_n is None or target_n <= len(y):
        return X, y, 0
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    need = target_n - len(y)
    if len(pos_idx) < 2 or need <= 0:
        # fallback to ROS
        return oversample_to_const(X, y, target_n, seed=seed)
    Xpos = X[pos_idx]
    k_eff = min(int(k), max(1, len(Xpos) - 1))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean").fit(Xpos)
    nbrs = nn.kneighbors(Xpos, return_distance=False)
    synth = []
    for _ in range(need):
        i = int(rng.integers(0, len(Xpos)))
        xi = Xpos[i]
        cand = [int(j) for j in nbrs[i] if int(j) != i]
        if not cand:
            j = int(rng.integers(0, len(Xpos) - 1))
            if j >= i: j += 1
        else:
            j = cand[int(rng.integers(0, len(cand)))]
        xj = Xpos[j]
        t = float(rng.random())
        synth.append(xi + t * (xj - xi))
    X_syn = np.vstack(synth).astype(np.float32)
    y_syn = np.ones(len(X_syn), dtype=y.dtype)
    X2 = np.vstack([X, X_syn])
    y2 = np.concatenate([y, y_syn])
    return X2, y2, int(len(X_syn))

# ---------------- misc ----------------
def pick_threshold(y_true, proba, which="balacc", grid=200):
    ts = np.linspace(0.0, 1.0, num=grid+1)[1:-1]
    if which == "f1":
        scorer = lambda yt, yp: f1_score(yt, yp, zero_division=0)
    elif which == "mcc":
        scorer = lambda yt, yp: matthews_corrcoef(yt, yp)
    elif which == "balacc":
        scorer = lambda yt, yp: balanced_accuracy_score(yt, yp)
    else:
        raise ValueError("unknown threshold metric")
    vals = [scorer(y_true, (proba >= t).astype(int)) for t in ts]
    best_i = int(np.nanargmax(vals)) if len(vals) else 0
    return float(ts[best_i])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["real", "ros", "smote", "gan"], required=True)
    ap.add_argument("--gan-generator", type=str, default=None, help="Path to generator.pth (for --method gan)")
    ap.add_argument("--seed", type=int, default=42)

    # scarcity and target size
    ap.add_argument("--scarce-real-frac", type=float, default=None, help="Fraction of TRAIN to keep as real; stratified")
    ap.add_argument("--min-train-pos", type=int, default=100)
    ap.add_argument("--min-train-neg", type=int, default=500)
    ap.add_argument("--const-train-size", type=int, default=None, help="Grow TRAIN to this size")

    # GAN controls
    ap.add_argument("--gan-synth-per-real", type=float, default=40.0, help="Cap synthetic positives as x per real positive")

    # RF
    ap.add_argument("--rf-n-est", type=int, default=160)
    ap.add_argument("--rf-max-depth", type=int, default=20)
    ap.add_argument("--rf-class-weight", choices=["balanced","none"], default="none")
    ap.add_argument("--val-threshold", choices=["none","balacc","f1","mcc"], default="balacc")

    # outputs
    ap.add_argument("--tag", type=str, default="iid_quick")
    ap.add_argument("--metrics-subdir", type=str, default="iid_quick")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    # 1) Load NPZ and split IID
    z = np.load(str(BODMAS_NPZ), allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=args.seed)

    # 2) Scarcity (optional, stratified; with floors)
    if args.scarce_real_frac is not None:
        n_total = max(1, int(round(args.scarce_real_frac * len(ytr))))
        idx_pos = np.where(ytr == 1)[0]
        idx_neg = np.where(ytr == 0)[0]
        n_pos = min(len(idx_pos), max(args.min_train_pos, int(round(n_total * (len(idx_pos)/len(ytr))))))
        n_neg = min(len(idx_neg), max(args.min_train_neg, n_total - n_pos))
        # top-up
        while (n_pos + n_neg) < n_total and (n_pos < len(idx_pos) or n_neg < len(idx_neg)):
            if n_pos < len(idx_pos): n_pos += 1
            if (n_pos + n_neg) < n_total and n_neg < len(idx_neg): n_neg += 1
        sel_pos = rng.choice(idx_pos, size=n_pos, replace=False) if n_pos > 0 else np.array([], int)
        sel_neg = rng.choice(idx_neg, size=n_neg, replace=False) if n_neg > 0 else np.array([], int)
        sel = rng.permutation(np.concatenate([sel_pos, sel_neg]))
        Xtr_small, ytr_small = Xtr[sel], ytr[sel]
    else:
        Xtr_small, ytr_small = Xtr, ytr

    n_train_real_before = len(ytr_small)
    n_real_pos = int((ytr_small == 1).sum())
    n_added = 0

    # 3) Augment (optional)
    method = args.method
    if args.const_train_size is not None and args.const_train_size > len(ytr_small):
        if method == "ros":
            Xtr_aug, ytr_aug, n_added = oversample_to_const(Xtr_small, ytr_small, args.const_train_size, seed=args.seed)

        elif method == "smote":
            Xtr_aug, ytr_aug, n_added = smote_to_const(Xtr_small, ytr_small, args.const_train_size, seed=args.seed, k=5)

        elif method == "gan":
            assert args.gan_generator, "--gan-generator is required for --method gan"
            G = load_generator(args.gan_generator)
            need = args.const_train_size - len(ytr_small)
            # cap by per-real limit
            cap = int(args.gan_synth_per_real * max(1, n_real_pos))
            n_synth = min(need, cap)
            X_syn = gan_generate(G, n_synth, seed=args.seed)
            y_syn = np.ones(n_synth, dtype=int)
            Xtr_aug = np.vstack([Xtr_small, X_syn])
            ytr_aug = np.concatenate([ytr_small, y_syn])
            n_added = int(n_synth)

        else:  # method == "real"
            Xtr_aug, ytr_aug = Xtr_small, ytr_small
    else:
        Xtr_aug, ytr_aug = Xtr_small, ytr_small

    # 4) Train RF (with optional val-threshold on augmented TRAIN)
    cw = None if args.rf_class_weight == "none" else "balanced"
    clf = RandomForestClassifier(
        n_estimators=args.rf_n_est, max_depth=args.rf_max_depth,
        class_weight=cw, n_jobs=-1, random_state=args.seed, oob_score=False
    )

    threshold = 0.5
    if args.val_threshold != "none" and len(np.unique(ytr_aug)) == 2 and len(ytr_aug) > 10:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        tr_i, va_i = next(sss.split(Xtr_aug, ytr_aug))
        clf.fit(Xtr_aug[tr_i], ytr_aug[tr_i])
        proba_val = clf.predict_proba(Xtr_aug[va_i])[:, 1]
        threshold = pick_threshold(ytr_aug[va_i], proba_val, which=args.val_threshold)
        clf.fit(Xtr_aug, ytr_aug)
    else:
        clf.fit(Xtr_aug, ytr_aug)

    # 5) Evaluate
    proba = clf.predict_proba(Xte)[:, 1]
    ypred = (proba >= threshold).astype(int)

    def _safe_auc(ytrue, p):
        try:
            return float(roc_auc_score(ytrue, p)) if len(set(ytrue)) == 2 else float("nan")
        except Exception:
            return float("nan")

    roc_auc = _safe_auc(yte, proba)
    try:
        pr_auc = float(average_precision_score(yte, proba))
    except Exception:
        pr_auc = float("nan")

    acc = float(accuracy_score(yte, ypred))
    prec = float(precision_score(yte, ypred, zero_division=0))
    rec  = float(recall_score(yte, ypred, zero_division=0))
    f1   = float(f1_score(yte, ypred, zero_division=0))
    bal  = float(balanced_accuracy_score(yte, ypred))
    mcc  = float(matthews_corrcoef(yte, ypred)) if len(set(yte)) == 2 else float("nan")

    tn = fp = fn = tp = 0
    if len(set(yte)) == 2:
        tn, fp, fn, tp = confusion_matrix(yte, ypred, labels=[0,1]).ravel()
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    # 6) Save
    out_dir_m = ROOT / "data" / "processed" / "metrics" / args.metrics_subdir
    out_dir_p = ROOT / "data" / "processed" / "preds"   / args.metrics_subdir
    out_dir_m.mkdir(parents=True, exist_ok=True)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    tag = f"_{args.tag}" if args.tag else ""
    metrics_path = out_dir_m / f"rf_iid_metrics{tag}.json"
    preds_path   = out_dir_p / f"rf_iid_preds{tag}.npz"

    metrics = {
        "kind": "iid",
        "method": method,
        "threshold": float(threshold),
        "n_train_real": int(n_train_real_before),
        "n_train_synth": int(max(0, n_added)),
        "n_train_total": int(len(ytr_aug)),
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "specificity": float(spec), "balanced_accuracy": bal, "mcc": mcc,
        "auc": roc_auc, "pr_auc": pr_auc,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "scarce_real_frac": args.scarce_real_frac,
        "const_train_size": args.const_train_size,
        "rf": {"n_estimators": args.rf_n_est, "max_depth": args.rf_max_depth,
               "class_weight": args.rf_class_weight, "random_state": args.seed},
    }

    metrics_path.write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(preds_path, y_true=yte, y_pred=ypred, proba=proba, threshold=threshold)

    def _fmt(x): 
        return "nan" if isinstance(x, float) and math.isnan(x) else f"{x:.4f}"
    print(
        f"[iid] {method} thr={_fmt(threshold)} AUC={_fmt(roc_auc)} AP={_fmt(pr_auc)} "
        f"Acc={acc:.4f} F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} BalAcc={bal:.4f} MCC={_fmt(mcc)} "
        f"(n_real={n_train_real_before}, n_added={n_added}, n_total={len(ytr_aug)})"
    )

if __name__ == "__main__":
    main()
