import argparse, json, numpy as np
from collections import Counter
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from src.paths import ROOT, TEMPORAL_SPLIT, FAMILY_SPLIT
from src.holdouts import SplitIndices
import math

METRICS_DIR = ROOT / "data" / "processed" / "metrics"
PRED_DIR    = ROOT / "data" / "processed" / "preds"


def load_xy():
    z = np.load(ROOT / "data" / "raw" / "bodmas.npz", allow_pickle=True)
    return z["X"], z["y"]


def sub(X, idx):
    return X[idx] if hasattr(X, "__getitem__") else np.asarray(X)[idx]


def maybe_sample(X, y, n_max=2000, seed=42):
    if len(y) <= n_max:
        return X, y, np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n_max, replace=False)
    return sub(X, idx), y[idx], idx


def tune_rf(X, y, grid="light"):
    presets = {
        "light":  {"n_estimators": [200, 400], "max_depth": [None, 20]},
        "medium": {"n_estimators": [300, 600], "max_depth": [None, 16, 24]},
        "heavy":  {"n_estimators": [400, 800], "max_depth": [None, 14, 20, 28]},
    }
    params = presets[grid]
    base = RandomForestClassifier(n_jobs=-1, random_state=42, oob_score=False)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(base, params, cv=cv, n_jobs=-1, scoring="roc_auc", verbose=1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_


def sample_scarce_indices(y, frac, min_pos=50, seed=42):
    """
    Stratified subset from y with at least `min_pos` positives (if available).
    frac is relative to the given pool (split.train).
    """
    rng = np.random.default_rng(seed)
    n_total = int(np.ceil(frac * len(y)))
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    # target positives ~ pool rate, but not below min_pos (if available)
    pool_pos_rate = len(pos_idx) / max(1, len(y))
    n_pos = max(min_pos, int(round(n_total * pool_pos_rate)))
    n_pos = min(n_pos, len(pos_idx))
    n_neg = max(0, n_total - n_pos)
    n_neg = min(n_neg, len(neg_idx))

    pos_sel = rng.choice(pos_idx, size=n_pos, replace=False) if n_pos > 0 else np.array([], dtype=int)
    neg_sel = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([], dtype=int)
    idx = np.concatenate([pos_sel, neg_sel])
    rng.shuffle(idx)
    return idx


def add_gan_synth(X_real, y_real, X_like_mal, n_synth, generator_path):
    """
    Adds n_synth synthetic malware rows. Assumes your augmentation utilities expose:
      - load_generator(path) -> generator object
      - sample_synthetic(G, n, like=<np.ndarray of malware rows>) -> np.ndarray
    """
    from src.augmentation.model import load_generator, sample_synthetic  # adjust names if needed
    if n_synth <= 0:
        return X_real, y_real
    G = load_generator(generator_path)
    X_syn = sample_synthetic(G, n=n_synth, like=X_like_mal)
    y_syn = np.ones(len(X_syn), dtype=y_real.dtype)
    X_aug = np.vstack([X_real, X_syn])
    y_aug = np.concatenate([y_real, y_syn])
    return X_aug, y_aug


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--use-temporal", action="store_true")
    g.add_argument("--use-family", action="store_true")
    ap.add_argument("--smoke", action="store_true")

    # Scarcity controls
    ap.add_argument("--scarce-real-frac", type=float, default=None,
                    help="Train on only this fraction of the TRAIN pool (e.g., 0.002 for 0.2%). If omitted, use full train.")
    ap.add_argument("--min-train-pos", type=int, default=50,
                    help="Minimum positives to include in scarce subset (if available).")
    ap.add_argument("--const-train-size", type=int, default=None,
                    help="If set and --use-gan, add synthetic to reach this total train size.")

    # GAN augmentation
    ap.add_argument("--use-gan", action="store_true")
    ap.add_argument("--gan-generator", type=str, default=None,
                    help="Path to generator.pth (ideally trained on the same split's TRAIN only).")
    ap.add_argument("--gan-like", choices=["scarce", "full"], default="scarce",
                    help="Use malware rows from the scarce subset ('scarce') or the full TRAIN pool ('full') as the 'like' distribution for the GAN sampler.")

    # RF settings / tuning
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--grid", choices=["light", "medium", "heavy"], default="light")
    ap.add_argument("--rf-n-est", type=int, default=400)
    ap.add_argument("--rf-max-depth", type=int, default=None)

    ap.add_argument("--model-out", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, y = load_xy()
    split_path = TEMPORAL_SPLIT if args.use_temporal else FAMILY_SPLIT
    kind = "temporal" if args.use_temporal else "family"
    split = SplitIndices.from_json(split_path)

    # Full train/test from the holdout split
    Xtr_full, ytr_full = sub(X, split.train), y[split.train]
    Xte, yte = sub(X, split.test), y[split.test]

    # Optionally reduce to a scarce subset of TRAIN
    if args.scarce_real_frac is not None:
        scarce_idx_local = sample_scarce_indices(ytr_full, args.scarce_real_frac,
                                                 min_pos=args.min_train_pos, seed=args.seed)
        Xtr, ytr = sub(Xtr_full, scarce_idx_local), ytr_full[scarce_idx_local]
        like_src = ("scarce", Xtr[ytr == 1])
    else:
        Xtr, ytr = Xtr_full, ytr_full
        like_src = ("full", Xtr_full[ytr_full == 1])

    # Optional smoke (quick) mode
    if args.smoke:
        Xtr, ytr, _ = maybe_sample(Xtr, ytr)
        Xte, yte, _ = maybe_sample(Xte, yte)

    print(f"[info] class_counts train={Counter(ytr)} test={Counter(yte)}")
    if len(set(ytr)) < 2:
        print("[warn] train has a single class; AUC undefined; model will be degenerate.")
    if len(set(yte)) < 2:
        print("[warn] test has a single class; AUC undefined.")

    # Optional augmentation to a constant total train size
    n_train_real_before = len(ytr)  # before any augmentation
    if args.use_gan:
        assert args.gan_generator, "--gan-generator is required with --use-gan"
        if args.const_train_size is None:
            print("[warn] --use-gan set but --const-train-size not provided; adding 0 synthetic.")
            n_synth = 0
        else:
            n_synth = max(0, int(args.const_train_size) - len(ytr))

        # Choose 'like' source for GAN sampling
        if args.gan_like == "scarce":
            X_like = like_src[1]  # malware rows from the scarce subset
            if X_like.size == 0:
                X_like = Xtr_full[ytr_full == 1]  # fallback to full TRAIN malware
        else:
            X_like = Xtr_full[ytr_full == 1]

        Xtr, ytr = add_gan_synth(Xtr, ytr, X_like, n_synth, args.gan_generator)
        print(f"[aug] target={args.const_train_size}  real={n_train_real_before}  synth={n_synth}  final_train_n={len(ytr)}")
    else:
        n_synth = 0

    # Train RF (tuned or fixed)
    if args.tune:
        clf, best = tune_rf(Xtr, ytr, grid=args.grid)
        print(f"[tune] best={best}")
    else:
        clf = RandomForestClassifier(
            n_estimators=args.rf_n_est, max_depth=args.rf_max_depth,
            n_jobs=-1, random_state=42, oob_score=False
        )
        clf.fit(Xtr, ytr)

    # Predictions and probabilities
    ypred = clf.predict(Xte)
    proba = None
    if hasattr(clf, "predict_proba"):
        P = clf.predict_proba(Xte)
        if hasattr(clf, "classes_") and 1 in clf.classes_:
            pos_idx = list(clf.classes_).index(1)
            proba = P[:, pos_idx]

    # METRICS (robust to single-class)
    acc = accuracy_score(yte, ypred)
    prec = precision_score(yte, ypred, zero_division=0)
    rec  = recall_score(yte, ypred, zero_division=0)
    f1   = f1_score(yte, ypred, zero_division=0)
    bal  = balanced_accuracy_score(yte, ypred)
    mcc  = matthews_corrcoef(yte, ypred) if len(set(yte)) == 2 else float("nan")

    tn = fp = fn = tp = 0
    if len(set(yte)) == 2:
        tn, fp, fn, tp = confusion_matrix(yte, ypred, labels=[0, 1]).ravel()
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    roc_auc = float("nan")
    pr_auc  = float("nan")
    if proba is not None and len(set(yte)) == 2:
        roc_auc = roc_auc_score(yte, proba)
        pr_auc  = average_precision_score(yte, proba)

    # Persist
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    metrics_path = METRICS_DIR / f"rf_{kind}_metrics{tag}.json"
    metrics = {
        "n_train_real": int(n_train_real_before),
        "n_train_synth": int(n_synth),
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
    }
    metrics_path.write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(PRED_DIR / f"rf_{kind}_preds{tag}.npz", y_true=yte, y_pred=ypred)

    print(
        f"[eval] {kind} "
        f"AUC={roc_auc if not math.isnan(roc_auc) else float('nan'):.4f} "
        f"AP={pr_auc if not math.isnan(pr_auc) else float('nan'):.4f} "
        f"Acc={acc:.4f} F1={f1:.4f} Prec={prec:.4f} Rec={rec:.4f} "
        f"Spec={spec:.4f} BalAcc={bal:.4f} MCC={mcc if not math.isnan(mcc) else float('nan'):.4f}"
    )

    if args.model_out:
        from joblib import dump
        dump(clf, args.model_out)


if __name__ == "__main__":
    main()
