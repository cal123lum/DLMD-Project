import argparse, json, numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

from src.paths import ROOT, TEMPORAL_SPLIT, FAMILY_SPLIT
from src.holdouts import SplitIndices

METRICS_DIR = ROOT / "data" / "processed" / "metrics"
PRED_DIR    = ROOT / "data" / "processed" / "preds"

def load_xy():
    z = np.load(ROOT / "data" / "raw" / "bodmas.npz", allow_pickle=True)
    return z["X"], z["y"]

def sub(X, idx): return X[idx] if hasattr(X, "__getitem__") else np.asarray(X)[idx]

def maybe_sample(X, y, n_max=2000, seed=42):
    if len(y) <= n_max: return X, y, np.arange(len(y))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(y), size=n_max, replace=False)
    return sub(X, idx), y[idx], idx

def tune_rf(X, y, grid="light"):
    presets = {
        "light":  {"n_estimators":[200,400], "max_depth":[None,20]},
        "medium": {"n_estimators":[300,600], "max_depth":[None,16,24]},
        "heavy":  {"n_estimators":[400,800], "max_depth":[None,14,20,28]},
    }
    params = presets[grid]
    base = RandomForestClassifier(n_jobs=-1, random_state=42, oob_score=False)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    gs = GridSearchCV(base, params, cv=cv, n_jobs=-1, scoring="roc_auc", verbose=1)
    gs.fit(X, y)
    return gs.best_estimator_, gs.best_params_

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--use-temporal", action="store_true")
    g.add_argument("--use-family", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--grid", choices=["light","medium","heavy"], default="light")
    ap.add_argument("--rf-n-est", type=int, default=400)
    ap.add_argument("--rf-max-depth", type=int, default=None)
    ap.add_argument("--model-out", type=str, default=None)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    X, y = load_xy()
    split_path = TEMPORAL_SPLIT if args.use_temporal else FAMILY_SPLIT
    kind = "temporal" if args.use_temporal else "family"
    split = SplitIndices.from_json(split_path)

    Xtr, ytr = sub(X, split.train), y[split.train]
    Xte, yte = sub(X, split.test),  y[split.test]

    if args.smoke:
        Xtr, ytr, _ = maybe_sample(Xtr, ytr)
        Xte, yte, _ = maybe_sample(Xte, yte)

    if args.tune:
        clf, best = tune_rf(Xtr, ytr, grid=args.grid)
        print(f"[tune] best={best}")
    else:
        clf = RandomForestClassifier(
            n_estimators=args.rf_n_est, max_depth=args.rf_max_depth,
            n_jobs=-1, random_state=42, oob_score=False
        )
        clf.fit(Xtr, ytr)

    ypred = clf.predict(Xte)
    proba = clf.predict_proba(Xte)[:,1] if hasattr(clf,"predict_proba") else None
    auc  = roc_auc_score(yte, proba) if proba is not None else float("nan")
    acc  = accuracy_score(yte, ypred)
    f1   = f1_score(yte, ypred)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    (METRICS_DIR / f"rf_{kind}_metrics{tag}.json").write_text(
        json.dumps({"n_train": int(len(ytr)), "n_test": int(len(yte)),
                    "accuracy": float(acc), "auc": float(auc), "f1": float(f1)}, indent=2)
    )
    np.savez_compressed(PRED_DIR / f"rf_{kind}_preds{tag}.npz",
                        y_true=yte, y_pred=ypred)

    if args.model_out:
        from joblib import dump
        dump(clf, args.model_out)

    print(f"[eval] {kind} acc={acc:.4f} auc={auc:.4f} f1={f1:.4f}")

if __name__ == "__main__":
    main()
