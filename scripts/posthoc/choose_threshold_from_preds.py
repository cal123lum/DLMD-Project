#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import f1_score, balanced_accuracy_score, matthews_corrcoef

PRED_DIR = Path("data/processed/preds")

# --- robust NPZ loader -------------------------------------------------------
def load_preds(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    keys = {k.lower(): k for k in z.files}

    # candidate keys for labels and scores
    y_keys = [k for k in keys if any(s in k for s in ["y_true","labels","y","target"])]
    p_keys = [k for k in keys if any(s in k for s in ["y_proba","proba","prob","score","scores","y_score","preds"])]

    def pick_y():
        for k in y_keys:
            a = z[keys[k]]
            if a.dtype.kind in "biu" and a.ndim == 1:
                return a.astype(int)
        # fallback: pick 1D int-like smallest array
        cand = [z[keys[k]] for k in z.files if z[keys[k]].ndim==1]
        cand = [a for a in cand if a.dtype.kind in "biu"]
        return (cand[0] if cand else z[z.files[0]]).astype(int)

    def pick_p():
        # prefer 1D float in [0,1]
        for k in p_keys:
            a = z[keys[k]].astype(float).ravel()
            if a.ndim==1 and np.isfinite(a).all():
                if a.min() >= -1e-6 and a.max() <= 1+1e-6:
                    return a.clip(0,1)
        # fallback: any 1D float array
        for k in z.files:
            a = z[k]
            if a.ndim==1 and a.dtype.kind == "f":
                return a.astype(float)
        # last resort: if we got a 2D [n,2], take column 1
        for k in z.files:
            a = z[k]
            if a.ndim==2 and a.shape[1]==2 and a.dtype.kind=="f":
                return a[:,1].astype(float)
        raise ValueError(f"no usable proba in {npz_path}")

    y = pick_y()
    p = pick_p()
    if y.shape[0] != p.shape[0]:
        # try to broadcast if p is [n,2]
        raise ValueError(f"shape mismatch {y.shape} vs {p.shape} in {npz_path}")
    return y, p

def infer_variant(tag: str, const_train_size):
    tag = (tag or "").lower()
    if "_gan" in tag: return "gan"
    if "_os" in tag or "_oversample" in tag: return "oversample"
    if "_sm" in tag or "_smote" in tag: return "smote"
    if "_real" in tag: return "real"
    return "real" if (pd.isna(const_train_size) or float(const_train_size) != float(const_train_size)) else "real"

# --- threshold search (non-degenerate) ---------------------------------------
def pick_threshold_non_degenerate(y, p, objective="balanced_accuracy", n_grid=512):
    p = np.asarray(p, float); y = np.asarray(y, int)
    pmin, pmax = float(p.min()), float(p.max())
    if not np.isfinite([pmin,pmax]).all() or pmin == pmax:
        return 0.5, None

    qs = np.linspace(0.005, 0.995, n_grid)
    grid = np.unique(np.quantile(p, qs))
    grid = grid[(grid > pmin) & (grid < pmax)]
    best_s, best_t = -1.0, 0.5

    for t in grid:
        yhat = (p >= t).astype(int)
        s1 = yhat.sum()
        if s1 == 0 or s1 == yhat.size:  # skip all-zeros/all-ones
            continue
        if objective == "balanced_accuracy":
            s = balanced_accuracy_score(y, yhat)
        elif objective == "mcc":
            try: s = matthews_corrcoef(y, yhat)
            except Exception: s = -1.0
        else:
            s = f1_score(y, yhat)
        if s > best_s:
            best_s, best_t = float(s), float(t)

    return best_t, (None if best_s < 0 else best_s)

# --- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_csv")
    ap.add_argument("--objective", default="balanced_accuracy",
                    choices=["balanced_accuracy","mcc","f1"])
    ap.add_argument("--n-grid", type=int, default=512)
    ap.add_argument("--out-suffix", default=None,
                    help="override output suffix (else _baopt/_mccopt/_f1opt)")
    args = ap.parse_args()

    raw = Path(args.raw_csv)
    df = pd.read_csv(raw)

    if "variant" not in df.columns:
        df["variant"] = [infer_variant(str(t), r.get("const_train_size"))
                         for t, (_, r) in zip(df["tag"], df.iterrows())]
    df["variant"] = df["variant"].astype(str).str.lower()

    keys = ["prefix","frac","variant"]
    thr_map, rows_info = {}, []

    for gvals, sub in df.groupby(keys):
        ys, ps = [], []
        for tag in sub["tag"].astype(str):
            npz = PRED_DIR / f"rf_temporal_preds_{tag}.npz"
            if not npz.exists(): continue
            try:
                y, p = load_preds(npz)
                ys.append(y); ps.append(p)
            except Exception as e:
                # comment the next line if too chatty
                # print(f"[skip] {npz}: {e}")
                continue
        if not ys:
            continue
        y_all = np.concatenate(ys); p_all = np.concatenate(ps)
        thr, score = pick_threshold_non_degenerate(y_all, p_all,
                                                   args.objective, args.n_grid)
        thr_map[gvals] = thr
        rows_info.append((*gvals, thr, score,
                          int((y_all==1).sum()), int((y_all==0).sum())))

    out_rows = []
    for _, r in df.iterrows():
        gkey = (r["prefix"], r["frac"], r["variant"])
        t = thr_map.get(gkey, r.get("threshold", 0.5))
        tag = str(r["tag"])
        npz = PRED_DIR / f"rf_temporal_preds_{tag}.npz"
        if npz.exists():
            try:
                y, p = load_preds(npz)
                yhat = (p >= t).astype(int)
                r["threshold"] = t
                r["f1"] = f1_score(y, yhat)
                r["balanced_accuracy"] = balanced_accuracy_score(y, yhat)
                try: r["mcc"] = matthews_corrcoef(y, yhat)
                except Exception: pass
            except Exception:
                pass
        out_rows.append(r)

    suffix = args.out_suffix or {"balanced_accuracy":"_baopt",
                                 "mcc":"_mccopt","f1":"_f1opt"}[args.objective]
    out = raw.with_name(raw.stem + suffix + raw.suffix)
    pd.DataFrame(out_rows).to_csv(out, index=False)
    print(f"[ok] wrote {out}")

    if rows_info:
        info = pd.DataFrame(rows_info,
                            columns=["prefix","frac","variant","threshold",
                                     "best_score","n_pos","n_neg"])
        print(info.head(20).to_string(index=False))

if __name__ == "__main__":
    main()
