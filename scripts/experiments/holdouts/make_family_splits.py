#!/usr/bin/env python3
import json, hashlib, numpy as np, pandas as pd
from pathlib import Path

META_CSV = Path("data/raw/bodmas_metadata.csv")   # <- you used this exact name
NPZ      = Path("data/raw/bodmas.npz")
OUT_DIR  = Path("data/processed/splits/family_lofo")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# knobs
MIN_TEST_POS      = 200     # require at least this many positives in the held-out family
BENIGN_TEST_FRAC  = 0.20    # ~20% of benigns go to test, deterministically by sha

# load
meta = pd.read_csv(META_CSV)
y = np.load(NPZ, allow_pickle=True)["y"].astype(int)
assert len(meta) == len(y), "metadata rows must align with X/y order"

# clean family labels
fam = meta["family"].fillna("").astype(str).str.strip()
fam = fam.replace({"": "UNKNOWN"})

sha = meta["sha"].astype(str)

idx_all = np.arange(len(y))
pos_idx = np.where(y == 1)[0]
neg_idx = np.where(y == 0)[0]

# deterministic benign split by hashing sha
def stable_hash01(s: str) -> float:
    h = hashlib.md5(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF  # [0,1]

benign_hash = np.array([stable_hash01(s) for s in sha[neg_idx]])
neg_test_mask = benign_hash < BENIGN_TEST_FRAC
neg_test_idx = neg_idx[neg_test_mask]            # go to test
neg_train_idx = neg_idx[~neg_test_mask]          # stay in train

# iterate named families (exclude UNKNOWN)
families = sorted(x for x in fam.unique() if x != "UNKNOWN")

for F in families:
    fam_pos_idx = np.where((y == 1) & (fam.values == F))[0]
    n_pos = len(fam_pos_idx)
    if n_pos < MIN_TEST_POS:
        print(f"[skip] {F}: only {n_pos} positives (< {MIN_TEST_POS})")
        continue

    test = np.sort(np.concatenate([fam_pos_idx, neg_test_idx]))
    train = np.setdiff1d(idx_all, test, assume_unique=False)

    out = {
        "train": train.tolist(),
        "test": test.tolist(),
        "family": F,
        "n_test_pos": int((y[test]==1).sum()),
        "n_test_neg": int((y[test]==0).sum()),
        "benign_test_frac": BENIGN_TEST_FRAC,
        "min_test_pos": MIN_TEST_POS,
    }
    path = OUT_DIR / f"holdout_{F}.json"
    path.write_text(json.dumps(out))
    print(f"[ok] wrote {path}  (test: pos={out['n_test_pos']}, neg={out['n_test_neg']})")
