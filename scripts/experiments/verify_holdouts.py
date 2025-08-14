# scripts/experiments/verify_holdouts.py
import numpy as np
from src.paths import TEMPORAL_SPLIT, FAMILY_SPLIT
from src.holdouts import SplitIndices
from src.data.metadata import load_metadata

def main():
    meta = load_metadata()

    # Temporal check
    try:
        t = SplitIndices.from_json(TEMPORAL_SPLIT)
        train_max = meta["timestamp"].iloc[t.train].dropna().max()
        test_min  = meta["timestamp"].iloc[t.test].dropna().min()
        ok = (test_min is None) or (train_max is None) or (test_min > train_max)
        print(f"[temporal] train_max={train_max} test_min={test_min} ok={ok}")
    except FileNotFoundError:
        print("[temporal] no indices; skip")

    # Family check
    try:
        f = SplitIndices.from_json(FAMILY_SPLIT)
        fam_train = set(meta["family"].iloc[f.train].unique())
        fam_test  = set(meta["family"].iloc[f.test].unique())
        inter = fam_train & fam_test - {"Benign"}  # allow Benign on both
        has_benign_train = "Benign" in fam_train
        has_benign_test  = "Benign" in fam_test
        ok = (len(inter) == 0) and has_benign_train and has_benign_test
        print(f"[family] overlap(excl. Benign)={sorted(inter)}  benign(train,test)=({has_benign_train},{has_benign_test}) ok={ok}")
    except FileNotFoundError:
        print("[family] no indices; skip")

if __name__ == "__main__":
    main()
