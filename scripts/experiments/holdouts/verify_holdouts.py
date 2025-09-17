# scripts/experiments/verify_holdouts.py
# Holdout split sanity checks for temporal and family regimes
# Author: Callum Musselwhite
# Last edit: 2025-09-17
# Purpose: verify that temporal splits respect time ordering and family LOFO splits have no family leakage except Benign

import numpy as np  # kept even if not used elsewhere
from src.paths import TEMPORAL_SPLIT, FAMILY_SPLIT
from src.holdouts import SplitIndices
from src.data.metadata import load_metadata


def main():
    meta = load_metadata()

    # temporal check: all train timestamps should be <= all test timestamps
    # if either side has missing timestamps, treat as ok and print what we found
    try:
        t = SplitIndices.from_json(TEMPORAL_SPLIT)
        train_max = meta["timestamp"].iloc[t.train].dropna().max()
        test_min = meta["timestamp"].iloc[t.test].dropna().min()
        ok = (test_min is None) or (train_max is None) or (test_min > train_max)
        print(f"[temporal] train_max={train_max} test_min={test_min} ok={ok}")
    except FileNotFoundError:
        print("[temporal] no indices; skip")

    # family check: no overlapping malware families between train and test
    # Benign is allowed to appear in both since family LOFO targets malware families only
    try:
        f = SplitIndices.from_json(FAMILY_SPLIT)
        fam_train = set(meta["family"].iloc[f.train].unique())
        fam_test = set(meta["family"].iloc[f.test].unique())
        inter = fam_train & fam_test - {"Benign"}
        has_benign_train = "Benign" in fam_train
        has_benign_test = "Benign" in fam_test
        ok = (len(inter) == 0) and has_benign_train and has_benign_test
        print(f"[family] overlap(excl. Benign)={sorted(inter)}  benign(train,test)=({has_benign_train},{has_benign_test}) ok={ok}")
    except FileNotFoundError:
        print("[family] no indices; skip")


if __name__ == "__main__":
    main()
