#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

NPZ = Path("data/raw/ember.npz")
OUT = Path("data/holdouts/ember_official_split.json")

def main():
    z = np.load(NPZ)
    n = int(z["y"].shape[0])

    # infer n_train_labeled by counting metadata rows from train chunks is annoying here;
    # simplest: reuse EMBER's known structure: train then test in build_ember_npz.py
    # We can detect the boundary because create_vectorized_features always returns train/test separately.
    # If you want it exact, print n_train_labeled in build_ember_npz.py and paste it here.
    raise SystemExit(
        "Add one print to build_ember_npz.py: print('n_train_labeled', int(m.sum()), 'n_test', X_test.shape[0]) "
        "then rerun and replace boundary below."
    )

if __name__ == "__main__":
    main()
