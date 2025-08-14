# src/create_mini_npz.py
import numpy as np
from src.paths import BODMAS_NPZ

data = np.load(str(BODMAS_NPZ))
X, y = data['X'][:1000], data['y'][:1000]
np.savez('data/raw/bodmas_mini.npz', X=X, y=y)
print("Mini NPZ saved: 1000 samples.")
