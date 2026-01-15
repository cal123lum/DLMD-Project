from pathlib import Path
import numpy as np
import ember

EMBER_DIR = Path("data/raw/ember2018")
OUT_NPZ = Path("data/raw/ember.npz")

def main():
    # Creates the vectorized binary feature files on disk
    ember.create_vectorized_features(str(EMBER_DIR))  # :contentReference[oaicite:2]{index=2}

    # Load vectorized matrices
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(str(EMBER_DIR))  # :contentReference[oaicite:3]{index=3}

    # Training set contains unlabeled samples (-1). Drop them.
    m = (y_train != -1)
    X = np.vstack([X_train[m], X_test])
    y = np.concatenate([y_train[m], y_test]).astype(np.int64)

    OUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, X=X, y=y)

    print("saved", OUT_NPZ)
    print("X", X.shape, "y", y.shape, "unique labels", np.unique(y, return_counts=True))
    print("n_train_labeled", int(m.sum()), "n_test", X_test.shape[0])

if __name__ == "__main__":
    main()
