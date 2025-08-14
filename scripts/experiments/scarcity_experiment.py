# src/augmentation/scarcity_experiment_npz_fixed.py

import os, sys
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from src.paths import BODMAS_NPZ, DATA_PROCESSED

# ─── allow import of augmentation/model.py & config_aug ──────────────────────
SCRIPT_DIR = os.path.dirname(__file__)
SRC_DIR    = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from src.augmentation.model import Generator
from src.augmentation import config_aug as C 



# ─── file paths ──────────────────────────────────────────────────────────────
NPZ_PATH    = str(BODMAS_NPZ)
GEN_PATH    = "models/augmentation/generator.pth"
OUTPUT_CSV  = str(DATA_PROCESSED / "scarcity_npz_fixed_results_zoom2.csv")

# ─── experiment settings ─────────────────────────────────────────────────────
FRACTIONS   = [0.002, 0.0045, 0.37, 0.99] 
CHUNK_SIZE  = 10000                   # batch size for synthetic generation

def generate_synthetic(G, n_samples, feature_dim):
    """Generate n_samples synthetic vectors in CHUNK_SIZE batches."""
    X_synth = np.empty((n_samples, feature_dim), dtype=np.float32)
    pos = 0
    while pos < n_samples:
        batch = min(CHUNK_SIZE, n_samples - pos)
        z = torch.randn(batch, C.LATENT_DIM)
        with torch.no_grad():
            X_synth[pos:pos+batch] = G(z).cpu().numpy()
        pos += batch
    return X_synth

def main():
    # 1) load X, y
    data = np.load(NPZ_PATH, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(int)
    print(f"Loaded {NPZ_PATH}: {X.shape[0]} samples, {X.shape[1]} features")

    # 2) 75/25 stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"Train={len(X_train)}, Test={len(X_test)}")

    # 3) load GAN generator
    G = Generator()
    G.load_state_dict(torch.load(GEN_PATH, map_location="cpu"))
    G.eval()

    results = []
    # 4) for each fraction
    for frac in FRACTIONS:
        n_real  = max(1, int(len(X_train) * frac))
        idx     = np.random.RandomState(42).choice(len(X_train), n_real, replace=False)
        X_real, y_real = X_train[idx], y_train[idx]

        # compute real‐only class balance
        p_mal = y_real.mean()      # fraction of malware
        p_ben = 1 - p_mal

        # --- REAL‐ONLY BASELINE ---
        clf_r = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_r.fit(X_real, y_real)
        auc_r = roc_auc_score(y_test, clf_r.predict_proba(X_test)[:,1])

        # --- SYNTHETIC AUGMENTATION ---
        n_synth = len(X_train) - n_real
        X_synth = generate_synthetic(G, n_synth, X_train.shape[1])

        # label synthetic according to real distribution
        y_synth = np.random.choice(
            [0, 1], size=n_synth, p=[p_ben, p_mal]
        )

        # clip extremes to ±3σ of the small real subset
        mean, std = X_real.mean(axis=0), X_real.std(axis=0)
        X_synth = np.clip(X_synth, mean - 3*std, mean + 3*std)

        # combine
        X_aug = np.vstack([X_real, X_synth])
        y_aug = np.concatenate([y_real, y_synth])

        clf_a = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_a.fit(X_aug, y_aug)
        auc_a = roc_auc_score(y_test, clf_a.predict_proba(X_test)[:,1])

        print(f"frac={frac:.3%} → AUC real={auc_r:.4f}, AUC aug={auc_a:.4f}")
        results.append({
            "real_frac": frac,
            "auc_real": auc_r,
            "auc_aug": auc_a,
            "p_mal":    p_mal
        })

    # 5) save
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
