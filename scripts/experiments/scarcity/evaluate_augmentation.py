import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))
import numpy as np
import pandas as pd
import torch
import joblib
from src.paths import GAN_GENERATOR_PTH, DATA_PROCESSED

from src.augmentation.model import Generator
from src.augmentation import config_aug as C 


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score



def load_generator():
    G = Generator().to('cpu')
    state = torch.load(str(GAN_GENERATOR_PTH), map_location='cpu')
    G.load_state_dict(state)
    G.eval()
    return G

def sample_synthetic(G, n_samples):
    """Generate n_samples synthetic feature vectors via G."""
    with torch.no_grad():
        z = torch.randn(n_samples, C.LATENT_DIM)
        fake = G(z).cpu().numpy()
    return fake

def main():
    # 1) Load real train/test
    train_df = pd.read_csv(config.TRAIN_PATH)
    test_df  = pd.read_csv(config.TEST_PATH)
    X_test = test_df.drop(columns=['label']).values
    y_test = test_df['label'].values

    # 2) Load generator
    G = load_generator()

    # 3) Fixed RF hyperparameters (defaults or tuned)
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth':    None,
        'random_state': 42,
        'n_jobs':      -1
    }

    # 4) Sweep synthetic ratios
    ratios = [0.25, 0.5, 0.75, 1.0]
    results = []

    for r in ratios:
        n_real = len(train_df)
        n_synth = int(n_real * r)

        # real data
        X_real = train_df.drop(columns=['label']).values
        y_real = train_df['label'].values

        # synthetic data
        X_synth = sample_synthetic(G, n_synth)
        y_synth = np.ones(n_synth, dtype=int)

        # combine
        X_aug = np.vstack([X_real, X_synth])
        y_aug = np.concatenate([y_real, y_synth])

        # train & evaluate
        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_aug, y_aug)
        y_proba = clf.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, y_proba)

        print(f"Synth ratio {r:0.2f} → AUC = {auc:.4f}")
        results.append({'ratio': r, 'auc': auc})

    # 5) Save results
    out = pd.DataFrame(results)
    csv_path = DATA_PROCESSED / 'augmentation_results.csv'
    out.to_csv(str(csv_path), index=False)
    print(f"Saved augmentation results to {csv_path}")

if __name__ == '__main__':
    main()
