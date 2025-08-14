# src/hyperparameter_tuning.py

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from scipy.stats import randint
import config

def main():
    # 1) Load full data
    data = np.load(config.NPZ_PATH)
    X, y = data['X'], data['y']
    print(f"Loaded {X.shape[0]} samples × {X.shape[1]} features")

    # 2) (Optional) Subsample for speed
    #    Comment out this block if you want to tune on the full set.
    X, _, y, _ = train_test_split(
        X, y,
        train_size=0.2,           # use 20% of data for tuning
        stratify=y,
        random_state=config.RANDOM_STATE
    )
    print(f"Subsampled to {X.shape[0]} for hyperparam search")

    # 3) Define RF and parameter distributions
    rf = RandomForestClassifier(
        oob_score=True,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    param_dist = {
        'n_estimators': randint(50, 200),       # try between 50 and 200 trees
        'max_depth': [None] + list(range(10, 101, 10)),  # None or depths 10,20,…100
        'min_samples_split': randint(2, 11),    # try 2–10
        'min_samples_leaf': randint(1, 11),     # try 1–10
    }

    # 4) Randomized search with 3-fold stratified CV
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=config.RANDOM_STATE
    )
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,                  # number of sampled parameter settings
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        verbose=2
    )

    print("Starting hyperparameter search…")
    search.fit(X, y)

    # 5) Report & save best
    print("\n🔍 Best parameters found:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"Best CV AUC = {search.best_score_:.4f}")

    os.makedirs('models', exist_ok=True)
    joblib.dump(search.best_estimator_, 'models/rf_tuned.joblib')
    print("✅ Saved tuned model to models/rf_tuned.joblib\n")

if __name__ == '__main__':
    main()
