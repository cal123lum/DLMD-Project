#!/usr/bin/env python3
# scripts/baselines/hyperparameter_tuning.py
# Hyperparameter tuning for RandomForest on BODMAS
# Author: Callum Musselwhite
# Last Edit: 2025-09-17
# Purpose: randomized hyperparameter search using stratified CV and AUC scoring

import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from scipy.stats import randint
import config


def main():
    """Randomized hyperparameter search for a RandomForest on BODMAS features"""

    # Load feature matrix and labels from the configured NPZ
    data = np.load(config.NPZ_PATH)
    X, y = data["X"], data["y"]
    print(f"Loaded {X.shape[0]} samples x {X.shape[1]} features")

    # Optional speed-up: tune on a 20% stratified subset
    # Comment this block out if you want to use the full dataset
    X, _, y, _ = train_test_split(
        X,
        y,
        train_size=0.2,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    print(f"Subsampled to {X.shape[0]} for hyperparameter search")

    # Base estimator and search space
    rf = RandomForestClassifier(
        oob_score=True,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    param_dist = {
        "n_estimators": randint(50, 200),
        "max_depth": [None] + list(range(10, 101, 10)),
        "min_samples_split": randint(2, 11),
        "min_samples_leaf": randint(1, 11),
    }

    # Randomized search with stratified 3-fold CV; AUC is the target metric
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=param_dist,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        verbose=2,
    )

    print("Starting hyperparameter search...")
    search.fit(X, y)

    # Report and persist the best model
    print("\nBest parameters found:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    print(f"Best CV AUC = {search.best_score_:.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(search.best_estimator_, "models/rf_tuned.joblib")
    print("Saved tuned model to models/rf_tuned.joblib\n")


if __name__ == "__main__":
    main()
