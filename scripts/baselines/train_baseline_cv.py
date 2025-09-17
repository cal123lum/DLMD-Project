#!/usr/bin/env python3
# scripts/baseline/train_baseline_cv.py
# Baseline RandomForest cross-validated training
# Author: Callum Musselwhite
# Last edit: 2025-09-17
# Purpose: run 5-fold stratified CV on BODMAS features and save final model

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from src.paths import BASELINE_RF_CV_JOB
from src import config


def main():
    """Baseline RF with 5-fold CV; reports mean/std of accuracy, F1, and AUC"""

    # Load dataset
    data = np.load(config.NPZ_PATH)
    X, y = data["X"], data["y"]
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)

    # Classifier and metrics
    clf = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE, n_jobs=-1)
    scoring = ["accuracy", "f1", "roc_auc"]

    # Cross-validate
    print("Running 5-fold CV...")
    results = cross_validate(clf, X, y, cv=skf, scoring=scoring, return_estimator=False, n_jobs=-1)

    # Report summary
    for metric in scoring:
        scores = results[f"test_{metric}"]
        print(f"{metric:>8}  mean = {scores.mean():.4f},  std = {scores.std():.4f}")

    # Fit on all data and save
    clf.fit(X, y)
    joblib.dump(clf, str(BASELINE_RF_CV_JOB))
    print(f"Saved final RF to {BASELINE_RF_CV_JOB}")


if __name__ == "__main__":
    main()
