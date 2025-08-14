import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score, roc_auc_score
from src.paths import BASELINE_RF_CV_JOB
from src import config as config 
import os, sys
from pathlib import Path 


def main():
    # --- 1) Load the full dataset ---
    data = np.load(config.NPZ_PATH)
    X, y = data['X'], data['y']
    print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features")

    # --- 2) Set up 5-fold stratified CV ---
    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=config.RANDOM_STATE
    )

    # --- 3) Define classifier & scoring ---
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    scoring = ['accuracy', 'f1', 'roc_auc']

    # --- 4) Run CV ---
    print("Running 5-fold CV…")
    results = cross_validate(
        clf,
        X, y,
        cv=skf,
        scoring=scoring,
        return_estimator=False,
        n_jobs=-1
    )

    # --- 5) Summarize ---
    for metric in scoring:
        scores = results[f'test_{metric}']
        print(f"{metric:>8}  mean = {scores.mean():.4f},  std = {scores.std():.4f}")

    # --- (Optional) Train final model on all data and save it ---
    clf.fit(X, y)
    joblib.dump(clf, str(BASELINE_RF_CV_JOB))
    print(f"Saved final RF to {BASELINE_RF_CV_JOB}")

if __name__ == '__main__':
    main()
