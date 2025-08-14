# src/train_baseline.py

import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import config

def main():
    # 1) Load train/test
    train = pd.read_csv(config.TRAIN_PATH)
    test  = pd.read_csv(config.TEST_PATH)

    X_train, y_train = train.drop('label', axis=1), train['label']
    X_test,  y_test  = test.drop('label',  axis=1), test['label']

    # 2) Train Random Forest
    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    print(f"Training RF on {X_train.shape[0]} samples with {X_train.shape[1]} features…")
    clf.fit(X_train, y_train)

    # 3) Evaluate
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred, digits=4))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")

    # 4) Save model
    os.makedirs('models', exist_ok=True)
    model_path = 'models/baseline_rf.joblib'
    joblib.dump(clf, model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == '__main__':
    main()
