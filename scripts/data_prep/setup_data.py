# src/setup_data.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import config

def main():
    print("➡️ Loading NPZ file…")
    data = np.load(config.NPZ_PATH)
    print("    keys available:", data.files)

    # pull out features & labels directly
    X = data[config.FEATURE_KEY]   # shape (N, D)
    y = data[config.LABEL_KEY]     # shape (N,)

    print(f"    loaded X shape = {X.shape}, y shape = {y.shape}")

    # build a DataFrame
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    df['label'] = y
    print("➡️ DataFrame built, shape:", df.shape)

    # sample malware & benign
    mal = df[df['label'] == 1].sample(
        n=config.MALWARE_SAMPLES, random_state=config.RANDOM_STATE)
    ben = df[df['label'] == 0].sample(
        n=config.BENIGN_SAMPLES, random_state=config.RANDOM_STATE)

    subset = pd.concat([mal, ben]) \
               .sample(frac=1, random_state=config.RANDOM_STATE) \
               .reset_index(drop=True)
    subset.to_csv(config.SUBSET_PATH, index=False)
    print(f"➡️ Subset saved to {config.SUBSET_PATH}")

    # train/test split
    X_sub, y_sub = subset.drop('label', axis=1), subset['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X_sub, y_sub,
        test_size=config.TEST_SIZE,
        stratify=y_sub,
        random_state=config.RANDOM_STATE
    )
    pd.concat([X_train, y_train], axis=1) \
      .to_csv(config.TRAIN_PATH, index=False)
    pd.concat([X_test,  y_test],  axis=1) \
      .to_csv(config.TEST_PATH,  index=False)
    print(f"➡️ Train/test saved to {config.TRAIN_PATH}, {config.TEST_PATH}")

    print("✅ All done!")

if __name__ == '__main__':
    main()
