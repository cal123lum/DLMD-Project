# verify_data.py
import pandas as pd
from pathlib import Path
from src.paths import DATA_PROCESSED
import os, sys


PATHS = {
    'Subset': str(DATA_PROCESSED / 'bodmas_subset.csv'),
    'Train' : str(DATA_PROCESSED / 'train.csv'),
    'Test'  : str(DATA_PROCESSED / 'test.csv'),
}

for name, path in PATHS.items():
    df = pd.read_csv(path)
    rows, cols = df.shape
    print(f"\n{name} ({path}):")
    print(f"  Rows = {rows}, Columns = {cols}")
    print(f"  Label counts:\n{df['label'].value_counts()}")
    print(f"  Missing values? {df.isnull().values.any()}")
