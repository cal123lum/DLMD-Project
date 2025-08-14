# src/data/metadata.py
from __future__ import annotations
import pandas as pd
from src.paths import BODMAS_META_CSV

def load_metadata() -> pd.DataFrame:
    """
    Returns columns: 'sha' (str), 'timestamp' (tz-aware), 'family' (str).
    Blank/NaN family -> 'Benign'.
    Row order preserved to align with X,y positions.
    """
    df = pd.read_csv(BODMAS_META_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    assert {"sha","timestamp","family"}.issubset(df.columns), "Expected sha,timestamp,family"
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    fam = df["family"].astype("string").str.strip()
    df["family"] = fam.fillna("Benign").mask(fam.eq(""), "Benign")
    return df.reset_index(drop=True)[["sha","timestamp","family"]]
