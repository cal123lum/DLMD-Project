#!/usr/bin/env python3
import json
from pathlib import Path
import pandas as pd

EMBER_DIR = Path("data/raw/ember2018")
OUT_CSV = Path("data/raw/ember_metadata.csv")

def iter_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def norm_ts(appeared: str):
    # appeared like "2018-11" -> UTC timestamp at month start
    if not appeared:
        return pd.NaT
    if len(appeared) == 7:
        appeared = appeared + "-01"
    return pd.to_datetime(appeared, utc=True, errors="coerce")

def main():
    train_parts = sorted(EMBER_DIR.glob("train_features_*.jsonl"), key=lambda p: p.name)
    test_path = EMBER_DIR / "test_features.jsonl"
    assert train_parts and test_path.exists(), "Missing train_features_*.jsonl or test_features.jsonl"

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    with OUT_CSV.open("w", encoding="utf-8") as out:
        out.write("sha,timestamp,family\n")

        # TRAIN: drop unlabeled label==-1 to match your build_ember_npz.py filter
        for part in train_parts:
            for obj in iter_jsonl(part):
                y = int(obj.get("label", -1))
                if y == -1:
                    continue
                sha = str(obj.get("sha256", ""))
                ts = norm_ts(str(obj.get("appeared", "")))
                fam = "Benign" if y == 0 else str(obj.get("avclass", "")).strip() or "UNKNOWN"
                out.write(f"{sha},{ts.isoformat() if pd.notna(ts) else ''},{fam}\n")
                rows_written += 1

        # TEST: always labeled
        for obj in iter_jsonl(test_path):
            y = int(obj.get("label", -1))
            if y == -1:
                continue
            sha = str(obj.get("sha256", ""))
            ts = norm_ts(str(obj.get("appeared", "")))
            fam = "Benign" if y == 0 else str(obj.get("avclass", "")).strip() or "UNKNOWN"
            out.write(f"{sha},{ts.isoformat() if pd.notna(ts) else ''},{fam}\n")
            rows_written += 1

    print("[ok] wrote", OUT_CSV, "rows=", rows_written)

if __name__ == "__main__":
    main()
