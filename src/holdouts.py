# src/holdouts.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json, numpy as np, pandas as pd, hashlib
from typing import List, Dict

@dataclass
class SplitIndices:
    train: List[int]
    test:  List[int]
    def to_json(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "train": sorted(set(map(int, self.train))),
            "test":  sorted(set(map(int, self.test))),
        }
        path.write_text(json.dumps(payload))
    @staticmethod
    def from_json(path: Path) -> "SplitIndices":
        d = json.loads(Path(path).read_text())
        return SplitIndices(train=list(map(int, d["train"])),
                            test=list(map(int, d["test"])))

def temporal_holdout(timestamps: pd.Series, cutoff: str) -> SplitIndices:
    # <= cutoff -> train; > cutoff -> test; NaT -> train (conservative)
    ts = pd.to_datetime(timestamps, utc=True, errors="coerce")
    c  = pd.to_datetime(cutoff, utc=True)
    idx = np.arange(len(ts))
    mask_train = ts.isna() | (ts <= c)
    mask_test  = ts > c
    return SplitIndices(train=idx[mask_train].tolist(),
                        test=idx[mask_test].tolist())

def _stable_hash_bucket(strings: pd.Series, modulo: int = 100) -> np.ndarray:
    # Deterministic bucket per string using md5 (fast, stable)
    return np.fromiter(
        (int(hashlib.md5(s.encode("utf-8")).hexdigest()[:8], 16) % modulo for s in strings),
        dtype=np.int64,
        count=len(strings)
    )

def family_holdout_with_benign(
    families: pd.Series,
    shas: pd.Series,
    heldout_families: List[str],
    *,
    benign_label: str = "Benign",
    benign_test_frac: float = 0.20,   # 20% of benign go to test, deterministically
) -> SplitIndices:
    """
    Test = all malware from held-out families  +  a deterministic subset of benign.
    Train = everything else. (No malware from held-out families leaks into train.)
    """
    fam = families.astype(str).str.strip()
    idx = np.arange(len(fam))

    # Guard: don't hold out Benign as a 'family'
    if any(benign_label.lower() == f.lower().strip() for f in heldout_families):
        raise ValueError(f"'{benign_label}' is the benign label; do not pass it as a held-out family.")

    is_benign = fam.eq(benign_label)

    # Positive test samples: malware of held-out families
    held = {f.strip() for f in heldout_families}
    test_pos_mask = fam.isin(held) & ~is_benign

    # Negative test samples: deterministic slice of benign
    benign_idx = idx[is_benign]
    if len(benign_idx) == 0:
        raise ValueError("No benign rows detected (family=='Benign'); cannot build a valid test set.")
    benign_sha = shas.iloc[benign_idx].astype(str)
    buckets = _stable_hash_bucket(benign_sha, modulo=100)
    benign_test_mask_local = buckets < int(round(benign_test_frac * 100))
    benign_test_idx = benign_idx[benign_test_mask_local]

    # Compose final sets
    test_idx = np.sort(np.unique(np.concatenate([idx[test_pos_mask], benign_test_idx])))
    train_mask = np.ones(len(fam), dtype=bool)
    train_mask[test_idx] = False
    train_idx = idx[train_mask]

    # Sanity: disjoint & non-empty both sides
    assert set(train_idx).isdisjoint(test_idx)
    if len(test_idx) == 0:
        raise ValueError("Empty test set: check heldout families / benign_test_frac.")

    return SplitIndices(train=train_idx.tolist(), test=test_idx.tolist())

def describe_split(split: SplitIndices, n_total: int) -> Dict[str, float]:
    return {
        "n_total": n_total,
        "n_train": len(split.train),
        "n_test":  len(split.test),
        "pct_test": round(100.0 * len(split.test) / max(1, n_total), 3),
    }
