#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-csv", required=True)
    ap.add_argument("--npz", required=True, help="NPZ containing y (aligned to meta-csv rows)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--exclude", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--benign-test-frac", type=float, default=0.2)
    ap.add_argument("--min-test-mal", type=int, default=50)
    ap.add_argument("--min-test-ben", type=int, default=50, help="ensure enough benign in test")
    args = ap.parse_args()

    meta = pd.read_csv(args.meta_csv).fillna("")
    if "group" not in meta.columns:
        raise ValueError("meta-csv missing 'group' column (expected SOREL tag group)")

    z = np.load(args.npz, allow_pickle=True)
    y = z["y"].astype(int)
    if len(y) != len(meta):
        raise ValueError(f"npz/meta length mismatch: len(y)={len(y)} len(meta)={len(meta)}")

    group = meta["group"].astype(str).str.strip().replace({"": "untagged"}).to_numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    exclude = {s.strip() for s in (args.exclude or "").split(",") if s.strip()}
    tags = sorted(set(group) - exclude)

    idx_all = np.arange(len(meta), dtype=int)
    ben_all = idx_all[y == 0]
    mal_all = idx_all[y == 1]

    if len(ben_all) == 0 or len(mal_all) == 0:
        raise ValueError(f"degenerate dataset: benign={len(ben_all)} malware={len(mal_all)}")

    # deterministic benign split used for ALL tag holdouts
    n_ben_test = int(round(args.benign_test_frac * len(ben_all)))
    n_ben_test = max(args.min_test_ben, n_ben_test)
    n_ben_test = min(n_ben_test, len(ben_all))
    ben_test = rng.choice(ben_all, size=n_ben_test, replace=False).astype(int)

    for tag in tags:
        mal_tag = idx_all[(y == 1) & (group == tag)]
        if len(mal_tag) < args.min_test_mal:
            continue

        test = np.unique(np.concatenate([mal_tag, ben_test])).astype(int)
        test_set = set(map(int, test))
        train = np.array([i for i in idx_all if int(i) not in test_set], dtype=int)

        # sanity: both classes in train/test
        if len(set(y[test])) < 2 or len(set(y[train])) < 2:
            continue

        out = out_dir / f"holdout_{tag}.json"
        out.write_text(json.dumps({"train": train.tolist(), "test": test.tolist()}, indent=2))

        # useful debug counts
        n_test_mal = int((y[test] == 1).sum())
        n_test_ben = int((y[test] == 0).sum())
        print(f"[ok] {tag:12s} train={len(train)} test={len(test)} (mal={n_test_mal} ben={n_test_ben}) -> {out}")

    print(f"[done] wrote splits to {out_dir}")

if __name__ == "__main__":
    main()
