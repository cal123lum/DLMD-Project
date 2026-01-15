#!/usr/bin/env python3
# scripts/experiments/holdouts/make_holdouts.py

import argparse
from pathlib import Path
import pandas as pd

from src.paths import ROOT, TEMPORAL_SPLIT, FAMILY_SPLIT
from src.holdouts import temporal_holdout, family_holdout_with_benign, describe_split


def load_meta(meta_path: str) -> pd.DataFrame:
    p = Path(meta_path)
    if p.suffix.lower() in (".parquet", ".pq"):
        meta = pd.read_parquet(p)
    else:
        meta = pd.read_csv(p)
    meta = meta.fillna("")
    if "timestamp" in meta.columns:
        meta["timestamp"] = pd.to_datetime(meta["timestamp"], utc=True, errors="coerce")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal-cutoff", type=str, help="e.g. 2016-01-01")
    ap.add_argument("--family", action="append", default=[],
                    help="repeatable flag, e.g., --family Emotet --family TrickBot")
    ap.add_argument("--benign-test-frac", type=float, default=0.20,
                    help="fraction of benign sent to test for family holdout")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument("--dataset", type=str, default="bodmas")
    ap.add_argument("--meta-csv", type=str, default=str(ROOT / "data" / "raw" / "bodmas_metadata.csv"))

    # allow the runner to control where splits get written
    ap.add_argument("--out-temporal", type=str, default=None)
    ap.add_argument("--out-family", type=str, default=None)

    args = ap.parse_args()

    meta = load_meta(args.meta_csv)

    # pick default output locations if not provided
    out_temporal = Path(args.out_temporal) if args.out_temporal else Path(TEMPORAL_SPLIT)
    out_family = Path(args.out_family) if args.out_family else Path(FAMILY_SPLIT)

    did = False

    if args.temporal_cutoff:
        assert "timestamp" in meta.columns, "meta must have a 'timestamp' column for temporal splits"
        t = temporal_holdout(meta["timestamp"], args.temporal_cutoff)
        print(f"[temporal] cutoff={args.temporal_cutoff} -> {describe_split(t, len(meta))}")
        if not args.dry_run:
            out_temporal.parent.mkdir(parents=True, exist_ok=True)
            t.to_json(out_temporal)
            print(f"[temporal] wrote {out_temporal}")
        did = True

    if args.family:
        assert "family" in meta.columns, "meta must have a 'family' column for family splits"
        assert "sha" in meta.columns, "meta must have a 'sha' column for family splits"
        f = family_holdout_with_benign(
            meta["family"], meta["sha"], args.family,
            benign_test_frac=args.benign_test_frac
        )
        print(f"[family] held_out={args.family} benign_test_frac={args.benign_test_frac} -> {describe_split(f, len(meta))}")
        if not args.dry_run:
            out_family.parent.mkdir(parents=True, exist_ok=True)
            f.to_json(out_family)
            print(f"[family] wrote {out_family}")
        did = True

    if not did:
        print("Nothing to do. Pass --temporal-cutoff and/or --family ...")


if __name__ == "__main__":
    main()
