#!/usr/bin/env python3
# scripts/experiments/holdouts/make_holdouts.py
# Holdout split maker for temporal and family regimes
# Author: Callum Musselwhite
# Last edit: 2025-09-17
# Purpose: write JSON split files for temporal cutoff and/or selected family holdouts

import argparse
from src.paths import TEMPORAL_SPLIT, FAMILY_SPLIT
from src.holdouts import temporal_holdout, family_holdout_with_benign, describe_split
from src.data.metadata import load_metadata


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temporal-cutoff", type=str, help="e.g. 2016-01-01")
    ap.add_argument("--family", action="append", default=[],
                    help="repeatable flag, e.g., --family Emotet --family TrickBot")
    ap.add_argument("--benign-test-frac", type=float, default=0.20,
                    help="fraction of benign sent to test for family holdout")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = load_metadata()

    did = False

    # temporal split
    if args.temporal_cutoff:
        t = temporal_holdout(meta["timestamp"], args.temporal_cutoff)
        print(f"[temporal] cutoff={args.temporal_cutoff} -> {describe_split(t, len(meta))}")
        if not args.dry_run:
            t.to_json(TEMPORAL_SPLIT)
            print(f"[temporal] wrote {TEMPORAL_SPLIT}")
        did = True

    # family holdout with benign hashing
    if args.family:
        f = family_holdout_with_benign(meta["family"], meta["sha"], args.family,
                                       benign_test_frac=args.benign_test_frac)
        print(f"[family] held_out={args.family} benign_test_frac={args.benign_test_frac} -> {describe_split(f, len(meta))}")
        if not args.dry_run:
            f.to_json(FAMILY_SPLIT)
            print(f"[family] wrote {FAMILY_SPLIT}")
        did = True

    if not did:
        print("Nothing to do. Pass --temporal-cutoff and/or --family ...")


if __name__ == "__main__":
    main()
