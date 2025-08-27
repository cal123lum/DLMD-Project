#!/usr/bin/env python3
import argparse, json, numpy as np, pandas as pd
from pathlib import Path

from src.holdouts import SplitIndices
from src.paths import ROOT

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-json", required=True,
                    help="data/processed/splits/family_lofo/holdout_<family>.json")
    ap.add_argument("--max-train-rows", type=int, default=None,
                    help="Cap #malware rows to speed GAN training (e.g., 20000).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stratify-by-family", action="store_true",
                    help="Sample malware across many families (recommended).")
    ap.add_argument("--out", required=True,
                    help="Where to write gan_subset_indices.json")
    args = ap.parse_args()

    split = SplitIndices.from_json(Path(args.split_json))
    # load labels + metadata
    y = np.load(ROOT / "data" / "raw" / "bodmas.npz", allow_pickle=True)["y"].astype(int)
    meta = pd.read_csv(ROOT / "data" / "raw" / "bodmas_metadata.csv").fillna("")
    fam_col = meta["family"].astype(str).str.strip().replace({"": "UNKNOWN"})

    train_idx = np.asarray(split.train, dtype=int)
    is_mal = (y[train_idx] == 1)
    mal_idx = train_idx[is_mal]

    if args.max_train_rows is not None and len(mal_idx) > args.max_train_rows:
        rng = np.random.default_rng(args.seed)
        if args.stratify_by_family:
            # partition by family; sample proportionally with a per-family ceiling
            fams = fam_col.iloc[mal_idx].values
            df = pd.DataFrame({"idx": mal_idx, "family": fams})
            grp = df.groupby("family", sort=False)
            # proportional allocation with min(ceil) to avoid one-family dominance
            per_fam = np.maximum(
                1,
                np.floor(args.max_train_rows * grp.size() / len(df)).astype(int)
            )
            take = []
            for fam, sub in grp:
                k = int(per_fam.loc[fam])
                k = max(1, min(k, len(sub)))
                take.append(sub.sample(n=k, random_state=int(rng.integers(0, 2**31-1))))
            df_take = pd.concat(take, ignore_index=True)
            # If rounding left us short/over, adjust by random add/drop
            if len(df_take) < args.max_train_rows:
                need = args.max_train_rows - len(df_take)
                rest = df.drop(df_take.index)
                if len(rest) > 0:
                    add = rest.sample(n=min(need, len(rest)),
                                      random_state=int(rng.integers(0, 2**31-1)))
                    df_take = pd.concat([df_take, add], ignore_index=True)
            elif len(df_take) > args.max_train_rows:
                df_take = df_take.sample(n=args.max_train_rows,
                                         random_state=int(rng.integers(0, 2**31-1)))
            mal_idx = df_take["idx"].to_numpy()
        else:
            mal_idx = rng.choice(mal_idx, size=int(args.max_train_rows), replace=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"train": list(map(int, mal_idx))}, indent=2))
    print(f"[ok] GAN subset → {out} (malware rows: {len(mal_idx)})")

if __name__ == "__main__":
    main()
