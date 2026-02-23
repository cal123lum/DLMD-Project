#!/usr/bin/env python3
# scripts/experiments/holdouts/make_family_gan_subset.py

import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd

from src.holdouts import SplitIndices
from src.paths import ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--split-json",
        required=True,
        help="path to family holdout split json (holdout_<family>.json)",
    )
    ap.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="cap number of malware rows to speed GAN training, e.g., 20000",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--stratify-by-family",
        action="store_true",
        help="sample malware across many families (within the training split)",
    )
    ap.add_argument(
        "--stratify-col",
        type=str,
        default="family",
        help="column in meta-csv to stratify over when --stratify-by-family is set "
             "(e.g., 'family' for BODMAS/EMBER, 'group' for SOREL tag-LOFO)",
    )
    ap.add_argument("--out", required=True, help="where to write gan_subset_indices.json")

    # IMPORTANT: dataset paths
    ap.add_argument("--npz", type=str, default=str(ROOT / "data" / "raw" / "bodmas.npz"))
    ap.add_argument("--meta-csv", type=str, default=str(ROOT / "data" / "raw" / "bodmas_metadata.csv"))

    args = ap.parse_args()

    split = SplitIndices.from_json(Path(args.split_json))

    # load labels
    z = np.load(args.npz, allow_pickle=True)
    y = z["y"].astype(int)

    train_idx = np.asarray(split.train, dtype=int)
    if train_idx.max(initial=-1) >= len(y):
        raise ValueError(
            f"split indices exceed dataset length: max(train)={train_idx.max()} vs n={len(y)} "
            f"(did you mix split-json from a different dataset?)"
        )

    mal_idx = train_idx[y[train_idx] == 1]

    # optional cap
    if args.max_train_rows is not None and len(mal_idx) > args.max_train_rows:
        rng = np.random.default_rng(args.seed)

        if args.stratify_by_family:
            # Only load meta if we actually need stratification
            meta = pd.read_csv(args.meta_csv).fillna("")
            col = (args.stratify_col or "family").strip()
            if col not in meta.columns:
                raise ValueError(f"meta-csv has no '{col}' column: {args.meta_csv}")

            label_col = meta[col].astype(str).str.strip().replace({"": "UNKNOWN"})

            labels = label_col.iloc[mal_idx].values
            df = pd.DataFrame({"idx": mal_idx, "label": labels})
            grp = df.groupby("label", sort=False)

            # proportional allocation with floor 1 each
            sizes = grp.size()
            alloc = np.maximum(
                1,
                np.floor(args.max_train_rows * (sizes / sizes.sum())).astype(int),
            )

            take = []
            for label, sub in grp:
                k = int(alloc.loc[label])
                k = max(1, min(k, len(sub)))
                take.append(sub.sample(n=k, random_state=int(rng.integers(0, 2**31 - 1))))

            df_take = pd.concat(take, ignore_index=True)

            # adjust for rounding to hit cap
            if len(df_take) < args.max_train_rows:
                need = args.max_train_rows - len(df_take)

                # easiest way to get remainder is boolean mask by idx
                chosen = set(df_take["idx"].tolist())
                rest = df[~df["idx"].isin(chosen)]

                if len(rest) > 0:
                    add = rest.sample(
                        n=min(need, len(rest)),
                        random_state=int(rng.integers(0, 2**31 - 1)),
                    )
                    df_take = pd.concat([df_take, add], ignore_index=True)
            elif len(df_take) > args.max_train_rows:
                df_take = df_take.sample(
                    n=args.max_train_rows,
                    random_state=int(rng.integers(0, 2**31 - 1)),
                )

            mal_idx = df_take["idx"].to_numpy(dtype=int)
        else:
            mal_idx = rng.choice(mal_idx, size=int(args.max_train_rows), replace=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"train": list(map(int, mal_idx))}, indent=2))
    print(f"[ok] GAN subset → {out} (malware rows: {len(mal_idx)})")


if __name__ == "__main__":
    main()
