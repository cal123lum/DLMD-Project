#!/usr/bin/env python3
"""
Create a SOREL-20M subset MANIFEST (metadata only) from meta.db.

Outputs a CSV with:
  rowid, sha256, is_malware, rl_fs_t, timestamp_utc, group

You still need ember_features later to extract X and build an NPZ.
"""

import argparse
import sqlite3
from pathlib import Path
import pandas as pd

TAG_COLS = [
    "adware", "flooder", "ransomware", "dropper", "spyware", "packed",
    "crypto_miner", "file_infector", "installer", "worm", "downloader",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="data/raw/sorel_20m/meta.db")
    ap.add_argument("--out", type=str, default="data/raw/sorel_20m/subsets/sorel_subset_manifest.csv")

    ap.add_argument("--n", type=int, default=100_000, help="total rows in manifest (mal+ben)")
    ap.add_argument("--mal-frac", type=float, default=0.5, help="fraction malware in manifest")
    ap.add_argument("--seed", type=int, default=42)

    # time-binning to spread across time (recommended)
    ap.add_argument("--time-bins", type=int, default=12, help="number of equal-count time bins")
    ap.add_argument("--tmin", type=float, default=None, help="min rl_fs_t (epoch seconds) filter")
    ap.add_argument("--tmax", type=float, default=None, help="max rl_fs_t (epoch seconds) filter")

    # group definition
    ap.add_argument("--group", choices=["none", "tags"], default="tags",
                    help="create a group column from tag columns (for LOFO-style splits later)")

    args = ap.parse_args()

    db = Path(args.db)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(str(db))
    cur = con.cursor()

    # confirm columns exist
    cols = [r[1] for r in cur.execute("PRAGMA table_info(meta)").fetchall()]
    for c in ["sha256", "is_malware", "rl_fs_t"]:
        if c not in cols:
            raise RuntimeError(f"meta.db missing column: {c}")

    have_tags = all(c in cols for c in TAG_COLS)

    # build WHERE filters
    where = ["rl_fs_t IS NOT NULL"]
    params = []

    if args.tmin is not None:
        where.append("rl_fs_t >= ?")
        params.append(float(args.tmin))
    if args.tmax is not None:
        where.append("rl_fs_t <= ?")
        params.append(float(args.tmax))

    where_sql = " AND ".join(where)

    # counts
    n_mal = int(args.n * args.mal_frac)
    n_ben = args.n - n_mal

    # choose query fields
    select_fields = ["rowid", "sha256", "is_malware", "rl_fs_t"]
    if args.group == "tags" and have_tags:
        select_fields += TAG_COLS

    sel = ", ".join(select_fields)

    # If time-bins > 1: we sample per bin to spread across time.
    # We approximate equal-count bins via quantiles on rl_fs_t using SQL order + limits.
    # Strategy:
    #  1) get bin boundaries by pulling a small sample of rl_fs_t ordered (fast enough)
    #  2) for each bin and each class, sample k rows ORDER BY RANDOM() with seed-like randomness
    #
    # Note: SQLite RANDOM() isn't seedable; reproducibility comes from deterministic binning
    # + stable post-shuffle with pandas using the provided seed.

    # Get approximate boundaries using percentiles on ordered timestamps
    # We do this by selecting rl_fs_t in order with LIMIT/OFFSET steps.
    # On 20M rows this can take a bit, but with 12 bins it’s reasonable.
    bins = args.time_bins
    if bins < 1:
        bins = 1

    # Get total eligible rows
    total = cur.execute(f"SELECT COUNT(*) FROM meta WHERE {where_sql}", params).fetchone()[0]
    total = int(total)
    if total == 0:
        raise RuntimeError("No rows match your time filters.")

    # Compute boundaries
    boundaries = []
    if bins == 1:
        mn, mx = cur.execute(
            f"SELECT MIN(rl_fs_t), MAX(rl_fs_t) FROM meta WHERE {where_sql}", params
        ).fetchone()
        boundaries = [float(mn), float(mx)]
    else:
        # positions for quantiles
        qs = [int(total * i / bins) for i in range(bins + 1)]
        # fetch timestamps at those offsets
        # We use a single ordered scan per offset (SQLite will do work; still acceptable for small bins)
        ts_vals = []
        for off in qs:
            r = cur.execute(
                f"SELECT rl_fs_t FROM meta WHERE {where_sql} ORDER BY rl_fs_t LIMIT 1 OFFSET ?",
                params + [off]
            ).fetchone()
            ts_vals.append(float(r[0]) if r else None)
        # clean up potential Nones and enforce monotonic
        ts_vals = [t for t in ts_vals if t is not None]
        boundaries = ts_vals
        if len(boundaries) < 2:
            mn, mx = cur.execute(
                f"SELECT MIN(rl_fs_t), MAX(rl_fs_t) FROM meta WHERE {where_sql}", params
            ).fetchone()
            boundaries = [float(mn), float(mx)]

    # per-bin allocations (roughly equal)
    per_bin_mal = [n_mal // bins] * bins
    per_bin_ben = [n_ben // bins] * bins
    # distribute remainder
    for i in range(n_mal - sum(per_bin_mal)):
        per_bin_mal[i % bins] += 1
    for i in range(n_ben - sum(per_bin_ben)):
        per_bin_ben[i % bins] += 1

    rows_out = []

    def compute_group(row):
        if args.group == "none":
            return ""
        if args.group == "tags" and have_tags:
            # choose one "primary" tag if any, else benign/malware generic
            for c in TAG_COLS:
                if int(row.get(c, 0)) == 1:
                    return c
            return "untagged"
        return ""

    # sample per bin
    for b in range(bins):
        t0 = boundaries[b]
        t1 = boundaries[b + 1] if b + 1 < len(boundaries) else boundaries[-1]
        # make the last bin inclusive
        if b == bins - 1:
            bin_where = f"{where_sql} AND rl_fs_t >= ? AND rl_fs_t <= ?"
        else:
            bin_where = f"{where_sql} AND rl_fs_t >= ? AND rl_fs_t < ?"

        bin_params = params + [t0, t1]

        km = per_bin_mal[b]
        kb = per_bin_ben[b]

        if km > 0:
            q = f"SELECT {sel} FROM meta WHERE {bin_where} AND is_malware=1 ORDER BY RANDOM() LIMIT {km}"
            rows_out += cur.execute(q, bin_params).fetchall()

        if kb > 0:
            q = f"SELECT {sel} FROM meta WHERE {bin_where} AND is_malware=0 ORDER BY RANDOM() LIMIT {kb}"
            rows_out += cur.execute(q, bin_params).fetchall()

    con.close()

    # build dataframe
    df = pd.DataFrame(rows_out, columns=select_fields)
    # stable shuffle for reproducibility
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # timestamp
    df["timestamp_utc"] = pd.to_datetime(df["rl_fs_t"], unit="s", utc=True, errors="coerce")

    # group
    if args.group == "tags":
        if have_tags:
            df["group"] = df.apply(lambda r: compute_group(r), axis=1)
        else:
            df["group"] = ""

    # keep minimal columns for manifest (plus group)
    keep = ["rowid", "sha256", "is_malware", "rl_fs_t", "timestamp_utc"]
    if "group" in df.columns:
        keep.append("group")
    df_out = df[keep]

    # quick report
    print("[ok] sampled rows:", len(df_out))
    print("  malware:", int((df_out["is_malware"] == 1).sum()),
          "benign:", int((df_out["is_malware"] == 0).sum()))
    print("  tmin:", df_out["timestamp_utc"].min(), "tmax:", df_out["timestamp_utc"].max())
    if "group" in df_out.columns:
        print("  top groups:")
        print(df_out["group"].value_counts().head(10))

    df_out.to_csv(out, index=False)
    print("[ok] wrote", out)

if __name__ == "__main__":
    main()
