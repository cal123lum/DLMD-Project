#!/usr/bin/env bash
set -euo pipefail

ROOT="$(pwd)"

# Fresh dirs
rm -rf tmp/plotcsv
mkdir -p \
  tmp/plotcsv/iid \
  tmp/plotcsv/temporal_bodmas tmp/plotcsv/temporal_ember tmp/plotcsv/temporal_sorel \
  tmp/plotcsv/family_bodmas  tmp/plotcsv/family_ember  tmp/plotcsv/family_sorel \
  tmp/plotcsv/temporal_heatmap

# IID (note: BODMAS = "final" in your folder naming)
ln -sf "$ROOT/data/processed/metrics/iid_final/raw.csv"  "$ROOT/tmp/plotcsv/iid/iid_bodmas.csv"
ln -sf "$ROOT/data/processed/metrics/iid_ember/raw.csv"  "$ROOT/tmp/plotcsv/iid/iid_ember.csv"
ln -sf "$ROOT/data/processed/metrics/iid_sorel/raw.csv"  "$ROOT/tmp/plotcsv/iid/iid_sorel.csv"

# TEMPORAL: per-cutoff CSVs (each becomes one "prefix" in plot_results)
for d in "$ROOT"/data/processed/metrics/temporal_final/*/ ; do
  b="$(basename "$d")"
  ln -sf "$d/raw.csv" "$ROOT/tmp/plotcsv/temporal_bodmas/${b}.csv"
done
for d in "$ROOT"/data/processed/metrics/temporal_ember/*/ ; do
  b="$(basename "$d")"
  ln -sf "$d/raw.csv" "$ROOT/tmp/plotcsv/temporal_ember/${b}.csv"
done
for d in "$ROOT"/data/processed/metrics/temporal_sorel/*/ ; do
  b="$(basename "$d")"
  ln -sf "$d/raw.csv" "$ROOT/tmp/plotcsv/temporal_sorel/${b}.csv"
done

# FAMILY LOFO: per-family CSVs
for d in "$ROOT"/data/processed/metrics/family_final/*/ ; do
  b="$(basename "$d")"
  ln -sf "$d/raw.csv" "$ROOT/tmp/plotcsv/family_bodmas/${b}.csv"
done
for d in "$ROOT"/data/processed/metrics/family_ember/*/ ; do
  b="$(basename "$d")"
  ln -sf "$d/raw.csv" "$ROOT/tmp/plotcsv/family_ember/${b}.csv"
done
for d in "$ROOT"/data/processed/metrics/family_sorel/*/ ; do
  b="$(basename "$d")"
  ln -sf "$d/raw.csv" "$ROOT/tmp/plotcsv/family_sorel/${b}.csv"
done

# Sanity: show broken links (should print nothing)
find "$ROOT/tmp/plotcsv" -type l ! -exec test -e {} \; -print

# Build "heatmap views" for temporal:
# plot_results heatmap needs test_start; your temporal CSVs have cutoff.
# So we concatenate all cutoffs and set test_start=cutoff.
python - <<'PY'
import glob, os
import pandas as pd

root = os.getcwd()

def make_heatmap_csv(name, pattern, out_path):
    paths = sorted(glob.glob(pattern))
    if not paths:
        print(f"[warn] no files matched: {pattern}")
        return
    frames = []
    for p in paths:
        df = pd.read_csv(p)
        # normalize headers similarly to plot_results
        df.columns = [c.strip().lower() for c in df.columns]
        if "test_start" not in df.columns:
            if "cutoff" in df.columns:
                df["test_start"] = df["cutoff"]
            else:
                # nothing we can do
                continue
        frames.append(df)
    if not frames:
        print(f"[warn] could not build heatmap CSV for {name}")
        return
    out = pd.concat(frames, ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"[ok] wrote {out_path} ({len(out)} rows)")

make_heatmap_csv(
    "bodmas",
    os.path.join(root, "tmp/plotcsv/temporal_bodmas/*.csv"),
    os.path.join(root, "tmp/plotcsv/temporal_heatmap/temporal_bodmas_all.csv"),
)
make_heatmap_csv(
    "ember",
    os.path.join(root, "tmp/plotcsv/temporal_ember/*.csv"),
    os.path.join(root, "tmp/plotcsv/temporal_heatmap/temporal_ember_all.csv"),
)
make_heatmap_csv(
    "sorel",
    os.path.join(root, "tmp/plotcsv/temporal_sorel/*.csv"),
    os.path.join(root, "tmp/plotcsv/temporal_heatmap/temporal_sorel_all.csv"),
)
PY

echo "[ok] plot inputs ready under tmp/plotcsv/"
