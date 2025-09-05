#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Always activate venv
source venv/bin/activate
export PYTHONPATH="$PWD:$PYTHONPATH"

# ---------- 1) IID FINAL ----------
venv/bin/python -m scripts.experiments.iid.run_iid_scarcity_sweep \
  --fractions "0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.0045,0.005,0.0055,0.006,0.0065,0.007,0.0075,0.008,0.0085,0.009,0.0095,0.01" \
  --seeds "42,1337,2025" \
  --const-train-size 20000 \
  --min-train-pos 50 --min-train-neg 50 \
  --epochs 80 --max-gan-malware 50000 --batch-size 128 --n-critic 4 --device auto \
  --rf-n-est 400 --rf-max-depth 20 --rf-class-weight none --val-threshold balacc

# ---------- 2) TEMPORAL FINAL ----------
# Assumes you edited run_temporal_sweep.sh as above (final dirs + GAN args + compare)
bash scripts/experiments/holdouts/run_temporal_sweep.sh

# ---------- 3) FAMILY FINAL ----------
venv/bin/python -m scripts.experiments.holdouts.run_family_sweep \
  --epochs 80 --max-gan-malware 50000 --batch-size 128 --n-critic 4 --device auto

echo "All FINAL runs completed."
