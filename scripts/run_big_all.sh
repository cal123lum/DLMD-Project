#!/usr/bin/env bash
set -euo pipefail

# ---- EDIT THESE ----
RUN_ID="big_$(date +%Y%m%d_%H%M)"
PY="python"

# EMBER paths
EMBER_NPZ="data/raw/ember.npz"
EMBER_META="data/raw/ember_metadata.csv"
EMBER_TIME_COL="timestamp"

# SOREL paths (150k)
SOREL_NPZ="data/raw/sorel_subset_150k.npz"
SOREL_META="data/raw/sorel_subset_150k_metadata.csv"
SOREL_TIME_COL="timestamp_utc"

# Cutoffs you chose (edit to yours)
EMBER_CUTS="2018-02-01,2018-03-01,2018-04-01,2018-05-01"
SOREL_CUTS="2018-02-01,2018-03-01,2018-04-01,2018-05-01"

# Fractions / seeds
FRACS="0.0005,0.001,0.002,0.003,0.005,0.01"
SEEDS="42,1337,2025"

# Methods (edit if you want)
METHODS_IID="real,oversample,smote,gan,evogan,evo"
METHODS_TEMP="real,oversample,smote,gan,evogan,evo"
METHODS_FAM="real,oversample,gan,evogan,evo"  # smote optional for family

# GAN training controls (match paper)
GAN_EPOCHS=80
GAN_MAX_MAL=20000
# --------------------

mkdir -p logs

run_block () {
  local name="$1"; shift
  local log="logs/${RUN_ID}_${name}.log"
  echo "==> ${name} -> ${log}"
  "$@" 2>&1 | tee -a "$log"
}

echo "RUN_ID=${RUN_ID}"

# ---------- EMBER ----------
run_block "EMBER_IID" \
  $PY -m scripts.experiments.iid.run_iid_scarcity_sweep \
    --dataset ember \
    --npz "$EMBER_NPZ" \
    --meta-csv "$EMBER_META" \
    --fractions "$FRACS" \
    --seeds "$SEEDS" \
    --methods "$METHODS_IID" \
    --epochs "$GAN_EPOCHS" \
    --max-gan-malware "$GAN_MAX_MAL" \

run_block "EMBER_TEMPORAL" \
  $PY -m scripts.experiments.holdouts.run_temporal_sweep \
    --dataset ember \
    --npz "$EMBER_NPZ" \
    --meta-csv "$EMBER_META" \
    --time-col "$EMBER_TIME_COL" \
    --cutoffs "$EMBER_CUTS" \
    --fractions "$FRACS" \
    --seeds-override "$SEEDS" \
    --methods "$METHODS_TEMP" \
    --epochs "$GAN_EPOCHS" \
    --max-gan-malware "$GAN_MAX_MAL" \
    --skip-existing

run_block "EMBER_FAMILY" \
  $PY -m scripts.experiments.holdouts.run_family_sweep \
    --dataset ember \
    --npz "$EMBER_NPZ" \
    --meta-csv "$EMBER_META" \
    --fractions "$FRACS" \
    --seeds-override "$SEEDS" \
    --methods "$METHODS_FAM" \
    --epochs "$GAN_EPOCHS" \
    --max-gan-malware "$GAN_MAX_MAL" \
    --skip-existing

# ---------- SOREL ----------
run_block "SOREL_IID" \
  $PY -m scripts.experiments.iid.run_iid_scarcity_sweep \
    --dataset sorel \
    --npz "$SOREL_NPZ" \
    --meta-csv "$SOREL_META" \
    --fractions "$FRACS" \
    --seeds "$SEEDS" \
    --methods "$METHODS_IID" \
    --epochs "$GAN_EPOCHS" \
    --max-gan-malware "$GAN_MAX_MAL" \

run_block "SOREL_TEMPORAL" \
  $PY -m scripts.experiments.holdouts.run_temporal_sweep \
    --dataset sorel \
    --npz "$SOREL_NPZ" \
    --meta-csv "$SOREL_META" \
    --time-col "$SOREL_TIME_COL" \
    --cutoffs "$SOREL_CUTS" \
    --fractions "$FRACS" \
    --seeds-override "$SEEDS" \
    --methods "$METHODS_TEMP" \
    --epochs "$GAN_EPOCHS" \
    --max-gan-malware "$GAN_MAX_MAL" \
    --skip-existing

run_block "SOREL_FAMILY_TAGLOFO" \
  $PY -m scripts.experiments.holdouts.run_family_sweep \
    --dataset sorel \
    --npz "$SOREL_NPZ" \
    --meta-csv "$SOREL_META" \
    --fractions "$FRACS" \
    --seeds-override "$SEEDS" \
    --methods "$METHODS_FAM" \
    --epochs "$GAN_EPOCHS" \
    --max-gan-malware "$GAN_MAX_MAL" \
    --skip-existing

echo "ALL DONE. Logs in ./logs/ (${RUN_ID}_*)"
