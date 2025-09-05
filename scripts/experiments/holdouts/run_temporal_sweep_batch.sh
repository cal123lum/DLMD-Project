#!/usr/bin/env bash
set -euo pipefail

# --- repo root & env ---
ROOT="${ROOT:-$PWD}"
cd "$ROOT"

# Activate venv if present
if [[ -d "venv" ]]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

# --- knobs (can be overridden via env) ---
CONST="${CONST:-20000}"   # total train size for +GAN
DO_CONTEXT="${DO_CONTEXT:-0}"
FRACS_SCARCITY="${FRACS_SCARCITY:-0.0005,0.001,0.0015,0.002,0.0025,0.003,0.0035,0.004,0.0045,0.005,0.0055,0.006,0.0065,0.007,0.0075,0.008,0.0085,0.009,0.0095,0.01}"
FRACS_CONTEXT="${FRACS_CONTEXT:-0.05,0.10,0.25,0.50,1.0}"
SEED="${SEED:-42}"

# --- robustness ---
SEEDS_LIST="${SEEDS_LIST:-42,1337,2025}"
WIN_DAYS="${WIN_DAYS:-60}"
WIN_STEP="${WIN_STEP:-30}"
N_WINS="${N_WINS:-6}"
ONLY_IDX="${ONLY_IDX:-}"         # 0..N-1 to run one cutoff for smoke tests
RUN_SUFFIX="${RUN_SUFFIX:-}"     # append to folder names
FORCE_TRAIN_GAN="${FORCE_TRAIN_GAN:-1}"

# If you want same MINPOS everywhere, export MINPOS_ALL=5
MINPOS_ALL="${MINPOS_ALL:-}"

# >>> Write EVERYTHING to *_rerun trees <<<
OUT_BASE="data/processed/metrics/temporal_rerun"
GAN_BASE="models/gan/temporal_rerun"

CUTS=(  "2019-10-01"  "2019-12-01"  "2020-03-01"  "2020-06-01"  "2020-09-01" )
PREFIXES=( "cut2019_10"  "cut2019_12"  "cut2020_03"  "cut2020_06"  "cut2020_09" )

minpos_for_prefix () {
  local p="$1"
  if [[ -n "$MINPOS_ALL" ]]; then
    echo "$MINPOS_ALL"; return
  fi
  case "$p" in
    cut2019_09) echo 5  ;;
    cut2019_10) echo 10 ;;
    cut2019_12) echo 15 ;;
    cut2020_03) echo 25 ;;
    cut2020_06) echo 25 ;;
    cut2020_09) echo 25 ;;
    *)          echo 10 ;;
  esac
}

run_one () {
  local prefix_base="$1"
  local cutoff="$2"
  local prefix="${prefix_base}${RUN_SUFFIX}"
  local minpos; minpos="$(minpos_for_prefix "$prefix_base")"

  # Per-cutoff folders (RERUN trees)
  local outdir="${OUT_BASE}/${prefix}"
  local gendir="${GAN_BASE}/${prefix}"
  mkdir -p "$outdir" "$gendir" logs

  echo "=== [${prefix}] cutoff=${cutoff} MINPOS=${minpos} CONST=${CONST} ==="

  # Always refresh split for this cutoff
  venv/bin/python -m scripts.experiments.holdouts.make_holdouts --temporal-cutoff "${cutoff}"
  venv/bin/python -m scripts.experiments.holdouts.verify_holdouts

  # Train GAN if forced OR missing for this cutoff
  if [[ "$FORCE_TRAIN_GAN" == "1" || ! -f "${gendir}/generator.pth" ]]; then
    # Train generator
    venv/bin/python scripts/gan/train_gan.py \
      --indices-json data/holdouts/temporal_indices.json \
      --malware-only \
      --out "${gendir}/generator.pth" | tee "logs/${prefix}_train_gan.log"
  else
    echo "[skip] generator exists at ${gendir}/generator.pth"
  fi

  # Ensure scaler exists for this cutoff (fits on same train malware subset)
  if [[ ! -f "${gendir}/scaler.npz" ]]; then
    venv/bin/python -m scripts.utils.make_gan_scaler \
      --indices-json data/holdouts/temporal_indices.json \
      --out "${gendir}/scaler.npz"
  fi

  # 3) Scarcity sweep → per-cutoff raw.csv
  venv/bin/python -m scripts.experiments.holdouts.sweep_holdouts_scarcity \
    --use-temporal \
    --fractions "$FRACS_SCARCITY" \
    --const-train-size "$CONST" \
    --gan-generator "${gendir}/generator.pth" \
    --gan-scaler    "${gendir}/scaler.npz" \
    --tag-prefix "$prefix" \
    --min-train-pos "$minpos" \
    --seeds "$SEEDS_LIST" \
    --val-threshold balacc \
    --compare gan,oversample,smote \
    --test-window-days "$WIN_DAYS" \
    --test-window-step-days "$WIN_STEP" \
    --n-test-windows "$N_WINS" \
    --metrics-subdir temporal_rerun \
    --rf-class-weight none \
    --rf-max-depth 20 \
    --rf-n-est 400 \
    --balance-after-augment \
    --gan-like full \
    --gan-synth-per-real 2 \
    --gan-quality nn_boundary \
    --gan-qmult 5 \
    --out-csv "${outdir}/raw.csv" \
    | tee "logs/${prefix}_sweep_scarcity.log"

  # 3b) Pair + aggregate (writes paired.csv and paired_cv.csv)
  venv/bin/python -m scripts.utils.pair_results \
    --prefix "$prefix" \
    --metrics-root "${OUT_BASE}" \
    --raw          "${outdir}/raw.csv" \
    --paired-out        "${outdir}/paired.csv" \
    --paired-cv-out     "${outdir}/paired_cv.csv" \
    | tee "logs/${prefix}_pair.log"

  echo ">>> Done: ${outdir}/paired.csv and paired_cv.csv"

  # 4) Optional context-tier sweep (unchanged)
  if [[ "$DO_CONTEXT" == "1" ]]; then
    make sweep PREFIX="$prefix" FRACTIONS="$FRACS_CONTEXT" MINPOS="$minpos" CONST="$CONST" \
      | tee "logs/${prefix}_sweep_context.log"
    make pair  PREFIX="$prefix" | tee -a "logs/${prefix}_pair_context.log"
  fi
}

# -------- main loop --------
if [[ -n "$ONLY_IDX" ]]; then
  run_one "${PREFIXES[$ONLY_IDX]}" "${CUTS[$ONLY_IDX]}"
else
  for i in "${!CUTS[@]}"; do
    run_one "${PREFIXES[$i]}" "${CUTS[$i]}"
  done
fi

echo "All temporal sweeps complete."
