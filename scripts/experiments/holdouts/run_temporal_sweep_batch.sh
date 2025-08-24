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

# --- knobs you can tweak (or override via env when calling) ---
CONST="${CONST:-20000}"   # total train size for +GAN
DO_CONTEXT="${DO_CONTEXT:-0}"  # 1 to also run context-tier sweep
FRACS_SCARCITY="${FRACS_SCARCITY:-0.0005,0.001,0.002,0.003,0.005,0.0075,0.01,0.015,0.02}"
FRACS_CONTEXT="${FRACS_CONTEXT:-0.05,0.10,0.25,0.50,1.0}"
SEED="${SEED:-42}"

# --- knobs for robustness
SEEDS_LIST="${SEEDS_LIST:-42,1337,2025}"      # multi-seed CV
WIN_DAYS="${WIN_DAYS:-60}"                    # fixed test window length
WIN_STEP="${WIN_STEP:-30}"                    # stride between windows
N_WINS="${N_WINS:-6}"                         # how many windows from test start  (I bumped default to 6; set what you want)
ONLY_IDX="${ONLY_IDX:-}"                      # 0..N-1 to run one cutoff for smoke tests
RUN_SUFFIX="${RUN_SUFFIX:-}"                  # append to results folders to avoid overwrite
FORCE_TRAIN_GAN="${FORCE_TRAIN_GAN:-1}"

# If you want the same MINPOS everywhere, export MINPOS_ALL=5 when calling.
MINPOS_ALL="${MINPOS_ALL:-}"

CUTS=(    "2019-09-01"  "2019-10-01"  "2019-12-01"  "2020-03-01"  "2020-06-01"  "2020-09-01" )
PREFIXES=("cut2019_09"  "cut2019_10"  "cut2019_12"  "cut2020_03"  "cut2020_06"  "cut2020_09" )

# helper: decide MINPOS per prefix (unless MINPOS_ALL is set)
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
    *)          echo 10 ;; # default
  esac
}

run_one () {
    local prefix_base="$1"
    local cutoff="$2"
    local prefix="${prefix_base}${RUN_SUFFIX}"
    local minpos
    minpos="$(minpos_for_prefix "$prefix_base")"

    # Artifacts
    local outdir="data/processed/metrics/temporal/${prefix}"
    local gendir="models/gan/temporal/${prefix}"   # <— train NEW GAN per run
    mkdir -p "$outdir" "$gendir" logs

    echo "=== [${prefix}] cutoff=${cutoff} MINPOS=${minpos} CONST=${CONST} ==="

    # Always refresh split
    make temporal-split PREFIX="$prefix" TEMP_CUTOFF="$cutoff" | tee "logs/${prefix}_split.log"

    # Train GAN (force if requested)
    if [[ "$FORCE_TRAIN_GAN" == "1" || ! -f "${gendir}/generator.pth" ]]; then
        make train-gan PREFIX="$prefix" TEMP_CUTOFF="$cutoff" | tee "logs/${prefix}_train_gan.log"
    else
        echo "[skip] generator exists at ${gendir}/generator.pth"
    fi

# 3) Scarcity sweep → raw.csv
  python -m scripts.experiments.holdouts.sweep_holdouts_scarcity \
    --use-temporal \
    --fractions "$FRACS_SCARCITY" \
    --const-train-size "$CONST" \
    --gan-generator "${gendir}/generator.pth" \
    --tag-prefix "$prefix" \
    --min-train-pos "$minpos" \
    --seeds "$SEEDS_LIST" \
    --val-threshold f1 \
    --compare gan,oversample,smote \
    --test-window-days "$WIN_DAYS" \
    --test-window-step-days "$WIN_STEP" \
    --n-test-windows "$N_WINS" \
    --out-csv "${outdir}/raw.csv" \
    | tee "logs/${prefix}_sweep_scarcity.log"

  # 3b) Pair + aggregate
  python -m scripts.utils.pair_results \
    --prefix "$prefix" \
    --metrics-root "data/processed/metrics/temporal" \
    --raw       "data/processed/metrics/temporal/${prefix}/raw.csv" \
    --paired-out    "data/processed/metrics/temporal/${prefix}/paired.csv" \
    --paired-cv-out "data/processed/metrics/temporal/${prefix}/paired_cv.csv" \
    | tee "logs/${prefix}_pair.log"

  echo ">>> Done: ${outdir}/paired.csv and paired_cv.csv"


  # 4) Optional: context-tier sweep (+ pair)
  if [[ "$DO_CONTEXT" == "1" ]]; then
    make sweep PREFIX="$prefix" FRACTIONS="$FRACS_CONTEXT" MINPOS="$minpos" CONST="$CONST" \
      | tee "logs/${prefix}_sweep_context.log"
    make pair  PREFIX="$prefix" | tee -a "logs/${prefix}_pair_context.log"
  fi

  echo ">>> Done: ${outdir}/paired.csv"
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
