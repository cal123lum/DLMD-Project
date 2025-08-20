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
  local prefix="$1"
  local cutoff="$2"
  local minpos
  minpos="$(minpos_for_prefix "$prefix")"

  # Where artifacts land
  local outdir="data/processed/metrics/temporal/${prefix}"
  local gendir="models/gan/temporal/${prefix}"
  mkdir -p "$outdir" "$gendir" logs

  echo "=== [$prefix] cutoff=${cutoff} MINPOS=${minpos} CONST=${CONST} ==="

  # 1) Split + verify
  make temporal-split PREFIX="$prefix" TEMP_CUTOFF="$cutoff" | tee "logs/${prefix}_split.log"

  # 2) Train leak-safe generator (only if not present)
  if [[ ! -f "${gendir}/generator.pth" ]]; then
    make train-gan PREFIX="$prefix" TEMP_CUTOFF="$cutoff" | tee "logs/${prefix}_train_gan.log"
  else
    echo "[skip] generator exists at ${gendir}/generator.pth"
  fi

  # 3) Scarcity-tier sweep (+ pair)
  make sweep PREFIX="$prefix" FRACTIONS="$FRACS_SCARCITY" MINPOS="$minpos" CONST="$CONST" \
    | tee "logs/${prefix}_sweep_scarcity.log"
  make pair  PREFIX="$prefix" | tee "logs/${prefix}_pair_scarcity.log"

  # 4) Optional: context-tier sweep (+ pair)
  if [[ "$DO_CONTEXT" == "1" ]]; then
    make sweep PREFIX="$prefix" FRACTIONS="$FRACS_CONTEXT" MINPOS="$minpos" CONST="$CONST" \
      | tee "logs/${prefix}_sweep_context.log"
    make pair  PREFIX="$prefix" | tee -a "logs/${prefix}_pair_context.log"
  fi

  echo ">>> Done: ${outdir}/paired.csv"
}

# -------- main loop --------
for i in "${!CUTS[@]}"; do
  run_one "${PREFIXES[$i]}" "${CUTS[$i]}"
done

echo "All temporal sweeps complete."
