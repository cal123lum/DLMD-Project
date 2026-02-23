
# Demo: IID micro-run (Real-only vs EVO) at one scarcity level, then plot ΔAUC.
# Fast: single fraction + single seed; no GAN training.

set -euo pipefail

FRACTION="0.005"        # 0.5% of TRAIN 
SEED="1337"
CONST_TRAIN=8000
RF_EST=200
RF_DEPTH=16

PLOT_DIR="data/processed/metrics/iid_final_demo/plots_demo"
RAW_METRICS="data/processed/metrics/iid_final_demo/raw.csv"
DATA_CHECK="data/raw/bodmas.npz"

say() { printf "\n\033[1m▶ %s\033[0m\n" "$*"; }
run() { echo "+ $*"; "$@"; }


say "Setting up PYTHONPATH (repo-local imports)"
export PYTHONPATH="$PWD"

say "1) Run a tiny IID sweep: Real-only vs EVO @ f=${FRACTION}, seed=${SEED}"
run python scripts/experiments/iid/run_iid_demo.py \
  --fractions "${FRACTION}" \
  --seeds "${SEED}" \
  --methods real,evo \
  --const-train-size "${CONST_TRAIN}" \
  --min-train-pos 30 --min-train-neg 30 \
  --rf-n-est "${RF_EST}" --rf-max-depth "${RF_DEPTH}" --rf-class-weight none \
  --val-threshold balacc \
  --skip-gan-train

if [[ ! -f "$RAW_METRICS" ]]; then
  echo "[err] Expected metrics CSV not found: $RAW_METRICS"
  exit 1
fi

say "2) Plot a single ΔAUC bar (EVO – Real) with bootstrap CI"
run python scripts/plots/plot_results.py \
  "$RAW_METRICS" \
  --mode delta-bars \
  --metric auc \
  --fractions "${FRACTION}" \
  --seed "${SEED}" \
  --out "${PLOT_DIR}"

PNG="${PLOT_DIR}/delta_auc_f${FRACTION}.png"
say "3) Result saved to: ${PNG}"

say "Talking points:"
echo "  - Identical TEST; only TRAIN composition changes (Real vs EVO)."
echo "  - RF, validation, and thresholding are fixed and shared."
echo "  - The bar shows ΔAUC at f=${FRACTION} with a bootstrap CI."
