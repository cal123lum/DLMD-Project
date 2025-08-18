#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Activate venv if present
if [ -d "venv" ]; then
  source venv/bin/activate
fi

mkdir -p logs

echo "=== START $(date) ==="

# --- Helper to verify splits
verify() {
  python scripts/experiments/verify_holdouts.py
}

# ---------- TEMPORAL SERIES ----------
for CUT in 2014-01-01 2016-01-01 2018-01-01; do
  echo
  echo "[make] temporal cutoff=$CUT"
  python scripts/experiments/make_holdouts.py --temporal-cutoff "$CUT"
  verify
  TAG="cut${CUT:0:4}"
  echo "[eval] temporal tag=$TAG"
  python scripts/experiments/eval_holdout.py --use-temporal --tune --grid medium --tag "$TAG"
done

# ---------- FAMILY: SINGLES ----------
for FAM in sfone wacatac upatre; do
  echo
  echo "[make] family=$FAM"
  python scripts/experiments/make_holdouts.py --family "$FAM" --benign-test-frac 0.25
  verify
  echo "[eval] family tag=$FAM"
  python scripts/experiments/eval_holdout.py --use-family --tune --grid medium --tag "$FAM"
done

# ---------- FAMILY: COMBOS ----------
echo
echo "[make] families=wacatac+upatre"
python scripts/experiments/make_holdouts.py --family wacatac --family upatre --benign-test-frac 0.25
verify
python scripts/experiments/eval_holdout.py --use-family --tune --grid heavy --tag wacatac_upatre

echo
echo "[make] families=wabot+small+ganelp"
python scripts/experiments/make_holdouts.py --family wabot --family small --family ganelp --benign-test-frac 0.25
verify
python scripts/experiments/eval_holdout.py --use-family --tune --grid heavy --tag wabot_small_ganelp

echo "=== END $(date) ==="
