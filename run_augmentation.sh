#!/usr/bin/env bash
set -eo pipefail

echo "➤ Activating virtual environment"
source venv/bin/activate

echo "➤ Waiting for generator model at models/augmentation/generator.pth"
while [ ! -f models/augmentation/generator.pth ]; do
  echo "   still waiting... (sleeping 60s)"
  sleep 60
done
echo "✅ Generator model found. Proceeding with augmentation pipeline."

echo "➤ Verifying generator output"
python src/augmentation/verify_generator.py

echo "➤ Evaluating augmentation ratios"
time python src/augmentation/evaluate_augmentation.py

echo "✅ All augmentation jobs complete"

