cat > scripts/run_ember_temporal_family.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python}
HPARAMS=${HPARAMS:-configs/standard_hparams.json}

NPZ=${NPZ:-data/raw/ember_uncompressed.npz}
META=${META:-data/raw/ember_metadata.csv}

SEEDS=${SEEDS:-0,1,2,3,4}
FRACTIONS=${FRACTIONS:-0.0005,0.001,0.005,0.01}

# Update cutoffs to your intended list
CUTOFFS=${CUTOFFS:-2018-07-01,2018-08-01,2018-09-01,2018-10-01,2018-11-01}

FAMILIES=${FAMILIES:-xtrat,zbot,ramnit,sality,installmonster,zusy,emotet,vtflooder,UNKNOWN,fareit,adposhel,high}

echo "[info] Using:"
echo "  python:   $PYTHON"
echo "  npz:      $NPZ"
echo "  meta:     $META"
echo "  hparams:  $HPARAMS"
echo "  seeds:    $SEEDS"
echo "  fractions:$FRACTIONS"
echo "  cutoffs:  $CUTOFFS"
echo "  families: $FAMILIES"

# Temporal sweep
$PYTHON -m scripts.experiments.holdouts.run_temporal_sweep \
  --dataset ember \
  --npz "$NPZ" \
  --meta-csv "$META" \
  --cutoffs "$CUTOFFS" \
  --seeds-override "$SEEDS" \
  --fractions "$FRACTIONS" \
  --hparams "$HPARAMS" \
  --skip-existing

# Family/LOFO sweep
$PYTHON -m scripts.experiments.holdouts.run_family_sweep \
  --dataset ember \
  --npz "$NPZ" \
  --meta-csv "$META" \
  --families "$FAMILIES" \
  --fractions "$FRACTIONS" \
  --seeds "$SEEDS" \
  --hparams "$HPARAMS" \
  --skip-existing
EOF

chmod +x scripts/run_ember_temporal_family.sh
