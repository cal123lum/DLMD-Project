# src/paths.py
from pathlib import Path

def _guess_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / ".git").exists():
            return p
    return here.parents[1]  # parent of src/

ROOT = _guess_root()

DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS         = ROOT / "models"
MODELS_BASE    = MODELS / "baselines"
MODELS_GAN     = MODELS / "gan" / "generator.pth"

BODMAS_NPZ         = DATA_RAW / "bodmas.npz"
BASELINE_RF_CV_JOB = MODELS_BASE / "baseline_rf_cv_full.joblib"
GAN_GENERATOR_PTH  = MODELS_GAN / "generator.pth"

HOLDOUTS_DIR     = ROOT / "data" / "holdouts"
TEMPORAL_SPLIT   = HOLDOUTS_DIR / "temporal_indices.json"
FAMILY_SPLIT     = HOLDOUTS_DIR / "family_indices.json"
BODMAS_META_CSV  = ROOT / "data" / "raw" / "bodmas_metadata.csv"
