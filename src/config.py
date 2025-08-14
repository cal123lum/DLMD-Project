# src/config.py
from src.paths import BODMAS_NPZ, DATA_PROCESSED
# Raw inputs

META_PATH   = 'data/raw/bodmas_metadata.csv'
NPZ_PATH    = str(BODMAS_NPZ)
SUBSET_PATH = str(DATA_PROCESSED / 'bodmas_subset.csv')
TRAIN_PATH  = str(DATA_PROCESSED / 'train.csv')
TEST_PATH   = str(DATA_PROCESSED / 'test.csv')

# Sampling params
MALWARE_SAMPLES = 1000
BENIGN_SAMPLES  = 1000
TEST_SIZE       = 0.2
RANDOM_STATE    = 42


FEATURE_KEY = 'X'
LABEL_KEY = 'y'
