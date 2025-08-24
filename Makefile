SHELL := /bin/bash
.ONESHELL:
.DEFAULT_GOAL := help

# -------- knobs (override per run) --------
PREFIX        ?= cut2020_01          # folder names under metrics/ and gan/
TEMP_CUTOFF   ?= 2020-01-01          # temporal cutoff date
FRACTIONS     ?= 0.0005,0.001,0.002  # comma list of real fractions
MINPOS        ?= 5                   # min positives for scarce subsampling
CONST         ?= 20000               # target train size for +GAN
SEED          ?= 42
METRICS_KIND  ?= temporal
SEEDS      ?=
WIN_DAYS   ?=
WIN_STEP   ?=
N_WINS     ?=
COMPARE    ?= gan,oversample,smote
VAL_THRESH ?= f1
DO_TUNE    ?=
GRID       ?= light

# -------- env/paths --------
VENV      ?= venv
PY        := $(VENV)/bin/python
export PYTHONPATH := $(PWD):$(PYTHONPATH)

OUTDIR    := data/processed/metrics/$(METRICS_KIND)/$(PREFIX)
RAWCSV    := $(OUTDIR)/raw.csv
PAIRED    := $(OUTDIR)/paired.csv

GEN_DIR   := models/gan/$(METRICS_KIND)/$(PREFIX)
GENERATOR := $(GEN_DIR)/generator.pth
SCALER    := $(GEN_DIR)/scaler.npz

HOLDOUT_JSON := data/holdouts/temporal_indices.json

# -------- targets --------
.PHONY: help dirs temporal-split verify train-gan sweep pair all

help:
	@echo "Usage examples:"
	@echo "  make all PREFIX=cut2020_01 TEMP_CUTOFF=2020-01-01 FRACTIONS=0.0005,0.001,0.002,0.005,0.01,0.02 MINPOS=5 CONST=20000"
	@echo "  make sweep PREFIX=cut2020_01 FRACTIONS=0.002 MINPOS=5 CONST=10000"
	@echo ""
	@echo "Artifacts:"
	@echo "  $(OUTDIR)/{raw.csv,paired.csv}"
	@echo "  $(GEN_DIR)/{generator.pth,scaler.npz}"

dirs:
	mkdir -p "$(OUTDIR)" "$(GEN_DIR)" logs

temporal-split: dirs
	$(PY) -m scripts.experiments.holdouts.make_holdouts --temporal-cutoff $(TEMP_CUTOFF)
	$(PY) -m scripts.experiments.holdouts.verify_holdouts

verify:
	$(PY) -m scripts.experiments.holdouts.verify_holdouts

train-gan: temporal-split
	$(PY) scripts/gan/train_gan.py \
	  --indices-json $(HOLDOUT_JSON) \
	  --malware-only \
	  --out $(GENERATOR)
	$(PY) -m scripts.utils.make_gan_scaler --out "$(SCALER)"



sweep: dirs
	$(PY) -m scripts.experiments.holdouts.sweep_holdouts_scarcity \
	  --use-temporal \
	  --fractions $(FRACTIONS) \
	  --const-train-size $(CONST) \
	  --gan-generator $(GENERATOR) \
	  --tag-prefix $(PREFIX) \
	  --min-train-pos $(MINPOS) \
	  --seed $(SEED) \
	  --val-threshold $(VAL_THRESH) \
	  --compare $(COMPARE) \
	  $(if $(SEEDS),--seeds $(SEEDS),) \
	  $(if $(WIN_DAYS),--test-window-days $(WIN_DAYS),) \
	  $(if $(WIN_STEP),--test-window-step-days $(WIN_STEP),) \
	  $(if $(N_WINS),--n-test-windows $(N_WINS),) \
	  $(if $(DO_TUNE),--tune --grid $(GRID),) \
	  --out-csv "$(RAWCSV)"


pair: dirs
	$(PY) -m scripts.utils.pair_results \
	  --in  "$(RAWCSV)" \
	  --out "$(PAIRED)"


all: train-gan sweep pair
