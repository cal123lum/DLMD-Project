SHELL := /bin/bash
.DEFAULT_GOAL := help

# -------- knobs (override per run) --------
PREFIX        ?= cut2020_01          # folder names under metrics/ and gan/
TEMP_CUTOFF   ?= 2020-01-01          # temporal cutoff date
FRACTIONS     ?= 0.0005,0.001,0.002  # comma list of real fractions
MINPOS        ?= 5                   # min positives for scarce subsampling
CONST         ?= 20000               # target train size for +GAN
SEED          ?= 42
METRICS_KIND  ?= temporal

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
	@$(PY) - <<-'PY'
		import numpy as np
		from pathlib import Path
		from src.paths import ROOT, TEMPORAL_SPLIT
		from src.holdouts import SplitIndices
		from sklearn.preprocessing import StandardScaler
		z = np.load(ROOT/'data'/'raw'/'bodmas.npz', allow_pickle=True)
		X, y = z['X'].astype(np.float32), z['y'].astype(int)
		s = SplitIndices.from_json(TEMPORAL_SPLIT)
		Xmal = X[s.train][y[s.train]==1]
		sc = StandardScaler().fit(Xmal)
		Path("$(SCALER)").parent.mkdir(parents=True, exist_ok=True)
		np.savez("$(SCALER)", mean_=sc.mean_, scale_=sc.scale_)
		print("[gan-scaler] wrote", "$(SCALER)", "using", Xmal.shape[0], "malware rows")
	PY

sweep: dirs
	$(PY) -m scripts.experiments.holdouts.sweep_holdouts_scarcity \
	  --use-temporal \
	  --fractions $(FRACTIONS) \
	  --const-train-size $(CONST) \
	  --gan-generator $(GENERATOR) \
	  --tag-prefix $(PREFIX) \
	  --min-train-pos $(MINPOS) \
	  --seed $(SEED) \
	  --out-csv "$(RAWCSV)"

pair: dirs
	@$(PY) - <<-'PY'
		import pandas as pd, numpy as np
		from pathlib import Path
		src = Path("$(RAWCSV)"); out = Path("$(PAIRED)")
		if not src.exists():
			raise SystemExit(f"[pair] missing raw csv: {src}")
		df = pd.read_csv(src)
		if df["used_gan"].dtype != bool:
			df["used_gan"] = df["used_gan"].astype(str).str.strip().str.lower().isin(["true","1","yes"])
		df["frac"] = pd.to_numeric(df["frac"], errors="coerce")
		df["const_train_size"] = pd.to_numeric(df["const_train_size"], errors="coerce")
		df["run_idx"] = np.arange(len(df))
		metrics = ["auc","pr_auc","f1","balanced_accuracy","mcc","precision","recall","accuracy","specificity"]
		other   = ["n_train_real","n_train_synth","n_train_total","tn","fp","fn","tp"]
		base = (df[df["used_gan"]==False].sort_values("run_idx")
				.groupby(["prefix","kind","frac"], as_index=False).tail(1)
				.rename(columns={c:f"{c}_real" for c in metrics+other}))
		aug  = (df[df["used_gan"]==True].sort_values("run_idx")
				.groupby(["prefix","kind","frac","const_train_size"], as_index=False).tail(1)
				.rename(columns={c:f"{c}_aug" for c in metrics+other}))
		keep = ["prefix","kind","frac"]+[f"{c}_real" for c in metrics+other]
		paired = aug.merge(base[keep], on=["prefix","kind","frac"], how="left")
		for m in ["auc","pr_auc","f1","balanced_accuracy","mcc"]:
			paired[f"delta_{m}"] = paired[f"{m}_aug"] - paired[f"{m}_real"]
		cols = ["prefix","kind","frac","const_train_size"] \
			+ [f"{m}_real" for m in ["auc","pr_auc","f1","balanced_accuracy","mcc"]] \
			+ [f"{m}_aug"  for m in ["auc","pr_auc","f1","balanced_accuracy","mcc"]] \
			+ [f"delta_{m}" for m in ["auc","pr_auc","f1","balanced_accuracy","mcc"]]
		paired = paired[cols].sort_values(["frac","const_train_size"])
		out.parent.mkdir(parents=True, exist_ok=True); paired.to_csv(out, index=False)
		print(f"[paired] wrote {out} ({len(paired)} rows)")
	PY

all: train-gan sweep pair
