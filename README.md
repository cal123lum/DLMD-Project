# DLMD Project (Publishable)

This repository contains the code used to run the DLMD experiments (IID, temporal, and LOFO regimes) and generate the paper figures across BODMAS, EMBER, and SOREL. Large datasets and run artifacts are intentionally not tracked in Git.

## Contents

- `configs/standard_hparams.json`  
  Shared hyperparameters used across all runs (RF settings, budgets, GAN/EVO/EvoGAN params, gates).

- `scripts/experiments/`  
  Experiment runners:
  - `scripts/experiments/iid/run_iid_scarcity_sweep.py`
  - `scripts/experiments/holdouts/run_temporal_sweep.py`
  - `scripts/experiments/holdouts/run_family_sweep.py`

- `scripts/plots/plot_results.py`  
  Plotting CLI for curves, deltas, heatmaps, and variability summaries.

- `data/`  
  Expected local layout (not committed):
  - `data/raw/` datasets
  - `data/processed/metrics/` metrics outputs from runs
  - `data/holdouts/` split indices (generated)

## Environment

Python 3.10+ recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

On CHPC or any managed cluster, use your preferred module environment and create a venv in your workspace.

## Local Data Placement

Place datasets under `data/raw/` (paths can be overridden via CLI flags).

**EMBER**
- `data/raw/ember_uncompressed.npz`
- `data/raw/ember_metadata.csv`

**SOREL subset**
- `data/raw/sorel_subset_150k.npz`
- `data/raw/sorel_subset_150k_metadata.csv`

**BODMAS**
- `data/raw/bodmas.npz` (or the filename expected by your dataset loader)

## Running experiments

All runners accept `--hparams configs/standard_hparams.json` and apply a standardised configuration.

### IID (example)

```bash
python -m scripts.experiments.iid.run_iid_scarcity_sweep   --dataset bodmas   --npz data/raw/bodmas.npz   --fractions 0.0005,0.001,0.005,0.01   --seeds 42,1337,2025   --methods real,gan,evogan,evo,smote,oversample   --hparams configs/standard_hparams.json
```

### Temporal sweep (example: EMBER)

```bash
python -m scripts.experiments.holdouts.run_temporal_sweep   --dataset ember   --npz data/raw/ember_uncompressed.npz   --meta-csv data/raw/ember_metadata.csv   --cutoffs 2018-07-01,2018-08-01,2018-09-01,2018-10-01,2018-11-01   --seeds-override 42,1337,2025   --fractions 0.0005,0.001,0.005,0.01   --hparams configs/standard_hparams.json   --skip-existing
```

### Family / LOFO sweep (example: EMBER)

```bash
python -m scripts.experiments.holdouts.run_family_sweep   --dataset ember   --npz data/raw/ember_uncompressed.npz   --meta-csv data/raw/ember_metadata.csv   --families xtrat,zbot,ramnit,sality,installmonster,zusy,emotet,vtflooder,UNKNOWN,fareit,adposhel,high   --seeds 42,1337,2025   --fractions 0.0005,0.001,0.005,0.01   --hparams configs/standard_hparams.json   --skip-existing
```

### Notes

- Temporal runs generate split indices under `data/holdouts/` and write metrics under `data/processed/metrics/temporal_<dataset>/`.
- LOFO runs write metrics under `data/processed/metrics/family_<dataset>/` (for SOREL, “family” corresponds to tag-group LOFO).

## Plotting

All plots are generated with `scripts/plots/plot_results.py`. The script accepts one or more CSVs (globs supported).

### Curves: AUC vs fraction (IID)

```bash
python -m scripts.plots.plot_results data/processed/metrics/iid_ember/raw.csv   --mode lines --metric auc --logx   --out figures/iid/ember
```

### Temporal aggregation over cutoffs/windows

```bash
python -m scripts.plots.plot_results "data/processed/metrics/temporal_ember/*/raw.csv"   --mode lines-agg --metric auc --logx   --out figures/temporal/ember
```

### LOFO aggregation over families/groups

```bash
python -m scripts.plots.plot_results "data/processed/metrics/family_ember/*/raw.csv"   --mode lines-agg --metric auc --logx   --out figures/family/ember
```

### ΔAUC snapshots (paired vs Real-only)

```bash
python -m scripts.plots.plot_results "data/processed/metrics/family_ember/*/raw.csv"   --mode delta-bars --metric auc --fractions 0.0005,0.005,0.01   --out figures/delta/family/ember
```

### Pooled ΔAUC across datasets (IID example)

```bash
python -m scripts.plots.plot_results   data/processed/metrics/iid_final/raw.csv   data/processed/metrics/iid_ember/raw.csv   data/processed/metrics/iid_sorel/raw.csv   --mode delta-bars --metric auc --fractions 0.0005,0.005,0.01   --out figures/delta_all/iid_all
```

## Reproducibility notes

- Datasets and large artifacts are intentionally excluded from Git history.
- Use the shared hyperparameter file (`configs/standard_hparams.json`) to keep runs consistent across datasets and regimes.
- Runners produce metrics CSVs under `data/processed/metrics/`, which are the inputs to all plotting commands.

## Contact

For questions about running on CHPC or reproducing specific paper figures, contact the project authors.
