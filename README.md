# DLMD Project – GAN/EVO Augmentation for Malware Detection

This repository contains the final code and artifacts for **“GAN‑based and EVO‑based Data Augmentation for Malware Classification under Data Scarcity.”**  
It reproduces the i.i.d., temporal, and family (LOFO) experiments and generates all figures used in the paper.

---

## 1) Quick Start

```bash
# 1) Create & activate a virtual environment
python3 -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) (Optional) Verify the three driver scripts are callable
python run_iid_scarcity_sweep.py --help || true
python run_temporal_sweep.py --help      || true
python run_family_sweep.py --help        || true
```

All experiments below assume you are in the repository root with the `venv` activated.

---

## 2) Data Layout

The runners expect processed metrics to land under these folders (created by the drivers). If you already have the results, your tree should look like this after running the sweeps:

```
data/
└── processed/
    └── metrics/
        ├── iid_final/
        │   └── raw.csv
        ├── temporal_final_window/
        │   ├── cut2019_10/raw.csv
        │   ├── cut2019_12/raw.csv
        │   ├── cut2020_03/raw.csv
        │   ├── cut2020_06/raw.csv
        │   └── cut2020_09/raw.csv   # may be absent if omitted
        └── family_final/
            ├── <familyA>/raw.csv
            ├── <familyB>/raw.csv
            └── ...
```

If you are starting from scratch, simply run the drivers in §3. They use the final defaults used for the paper (no extra CLI args required).

---

## 3) Reproducing the Experiments (Drivers)

> **Important:** The three driver scripts pick up the **final defaults** for splits, seeds, scarcity fractions, augmentation budgets, and evaluation. You can run them **without any arguments** to reproduce the paper’s results.

### 3.1 i.i.d. Scarcity Calibration
Creates `data/processed/metrics/iid_final/raw.csv`.

```bash
python run_iid_scarcity_sweep.py
```

### 3.2 Temporal Holdout with Rolling Windows
Creates one `raw.csv` per cutoff under `data/processed/metrics/temporal_final_window/*/raw.csv`.

```bash
python run_temporal_sweep.py
```

### 3.3 Family Leave‑One‑Family‑Out (LOFO)
Creates one `raw.csv` per family under `data/processed/metrics/family_final/*/raw.csv`.

```bash
python run_family_sweep.py
```

Each driver will:
- iterate the final scarcity grid `f ∈ {0.0005 … 0.01}` with floors on class counts,
- run seeds `{42, 1337, 2025}`,
- apply the shared augmentation budget and NN quality gate for GAN/EVO/EvoGAN,
- evaluate the fixed RF classifier,
- and write a consolidated `raw.csv` per regime/slice ready for plotting.

---

## 4) Make the Figures (Single CLI Plotter)

All plots are produced by a single pure‑matplotlib script:

```
scripts/plots/plot_results.py
```

> Tip: use `--out <dir>` to keep figures neatly grouped; paths below mirror the paper.

### 4.1 i.i.d. Figures

**AUC vs fraction (lines):**
```bash
python scripts/plots/plot_results.py \
  data/processed/metrics/iid_final/raw.csv \
  --mode lines --metric auc --logx \
  --out figures/iid
```

**ΔAUC bars @ f ∈ {0.0005, 0.005, 0.01} (paired bootstrap CIs + stars):**
```bash
python scripts/plots/plot_results.py \
  data/processed/metrics/iid_final/raw.csv \
  --mode delta-bars --metric auc --fractions 0.0005,0.005,0.01 \
  --out figures/iid
```

### 4.2 Temporal Figures

**Aggregate trend across cutoffs/windows (AUC vs fraction):**
```bash
python scripts/plots/plot_results.py \
  data/processed/metrics/temporal_final_window/*/raw.csv \
  --mode lines-agg --metric auc --logx \
  --out figures/temporal
```

**Per‑method ΔAUC heatmaps @ f=0.0005:**
```bash
# GAN
python scripts/plots/plot_results.py \
  data/processed/metrics/temporal_final_window/*/raw.csv \
  --mode heatmap --metric auc --method gan --fraction 0.0005 \
  --out figures/temporal

# EvoGAN
python scripts/plots/plot_results.py \
  data/processed/metrics/temporal_final_window/*/raw.csv \
  --mode heatmap --metric auc --method evogan --fraction 0.0005 \
  --out figures/temporal

# EVO
python scripts/plots/plot_results.py \
  data/processed/metrics/temporal_final_window/*/raw.csv \
  --mode heatmap --metric auc --method evo --fraction 0.0005 \
  --out figures/temporal
```

### 4.3 Family (LOFO) Figures

**AUC vs fraction, aggregated over families (lines):**
```bash
python scripts/plots/plot_results.py \
  data/processed/metrics/family_final/*/raw.csv \
  --mode lines-agg --metric auc --logx \
  --out figures/family
```

**Per‑family ΔAUC bars @ f=0.0005 (with CIs):**
```bash
python scripts/plots/plot_results.py \
  data/processed/metrics/family_final/*/raw.csv \
  --mode family-bars --metric auc --fraction 0.0005 \
  --out figures/family
```

**Seed variability (box) @ f=0.0005 (optional for appendix):**
```bash
python scripts/plots/plot_results.py \
  data/processed/metrics/family_final/*/raw.csv \
  --mode seed-box-agg --metric auc --fraction 0.0005 \
  --out figures/family
```

> You can also generate PR‑AUC / F1 / BalAcc / MCC by switching `--metric`.

---


## 5) Troubleshooting

- **FileNotFoundError for `raw.csv`**  
  Run the corresponding driver first (see §3). The plotter reads consolidated CSVs only.

- **Matplotlib font / backend warnings**  
  These are harmless in headless environments; figures are still written to disk.

- **Out-of-memory during drivers**  
  Activate the virtualenv and ensure dependencies match `requirements.txt`. If you’re on a low‑RAM machine, close other apps. The defaults used in the paper run on a standard workstation.

- **Different colors/order in legends**  
  The plotter enforces a fixed method order and color map. If your CSV contains different variant labels, they must be one of: `real, gan, evogan, evo, smote, oversample`.

---

## 6) Repro Notes (what’s fixed)

- Fixed RF classifier and validation/threshold protocol across all regimes.
- Shared augmentation budget & nearest‑neighbour gate across GAN/EVO/EvoGAN.
- Identical test partitions across methods; differences reflect **training‑set composition** only.
- Seeds `{42, 1337, 2025}`; fractions `{0.0005 … 0.01}`; 60‑day windows with 30‑day stride for temporal.

---

## 7) Outputs

Figures are written under `figures/` with informative names, e.g.:

```
figures/
├── iid/
│   ├── iid_auc_lines.png
│   ├── delta_auc_f0.0005.png
│   ├── delta_auc_f0.005.png
│   └── delta_auc_f0.01.png
├── temporal/
│   ├── familyALL_auc_lines.png            # aggregate trend (lines-agg)
│   ├── heatmap_gan_auc_f0.0005.png
│   ├── heatmap_evogan_auc_f0.0005.png
│   └── heatmap_evo_auc_f0.0005.png
└── family/
    ├── familyALL_auc_lines.png
    ├── family_delta_auc_f0.0005.png
    └── agg_auc_seed_box_f0.0005_.png
```

