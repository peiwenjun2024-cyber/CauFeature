# CauFeature

CauFeature is a causality-informed feature selection framework for compact and interpretable downstream classification. This repository keeps the final experiment code and datasets for reproducing the CauFeature paper experiments.

## Repository Scope

This GitHub repository intentionally tracks only the source code, datasets, and environment description needed to rerun the experiments. Generated result folders, paper figures, LaTeX paper files, zip archives, and plain-text manifest or requirements files are not versioned.

Ignored generated paths include:

- `results/`
- `figure/`
- `paper/`
- `*.txt`
- `*.zip`

The method uses directed-dependence evidence for feature selection and does not claim to recover the true data-generating causal graph.

## Environment

Create an environment from:

```bash
conda env create -f environment.yml
conda activate caufeature
```

If you use `pip` instead of conda, install the packages listed in `environment.yml`.

## Core Files

- `model_train_v6.py`: shared CNN training and evaluation pipeline.
- `feature_perturbation_v7_english.py`: perturbation-based feature contribution estimation.
- `causal_graph_build_v19_english.py`: directed-dependence graph construction and path verification.
- `redundantRecover_v10_english.py`: supplementary informative feature recovery.
- `shared_globals.py`: shared experiment state used by the implementation.

## Experiment Entrypoints

- `run_caufeature_experiments.py`: runs CauFeature on one or more datasets and seeds.
- `run_baseline_experiments.py`: runs VT, MI, CHI, RFE, and RFC baselines under fixed-budget or matched-dimension settings.
- `run_ablation_experiments.py`: runs repeated-seed ablation for Original, Phase 1, Phase 1+2, and Full CauFeature.
- `summarize_caufeature_results.py`: summarizes CauFeature runs.
- `summarize_experiment_results.py`: summarizes multiple experiment CSV files by dataset and method.

Optional single-configuration comparison scripts:

- `shap_cnn.py`
- `lime_cnn_fast.py`
- `tff.py`

## Datasets

Datasets are stored in `dataset/`. The representative repeated-seed datasets are:

- `MagicGamma.csv`
- `BrainMethod.csv`
- `SoftwareQuality.csv`

The cross-dataset analysis uses the dataset list defined in `run_caufeature_experiments.py`.

## Basic Usage

Run CauFeature on the representative datasets:

```bash
python run_caufeature_experiments.py \
  --datasets MagicGamma.csv BrainMethod.csv SoftwareQuality.csv \
  --seeds 0 1 2 3 4 \
  --output results/caufeature_runs.csv
```

Run fixed-budget traditional baselines:

```bash
python run_baseline_experiments.py \
  --datasets MagicGamma.csv BrainMethod.csv SoftwareQuality.csv \
  --methods VT MI CHI RFE RFC \
  --retention 0.75 \
  --seeds 0 1 2 3 4 \
  --original-metrics-csv results/caufeature_runs.csv \
  --output results/baseline_fixed_runs.csv
```

Run matched-dimension traditional baselines:

```bash
python run_baseline_experiments.py \
  --datasets MagicGamma.csv BrainMethod.csv SoftwareQuality.csv \
  --methods VT MI CHI RFE RFC \
  --retention 0.75 \
  --seeds 0 1 2 3 4 \
  --original-metrics-csv results/caufeature_runs.csv \
  --matched-dimensions-csv results/caufeature_runs.csv \
  --output results/baseline_matched_runs.csv
```

Run repeated-seed ablation:

```bash
python run_ablation_experiments.py \
  --datasets MagicGamma.csv BrainMethod.csv SoftwareQuality.csv \
  --seeds 0 1 2 3 4 \
  --output results/ablation_runs.csv \
  --resume
```

Summarize completed runs:

```bash
python summarize_experiment_results.py \
  --inputs results/caufeature_runs.csv results/baseline_fixed_runs.csv results/baseline_matched_runs.csv \
  --output results/experiment_summary.csv
```

## Notes

- `reduction_pct` is computed as `(original_dim - selected_dim) / original_dim * 100`.
- Higher dimensionality reduction means stronger compression, but it should be interpreted together with Accuracy and F-measure.
- Runtime values are implementation-level measurements, not theoretical complexity rankings.
