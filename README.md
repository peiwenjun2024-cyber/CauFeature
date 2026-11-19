# CauFeature

CauFeature is a structure-aware feature selection framework for compact and
interpretable downstream classification. This repository contains the source
code, datasets, and environment description used to reproduce the paper
experiments.

## 1026 Reproduction Package

The 1026 package provides the RQ1–RQ8 code and the public-data workflow
for RQ6 Table 9. [Reproduction instructions](reproduction/1026/README.md)
· [Run from public data](reproduction/1026/PUBLIC_DATA.md)

## Clean Reproduction Protocol

`run_clean_experiments.py` is the default experiment entrypoint. It enforces the
following order for every dataset and seed:

1. Create the outer train/test split from the raw data.
2. Fit missing-value handling and categorical encoding on the outer-training
   split, then transform the outer test split.
3. Create an inner validation split from the outer-training data.
4. Apply SMOTEENN only to the inner fitting split.
5. Perform feature selection using outer-training data only.
6. Use the inner validation split for early stopping and evaluate the outer test
   split once.

The output directory includes a `manifest.json` file with the protocol,
dataset list, seeds, and SHA-256 hashes of the source files used for the run.

## Environment

```bash
conda env create -f environment.yml
conda activate caufeature
```

## Core Files

- `run_clean_experiments.py`: clean entrypoint for RQ1--RQ4 and RQ6.
- `model_train_v6.py`: shared 1D-CNN construction and training helpers.
- `feature_perturbation_v7_english.py`: single-feature perturbation and
  contribution estimation.
- `causal_graph_build_v19_english.py`: directed dependency graph construction
  and stable path screening.
- `redundantRecover_v10_english.py`: dual-feature perturbation and synergistic
  feature recovery.
- `run_baseline_experiments.py`: traditional feature-selection helpers used by
  the clean entrypoint.
- `run_explainable_experiments.py`: SHAP, LIME, and FeatureX helpers used by the
  clean entrypoint.
- `shared_globals.py`: shared state required by the implementation.

## Default Run

The default command runs RQ1--RQ3 with seeds 0--4. RQ1 uses the 15 paper
datasets; RQ2 and RQ3 use `MagicGamma.csv`, `BrainMethod.csv`, and
`SoftwareQuality.csv`.

```bash
python run_clean_experiments.py
```

Results are written to `reproduction_clean_20260720/raw` by default. Use
`--output-dir` to select another location.

Run a small RQ1 check before a full experiment:

```bash
python run_clean_experiments.py \
  --stages rq1 \
  --datasets MagicGamma.csv \
  --seeds 0 \
  --output-dir results/pilot
```

Run the repeated-seed ablation:

```bash
python run_clean_experiments.py \
  --stages rq4 \
  --output-dir reproduction_clean_20260720/raw
```

Generate the fixed BrainMethod process trace used by RQ6:

```bash
python run_clean_experiments.py \
  --stages rq6 \
  --datasets BrainMethod.csv \
  --seeds 3 \
  --output-dir reproduction_clean_20260720/raw
```

## Datasets

Datasets are stored in `dataset/`. The RQ1 dataset list is defined by
`RQ1_DATASETS` in `run_clean_experiments.py`; the default list contains the 15
datasets reported in the paper.

## Notes

- Generated results, figures, paper files, archives, and logs are not versioned.
- `reduction_pct` is `(original_dim - selected_dim) / original_dim * 100`.
- Compression should be interpreted together with Accuracy and F-measure.
- Runtime values are implementation-level measurements, not cross-method
  complexity rankings.
