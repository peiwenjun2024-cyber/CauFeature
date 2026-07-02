# CauFeature Paper Experiments

This repository contains the experiment code, datasets, result CSV files, and exported figures used for the CauFeature paper. The code is organized for paper reproduction rather than the earlier GUI demonstration.

## Scope

The paper evaluates CauFeature as a causality-informed feature selection framework. The experiments do not claim to recover the true causal structure. Reported selected features are interpreted as target-local directed features (TLDFs) or causality-informed selected features under the implemented perturbation, directed-dependence, and recovery procedure.

The main reported metrics are:

- `selected_dim`: number of selected features.
- `reduction_pct`: dimensionality reduction percentage, computed as `(original_dim - selected_dim) / original_dim * 100`. A larger value means stronger compression, but it is meaningful only together with predictive performance.
- `selected_accuracy`: downstream CNN accuracy after feature selection.
- `selected_f1`: downstream weighted F-measure after feature selection.
- `accuracy_delta_pct_points` and `f1_delta_pct_points`: change from the original-feature CNN in percentage points.
- `total_time_sec`: implementation-level running time measurement, not a theoretical complexity bound.

## Environment

The experiments were prepared with Python 3.12, TensorFlow/Keras, scikit-learn, imbalanced-learn, NumPy, Pandas, SciPy, NetworkX, Matplotlib, tqdm, and seaborn. Install the experiment dependencies with:

```bash
pip install -r requirements-experiments.txt
```

The paper reports experiments on a server with Intel Xeon Gold 6430 CPUs, an NVIDIA RTX 3090 GPU, and 120 GB RAM. GPU model may change runtime and small floating-point variations, but the experiment scripts fix seeds where supported.

## Key Files

Core CauFeature implementation:

- `model_train_v6.py`: shared CNN training and evaluation pipeline.
- `feature_perturbation_v7_english.py`: perturbation-based feature contribution estimation.
- `causal_graph_build_v19_english.py`: directed-dependence graph construction and path verification.
- `redundantRecover_v10_english.py`: supplementary informative feature recovery.
- `shared_globals.py`: shared experiment state used by the original implementation.

Experiment entry points:

- `run_caufeature_experiments.py`: runs CauFeature on one or more datasets and seeds.
- `run_baseline_experiments.py`: runs VT, MI, CHI, RFE, and RFC baselines under fixed-budget or matched-dimension settings.
- `run_ablation_experiments.py`: runs repeated-seed ablation for Original, Phase 1, Phase 1+2, and Full CauFeature.
- `summarize_caufeature_results.py`: summarizes CauFeature runs.
- `summarize_experiment_results.py`: summarizes multiple experiment CSV files by dataset and method.

Figure-generation Python scripts are intentionally not included in this repository. The final exported paper figures and the CSV files from which the reported values were derived are provided directly.

Optional single-configuration comparison scripts:

- `shap_cnn.py`
- `lime_cnn_fast.py`
- `tff.py`

These optional scripts correspond to broad SHAP/LIME/FeatureX-style comparisons. In the paper they are treated as single-configuration supplementary comparisons, not repeated-seed robustness experiments.

## Datasets

Datasets are stored in `dataset/`. The main representative repeated-seed datasets are:

- `MagicGamma.csv`
- `BrainMethod.csv`
- `SoftwareQuality.csv`

The RQ1 cross-dataset compression analysis uses the dataset list defined in `run_caufeature_experiments.py`:

```text
iris.csv
Rice_Cammeo_Osmancik.csv
MagicGamma.csv
HeartDisease.csv
Thyroid_Diff.csv
MentalStress.csv
students dropout academic success.csv
CustomerChurn.csv
DriedBean.csv
online_shoppers_intention.csv
SmokeAlarm.csv
Income.csv
bank-full.csv
DiabetesHealth.csv
BrainMethod.csv
SoftwareQuality.csv
pirvision_office_dataset2.csv
```

## Experiment Logic

### 1. CauFeature Cross-Dataset RQ1

This experiment evaluates the adaptive reduction behavior of CauFeature across datasets with different feature dimensions and sample sizes.

```bash
python run_caufeature_experiments.py \
  --seeds 42 \
  --output results/caufeature_rq1_cross_dataset.csv
```

This repository provides the exported RQ1 figures and archived result tables. Rerunning this command is for verification or extension rather than for overwriting the archived repeated-seed files.

### 2. CauFeature Repeated-Seed Runs

The representative repeated-seed comparison uses MagicGamma, BrainMethod, and SoftwareQuality.

```bash
python run_caufeature_experiments.py \
  --datasets MagicGamma.csv BrainMethod.csv \
  --seeds 0 1 2 3 4 \
  --output results/caufeature_full.csv

python run_caufeature_experiments.py \
  --datasets SoftwareQuality.csv \
  --seeds 0 1 2 3 4 \
  --output results/caufeature_software_241_full_default.csv
```

For commands that need one representative CauFeature metric file, concatenate the two repeated-seed outputs:

```bash
python - <<'PY'
from pathlib import Path
paths = [Path("results/caufeature_full.csv"), Path("results/caufeature_software_241_full_default.csv")]
out = Path("results/caufeature_representative_3datasets.csv")
with out.open("w", encoding="utf-8") as w:
    for i, path in enumerate(paths):
        lines = path.read_text(encoding="utf-8").splitlines()
        if i == 0:
            w.write("\n".join(lines) + "\n")
        else:
            w.write("\n".join(lines[1:]) + "\n")
print(out)
PY
```

### 3. Fixed-Budget Baselines

The fixed-budget baseline protocol keeps a fixed retention ratio for VT, MI, CHI, RFE, and RFC. In the paper results, this is used as the fixed-budget (FB) comparison.

```bash
python run_baseline_experiments.py \
  --datasets MagicGamma.csv \
  --methods VT \
  --retention 0.75 \
  --seeds 0 1 2 3 4 \
  --original-metrics-csv results/caufeature_representative_3datasets.csv \
  --output results/baseline_mg_brain.csv

python run_baseline_experiments.py \
  --datasets BrainMethod.csv \
  --methods VT \
  --retention 0.75 \
  --seeds 0 1 2 3 4 \
  --original-metrics-csv results/caufeature_representative_3datasets.csv \
  --output results/baseline_brain_vt.csv

python run_baseline_experiments.py \
  --datasets MagicGamma.csv BrainMethod.csv \
  --methods MI CHI RFE RFC \
  --retention 0.75 \
  --seeds 0 1 2 3 4 \
  --original-metrics-csv results/caufeature_representative_3datasets.csv \
  --output results/baseline_mg_brain_more.csv

python run_baseline_experiments.py \
  --datasets SoftwareQuality.csv \
  --methods VT MI CHI RFE RFC \
  --retention 0.75 \
  --seeds 0 1 2 3 4 \
  --original-metrics-csv results/caufeature_representative_3datasets.csv \
  --output results/baseline_software_241_fixed.csv
```

The archived paper CSVs are split by run batch:

- `results/baseline_mg_brain.csv`
- `results/baseline_brain_vt.csv`
- `results/baseline_mg_brain_more.csv`
- `results/baseline_software_241_fixed.csv`

In the archived run, `results/baseline_mg_brain.csv` contains MagicGamma/VT and `results/baseline_brain_vt.csv` contains BrainMethod/VT.

### 4. Matched-Dimension Baselines

The matched-dimension (MD) protocol makes each baseline select the same number of features as CauFeature for the same dataset and seed. This directly tests whether the CauFeature subset remains competitive under the same feature budget.

```bash
python run_baseline_experiments.py \
  --datasets MagicGamma.csv BrainMethod.csv \
  --methods VT MI CHI RFE RFC \
  --retention 0.75 \
  --seeds 0 1 2 3 4 \
  --original-metrics-csv results/caufeature_representative_3datasets.csv \
  --matched-dimensions-csv results/caufeature_representative_3datasets.csv \
  --output results/baseline_matched_mg_brain.csv

python run_baseline_experiments.py \
  --datasets SoftwareQuality.csv \
  --methods VT MI CHI RFE RFC \
  --retention 0.75 \
  --seeds 0 1 2 3 4 \
  --original-metrics-csv results/caufeature_representative_3datasets.csv \
  --matched-dimensions-csv results/caufeature_representative_3datasets.csv \
  --output results/baseline_software_241_matched.csv
```

The archived paper CSVs are:

- `results/baseline_matched_mg_brain.csv`
- `results/baseline_software_241_matched.csv`

### 5. Repeated-Seed Ablation

The ablation evaluates the contribution of each stage:

- `Original`: original full feature set.
- `Phase 1`: contribution-guided TLDF candidates.
- `Phase 1+2`: TLDF screening with directed-dependence graph and path verification.
- `Full`: complete CauFeature with supplementary informative feature recovery.

```bash
python run_ablation_experiments.py \
  --datasets MagicGamma.csv BrainMethod.csv SoftwareQuality.csv \
  --seeds 0 1 2 3 4 \
  --output results/ablation_runs.csv \
  --resume
```

### 6. Summaries and Paper Figures

After the required result CSV files are available, numerical summaries can be regenerated with:

```bash
python summarize_experiment_results.py \
  --inputs results/caufeature_full.csv results/caufeature_software_241_full_default.csv \
  --output results/caufeature_representative_summary.csv
```

This repository already contains the aligned paper summary CSV files and final exported figures. Plotting scripts are not included by request.

Important generated summary files:

- `results/aligned_caufeature_3datasets.csv`
- `results/aligned_baseline_fixed_3datasets.csv`
- `results/aligned_baseline_matched_3datasets.csv`
- `results/aligned_fixed_summary_mean_std.csv`
- `results/aligned_matched_summary_mean_std.csv`
- `results/aligned_ablation_summary.csv`
- `results/ablation_summary_mean_std.csv`

Important paper figures:

- `figure/overview20260702.pdf`
- `figure/rq1_dataset_compression_map.pdf`
- `figure/rq1_grouped_summary.pdf`
- `figure/rq2_broad_method_tradeoff.pdf`
- `figure/rq2_baseline_fmeasure_heatmaps.pdf`
- `figure/rq2_matched_subset_quality.pdf`
- `figure/compression_performance_tradeoff.pdf`
- `figure/rq3_ablation_trajectory.pdf`
- `figure/rq5_runtime_comparison.pdf`
- `figure/causalgraph (1).png`

## Reproducing the Current Paper Results

The current paper result values can be checked from the archived CSV files in `results/`, and the exported figures are stored in `figure/`. To rerun experiments from scratch, run the experiment commands in the order above. Full reruns are computationally expensive, especially on high-dimensional datasets and repeated-seed ablations.

## Notes on Interpretation

- Higher `reduction_pct` means fewer retained features, but the paper evaluates reduction and performance jointly.
- CauFeature is positioned as a compact, causality-informed feature selection method, not as a guaranteed causal discovery method.
- Fixed-budget baselines answer whether CauFeature compares favorably to standard selectors under a common retained-feature ratio.
- Matched-dimension baselines answer whether CauFeature selects higher-quality features under the same feature count.
- SHAP/LIME/FeatureX comparisons are supplementary single-configuration comparisons and should not be interpreted as repeated-seed robustness evidence.
