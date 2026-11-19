# Run from public data

Use Bash from the unpacked package root with the GPU environment described
in `ENVIRONMENT.md`. Install the main dependencies with
`python -m pip install -r requirements-core.txt`. Install the relevant
entries in `requirements-optional.txt` for additional RQs.

```bash
export PYTHON="${PYTHON:-python}"
export WORK="${WORK:-$(mktemp -d /tmp/caufeature-public.XXXXXX)}"
export FAST30_SOURCE_ROOT="$WORK/inputs"
export FAST30_FRESH_CASES_ROOT="$WORK/core"
export CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
mkdir -p "$FAST30_SOURCE_ROOT"
sha256sum -c SHA256SUMS.txt --quiet
"$PYTHON" scripts/restore_archives.py
```

## Prepare the project CSVs and fixed splits

For each row below, download the public dataset release. Pass the ZIP or
the specified member to `scripts/prepare_public_data.py DATASET INPUT
--data-root "$FAST30_SOURCE_ROOT"`. It writes the canonical path and checks
the exact bytes against `provenance/rq1-inputs.csv`.

| Dataset key | Public input | Member or file |
| --- | --- | --- |
| BankFull | [UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) | `bank.zip` → `bank-full.csv` |
| BrainMethod | [Pinned FeatureX archive](https://raw.githubusercontent.com/aaaa-res/FeatureX/dd7d48ce3cce58d3bca98322244f09cfa3303cc5/Dataset.rar) | `Dataset/BM.csv` |
| CustomerChurn | [Pinned FeatureX archive](https://raw.githubusercontent.com/aaaa-res/FeatureX/dd7d48ce3cce58d3bca98322244f09cfa3303cc5/Dataset.rar) | `Dataset/churn.csv` |
| DiabetesHealth_sample10 | [UCI CDC Diabetes Health Indicators binary CSV](https://archive.ics.uci.edu/static/public/891/data.csv) | `data.csv` |
| DriedBean | [UCI Dry Bean](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) | `DryBeanDataset/Dry_Bean_Dataset.arff` |
| HeartDisease | [Pinned public mirror](https://raw.githubusercontent.com/vngak-ai/sit307-11.1HD-heart-stacking/752356b7d7591990fcb76017c587c7b4f0a93f84/data/heart.csv) | `heart.csv` |
| Income | [UCI Adult ZIP](https://archive.ics.uci.edu/static/public/2/adult.zip) | ZIP with both `adult.data` and `adult.test` |
| MagicGamma | [UCI MAGIC Gamma Telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope) | `magic04.data` |
| MentalStress | [Student Stress Factors](https://www.kaggle.com/datasets/rxnach/student-stress-factors-a-comprehensive-analysis) | `StressLevelDataset.csv` |
| OnlineShoppers | [UCI Online Shoppers](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) | `online_shoppers_intention.csv` |
| SmokeAlarm | [Pinned original author's CSV](https://raw.githubusercontent.com/Blatts01/ai-smokedetector/e1b9d33cbca5885e2bb68fe4467e6e6960eebf91/data/final.csv) | `final.csv` or `smoke_detection_iot.csv` |
| SoftwareQuality | [UCI TUANDROMD](https://archive.ics.uci.edu/dataset/855/tuandromd+%28tezpur+university+android+malware+dataset%29) | `TUANDROMD.csv` |
| StudentsAcademic | [UCI Students’ Dropout and Academic Success](https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success) | `data.csv` |
| ThyroidDiff | [UCI Thyroid Cancer Recurrence](https://archive.ics.uci.edu/dataset/915/differentiated+thyroid+cancer+recurrence) | `Thyroid_Diff.csv` |

For MentalStress, the generator applies the project's target recoding
`stress_level=2` to `stress_level=1` and preserves the row order.

Extract the two named CSV members from the FeatureX RAR with `bsdtar`
(the `libarchive-tools` package on Debian and Ubuntu).

```bash
curl -fL 'https://raw.githubusercontent.com/aaaa-res/FeatureX/dd7d48ce3cce58d3bca98322244f09cfa3303cc5/Dataset.rar' \
  -o "$WORK/FeatureX-Dataset.rar"
printf '%s  %s\n' '3c1f28f2cc47c59ac61957ad93edbeba38dbc2f7a3a6949d648fedade40c9cee' \
  "$WORK/FeatureX-Dataset.rar" | sha256sum -c
bsdtar -xOf "$WORK/FeatureX-Dataset.rar" Dataset/BM.csv > "$WORK/BM.csv"
bsdtar -xOf "$WORK/FeatureX-Dataset.rar" Dataset/churn.csv > "$WORK/churn.csv"
"$PYTHON" scripts/prepare_public_data.py BrainMethod "$WORK/BM.csv" --data-root "$FAST30_SOURCE_ROOT"
"$PYTHON" scripts/prepare_public_data.py CustomerChurn "$WORK/churn.csv" --data-root "$FAST30_SOURCE_ROOT"
```

The hash in `provenance/rq1-inputs.csv` applies to the project CSV after
any preprocessing. See [dataset provenance](provenance/data-lineage.md)
for archived versions and preparation rules.

### RQ1 · Common pipeline and standardized CNNs

```bash
curl -fL 'https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip' \
  -o "$WORK/online-shoppers.zip"
"$PYTHON" scripts/prepare_public_data.py OnlineShoppers "$WORK/online-shoppers.zip" \
  --data-root "$FAST30_SOURCE_ROOT"
bash scripts/run_rq1_public.sh OnlineShoppers "$FAST30_SOURCE_ROOT" "$WORK" S01
```

The launcher uses `α=0.01` for every common dataset, including
`DiabetesHealth_sample10`, `Income` and `MagicGamma`. It regenerates the
registered splits and verifies SHA-256 before running P1–P3. It leaves
`TF_NUM_INTRAOP_THREADS` unset for P1 and sets `OMP_NUM_THREADS=4`,
`TF_NUM_INTRAOP_THREADS=2` and `TF_NUM_INTEROP_THREADS=1` for the
standardized CNNs. P1 sets its own `OMP_NUM_THREADS=4`.
For all 30 seeds, pass `all` after preparing the corresponding public CSV.
Selection records go to `$WORK/core/<dataset>/<seed>/` and CNN records go
to `$WORK/scaled/<dataset>/<seed>/cases/<dataset>/<seed>/`.
Choose a fresh work directory when changing the experiment configuration.
The `selection_branch` field in the input registry identifies the old
archive source for each registered case.

### RQ1 · Optdigits

```bash
curl -fL 'https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip' \
  -o "$WORK/optdigits.zip"
"$PYTHON" scripts/prepare_optdigits.py "$WORK/optdigits.zip" \
  "$FAST30_SOURCE_ROOT/pilot_optdigits_30_20261011/dataset/Optdigits.csv"
"$PYTHON" scripts/prepare_inputs.py --experiment rq1 \
  --data-root "$FAST30_SOURCE_ROOT" --dataset Optdigits --seed S01 --write-missing-splits
"$PYTHON" experiments/paper/rq1_optdigits/run_case.py \
  --data-root "$FAST30_SOURCE_ROOT" --seed S01 --output "$FAST30_FRESH_CASES_ROOT"
```

The Optdigits launcher generates its own manifest and writes its two
standardized CNN records inside the case directory. Run it separately for
each requested seed after generating the corresponding split.
For all 30 splits, omit `--seed` from `scripts/prepare_inputs.py`.
Then run the frozen launcher for S01–S30.

### Table 3 and RQ6 Spearman analysis

After generating the RQ1 cases and both standardized CNN evaluations, run
the summary entrypoint. It reads the fresh per-seed records and writes the
Table 3 statistics and RQ6 correlations to the specified JSON file.

```bash
"$PYTHON" experiments/analysis/reporting/summarize_main_results.py \
  --case-root "$WORK/core" --scaled-root "$WORK/scaled" \
  --output "$WORK/main-summary.json"
```

The command defaults to S01–S30 for each dataset found in `$WORK/core`.
For the full paper scope, generate all 15 datasets before aggregating. A
limited selection can be requested with `--datasets` and `--seeds`, for
example `--datasets OnlineShoppers --seeds S01` for a one-case smoke test.

### RQ2 · Native baselines

```bash
export RQ2_MANIFEST="$WORK/rq2-manifest.json"
export RQ2_OUTPUT_ROOT="$WORK/rq2"
"$PYTHON" scripts/prepare_inputs.py --experiment rq2 --data-root "$FAST30_SOURCE_ROOT" \
  --dataset OnlineShoppers --seed S01 --write-missing-splits --rq2-manifest "$RQ2_MANIFEST"
"$PYTHON" experiments/paper/rq2/orchestrate.py \
  --dataset OnlineShoppers --seed S01 --methods PPFS --workers 1
```

SHAP, LIME, FeatureX and PPFS use the bundled code and Python dependencies.
BAMB-Z and EAMB-Z additionally need GNU Octave with Statistics. On Debian
or Ubuntu the system packages are `octave` and `octave-statistics`. Use
these vendor revisions.

```bash
export RQ2_THIRD_PARTY_ROOT="$WORK/vendor"
mkdir -p "$RQ2_THIRD_PARTY_ROOT"
git clone https://github.com/kuiy/CausalLearner.git "$RQ2_THIRD_PARTY_ROOT/CausalLearner"
git -C "$RQ2_THIRD_PARTY_ROOT/CausalLearner" checkout --detach 429d13403be22a5df68631135b851a9acb05a513
unzip -q "$RQ2_THIRD_PARTY_ROOT/CausalLearner/Causal_learner.zip" -d "$RQ2_THIRD_PARTY_ROOT/CausalLearner/source"
git clone https://github.com/Xianjie-Guo/EAMB.git "$RQ2_THIRD_PARTY_ROOT/EAMB"
git -C "$RQ2_THIRD_PARTY_ROOT/EAMB" checkout --detach 9df90ff9687e00f16db28a7efecae74cb7400e5e
```

To enable the paper's identity-reuse rule with fresh RQ1 evaluations, use
`export RQ2_ORIGINAL_EVAL_ROOT="$WORK/scaled/OnlineShoppers/S01/cases"` for
the common dataset and seed in the example above,
or `export RQ2_ORIGINAL_EVAL_ROOT="$FAST30_FRESH_CASES_ROOT"` for Optdigits. The RQ2
worker checks the ordered feature list and split before reusing a record.

### RQ3 · Equal feature budgets

Run RQ1 first. The following uses the number of features in that seed's
fresh `record.json` as the fixed budget and the split from its P1 cache.

```bash
FAST30_MATCHED_OUTPUT_ROOT="$WORK/rq3" \
  "$PYTHON" experiments/main/run_5ds_s30_trainonly_20261011.py \
  --datasets OnlineShoppers --seeds S01 --methods MI --modes scaled
```

### RQ4 · Real-data ablation and Deep3 inputs

```bash
export RQ4_OUTPUT_ROOT="$WORK/rq4"
"$PYTHON" experiments/ablation/run_worker.py prepare OnlineShoppers S01
"$PYTHON" experiments/ablation/run_worker.py policy OnlineShoppers S01 all_pairs
"$PYTHON" experiments/ablation/run_worker.py evaluate OnlineShoppers S01 p3_all_pairs
```

`prepare` builds the case manifest from the fresh RQ1 record, P1 cache,
selector model and development CSV. The evaluation trains the bundled
standardized CNN on the generated feature subset.

```bash
"$PYTHON" scripts/prepare_deep3.py --dimension 20 --seed 1 --output "$WORK/deep3-d20-s01"
```

The Deep3 generator writes the synthetic data, split, graph truth and
manifest. The archived Deep3 audit in the README requires the original
per-seed path sidecars, run records and frozen runtime metadata. The
component-aligned Table 8 replay uses its own per-variant records.
The full historical ablation scheduler also uses the frozen study records
under `results/ablation_five_20261028/`, including the SoftwareQuality
global-PC kernel verification inputs.

### RQ5 · Screening level and perturbation supports

```bash
"$PYTHON" experiments/paper/rq5_alpha/run_case.py OnlineShoppers S01 \
  --alpha 0.005 --output "$WORK/rq5-alpha"
```

The alpha runner accepts `0.005`, `0.01` and `0.05`. For the default-alpha
comparison, use a fresh RQ1 case run with `--alpha 0.01`. Leave the GPU
visible for the fresh CNN evaluation.

```bash
export RQ5_CASES_ROOT="$WORK/rq5e-inputs"
export RQ5_OUTPUT_ROOT="$WORK/rq5e"
"$PYTHON" experiments/paper/rq5_fig3e/prepare_supports.py OnlineShoppers S01
CUDA_VISIBLE_DEVICES=-1 "$PYTHON" experiments/paper/rq5_fig3e/equal_budget_max_js.py OnlineShoppers S01
```

For the complete panel, generate RQ1 cases and both support grids for the
five study datasets across S01–S30. Then run `equal_budget_max_js.py` without
case arguments, followed by these commands.

```bash
"$PYTHON" experiments/paper/rq5_fig3e/summarize_equal_budget_max_js.py
"$PYTHON" experiments/paper/rq5_fig3e/plot_panel.py --summary "$RQ5_OUTPUT_ROOT/summary.json"
```

The remaining sensitivity panels use the archived case inputs and variant
records in `results/sensitivity_five_20261028/`. The historical bootstrap
uses `results/bootstrap_repeatability_full16_20261112/` and the corresponding
P1 model and development inputs.

### RQ6 · Stability and frozen scorer comparison

Aggregate any completed seed subset of size two or greater. For Table 10,
generate all 30 seed records per dataset and omit `--seeds`. The Table 3
summary entrypoint above checks the ordered final features against the
per-seed standardized CNN files.

```bash
"$PYTHON" experiments/analysis/case_studies/stability/analyze_stability.py \
  --case-root "$FAST30_FRESH_CASES_ROOT" --datasets OnlineShoppers --seeds S01 S02 \
  --output "$WORK/rq6-stability.csv"
```

### RQ6 · Regenerate the fixed CNN study (Table 9)

Use the pinned environment in `requirements-core.txt` and `ENVIRONMENT.md`,
including a TensorFlow-visible GPU. Start with the public-data setup and
the RQ1 OnlineShoppers S01 example above. Its fresh case provides the
original CNN scorer recipe's checkpoint, development CSV and P1–P3 caches,
plus the two standardized CNN reference evaluations.

```bash
export TABLE9_OUT="$WORK/rq6-table9"
"$PYTHON" experiments/paper/rq6_fixed_cnn/run_fresh_case.py OnlineShoppers S01 \
  --run-root "$WORK" --output "$TABLE9_OUT" --scorers RF
"$PYTHON" experiments/paper/rq6_fixed_cnn/summarize.py \
  --case-root "$TABLE9_OUT/cases" --output "$WORK/table9-s01.csv" \
  --datasets OnlineShoppers --seeds S01 --scorers RF
```

The first command regenerates the shared unscaled scorer inputs and the RF
checkpoint, restores the original contribution and reliability calculation,
and rebuilds the shared shallow P2 state at `α=0.01`. It runs the bundled
restricted P2/P3 and checks the regenerated CNN reference before evaluating
the alternative subset with the independent standardized CNN. It can reuse
an evaluation on the same ordered features and split. The work directory
contains all newly generated artifacts. Use a new output directory to retry
an interrupted case.

For the full table, prepare the RQ1 cases for BankFull, OnlineShoppers,
StudentsAcademic, Optdigits and SoftwareQuality, each for S01–S30. Run
`run_fresh_case.py DATASET SEED --run-root "$WORK" --output "$TABLE9_OUT"`
for each case. Omitting `--scorers` regenerates both MLP and RF. Run
`summarize.py --case-root "$TABLE9_OUT/cases" --output "$WORK/table9.csv"`
with the default dataset, seed and scorer scope.

The archive replay in the README uses the original byte-locked files listed
in `experiments/paper/rq6_fixed_cnn/protocol.lock.json`, including the shared
P1 arrays, checkpoints, manifests, P2 states and paper CNN records. The
public-data command regenerates these inputs in its own workspace.

### RQ7 · OnlineShoppers explanation

Run the RQ1 OnlineShoppers S01 case at α=0.01, then prepare the full-control
P2 state and original P3 policy in the same RQ4 output root.

```bash
export RQ4_OUTPUT_ROOT="$WORK/rq4"
CUDA_VISIBLE_DEVICES=-1 "$PYTHON" experiments/ablation/run_worker.py shallow OnlineShoppers S01
CUDA_VISIBLE_DEVICES=-1 "$PYTHON" experiments/ablation/run_worker.py deep OnlineShoppers S01 full_control
"$PYTHON" experiments/ablation/run_worker.py policy OnlineShoppers S01 original
export RQ7_OUTPUT_ROOT="$WORK/rq7"
CUDA_VISIBLE_DEVICES=-1 "$PYTHON" experiments/paper/rq7_onlineshoppers/export_evidence.py --ablation-root "$RQ4_OUTPUT_ROOT"
CUDA_VISIBLE_DEVICES=-1 "$PYTHON" experiments/paper/rq7_onlineshoppers/plot.py
```

The plotting entry uses the paper case's graph and feature counts. The
archive replay takes the original P1, shallow P2, main record and P3
witness files as inputs.

### RQ8 · Local stage timings

```bash
"$PYTHON" experiments/analysis/reporting/aggregate_runtime_table.py \
  --results-root "$FAST30_FRESH_CASES_ROOT" --datasets OnlineShoppers --expected-seeds 1 \
  --output "$WORK/rq8-local.csv"
```

Use 30 completed seed records for the full per-dataset aggregation.
The original elapsed times are read from the paper run's archived records.
The historical comparison script in the README requires the external
reference records and archive-mode output layout.
