# CauFeature experiment reproduction

Run these commands with Bash from the unpacked package root. Use the original
AutoDL environment for the frozen RF/CNN evaluation. On a new environment,
install the dependencies for the entrypoints you intend to run from
`requirements-core.txt` and `requirements-optional.txt`.

See [Run from public data](PUBLIC_DATA.md) for the input generators and
commands for running the main experiment at `α=0.01` and summarizing
Table 3 and the RQ6 Spearman analysis. The commands below use archived
inputs and reference records.

```bash
# setup
export PACKAGE_ROOT="$PWD"
export FAST30_SOURCE_ROOT="${FAST30_SOURCE_ROOT:-/root/autodl-tmp/fast30_final}"
export SMOKE_OUT="${SMOKE_OUT:-$(mktemp -d /tmp/caufeature-smoke.XXXXXX)}"
export PYTHON="${PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1 TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
mkdir -p "$SMOKE_OUT"
sha256sum -c SHA256SUMS.txt --quiet
"$PYTHON" scripts/restore_archives.py
```

## RQ1 · main pipeline, paper evaluator and Optdigits

The first command needs the original data and `manifest/split_manifest.jsonl`.
The paper evaluator also reads the archived official case manifest
`results/alpha001_full30_20261009/official_main_results_15x30_manifest.csv`,
its per-seed selection record, and the matching P1 cache under
`results/fast30_semanticperturb_flat_v6_20260930/`.

```bash
# rq1-core
mkdir -p "$SMOKE_OUT/rq1-core-work"
(
  cd "$SMOKE_OUT/rq1-core-work"
  env -u TF_NUM_INTRAOP_THREADS \
    FAST30_MANIFEST="$FAST30_SOURCE_ROOT/manifest/split_manifest.jsonl" \
    FAST30_DATASET_DIR="$FAST30_SOURCE_ROOT/code_revised_20260910/dataset" \
    FAST30_BASELINE_CACHE="$SMOKE_OUT/rq1-baseline-cache" \
    FAST30_OUTPUT_ROOT="$SMOKE_OUT/rq1-core" \
    "$PYTHON" "$PACKAGE_ROOT/CauFeature/runner.py" --dataset ThyroidDiff.csv --seed 4 --alpha 0.01
)
```

```bash
# rq1
OMP_NUM_THREADS=4 FAST30_EVAL_OUTPUT_ROOT="$SMOKE_OUT/rq1" \
  FAST30_EVAL_DATASET=ThyroidDiff FAST30_EVAL_SEED=S04 \
  "$PYTHON" experiments/main/run_scaled_evaluator_full_vs_selected.py
```

Optdigits needs `pilot_optdigits_30_20261011/dataset/Optdigits.csv` and
`pilot_optdigits_30_20261011/splits/Optdigits_S01.csv` under the input root.
The seed-specific launcher verifies both files against `provenance/rq1-inputs.csv`.
To reconstruct a missing fixed split, run `scripts/prepare_inputs.py` with
`--experiment rq1 --dataset Optdigits --seed S01 --write-missing-splits` and
`--data-root` pointing at an input working copy. The project CSV can be generated
with `python scripts/prepare_optdigits.py optdigits.zip DESTINATION` using the
[original UCI archive](https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip).

```bash
# optdigits
"$PYTHON" experiments/paper/rq1_optdigits/run_case.py --seed S01 --output "$SMOKE_OUT/rq1-optdigits"
```

## RQ2 · native baselines

The input registry checks the exact raw dataset and split bytes. Supply the
files at the paths in `provenance/rq2-inputs.csv`. The PPFS example needs
PyImpetus. `--methods` accepts `SHAP`, `LIME`, `FeatureX`, `PPFS`, `BAMB-Z` and
`EAMB-Z`. BAMB-Z and EAMB-Z also need GNU Octave with Statistics and the
original `CausalLearner/source/Causal_Learner` and `EAMB/EAMB-code (matlab)`
trees under `RQ2_THIRD_PARTY_ROOT`. The default vendor root is
`$FAST30_SOURCE_ROOT/code_revised_20260910/third_party`.

```bash
# rq2
export RQ2_MANIFEST="$SMOKE_OUT/rq2-manifest.json"
export RQ2_OUTPUT_ROOT="$SMOKE_OUT/rq2"
"$PYTHON" scripts/prepare_inputs.py --data-root "$FAST30_SOURCE_ROOT" \
  --dataset OnlineShoppers --seed S01 --rq2-manifest "$RQ2_MANIFEST"
"$PYTHON" experiments/paper/rq2/orchestrate.py \
  --dataset OnlineShoppers --seed S01 --methods PPFS --workers 1
```

## RQ3 · matched feature budgets

This needs the RQ1 official case manifest, the recorded per-seed selection
budget and P1 split metadata, and the original raw data and split CSVs.

```bash
# rq3
FAST30_MATCHED_OUTPUT_ROOT="$SMOKE_OUT/rq3" \
  "$PYTHON" experiments/main/run_5ds_s30_trainonly_20261011.py \
  --datasets OnlineShoppers --seeds S01 --methods MI --modes scaled
```

## RQ4 · ablation and known-structure audit

The ablation runner needs `results/ablation_five_20261028/cases/OnlineShoppers/S01/manifest.json`,
its original P1 cache, scoring model, development CSV and main records.
The evaluation can also require the original standardized CNN evaluator.

```bash
# rq4
export RQ4_OUTPUT_ROOT="$SMOKE_OUT/rq4"
"$PYTHON" experiments/ablation/run_worker.py policy OnlineShoppers S01 all_pairs
"$PYTHON" experiments/ablation/run_worker.py evaluate OnlineShoppers S01 p3_all_pairs
```

The Deep3 audit reads the original synthetic datasets, splits, truth records
and per-seed artifacts under `RQ4_EXPERIMENT_HUB_20261014/deep3_extension_20261015/`.

```bash
# rq4-deep3
RQ4_DEEP3_AUDIT_OUT="$SMOKE_OUT/rq4-deep3" \
  "$PYTHON" historical/fast30_final/RQ4_EXPERIMENT_HUB_20261014/reports/paper_s4_validation_20261015/audit_and_summarize.py
```

## RQ5 · screening level and equal-budget perturbations

Figure 3(c) needs the original per-case manifest under
`results/sensitivity_five_20261028/cases/` and the original data CSV, split
CSV, P1 cache and selector development CSV it references. Keep the original
ablation evaluation records available for ordered-feature matches.

```bash
# rq5-alpha
CUDA_VISIBLE_DEVICES=-1 "$PYTHON" experiments/paper/rq5_alpha/run_case.py \
  BankFull S01 --alpha 0.005 --output "$SMOKE_OUT/rq5-alpha"
```

Figure 3(e) needs the original case manifests, P1 cache, selector model and
development CSV, and the archived quantile and uniform supports. The plot
command also reads the external full-grid summary from the original campaign.

```bash
# rq5-e
export RQ5_OUTPUT_ROOT="$SMOKE_OUT/rq5-e"
export RQ5_JS_ARCHIVE="${RQ5_JS_ARCHIVE:-/root/autodl-fs/multibranch_ablation_20261113/equal_budget_max_js_20261114}"
CUDA_VISIBLE_DEVICES=-1 "$PYTHON" experiments/paper/rq5_fig3e/equal_budget_max_js.py OnlineShoppers S01
CUDA_VISIBLE_DEVICES=-1 "$PYTHON" experiments/paper/rq5_fig3e/plot_panel.py --summary "$RQ5_JS_ARCHIVE/summary.json"
```

## RQ6 · fixed CNN and selection stability

For a new Table 9 run, see the [public-data instructions](PUBLIC_DATA.md#rq6--regenerate-the-fixed-cnn-study-table-9).
The archive replay below needs the original AutoDL path
`/root/autodl-tmp/fast30_final`, its archived RF scorer and shared P1 state,
the original split/data and the shallow P2 state. The full list of absolute
input paths, hashes and required Python package versions is in
`experiments/paper/rq6_fixed_cnn/protocol.lock.json`.

```bash
# rq6
RQ6_OUTPUT_ROOT="$SMOKE_OUT/rq6" \
  "$PYTHON" experiments/paper/rq6_fixed_cnn/run_case.py OnlineShoppers S01 --scorer RF
```

Table 10 reads archived main selection records and the corresponding P1 and
scaled CNN records. The selected CNN records are at
`results/scaled_evaluator_full_vs_selected_20261010/cases/<dataset>/<seed>/selected_scaled.json`.
For Optdigits, `record.json`, `p1_cache.json` and `selected_scaled.json` are
under `results/alpha001_full30_20261009/Optdigits/<seed>/`. The dataset filter
selects which archived seed subsets to aggregate.

```bash
# rq6-stability
"$PYTHON" experiments/analysis/case_studies/stability/analyze_stability.py \
  --source-root "$FAST30_SOURCE_ROOT" --datasets OnlineShoppers --output "$SMOKE_OUT/rq6-stability.csv"
```

## RQ7 · explanation evidence

This entry reads the original OnlineShoppers S01 data, split, P1 scores,
frozen shallow P2 state and P3 witnesses. The original evidence file is
`results/rq7_onlineshoppers_s01_20261110/evidence.json`.
The contribution and dependency PNGs map to the manuscript's
`fig3_recovery_threshold.png` and `fig3b_numbered_matrix.png`, respectively.

```bash
# rq7
export RQ7_OUTPUT_ROOT="$SMOKE_OUT/rq7"
CUDA_VISIBLE_DEVICES=-1 "$PYTHON" experiments/paper/rq7_onlineshoppers/export_evidence.py
CUDA_VISIBLE_DEVICES=-1 "$PYTHON" experiments/paper/rq7_onlineshoppers/plot.py
```

## RQ8 · archived runtime aggregation

This entry reads the saved `timing_seconds` fields in the corresponding
`results/alpha001_full30_20261009/<dataset>/<seed>/record.json`.

```bash
# rq8
"$PYTHON" experiments/analysis/reporting/aggregate_runtime_table.py \
  --results-root "$FAST30_SOURCE_ROOT/results/alpha001_full30_20261009" \
  --datasets SoftwareQuality --expected-seeds 1 --output "$SMOKE_OUT/rq8.csv"
```

## Verify the smoke outputs

The verifier reads the original per-case and aggregate references under the
same source root, including `2026_peiwenjun/statistics/`. For Fig. 3(e), supply
`--js-source` when the original equal-budget archive is mounted elsewhere.
Use `--checks` to select the examples you ran. Omit it to check every example
above. The verifier writes to standard output. Exit status is 0 for matches,
1 for a missing output, failed run or structural difference, and 2 for a
score difference with matching structural fields.

```bash
# verify
"$PYTHON" scripts/verify_paper_smoke.py --source "$FAST30_SOURCE_ROOT" --output "$SMOKE_OUT"
```
