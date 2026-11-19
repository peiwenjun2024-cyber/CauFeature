#!/usr/bin/env bash
# Run the common RQ1 pipeline against a prepared, hash-locked public dataset.
set -euo pipefail

if (( $# < 3 || $# > 4 )); then
    echo 'Usage: bash scripts/run_rq1_public.sh DATASET INPUT_ROOT RUN_ROOT [S01..S30|all]' >&2
    exit 2
fi

dataset=$1
seed=${4:-S01}
if [[ ! $dataset =~ ^[A-Za-z][A-Za-z0-9_]*$ || $dataset == Optdigits ]]; then
    echo 'Use a common RQ1 dataset key. Optdigits uses its separate frozen launcher.' >&2
    exit 2
fi
if [[ $seed != all && ! $seed =~ ^S(0[1-9]|[12][0-9]|30)$ ]]; then
    echo 'Choose a seed from S01 through S30, or all.' >&2
    exit 2
fi

package=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
input_root=$(realpath -e -- "$2")
run_root=$(realpath -m -- "$3")
if [[ ! -d $input_root || $run_root == / || "$run_root/" == "$input_root/"* || "$run_root/" == "$package/"* ]]; then
    echo 'Use an existing input root and a run root outside the input and package directories.' >&2
    exit 2
fi
python=$(command -v -- "${PYTHON:-python}")
[[ $python == /* ]] || python="$PWD/$python"

export FAST30_SOURCE_ROOT="$input_root"
export FAST30_DATASET_DIR="$input_root/code_revised_20260910/dataset"
export FAST30_FRESH_CASES_ROOT="$run_root/core"
export FAST30_OUTPUT_ROOT="$FAST30_FRESH_CASES_ROOT"
export FAST30_BASELINE_CACHE="$run_root/baseline-cache"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_NUM_INTEROP_THREADS=1

mkdir -p -- "$run_root/work"
scratch=$(mktemp -d "$run_root/work/${dataset}.XXXXXX")
export FAST30_MANIFEST="$scratch/splits.jsonl"
seed_filter=()
seeds=("$seed")
if [[ $seed == all ]]; then
    seeds=(S{01..30})
else
    seed_filter=(--seed "$seed")
fi
"$python" "$package/scripts/prepare_inputs.py" --experiment rq1 \
    --data-root "$input_root" --dataset "$dataset" "${seed_filter[@]}" \
    --check-regeneration --write-missing-splits --rq1-manifest "$FAST30_MANIFEST"

for seed_id in "${seeds[@]}"; do
    case_work="$scratch/$seed_id"
    mkdir -p -- "$case_work"
    (
        cd -- "$case_work"
        env -u TF_NUM_INTRAOP_THREADS "$python" "$package/CauFeature/runner.py" \
            --dataset "$dataset.csv" --seed "${seed_id#S}" --alpha 0.01
        OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=2 EVAL_SHARDS=1 EVAL_SHARD_ID=0 \
            FAST30_EVAL_OUTPUT_ROOT="$run_root/scaled/$dataset/$seed_id" \
            FAST30_EVAL_DATASET="$dataset" FAST30_EVAL_SEED="$seed_id" \
            "$python" "$package/experiments/main/run_scaled_evaluator_full_vs_selected.py"
    )
done
