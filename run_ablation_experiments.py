import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def _patch_training_epochs(max_epochs: int | None) -> None:
    if max_epochs is None:
        return

    import model_train_v6

    original_train_model = model_train_v6.train_model

    def train_model_with_epoch_limit(*args, **kwargs):
        kwargs["epochs"] = min(int(kwargs.get("epochs", 100)), max_epochs)
        kwargs["early_stopping_patience"] = min(int(kwargs.get("early_stopping_patience", 10)), max(2, max_epochs // 4))
        return original_train_model(*args, **kwargs)

    model_train_v6.train_model = train_model_with_epoch_limit


def _evaluate_subset(
    data_path: Path,
    seed: int,
    features: list[str] | None,
    original_accuracy: float,
    original_fmeasure: float,
    base_state: dict,
) -> tuple[float, float]:
    import shared_globals
    from model_train_v6 import train

    shared_globals.data = base_state["data"].copy()
    shared_globals.feature_data = base_state["feature_data"].copy()
    shared_globals.feature_names = list(base_state["feature_names"])
    shared_globals.target_name = list(base_state["target_name"])
    shared_globals.all_names = list(base_state["all_names"])
    shared_globals.sample_num = base_state["sample_num"]
    shared_globals.model = None
    shared_globals.filtered_features = features
    shared_globals.random_seed = seed
    shared_globals.accuracy = original_accuracy
    shared_globals.f_score = original_fmeasure
    train(str(data_path), features)
    return shared_globals.accuracy, shared_globals.f_score


def run_ablation_once(data_path: Path, seed: int, *, max_epochs: int | None = None) -> list[dict]:
    import numpy as np
    from scipy.stats import spearmanr

    import shared_globals
    from causal_graph_build_v19_english import Feature, build_causal_graph
    from feature_perturbation_v7_english import CausalFeaturePerturbation
    from model_train_v6 import train
    from redundantRecover_v10_english import PathShapleyModule

    _patch_training_epochs(max_epochs)
    shared_globals.init()
    shared_globals.random_seed = seed

    start_total = time.perf_counter()
    model = train(str(data_path), None)
    shared_globals.model = model
    original_dim = shared_globals.feature_data.shape[1]
    original_accuracy = shared_globals.accuracy
    original_fmeasure = shared_globals.f_score
    base_state = {
        "data": shared_globals.data.copy(),
        "feature_data": shared_globals.feature_data.copy(),
        "feature_names": list(shared_globals.feature_names),
        "target_name": list(shared_globals.target_name),
        "all_names": list(shared_globals.all_names),
        "sample_num": shared_globals.sample_num,
    }
    original_feature_names = list(shared_globals.feature_names)

    perturb = CausalFeaturePerturbation(shared_globals.model, shared_globals.data)
    perturb.run_perturbation()
    perturb.initialize_original_preds(shared_globals.data, shared_globals.model)

    features = [
        Feature(name=name, feature_names=shared_globals.feature_names, all_con_values=shared_globals.con_list)
        for name in shared_globals.all_names
    ]
    phase1_features = sorted(
        feature.name
        for feature in features
        if feature.name in shared_globals.feature_names and getattr(feature, "is_high_contrib", False)
    )
    if not phase1_features:
        phase1_features = [name for name, _ in shared_globals.feature_max_abs[: max(1, original_dim // 2)]]

    is_numeric = all(feature.data.dtype.kind in "iufc" for feature in features)
    if is_numeric:
        shared_globals.corr_matrix = np.corrcoef([feature.data for feature in features])
    else:
        shared_globals.corr_matrix, _ = spearmanr([feature.data for feature in features])

    valid_paths, _, _ = build_causal_graph(
        features,
        shared_globals.corr_matrix,
        shared_globals.target_name,
        is_numeric,
    )
    phase12_features = sorted(
        name for name in (shared_globals.filtered_features or [])
        if name in original_feature_names
    )

    shapley_module = PathShapleyModule(log_verbose=False, decay_type="exponential", alpha=0.7)
    shapley_module.run_redundant_check()
    full_features = sorted(
        name for name in (shared_globals.filtered_features or [])
        if name in original_feature_names
    )

    variants = [
        ("Original", None, original_dim, original_accuracy, original_fmeasure),
        ("Phase 1", phase1_features, len(phase1_features), None, None),
        ("Phase 1+2", phase12_features, len(phase12_features), None, None),
        ("Full", full_features, len(full_features), None, None),
    ]

    rows = []
    for variant, selected_features, selected_dim, acc, fmeasure in variants:
        if acc is None or fmeasure is None:
            delta_acc, delta_fmeasure = _evaluate_subset(
                data_path,
                seed,
                selected_features,
                original_accuracy,
                original_fmeasure,
                base_state,
            )
            acc = original_accuracy + delta_acc / 100.0
            fmeasure = original_fmeasure + delta_fmeasure / 100.0
        rows.append(
            {
                "dataset": data_path.name,
                "seed": seed,
                "variant": variant,
                "selected_dim": selected_dim,
                "reduction_pct": 100.0 * (original_dim - selected_dim) / original_dim if original_dim else 0.0,
                "accuracy": acc,
                "fmeasure": fmeasure,
                "accuracy_delta_pct_points": (acc - original_accuracy) * 100.0,
                "fmeasure_delta_pct_points": (fmeasure - original_fmeasure) * 100.0,
                "valid_path_count": len(valid_paths),
                "total_time_sec": time.perf_counter() - start_total,
                "selected_features": "" if selected_features is None else ";".join(selected_features),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated-seed CauFeature ablation experiments.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--datasets", nargs="+", default=["MagicGamma.csv", "BrainMethod.csv"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", default="results/ablation_runs.csv")
    parser.add_argument("--epochs", type=int, default=None, help="Optional maximum epochs for quick validation runs.")
    parser.add_argument("--resume", action="store_true", help="Skip dataset/seed pairs already present in the output CSV.")
    parser.add_argument("--one-shot", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def append_rows(output_path: Path, rows: list[dict]) -> None:
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with output_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
        fh.flush()


def completed_pairs(output_path: Path) -> set[tuple[str, int]]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    done: set[tuple[str, int]] = set()
    with output_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("variant") == "Full":
                done.add((row["dataset"], int(row["seed"])))
    return done


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(args.dataset_dir)
    done = completed_pairs(output_path) if args.resume else set()

    if not args.one_shot and (len(args.datasets) * len(args.seeds) > 1):
        for dataset in args.datasets:
            data_path = dataset_dir / dataset
            if not data_path.exists():
                print(f"Skipping missing dataset: {data_path}")
                continue
            for seed in args.seeds:
                if (dataset, seed) in done:
                    print(f"Skipping completed ablation: dataset={dataset}, seed={seed}")
                    continue
                cmd = [
                    sys.executable,
                    __file__,
                    "--dataset-dir",
                    str(dataset_dir),
                    "--datasets",
                    dataset,
                    "--seeds",
                    str(seed),
                    "--output",
                    str(output_path),
                    "--one-shot",
                ]
                if args.epochs is not None:
                    cmd.extend(["--epochs", str(args.epochs)])
                if args.resume:
                    cmd.append("--resume")
                print(f"Launching isolated ablation process: dataset={dataset}, seed={seed}")
                subprocess.run(cmd, check=True)
        return

    for dataset in args.datasets:
        data_path = dataset_dir / dataset
        if not data_path.exists():
            print(f"Skipping missing dataset: {data_path}")
            continue
        for seed in args.seeds:
            if (dataset, seed) in done:
                print(f"Skipping completed ablation: dataset={dataset}, seed={seed}")
                continue
            print(f"Running ablation: dataset={dataset}, seed={seed}")
            rows = run_ablation_once(data_path, seed, max_epochs=args.epochs)
            append_rows(output_path, rows)
            print(f"Appended {len(rows)} ablation rows to {output_path}")
            if args.one_shot:
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(0)


if __name__ == "__main__":
    main()
