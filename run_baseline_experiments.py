import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


DEFAULT_DATASETS = [
    "MagicGamma.csv",
    "BrainMethod.csv",
    "SoftwareQuality.csv",
]

DEFAULT_METHODS = ["VT", "MI", "CHI", "RFE", "RFC"]


def _encode_features_and_target(data_path: Path):
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    try:
        df = pd.read_csv(data_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="gbk")
    x = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()

    for col in x.select_dtypes(exclude=["number"]).columns:
        x[col] = LabelEncoder().fit_transform(x[col].astype(str))
    if x.isnull().any().any():
        x = x.fillna(x.mean(numeric_only=True))

    if not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y.astype(str))

    return x, y


def _select_features(
    method: str,
    x,
    y,
    retention: float,
    seed: int,
    target_count: Optional[int] = None,
) -> tuple[List[str], str]:
    import numpy as np

    method = method.upper()
    feature_names = x.columns.tolist()
    if target_count is None:
        target_count = int(round(len(feature_names) * retention))
        rule_suffix = f"retention={retention:.2f}"
    else:
        rule_suffix = "matched_k"
    target_count = max(1, min(len(feature_names), target_count))

    if method == "VT":
        variances = x.var(numeric_only=True).sort_values(ascending=False)
        selected = variances.index[:target_count].tolist()
        return selected, f"top variance features, {rule_suffix}, k={target_count}"

    if method == "MI":
        from sklearn.feature_selection import SelectKBest, mutual_info_classif

        selector = SelectKBest(mutual_info_classif, k=target_count)
        selector.fit(x, y)
        selected = [name for name, keep in zip(feature_names, selector.get_support()) if keep]
        return selected, f"SelectKBest(mutual_info_classif), {rule_suffix}, k={target_count}"

    if method == "CHI":
        from sklearn.feature_selection import SelectKBest, chi2
        from sklearn.preprocessing import MinMaxScaler

        x_non_negative = MinMaxScaler().fit_transform(x)
        selector = SelectKBest(chi2, k=target_count)
        selector.fit(x_non_negative, y)
        selected = [name for name, keep in zip(feature_names, selector.get_support()) if keep]
        return selected, f"SelectKBest(chi2 after MinMaxScaler), {rule_suffix}, k={target_count}"

    if method == "RFE":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.feature_selection import RFE

        estimator = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        selector = RFE(estimator=estimator, n_features_to_select=target_count, step=1)
        selector.fit(x, y)
        selected = [name for name, keep in zip(feature_names, selector.get_support()) if keep]
        return selected, f"RFE(RandomForestClassifier), {rule_suffix}, k={target_count}"

    if method == "RFC":
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        model.fit(x, y)
        ranking = np.argsort(model.feature_importances_)[::-1]
        selected = [feature_names[i] for i in ranking[:target_count]]
        return selected, f"RandomForest feature importance top-k, {rule_suffix}, k={target_count}"

    raise ValueError(f"Unsupported baseline method: {method}")


def _load_original_metrics(metrics_csv: Optional[Path]) -> Dict[Tuple[str, int], Tuple[float, float]]:
    if metrics_csv is None:
        return {}
    metrics: Dict[Tuple[str, int], Tuple[float, float]] = {}
    with metrics_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            seed = int(row["seed"])
            values = (
                float(row["original_accuracy"]),
                float(row["original_f1"]),
            )
            metrics[(row["dataset"], seed)] = values
            if row.get("run_dataset"):
                metrics[(row["run_dataset"], seed)] = values
    return metrics


def _load_matched_dimensions(dimensions_csv: Optional[Path]) -> Dict[Tuple[str, int], int]:
    if dimensions_csv is None:
        return {}
    dimensions: Dict[Tuple[str, int], int] = {}
    with dimensions_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            seed = int(row["seed"])
            selected_dim = int(round(float(row["selected_dim"])))
            dimensions[(row["dataset"], seed)] = selected_dim
            if row.get("run_dataset"):
                dimensions[(row["run_dataset"], seed)] = selected_dim
    return dimensions


def run_baseline_once(
    data_path: Path,
    method: str,
    seed: int,
    retention: float,
    original_metrics: Optional[Tuple[float, float]] = None,
    target_count: Optional[int] = None,
) -> Dict[str, object]:
    import shared_globals
    from model_train_v6 import load_and_preprocess_data, train

    x, y = _encode_features_and_target(data_path)
    selected_features, rule = _select_features(method, x, y, retention, seed, target_count)

    shared_globals.init()
    shared_globals.random_seed = seed

    start_total = time.perf_counter()
    if original_metrics is None:
        train(str(data_path), None)
        original_dim = shared_globals.feature_data.shape[1]
        original_accuracy = shared_globals.accuracy
        original_f1 = shared_globals.f_score
    else:
        load_and_preprocess_data(str(data_path), None)
        original_dim = shared_globals.feature_data.shape[1]
        original_accuracy, original_f1 = original_metrics
        shared_globals.accuracy = original_accuracy
        shared_globals.f_score = original_f1

    shared_globals.model = None
    shared_globals.filtered_features = selected_features
    train(str(data_path), selected_features)
    total_time = time.perf_counter() - start_total

    return {
        "dataset": data_path.name,
        "method": method.upper(),
        "seed": seed,
        "retention": retention,
        "matched_target_dim": target_count if target_count is not None else "",
        "selection_rule": rule,
        "original_dim": original_dim,
        "selected_dim": len(selected_features),
        "reduction_pct": 100.0 * (original_dim - len(selected_features)) / original_dim if original_dim else 0.0,
        "original_accuracy": original_accuracy,
        "selected_accuracy": original_accuracy + shared_globals.accuracy / 100.0,
        "accuracy_delta_pct_points": shared_globals.accuracy,
        "original_f1": original_f1,
        "selected_f1": original_f1 + shared_globals.f_score / 100.0,
        "f1_delta_pct_points": shared_globals.f_score,
        "total_time_sec": total_time,
        "selected_features": ";".join(selected_features),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run traditional feature-selection baselines through the shared CNN evaluation pipeline."
    )
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing CSV datasets.")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS, help="Dataset file names.")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS, help="Baseline methods: VT MI CHI RFE RFC.")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42], help="Random seeds to evaluate.")
    parser.add_argument(
        "--retention",
        type=float,
        default=0.75,
        help="Fraction of original features retained by fixed top-k baselines.",
    )
    parser.add_argument("--output", default="results/baseline_runs.csv", help="Output CSV path.")
    parser.add_argument(
        "--seeded-dataset-pattern",
        default=None,
        help=(
            "Optional dataset filename pattern used to select a different CSV for each seed, "
            "for example results/prefiltered_datasets/SoftwareQuality_prefilter_top50_seed{seed}.csv. "
            "When supplied, --datasets is used as the reported dataset name."
        ),
    )
    parser.add_argument(
        "--original-metrics-csv",
        default=None,
        help="Optional CauFeature CSV containing original_accuracy/original_f1 by dataset and seed. "
             "When supplied, baseline runs reuse those original metrics and train only selected features.",
    )
    parser.add_argument(
        "--matched-dimensions-csv",
        default=None,
        help="Optional CauFeature CSV containing selected_dim by dataset and seed. "
             "When supplied, each baseline selects the same number of features as CauFeature for that seed.",
    )
    return parser.parse_args()


def append_row(output_path: Path, row: Dict[str, object]) -> None:
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with output_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def main() -> None:
    args = parse_args()
    if not 0 < args.retention <= 1:
        raise SystemExit("--retention must be in the interval (0, 1].")

    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    original_metric_map = _load_original_metrics(
        Path(args.original_metrics_csv) if args.original_metrics_csv else None
    )
    matched_dimension_map = _load_matched_dimensions(
        Path(args.matched_dimensions_csv) if args.matched_dimensions_csv else None
    )

    row_count = 0
    for dataset_name in args.datasets:
        default_data_path = dataset_dir / dataset_name
        for method in args.methods:
            for seed in args.seeds:
                if args.seeded_dataset_pattern:
                    data_path = Path(args.seeded_dataset_pattern.format(seed=seed))
                else:
                    data_path = default_data_path
                if not data_path.exists():
                    print(f"Skipping missing dataset: {data_path}")
                    continue
                print(f"Running baseline: dataset={data_path.name}, method={method}, seed={seed}")
                original_metrics = (
                    original_metric_map.get((data_path.name, seed))
                    or original_metric_map.get((dataset_name, seed))
                )
                target_count = (
                    matched_dimension_map.get((data_path.name, seed))
                    or matched_dimension_map.get((dataset_name, seed))
                )
                row = run_baseline_once(
                    data_path,
                    method,
                    seed,
                    args.retention,
                    original_metrics,
                    target_count,
                )
                if args.seeded_dataset_pattern:
                    row["run_dataset"] = row["dataset"]
                    row["dataset"] = dataset_name
                append_row(output_path, row)
                row_count += 1
                print(f"Appended result row to {output_path}")

    if row_count == 0:
        raise SystemExit("No experiment rows were produced.")

    print(f"Wrote {row_count} rows to {output_path}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as exc:
        sys.stdout.flush()
        sys.stderr.flush()
        code = exc.code if isinstance(exc.code, int) else 1
        os._exit(code)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
