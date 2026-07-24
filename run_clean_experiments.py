"""Leakage-free reruns for RQ1--RQ3.

The outer test split is created before preprocessing, model fitting, or feature
selection.  All selection and early-stopping decisions use only outer-training
data; the outer test set is evaluated exactly once per fitted model.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


RQ1_DATASETS = [
    "MagicGamma.csv", "HeartDisease.csv", "Thyroid_Diff.csv", "MentalStress.csv",
    "students dropout academic success.csv", "CustomerChurn.csv", "DriedBean.csv",
    "online_shoppers_intention.csv", "SmokeAlarm.csv", "Income.csv", "bank-full.csv",
    "DiabetesHealth.csv", "BrainMethod.csv", "SoftwareQuality.csv",
    "pirvision_office_dataset2.csv",
]
REPRESENTATIVE_DATASETS = ["MagicGamma.csv", "BrainMethod.csv", "SoftwareQuality.csv"]
SEEDS = [0, 1, 2, 3, 4]
TRADITIONAL_METHODS = ["VT", "MI", "CHI", "RFE", "RFC"]
EXPLAINABLE_METHODS = ["SHAP", "LIME", "FeatureX"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def can_stratify(y: pd.Series) -> bool:
    return y.nunique() > 1 and y.value_counts().min() >= 2


def split_frame(frame: pd.DataFrame, target: pd.Series, seed: int, test_size: float):
    stratify = target if can_stratify(target) else None
    return train_test_split(frame, target, test_size=test_size, random_state=seed, stratify=stratify)


def prepare_outer_split(data_path: Path, seed: int):
    from model_train_v6 import read_tabular_data

    raw = read_tabular_data(data_path)
    raw_x = raw.iloc[:, :-1].copy()
    raw_y = raw.iloc[:, -1].copy()
    x_train_raw, x_test_raw, y_train_raw, y_test_raw = split_frame(raw_x, raw_y, seed, 0.2)

    numeric_columns = x_train_raw.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in x_train_raw.columns if column not in numeric_columns]
    x_train = x_train_raw.copy()
    x_test = x_test_raw.copy()
    for column in numeric_columns:
        value = x_train[column].median()
        x_train[column] = x_train[column].fillna(value)
        x_test[column] = x_test[column].fillna(value)
    if categorical_columns:
        x_train[categorical_columns] = x_train[categorical_columns].fillna("__missing__").astype(str)
        x_test[categorical_columns] = x_test[categorical_columns].fillna("__missing__").astype(str)
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        x_train[categorical_columns] = encoder.fit_transform(x_train[categorical_columns])
        x_test[categorical_columns] = encoder.transform(x_test[categorical_columns])

    label_encoder = LabelEncoder().fit(y_train_raw)
    unseen = set(y_test_raw.unique()) - set(label_encoder.classes_)
    if unseen:
        raise ValueError(f"Outer test set contains unseen classes: {sorted(unseen)}")
    y_train = pd.Series(label_encoder.transform(y_train_raw), index=x_train.index, name=raw.columns[-1])
    y_test = pd.Series(label_encoder.transform(y_test_raw), index=x_test.index, name=raw.columns[-1])
    split_hash = hashlib.sha256(",".join(map(str, sorted(x_test.index))).encode("utf-8")).hexdigest()[:16]
    return x_train, x_test, y_train, y_test, raw.columns[-1], split_hash


def one_hot(values: pd.Series, class_count: int) -> np.ndarray:
    return np.eye(class_count, dtype=np.float32)[values.to_numpy(dtype=int)]


def fit_and_evaluate(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame,
                     y_test: pd.Series, selected: list[str], seed: int):
    """Fit on the outer train split, validate inside it, and score only outer test."""
    from model_train_v6 import build_model, set_seed, smote_enn_resample, train_model
    import shared_globals

    selected = [name for name in selected if name in x_train.columns]
    if not selected:
        raise ValueError("No valid selected features.")
    train_selected = x_train[selected]
    test_selected = x_test[selected]
    x_fit, x_validation, y_fit, y_validation = split_frame(train_selected, y_train, seed + 10_000, 0.2)
    try:
        x_fit_resampled, y_fit_resampled = smote_enn_resample(x_fit, y_fit, random_state=seed)
    except ValueError as exc:
        print(f"SMOTEENN fallback for seed {seed}: {exc}")
        x_fit_resampled, y_fit_resampled = x_fit, y_fit

    class_count = int(y_train.max() + 1)
    x_fit_array = x_fit_resampled.to_numpy(dtype=np.float32).reshape(-1, len(selected), 1)
    x_validation_array = x_validation.to_numpy(dtype=np.float32).reshape(-1, len(selected), 1)
    x_test_array = test_selected.to_numpy(dtype=np.float32).reshape(-1, len(selected), 1)
    y_fit_array = one_hot(pd.Series(y_fit_resampled), class_count)
    y_validation_array = one_hot(y_validation, class_count)

    set_seed(seed)
    shared_globals.filtered_features = None
    model = build_model((len(selected), 1), class_count, len(x_train))
    model, _ = train_model(
        model, x_fit_array, y_fit_array, x_validation_array, y_validation_array, len(x_train)
    )
    predicted = np.argmax(model.predict(x_test_array, verbose=0), axis=1)
    actual = y_test.to_numpy(dtype=int)
    return model, accuracy_score(actual, predicted), f1_score(actual, predicted, average="macro", zero_division=0)


def configure_caufeature_state(x_train, y_train, target_name, model, seed):
    import shared_globals

    shared_globals.init()
    shared_globals.random_seed = seed
    shared_globals.model = model
    shared_globals.feature_data = x_train.copy()
    shared_globals.feature_names = x_train.columns.tolist()
    shared_globals.target_name = [target_name]
    shared_globals.all_names = shared_globals.feature_names + shared_globals.target_name
    shared_globals.data = pd.concat([x_train, y_train.rename(target_name)], axis=1)
    shared_globals.sample_num = len(x_train)


def select_caufeature(x_train, y_train, target_name, full_model, seed):
    from scipy.stats import spearmanr
    import shared_globals
    from causal_graph_build_v19_english import Feature, build_causal_graph, config
    from feature_perturbation_v7_english import CausalFeaturePerturbation
    from redundantRecover_v10_english import PathShapleyModule

    original_config = config.__dict__.copy()
    configure_caufeature_state(x_train, y_train, target_name, full_model, seed)
    try:
        phase_one = time.perf_counter()
        perturbation = CausalFeaturePerturbation(full_model, shared_globals.data)
        perturbation.run_perturbation()
        perturbation.initialize_original_preds(shared_globals.data, full_model)
        phase_one_time = time.perf_counter() - phase_one

        phase_two = time.perf_counter()
        features = [
            Feature(name=name, feature_names=shared_globals.feature_names, all_con_values=shared_globals.con_list)
            for name in shared_globals.all_names
        ]
        is_numeric = all(feature.data.dtype.kind in "iufc" for feature in features)
        if is_numeric:
            shared_globals.corr_matrix = np.corrcoef([feature.data for feature in features])
        else:
            shared_globals.corr_matrix, _ = spearmanr([feature.data for feature in features])
        valid_paths, _, _ = build_causal_graph(
            features, shared_globals.corr_matrix, shared_globals.target_name, is_numeric
        )
        phase_two_time = time.perf_counter() - phase_two
        phase_one_features = [
            feature.name for feature in features
            if feature.name in x_train.columns and feature.is_high_contrib
        ]
        phase_two_features = [
            name for name in (shared_globals.filtered_features or []) if name in x_train.columns
        ]

        phase_three = time.perf_counter()
        PathShapleyModule(log_verbose=False, decay_type="exponential", alpha=0.7).run_redundant_check()
        phase_three_time = time.perf_counter() - phase_three
        selected = [name for name in (shared_globals.filtered_features or []) if name in x_train.columns]
        if not selected:
            selected = [x_train.columns[int(np.argmax(shared_globals.con_list))]]
        details = {
            "contribution_scores": dict(zip(shared_globals.feature_names, shared_globals.con_list)),
            "phase_one_features": phase_one_features,
            "phase_two_features": phase_two_features,
            "final_features": selected,
            "valid_paths": [str(path) for path in valid_paths],
            "accepted_recovery_log": list(shared_globals.significant_feature_pairs_log),
        }
        return selected, len(valid_paths), phase_one_time, phase_two_time, phase_three_time, details
    finally:
        config.__dict__.clear()
        config.__dict__.update(original_config)


def select_traditional(method: str, x_train: pd.DataFrame, y_train: pd.Series, k: int, seed: int):
    from run_baseline_experiments import _select_features
    return _select_features(method, x_train, y_train, retention=1.0, seed=seed, target_count=k)


def select_explainable(method: str, model, x_train: pd.DataFrame, y_train: pd.Series, seed: int):
    from run_explainable_experiments import _select_featurex, _select_lime, _select_shap

    explanation_train, explanation_holdout, _, _ = split_frame(x_train, y_train, seed + 20_000, 0.2)
    if method == "FeatureX":
        return _select_featurex(x_train, y_train, model, seed)[0], "FeatureX thresholds from commit 5b3d1d4 on outer-training data"
    if method == "SHAP":
        return _select_shap(model, explanation_train, explanation_holdout, seed)[0], "Permutation SHAP with fixed 0.2% attribution cutoff on outer-training data"
    if method == "LIME":
        return _select_lime(model, explanation_train, explanation_holdout, seed)[0], "Global mean absolute LIME with fixed 0.2% attribution cutoff on outer-training data"
    raise ValueError(f"Unsupported explainable method: {method}")


def append_row(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()


def existing_keys(path: Path, methods: bool = False):
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not methods:
        return {(row["dataset"], int(row["seed"])) for row in rows}
    field = "method" if not rows or "method" in rows[0] else "variant"
    return {(row["dataset"], int(row["seed"]), row[field]) for row in rows}


def read_caufeature_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["dataset"], int(row["seed"])): row for row in csv.DictReader(handle)}


def run_rq1(dataset_dir: Path, output: Path, datasets: list[str], seeds: list[int]):
    done = existing_keys(output)
    for dataset in datasets:
        for seed in seeds:
            key = (dataset, seed)
            if key in done:
                continue
            print(f"RQ1 clean: {dataset}, seed={seed}", flush=True)
            started = time.perf_counter()
            x_train, x_test, y_train, y_test, target, split_hash = prepare_outer_split(dataset_dir / dataset, seed)
            full_model, full_acc, full_f1 = fit_and_evaluate(x_train, y_train, x_test, y_test, list(x_train.columns), seed)
            selected, paths, phase_one, phase_two, phase_three, _ = select_caufeature(x_train, y_train, target, full_model, seed)
            _, selected_acc, selected_f1 = fit_and_evaluate(x_train, y_train, x_test, y_test, selected, seed)
            append_row(output, {
                "dataset": dataset, "seed": seed, "protocol": "outer_holdout_train_only_selection_v1",
                "outer_test_hash": split_hash, "outer_train_n": len(x_train), "outer_test_n": len(x_test),
                "original_dim": len(x_train.columns), "selected_dim": len(selected),
                "reduction_pct": 100 * (len(x_train.columns) - len(selected)) / len(x_train.columns),
                "original_accuracy": full_acc, "selected_accuracy": selected_acc,
                "accuracy_delta_pct_points": 100 * (selected_acc - full_acc),
                "original_f1": full_f1, "selected_f1": selected_f1,
                "f1_delta_pct_points": 100 * (selected_f1 - full_f1), "valid_path_count": paths,
                "feature_perturb_time_sec": phase_one, "graph_time_sec": phase_two,
                "recovery_time_sec": phase_three, "total_time_sec": time.perf_counter() - started,
                "selected_features": ";".join(selected),
            })


def run_rq4(dataset_dir: Path, output: Path, datasets: list[str], seeds: list[int]):
    done = existing_keys(output, methods=True)
    for dataset in datasets:
        for seed in seeds:
            x_train, x_test, y_train, y_test, target, split_hash = prepare_outer_split(dataset_dir / dataset, seed)
            print(f"RQ4 clean selection: {dataset}, seed={seed}", flush=True)
            full_model, full_acc, full_f1 = fit_and_evaluate(x_train, y_train, x_test, y_test, list(x_train.columns), seed)
            final, paths, _, _, _, details = select_caufeature(x_train, y_train, target, full_model, seed)
            variants = {
                "Full features": list(x_train.columns),
                "Phase 1": details["phase_one_features"],
                "Phase 1+2": details["phase_two_features"],
                "CauFeature": final,
            }
            for variant, selected in variants.items():
                key = (dataset, seed, variant)
                if key in done:
                    continue
                print(f"RQ4 clean evaluation: {dataset}, {variant}, seed={seed}", flush=True)
                started = time.perf_counter()
                if variant == "Full features":
                    selected_acc, selected_f1 = full_acc, full_f1
                else:
                    _, selected_acc, selected_f1 = fit_and_evaluate(x_train, y_train, x_test, y_test, selected, seed)
                append_row(output, {
                    "dataset": dataset, "variant": variant, "seed": seed,
                    "protocol": "outer_holdout_train_only_selection_v1", "outer_test_hash": split_hash,
                    "original_dim": len(x_train.columns), "selected_dim": len(selected),
                    "reduction_pct": 100 * (len(x_train.columns) - len(selected)) / len(x_train.columns),
                    "original_accuracy": full_acc, "selected_accuracy": selected_acc,
                    "accuracy_delta_pct_points": 100 * (selected_acc - full_acc),
                    "original_f1": full_f1, "selected_f1": selected_f1,
                    "f1_delta_pct_points": 100 * (selected_f1 - full_f1),
                    "valid_path_count": paths if variant != "Phase 1" else "",
                    "total_time_sec": time.perf_counter() - started, "selected_features": ";".join(selected),
                })


def run_rq6(dataset_dir: Path, output_dir: Path, dataset: str, seed: int):
    x_train, x_test, y_train, y_test, target, split_hash = prepare_outer_split(dataset_dir / dataset, seed)
    full_model, full_acc, full_f1 = fit_and_evaluate(x_train, y_train, x_test, y_test, list(x_train.columns), seed)
    final, paths, phase_one, phase_two, phase_three, details = select_caufeature(
        x_train, y_train, target, full_model, seed
    )
    _, selected_acc, selected_f1 = fit_and_evaluate(x_train, y_train, x_test, y_test, final, seed)
    payload = {
        "dataset": dataset, "seed": seed, "protocol": "outer_holdout_train_only_selection_v1",
        "outer_test_hash": split_hash, "outer_train_n": len(x_train), "outer_test_n": len(x_test),
        "full_accuracy": full_acc, "full_f_measure": full_f1,
        "selected_accuracy": selected_acc, "selected_f_measure": selected_f1,
        "valid_path_count": paths, "phase_times_sec": {
            "phase_1": phase_one, "phase_2": phase_two, "phase_3": phase_three,
        },
        **details,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"rq6_{Path(dataset).stem}_seed{seed}_clean.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def run_rq2(dataset_dir: Path, cau_path: Path, output: Path, datasets: list[str], seeds: list[int]):
    cau_rows = read_caufeature_rows(cau_path)
    done = existing_keys(output, methods=True)
    for dataset in datasets:
        for seed in seeds:
            reference = cau_rows[(dataset, seed)]
            x_train, x_test, y_train, y_test, _, split_hash = prepare_outer_split(dataset_dir / dataset, seed)
            k = int(reference["selected_dim"])
            for method in TRADITIONAL_METHODS:
                key = (dataset, seed, method)
                if key in done:
                    continue
                print(f"RQ2 clean: {dataset}, {method}, seed={seed}", flush=True)
                started = time.perf_counter()
                selected, rule = select_traditional(method, x_train, y_train, k, seed)
                _, selected_acc, selected_f1 = fit_and_evaluate(x_train, y_train, x_test, y_test, selected, seed)
                full_acc, full_f1 = float(reference["original_accuracy"]), float(reference["original_f1"])
                append_row(output, {
                    "dataset": dataset, "method": method, "seed": seed,
                    "protocol": "outer_holdout_train_only_selection_v1", "outer_test_hash": split_hash,
                    "matched_target_dim": k, "selection_rule": rule, "original_dim": len(x_train.columns),
                    "selected_dim": len(selected), "reduction_pct": 100 * (len(x_train.columns) - len(selected)) / len(x_train.columns),
                    "original_accuracy": full_acc, "selected_accuracy": selected_acc,
                    "accuracy_delta_pct_points": 100 * (selected_acc - full_acc),
                    "original_f1": full_f1, "selected_f1": selected_f1,
                    "f1_delta_pct_points": 100 * (selected_f1 - full_f1),
                    "total_time_sec": time.perf_counter() - started, "selected_features": ";".join(selected),
                })


def run_rq3(dataset_dir: Path, cau_path: Path, output: Path, datasets: list[str], seeds: list[int]):
    cau_rows = read_caufeature_rows(cau_path)
    done = existing_keys(output, methods=True)
    for dataset in datasets:
        for seed in seeds:
            x_train, x_test, y_train, y_test, _, split_hash = prepare_outer_split(dataset_dir / dataset, seed)
            full_model, full_acc, full_f1 = fit_and_evaluate(x_train, y_train, x_test, y_test, list(x_train.columns), seed)
            for method in EXPLAINABLE_METHODS:
                key = (dataset, seed, method)
                if key in done:
                    continue
                print(f"RQ3 clean: {dataset}, {method}, seed={seed}", flush=True)
                started = time.perf_counter()
                selected, rule = select_explainable(method, full_model, x_train, y_train, seed)
                _, selected_acc, selected_f1 = fit_and_evaluate(x_train, y_train, x_test, y_test, selected, seed)
                append_row(output, {
                    "dataset": dataset, "method": method, "seed": seed,
                    "protocol": "outer_holdout_train_only_selection_v1", "outer_test_hash": split_hash,
                    "selection_rule": rule, "original_dim": len(x_train.columns), "selected_dim": len(selected),
                    "reduction_pct": 100 * (len(x_train.columns) - len(selected)) / len(x_train.columns),
                    "original_accuracy": full_acc, "selected_accuracy": selected_acc,
                    "accuracy_delta_pct_points": 100 * (selected_acc - full_acc),
                    "original_f1": full_f1, "selected_f1": selected_f1,
                    "f1_delta_pct_points": 100 * (selected_f1 - full_f1),
                    "total_time_sec": time.perf_counter() - started, "selected_features": ";".join(selected),
                })


def write_manifest(output_dir: Path):
    source_files = [
        Path(__file__), Path("model_train_v6.py"), Path("run_baseline_experiments.py"),
        Path("run_explainable_experiments.py"), Path("feature_perturbation_v7_english.py"),
        Path("causal_graph_build_v19_english.py"), Path("redundantRecover_v10_english.py"),
    ]
    payload = {
        "protocol": "outer_holdout_train_only_selection_v1",
        "outer_split": "stratified 80/20 when possible, determined by dataset and seed",
        "preprocessing": "fit on outer training data only",
        "early_stopping": "validation split drawn from outer training data only",
        "selection": "CauFeature, traditional baselines, SHAP, LIME, and FeatureX use outer training data only",
        "evaluation": "outer test data is used only after feature selection and model fitting",
        "seeds": SEEDS, "rq1_datasets": RQ1_DATASETS, "representative_datasets": REPRESENTATIVE_DATASETS,
        "source_hashes": {str(path): sha256(path) for path in source_files},
    }
    (output_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run leakage-free RQ1--RQ7 experiments.")
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--output-dir", default="reproduction_clean_20260720/raw")
    parser.add_argument("--stages", nargs="+", choices=["rq1", "rq2", "rq3", "rq4", "rq6"], default=["rq1", "rq2", "rq3"])
    parser.add_argument("--datasets", nargs="*", default=None, help="Override RQ1 datasets for a pilot run.")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(output_dir)
    datasets = args.datasets if args.datasets is not None else RQ1_DATASETS
    comparison_datasets = args.datasets if args.datasets is not None else REPRESENTATIVE_DATASETS
    seeds = args.seeds if args.seeds is not None else SEEDS
    dataset_dir = Path(args.dataset_dir)
    cau_path = output_dir / "caufeature_rq1_clean.csv"
    if "rq1" in args.stages:
        run_rq1(dataset_dir, cau_path, datasets, seeds)
    if "rq2" in args.stages:
        if not cau_path.exists():
            raise SystemExit("RQ2 requires the clean RQ1 output in the same output directory.")
        run_rq2(dataset_dir, cau_path, output_dir / "rq2_matched_traditional_clean.csv", comparison_datasets, seeds)
    if "rq3" in args.stages:
        if not cau_path.exists():
            raise SystemExit("RQ3 requires the clean RQ1 output in the same output directory.")
        run_rq3(dataset_dir, cau_path, output_dir / "rq3_explainable_broad_clean.csv", comparison_datasets, seeds)
    if "rq4" in args.stages:
        run_rq4(dataset_dir, output_dir / "rq4_ablation_clean.csv", comparison_datasets, seeds)
    if "rq6" in args.stages:
        if len(comparison_datasets) != 1 or len(seeds) != 1:
            raise SystemExit("RQ6 requires exactly one dataset and one seed.")
        run_rq6(dataset_dir, output_dir, comparison_datasets[0], seeds[0])


if __name__ == "__main__":
    main()
