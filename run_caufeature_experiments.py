import argparse
import csv
import os
import sys
import time
import traceback
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


DEFAULT_DATASETS = [
    "iris.csv",
    "Rice_Cammeo_Osmancik.csv",
    "MagicGamma.csv",
    "HeartDisease.csv",
    "Thyroid_Diff.csv",
    "MentalStress.csv",
    "students dropout academic success.csv",
    "CustomerChurn.csv",
    "DriedBean.csv",
    "online_shoppers_intention.csv",
    "SmokeAlarm.csv",
    "Income.csv",
    "bank-full.csv",
    "DiabetesHealth.csv",
    "BrainMethod.csv",
    "SoftwareQuality.csv",
    "pirvision_office_dataset2.csv",
]


def _prefilter_dataset(data_path: Path, output_dir: Path, top_k: int, seed: int) -> Path:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
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

    keep_count = max(1, min(top_k, x.shape[1]))
    selector = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    selector.fit(x, y)
    ranking = selector.feature_importances_.argsort()[::-1][:keep_count]
    selected = x.columns[ranking].tolist()

    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = output_dir / f"{data_path.stem}_prefilter_top{keep_count}_seed{seed}.csv"
    pd.concat([df[selected], df.iloc[:, -1]], axis=1).to_csv(filtered_path, index=False)
    return filtered_path


def run_caufeature_once(
    data_path: Path,
    seed: int,
    *,
    n_bootstrap: int | None = None,
    alpha_list: list[float] | None = None,
    max_path_length: int | None = None,
    prefilter_top_k: int | None = None,
    prefilter_dir: Path | None = None,
) -> dict:
    import numpy as np
    from scipy.stats import spearmanr

    import shared_globals
    from causal_graph_build_v19_english import Feature, build_causal_graph, config
    from feature_perturbation_v7_english import CausalFeaturePerturbation
    from model_train_v6 import train
    from redundantRecover_v10_english import PathShapleyModule

    original_data_path = data_path
    if prefilter_top_k is not None:
        data_path = _prefilter_dataset(
            data_path,
            prefilter_dir or Path("results/prefiltered_datasets"),
            prefilter_top_k,
            seed,
        )

    original_n_bootstrap = config.n_bootstrap
    if n_bootstrap is not None:
        config.n_bootstrap = n_bootstrap

    shared_globals.init()
    shared_globals.random_seed = seed

    try:
        start_total = time.perf_counter()
        model = train(str(data_path), shared_globals.filtered_features)
        shared_globals.model = model

        original_dim = shared_globals.feature_data.shape[1]
        original_accuracy = shared_globals.accuracy
        original_f1 = shared_globals.f_score

        start = time.perf_counter()
        perturb = CausalFeaturePerturbation(shared_globals.model, shared_globals.data)
        perturb.run_perturbation()
        perturb.initialize_original_preds(shared_globals.data, shared_globals.model)
        feature_perturb_time = time.perf_counter() - start

        start = time.perf_counter()
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
            features,
            shared_globals.corr_matrix,
            shared_globals.target_name,
            is_numeric,
            alpha_list=alpha_list,
            max_path_length=max_path_length,
        )
        graph_time = time.perf_counter() - start

        start = time.perf_counter()
        shapley_module = PathShapleyModule(log_verbose=False, decay_type="exponential", alpha=0.7)
        shapley_module.run_redundant_check()
        recovery_time = time.perf_counter() - start

        selected_features = list(shared_globals.filtered_features or [])
        shared_globals.model = None
        train(str(data_path), selected_features)

        total_time = time.perf_counter() - start_total
        setting_parts = []
        if prefilter_top_k is not None:
            setting_parts.append(f"prefilter_top_k={prefilter_top_k}")
        if n_bootstrap is not None:
            setting_parts.append(f"n_bootstrap={n_bootstrap}")
        if alpha_list is not None:
            setting_parts.append("alpha_list=" + ";".join(f"{a:g}" for a in alpha_list))
        if max_path_length is not None:
            setting_parts.append(f"max_path_length={max_path_length}")
        experiment_setting = ", ".join(setting_parts) if setting_parts else "default"
        return {
            "dataset": original_data_path.name,
            "run_dataset": data_path.name,
            "seed": seed,
            "experiment_setting": experiment_setting,
            "original_dim": original_dim,
            "selected_dim": len(selected_features),
            "reduction_pct": 100.0 * (original_dim - len(selected_features)) / original_dim if original_dim else 0.0,
            "original_accuracy": original_accuracy,
            "selected_accuracy": original_accuracy + shared_globals.accuracy / 100.0,
            "accuracy_delta_pct_points": shared_globals.accuracy,
            "original_f1": original_f1,
            "selected_f1": original_f1 + shared_globals.f_score / 100.0,
            "f1_delta_pct_points": shared_globals.f_score,
            "valid_path_count": len(valid_paths),
            "feature_perturb_time_sec": feature_perturb_time,
            "graph_time_sec": graph_time,
            "recovery_time_sec": recovery_time,
            "total_time_sec": total_time,
            "selected_features": ";".join(selected_features),
        }
    finally:
        config.n_bootstrap = original_n_bootstrap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CauFeature over one or more datasets and write per-run metrics to CSV."
    )
    parser.add_argument("--dataset-dir", default="dataset", help="Directory containing CSV datasets.")
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Dataset file names relative to --dataset-dir.",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[42], help="Random seeds to evaluate.")
    parser.add_argument("--output", default="results/caufeature_runs.csv", help="Output CSV path.")
    parser.add_argument("--n-bootstrap", type=int, default=None, help="Override bootstrap count for graph pruning.")
    parser.add_argument(
        "--alpha-list",
        nargs="*",
        type=float,
        default=None,
        help="Override alpha thresholds used in graph construction.",
    )
    parser.add_argument("--max-path-length", type=int, default=None, help="Override maximum causal path length.")
    parser.add_argument(
        "--prefilter-top-k",
        type=int,
        default=None,
        help="Optionally prefilter each dataset to top-k RandomForest features before CauFeature.",
    )
    parser.add_argument(
        "--prefilter-dir",
        default="results/prefiltered_datasets",
        help="Directory for generated prefiltered datasets.",
    )
    return parser.parse_args()


def append_row(output_path: Path, row: dict) -> None:
    file_exists = output_path.exists() and output_path.stat().st_size > 0
    with output_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        fh.flush()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_count = 0
    for dataset_name in args.datasets:
        data_path = dataset_dir / dataset_name
        if not data_path.exists():
            print(f"Skipping missing dataset: {data_path}")
            continue
        for seed in args.seeds:
            print(f"Running CauFeature: dataset={data_path.name}, seed={seed}")
            row = run_caufeature_once(
                data_path,
                seed,
                n_bootstrap=args.n_bootstrap,
                alpha_list=args.alpha_list,
                max_path_length=args.max_path_length,
                prefilter_top_k=args.prefilter_top_k,
                prefilter_dir=Path(args.prefilter_dir),
            )
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
