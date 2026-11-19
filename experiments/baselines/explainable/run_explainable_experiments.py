import argparse
import csv
import os
import random
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


DEFAULT_DATASETS = ["MagicGamma.csv", "BrainMethod.csv", "SoftwareQuality.csv"]
DEFAULT_METHODS = ["SHAP", "LIME", "FeatureX"]


def _replacement_values(series, rng):
    """Reproduce the public FeatureX perturbation-value rules."""
    import numpy as np

    values = series.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return []

    minimum = float(values.min())
    maximum = float(values.max())
    unique_values = np.unique(values)
    unique_count = len(unique_values)
    if minimum == 0.0 and maximum == 1.0 and unique_count != 2:
        return np.linspace(0.0, 1.0, 11).tolist()

    difference = maximum - minimum
    if unique_count and difference / unique_count > 100:
        augmented = unique_values.tolist()
        lower = int(np.floor(minimum))
        upper = int(np.ceil(maximum))
        augmented.extend(rng.randint(lower, upper) for _ in range(30))
        return sorted(augmented)

    if difference > 10000:
        return np.linspace(minimum, maximum, 1000, endpoint=False).tolist()

    lower = int(np.floor(minimum))
    upper = int(np.ceil(maximum))
    return list(range(lower, upper + 1))


def _featurex_scores(model, features, seed):
    import numpy as np

    baseline = features.mean(axis=0).to_numpy(dtype=np.float32)
    baseline_input = baseline.reshape(1, -1, 1)
    baseline_prediction = model.predict(baseline_input, verbose=0)[0]
    predicted_class = int(np.argmax(baseline_prediction))
    rng = random.Random(seed)
    scores = []

    for index, column in enumerate(features.columns):
        candidates = _replacement_values(features[column], rng)
        if not candidates:
            scores.append(0.0)
            continue
        modified = np.repeat(baseline[None, :], len(candidates), axis=0)
        modified[:, index] = candidates
        predictions = model.predict(modified.reshape(-1, features.shape[1], 1), verbose=0)
        changes = np.abs(predictions[:, predicted_class] - baseline_prediction[predicted_class])
        scores.append(float(np.max(changes)) if changes.size else 0.0)
    return np.asarray(scores, dtype=float)


def _featurex_components(features):
    import numpy as np
    from scipy.stats import spearmanr

    if features.shape[1] < 2:
        return []
    correlation, _ = spearmanr(features.to_numpy(), axis=0)
    correlation = np.atleast_2d(correlation)
    adjacency = {name: set() for name in features.columns}
    for left in range(features.shape[1]):
        for right in range(left + 1, features.shape[1]):
            if np.isfinite(correlation[left, right]) and abs(correlation[left, right]) >= 0.7:
                left_name = features.columns[left]
                right_name = features.columns[right]
                adjacency[left_name].add(right_name)
                adjacency[right_name].add(left_name)

    components = []
    seen = set()
    for name, neighbors in adjacency.items():
        if name in seen or not neighbors:
            continue
        stack = [name]
        component = set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(adjacency[current] - component)
        seen.update(component)
        components.append(component)
    return components


def _featurex_target_correlation(features, target):
    import numpy as np
    from scipy.stats import pointbiserialr, spearmanr

    binary = len(np.unique(target)) == 2
    correlations = {}
    for column in features.columns:
        if binary:
            value, _ = pointbiserialr(features[column], target)
        else:
            value, _ = spearmanr(features[column], target)
        correlations[column] = 0.0 if not np.isfinite(value) else float(value)
    return correlations


def _select_featurex(features, target, model, seed):
    """Adapt the official FeatureX notebooks to the shared experiment pipeline."""
    scores = _featurex_scores(model, features, seed)
    names = list(features.columns)
    score_by_name = dict(zip(names, scores))
    target_correlation = _featurex_target_correlation(features, target)
    components = _featurex_components(features)
    memberships = {name: 0 for name in names}
    for component in components:
        for name in component:
            memberships[name] = len(component) - 1

    selected = set(names)
    high_or_low = {name for name in names if score_by_name[name] >= 0.5 or score_by_name[name] <= 0.1}
    sufficient_non_low = sum(score_by_name[name] >= 0.1 for name in names) / len(names) >= 0.15

    if sufficient_non_low:
        active_components = [set(component) - high_or_low for component in components]
        for name in names:
            score = score_by_name[name]
            if score <= 0.1:
                selected.discard(name)
            elif 0.1 < score < 0.5 and abs(target_correlation[name]) < 0.3 and memberships[name] > 0:
                for component in active_components:
                    if name in component:
                        best = max(component, key=lambda item: score_by_name[item])
                        selected.intersection_update({best} | (set(names) - component))
                        break
    else:
        active_components = [{name for name in component if score_by_name[name] < 0.1} for component in components]
        for name in names:
            if abs(target_correlation[name]) < 0.7 and memberships[name] > 0:
                for component in active_components:
                    if name in component:
                        best = max(component, key=lambda item: score_by_name[item])
                        selected.intersection_update({best} | (set(names) - component))
                        break

    if not selected:
        selected.add(max(names, key=lambda name: score_by_name[name]))
    return [name for name in names if name in selected], scores


def _mean_absolute_shap_values(values, n_features):
    import numpy as np

    if isinstance(values, list):
        array = np.stack([np.asarray(item) for item in values], axis=0)
    else:
        array = np.asarray(values)
    feature_axes = [axis for axis, size in enumerate(array.shape) if size == n_features]
    if not feature_axes:
        raise ValueError(f"Cannot locate a SHAP feature axis in shape {array.shape}.")
    feature_axis = feature_axes[-1]
    array = np.moveaxis(array, feature_axis, -1)
    return np.mean(np.abs(array), axis=tuple(range(array.ndim - 1)))


def _select_shap(model, train_features, test_features, seed):
    import numpy as np
    import shap

    def predict(values):
        values = np.asarray(values, dtype=np.float32)
        return model.predict(values.reshape(-1, train_features.shape[1], 1), verbose=0)

    background = train_features.iloc[: min(50, len(train_features))].to_numpy(dtype=np.float32)
    explained = test_features.iloc[: min(50, len(test_features))].to_numpy(dtype=np.float32)
    explainer = shap.PermutationExplainer(predict, background, seed=seed)
    explanation = explainer(
        explained,
        max_evals=2 * train_features.shape[1] + 1,
        batch_size=50,
    )
    scores = _mean_absolute_shap_values(explanation.values, test_features.shape[1])
    normalized = scores / scores.sum() if scores.sum() > 0 else np.ones_like(scores) / len(scores)
    selected = [name for name, value in zip(test_features.columns, normalized) if value >= 0.002]
    if not selected:
        selected = [test_features.columns[int(np.argmax(scores))]]
    return selected, scores


def _select_lime(model, train_features, test_features, seed):
    import numpy as np
    from lime.lime_tabular import LimeTabularExplainer

    def predict(values):
        values = np.asarray(values, dtype=np.float32)
        return model.predict(values.reshape(-1, train_features.shape[1], 1), verbose=0)

    explainer = LimeTabularExplainer(
        training_data=train_features.to_numpy(dtype=float),
        feature_names=list(train_features.columns),
        mode="classification",
        random_state=seed,
        discretize_continuous=True,
        discretizer="quartile",
        sample_around_instance=True,
    )
    rng = np.random.default_rng(seed)
    count = min(200, len(test_features))
    indices = rng.choice(len(test_features), size=count, replace=False)
    scores = np.zeros(train_features.shape[1], dtype=float)
    for index in indices:
        explanation = explainer.explain_instance(
            test_features.iloc[index].to_numpy(dtype=float),
            predict,
            num_features=train_features.shape[1],
            top_labels=1,
            num_samples=500,
        )
        for feature_index, weight in explanation.as_map()[explanation.top_labels[0]]:
            scores[feature_index] += abs(weight)
    scores /= count
    normalized = scores / scores.sum() if scores.sum() > 0 else np.ones_like(scores) / len(scores)
    selected = [name for name, value in zip(train_features.columns, normalized) if value >= 0.002]
    if not selected:
        selected = [train_features.columns[int(np.argmax(scores))]]
    return selected, scores


def _fit_full_model(data_path, seed):
    import shared_globals
    from model_train_v6 import train

    shared_globals.init()
    shared_globals.random_seed = seed
    model = train(str(data_path), None)
    shared_globals.model = model
    features = shared_globals.feature_data.copy()
    target = shared_globals.data.iloc[:, -1].copy()
    return model, features, target, shared_globals.accuracy, shared_globals.f_score


def _evaluate_subset(data_path, seed, full_features, original_accuracy, original_f1, selected):
    import shared_globals
    from model_train_v6 import train

    shared_globals.feature_data = full_features.copy()
    shared_globals.filtered_features = list(selected)
    shared_globals.random_seed = seed
    shared_globals.accuracy = original_accuracy
    shared_globals.f_score = original_f1
    train(str(data_path), list(selected))
    return (
        original_accuracy + shared_globals.accuracy / 100.0,
        original_f1 + shared_globals.f_score / 100.0,
        shared_globals.accuracy,
        shared_globals.f_score,
    )


def _split_for_explanations(features, target, seed):
    from sklearn.model_selection import train_test_split

    return train_test_split(features, target, test_size=0.2, random_state=seed)


def run_once(data_path, seed, methods):
    model, features, target, original_accuracy, original_f1 = _fit_full_model(data_path, seed)
    train_features, test_features, _, _ = _split_for_explanations(features, target, seed)
    selectors = {}
    selection_times = {}
    if "FeatureX" in methods:
        start = time.perf_counter()
        selectors["FeatureX"] = _select_featurex(features, target, model, seed)
        selection_times["FeatureX"] = time.perf_counter() - start
    if "SHAP" in methods:
        start = time.perf_counter()
        selectors["SHAP"] = _select_shap(model, train_features, test_features, seed)
        selection_times["SHAP"] = time.perf_counter() - start
    if "LIME" in methods:
        start = time.perf_counter()
        selectors["LIME"] = _select_lime(model, train_features, test_features, seed)
        selection_times["LIME"] = time.perf_counter() - start

    rows = []
    for method in methods:
        selected, _ = selectors[method]
        start = time.perf_counter()
        selected_accuracy, selected_f1, accuracy_delta, f1_delta = _evaluate_subset(
            data_path,
            seed,
            features,
            original_accuracy,
            original_f1,
            selected,
        )
        retrain_time = time.perf_counter() - start
        rows.append(
            {
                "dataset": data_path.name,
                "method": method,
                "seed": seed,
                "original_dim": features.shape[1],
                "selected_dim": len(selected),
                "reduction_pct": 100.0 * (features.shape[1] - len(selected)) / features.shape[1],
                "original_accuracy": original_accuracy,
                "selected_accuracy": selected_accuracy,
                "accuracy_delta_pct_points": accuracy_delta,
                "original_f1": original_f1,
                "selected_f1": selected_f1,
                "f1_delta_pct_points": f1_delta,
                "selection_rule": {
                    "FeatureX": "official FeatureX commit 5b3d1d4; thresholds delta=0.1, epsilon=0.5, gamma=0.3, beta=0.7",
                    "SHAP": "PermutationExplainer with 50 background and 50 explained samples; retain normalized mean absolute attribution >= 0.2%",
                    "LIME": "global mean absolute LIME attribution over up to 200 test samples; retain normalized attribution >= 0.2%",
                }[method],
                "selection_time_sec": selection_times[method],
                "retrain_time_sec": retrain_time,
                "total_time_sec": selection_times[method] + retrain_time,
                "selected_features": ";".join(selected),
            }
        )
    return rows


def append_rows(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exists = output_path.exists() and output_path.stat().st_size > 0
    with output_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SHAP, LIME, and FeatureX with CauFeature's shared preprocessing and CNN pipeline."
    )
    parser.add_argument("--dataset-dir", default="dataset")
    parser.add_argument("--datasets", nargs="*", default=DEFAULT_DATASETS)
    parser.add_argument("--methods", nargs="*", choices=DEFAULT_METHODS, default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", default="results/explainable_runs.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    for dataset_name in args.datasets:
        data_path = Path(args.dataset_dir) / dataset_name
        if not data_path.exists():
            raise SystemExit(f"Missing dataset: {data_path}")
        for seed in args.seeds:
            print(f"Running explainable methods: dataset={dataset_name}, seed={seed}")
            append_rows(output_path, run_once(data_path, seed, args.methods))


if __name__ == "__main__":
    main()
