#!/usr/bin/env python3
"""Rebuild Table 3 and the RQ6 Spearman analysis from completed RQ1 cases."""

import argparse
import json
import math
import re
import statistics
import tempfile
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.preprocessing import LabelEncoder


SEEDS = tuple(f"S{index:02d}" for index in range(1, 31))
METRICS = {"accuracy": "acc", "f1_macro": "f1"}
SPEARMAN_VARIABLES = (
    "n_samples", "original_dimension", "reduction_percent", "high_correlation_pair_fraction",
)


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def scaled_pair(case, scaled_root, dataset, seed):
    candidates = [case]
    if scaled_root is not None:
        candidates.extend((
            scaled_root / dataset / seed,
            scaled_root / "cases" / dataset / seed,
            scaled_root / dataset / seed / "cases" / dataset / seed,
        ))
    matches = {
        path.resolve()
        for path in candidates
        if all((path / f"{mode}_scaled.json").is_file() for mode in ("full", "selected"))
    }
    if len(matches) != 1:
        raise ValueError(f"{dataset}/{seed}: expected exactly one complete scaled CNN pair, found {len(matches)}")
    directory, = matches
    return (read_json(directory / "full_scaled.json"), read_json(directory / "selected_scaled.json"))


def read_case(case_root, scaled_root, dataset, seed):
    case = case_root / dataset / seed
    record = read_json(case / "record.json")
    p1 = read_json(case / "p1_cache.json")
    if any(obj["status"] != "complete" or obj["dataset"] not in (dataset, f"{dataset}.csv")
           for obj in (record, p1)):
        raise ValueError(f"{case}: incomplete case or mismatched dataset")
    if (record["split_seed_id"], p1["seed_id"]) != (seed, seed):
        raise ValueError(f"{case}: mismatched seed")

    full, selected = scaled_pair(case, scaled_root, dataset, seed)
    split_hash = record["split_hash"]
    if not split_hash or any(obj["split_hash"] != split_hash for obj in (p1, full, selected)):
        raise ValueError(f"{case}: split fingerprints differ")

    for mode, evaluated in (("full", full), ("selected", selected)):
        if (evaluated["dataset"], evaluated["seed_id"], evaluated["mode"]) != (dataset, seed, mode):
            raise ValueError(f"{case}: mismatched {mode} CNN identity")
        if len(evaluated["features"]) != evaluated["dimension"] or not all(
            math.isfinite(evaluated[key]) and 0 <= evaluated[key] <= 1 for key in METRICS
        ):
            raise ValueError(f"{case}: invalid {mode} CNN dimension or score")

    features = p1["feature_names"]
    selection = record["selection"]
    final = selection["final_features"]
    if not features or len(set(features)) != len(features) or (
        len(features) != selection["raw_dimension"] or len(features) != record["taxonomy"]["n_features"]
    ):
        raise ValueError(f"{case}: invalid feature universe")
    if (not final or len(final) != len(set(final)) or len(final) != selection["final_dimension"]
            or not set(selection["p2_features"]) <= set(final) <= set(features)):
        raise ValueError(f"{case}: invalid final selection")
    if (set(full["features"]) != set(features) or full["dimension"] != len(features)
            or selected["features"] != final):
        raise ValueError(f"{case}: evaluated columns differ from the full or final subset")

    return {
        "features": features,
        "n_samples": int(record["taxonomy"]["n_samples"]),
        "dataset_sha256": p1["dataset_sha256"],
        "development_csv": case / "selector_input_train_validation.csv",
        "full": full,
        "selected": selected,
    }


def describe(values):
    return {"mean": statistics.mean(values), "sd": statistics.stdev(values) if len(values) > 1 else None}


def holm(raw_p):
    adjusted = {}
    previous = 0.0
    order = sorted(raw_p, key=lambda key: 1.0 if raw_p[key] is None else raw_p[key])
    for index, key in enumerate(order):
        value = raw_p[key]
        if value is None:
            adjusted[key] = None
        else:
            previous = max(previous, min(1.0, (len(order) - index) * value))
            adjusted[key] = previous
    return adjusted


def paired_p(differences):
    if not any(differences):
        return 1.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sample size too small for normal approximation.")
        warnings.filterwarnings("ignore", message="Exact p-value calculation does not work if there are zeros.*")
        return float(wilcoxon(
            differences, zero_method="wilcox", alternative="two-sided", correction=False, method="auto",
        ).pvalue)


def high_correlation_pair_fraction(path, features):
    frame = pd.read_csv(path, usecols=features)
    for name in frame.select_dtypes(exclude=["number"]).columns:
        frame[name] = LabelEncoder().fit_transform(frame[name])
    if frame.isnull().any().any():
        frame = frame.fillna(frame.mean())
    frame = frame.loc[:, frame.nunique(dropna=False) > 1]
    corr = frame.corr(method="spearman").to_numpy()
    pairs = np.abs(corr[np.triu_indices(len(frame.columns), 1)])
    if not np.isfinite(pairs).all():
        raise ValueError(f"{path}: nonfinite development-set feature correlations")
    return float(np.mean(pairs >= 0.8)) if len(pairs) else 0.0


def summarize_dataset(entries):
    first = entries[0]
    shared = ("features", "n_samples", "dataset_sha256")
    if any(case[key] != first[key] for case in entries for key in shared):
        raise ValueError(f"{first['full']['dataset']}: data or feature universe differs across seeds")
    n_features = len(first["features"])
    subsets = [set(case["selected"]["features"]) for case in entries]
    dimensions = list(map(len, subsets))
    reduction = describe([100 * (1 - size / n_features) for size in dimensions])
    table3 = {
        "n": len(entries),
        "n_samples": first["n_samples"],
        "original_dimension": n_features,
        "selected_dimension": describe(dimensions),
        "reduction": reduction,
    }
    raw_p = {}
    for metric, label in METRICS.items():
        for mode in ("full", "selected"):
            table3[f"{mode}_{label}"] = describe([100 * case[mode][metric] for case in entries])
        differences = [100 * (case["selected"][metric] - case["full"][metric]) for case in entries]
        table3[f"delta_{label}_pp"] = describe(differences)
        raw_p[metric] = paired_p(differences)

    jaccard = [len(a & b) / len(a | b) for a, b in combinations(subsets, 2)]
    similarity = describe(jaccard) if jaccard else {"mean": None, "sd": None}
    redundancy = describe([
        high_correlation_pair_fraction(case["development_csv"], case["features"]) for case in entries
    ])
    rq6 = {
        "n_samples": first["n_samples"],
        "original_dimension": n_features,
        "reduction_percent": reduction["mean"],
        "high_correlation_pair_fraction": redundancy["mean"],
        "high_correlation_pair_fraction_sd": redundancy["sd"],
        "n_pairs": len(jaccard),
        "jaccard_mean": similarity["mean"],
        "jaccard_sd": similarity["sd"],
    }
    return table3, raw_p, rq6


def spearman_tests(rows):
    rho, raw_p = {}, {}
    response = [row["jaccard_mean"] for row in rows]
    for key in SPEARMAN_VARIABLES:
        values = [row[key] for row in rows]
        rho[key] = raw_p[key] = None
        if len(rows) >= 3 and None not in response and len(set(response)) > 1 and len(set(values)) > 1:
            result = spearmanr(values, response)
            rho[key], raw_p[key] = float(result.statistic), float(result.pvalue)
    adjusted = holm(raw_p)
    return {
        key: {"rho": rho[key], "p_raw": raw_p[key], "p_holm": adjusted[key]}
        for key in SPEARMAN_VARIABLES
    }


def summarize(case_root, scaled_root, datasets, seeds):
    table3, raw_p, rq6 = {}, {}, {}
    for dataset in datasets:
        entries = [read_case(case_root, scaled_root, dataset, seed) for seed in seeds]
        table3[dataset], tests, rq6[dataset] = summarize_dataset(entries)
        raw_p.update({(dataset, metric): p for metric, p in tests.items()})
    adjusted = holm(raw_p)
    for dataset, row in table3.items():
        row["paired_tests"] = {
            metric: {
                "wilcoxon_p": raw_p[dataset, metric],
                "holm_p": adjusted[dataset, metric],
                "significant": adjusted[dataset, metric] < 0.05,
            }
            for metric in METRICS
        }
    mean_row = {
        key: statistics.mean(row[key]["mean"] for row in table3.values())
        for key in ("reduction", "delta_acc_pp", "delta_f1_pp")
    }
    return {
        "seed_ids": list(seeds),
        "table3": {"datasets": table3, "mean_row": mean_row},
        "rq6": {"datasets": rq6, "spearman": spearman_tests(list(rq6.values()))},
    }


def write_summary(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, required=True, help="root of DATASET/Sxx/record.json")
    parser.add_argument("--scaled-root", type=Path, help="root of scaled CNN pairs stored separately from cases")
    parser.add_argument("--datasets", nargs="+", help="dataset keys (default: discover cases under --case-root)")
    seed_args = parser.add_mutually_exclusive_group()
    seed_args.add_argument("--seeds", nargs="+", choices=SEEDS, help="exact seed IDs to report")
    seed_args.add_argument(
        "--expected-seeds", type=int, choices=range(1, len(SEEDS) + 1), default=len(SEEDS),
        metavar="N", help="require seeds S01 through SNN for each dataset (default: 30)",
    )
    parser.add_argument("--output", type=Path, required=True, help="summary JSON outside input roots")
    args = parser.parse_args()

    case_root = args.case_root.expanduser().resolve()
    scaled_root = args.scaled_root.expanduser().resolve() if args.scaled_root else None
    output = args.output.expanduser().resolve()
    if any(output.is_relative_to(root) for root in (case_root, scaled_root) if root is not None):
        parser.error("--output must be outside the input roots")
    seeds = args.seeds or SEEDS[:args.expected_seeds]
    if len(set(seeds)) != len(seeds):
        parser.error("seed IDs must be unique")
    try:
        datasets = args.datasets or sorted({
            record.parent.parent.name for record in case_root.glob("*/S??/record.json")
        })
        if not datasets or len(set(datasets)) != len(datasets) or any(
            not re.fullmatch(r"[A-Za-z0-9_]+", dataset) for dataset in datasets
        ):
            raise ValueError("supply unique dataset keys or a --case-root with discoverable cases")
        write_summary(output, summarize(case_root, scaled_root, datasets, seeds))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
