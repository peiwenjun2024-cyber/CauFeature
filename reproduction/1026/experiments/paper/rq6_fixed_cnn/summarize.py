#!/usr/bin/env python3
"""Summarize Table 9 from completed case outputs."""

import argparse
import csv
import json
import math
import statistics
import tempfile
from pathlib import Path

from scipy.stats import t

from common import DS, KINDS, load


SEEDS = tuple(f"S{index:02d}" for index in range(1, 31))
VALUES = (
    "dimension", "accuracy_percent", "f_measure_percent",
    "reduction_percent", "delta_accuracy_pp", "delta_f_measure_pp",
)


def read_evaluation(case, dataset, seed, scorer):
    path = case / "evaluation" / f"{scorer}.json"
    row = load(path)
    if (row["dataset"], row["seed_id"], row["kind"], row["evaluator"]) != (dataset, seed, scorer, "CNN"):
        raise ValueError(f"{path}: inconsistent evaluation identity")
    if not row["features"] or row["dimension"] != len(set(row["features"])) or row["dimension"] != len(row["features"]):
        raise ValueError(f"{path}: invalid feature list or dimension")
    for metric in ("accuracy", "f1_macro"):
        if not math.isfinite(row[metric]) or not 0 <= row[metric] <= 1:
            raise ValueError(f"{path}: invalid {metric}")
    if scorer != "FULL":
        selection = load(case / "selection" / f"{scorer}.json")
        if any(selection[key] != row[key] for key in ("protocol", "split_hash", "features")):
            raise ValueError(f"{path}: evaluation differs from the selection")
    return row


def paired_values(full, row):
    return (
        row["dimension"],
        100 * row["accuracy"],
        100 * row["f1_macro"],
        100 * (1 - row["dimension"] / full["dimension"]),
        100 * (row["accuracy"] - full["accuracy"]),
        100 * (row["f1_macro"] - full["f1_macro"]),
    )


def summarize(case_root, datasets, seeds, scorers):
    rows = []
    experiment_protocol = None
    for dataset in datasets:
        groups = {scorer: [] for scorer in scorers}
        for seed in seeds:
            case = case_root / dataset / seed
            full = read_evaluation(case, dataset, seed, "FULL")
            if experiment_protocol is None:
                experiment_protocol = full["protocol"]
            if not full["split_hash"] or full["protocol"] != experiment_protocol:
                raise ValueError(f"{case}: invalid split or mixed experiment protocols")
            for scorer in scorers:
                row = full if scorer == "FULL" else read_evaluation(case, dataset, seed, scorer)
                if row["protocol"] != full["protocol"] or row["split_hash"] != full["split_hash"]:
                    raise ValueError(f"{case}: {scorer} and FULL use different protocols or splits")
                if not set(row["features"]).issubset(full["features"]):
                    raise ValueError(f"{case}: {scorer} uses features outside the FULL evaluation")
                groups[scorer].append(paired_values(full, row))

        for scorer, group in groups.items():
            count = len(group)
            row = dict(dataset=dataset, scorer=scorer, evaluator="CNN", n=count, complete=count == len(SEEDS))
            for key, values in zip(VALUES, zip(*group)):
                mean = statistics.mean(values)
                sd = statistics.stdev(values) if count > 1 else None
                row.update({f"{key}_n": count, f"{key}_mean": mean, f"{key}_sd": sd})
                if key.startswith("delta_"):
                    # Paired t intervals are descriptive on overlapping splits.
                    half = float(t.ppf(.975, count - 1)) * sd / math.sqrt(count) if sd is not None else None
                    row[f"{key}_ci95_low"] = mean - half if half is not None else None
                    row[f"{key}_ci95_high"] = mean + half if half is not None else None
            rows.append(row)
    return rows


def write_summary(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False)
    temporary = Path(stream.name)
    try:
        with stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", required=True, type=Path, help="directory containing DATASET/Sxx cases")
    parser.add_argument("--output", required=True, type=Path, help="summary CSV outside the case directory")
    parser.add_argument("--datasets", nargs="+", choices=DS, default=DS)
    parser.add_argument("--seeds", nargs="+", choices=SEEDS, default=SEEDS)
    parser.add_argument("--scorers", nargs="+", choices=KINDS[1:], default=KINDS[1:],
                        help="alternative scorers to summarize alongside FULL and CNN")
    args = parser.parse_args()
    if any(len(set(values)) != len(values) for values in (args.datasets, args.seeds, args.scorers)):
        parser.error("dataset, seed, and scorer lists must not contain duplicates")
    case_root = args.case_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if output.is_relative_to(case_root):
        parser.error("--output must be outside --case-root")
    try:
        write_summary(output, summarize(case_root, args.datasets, args.seeds, ("FULL", "CNN", *args.scorers)))
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
