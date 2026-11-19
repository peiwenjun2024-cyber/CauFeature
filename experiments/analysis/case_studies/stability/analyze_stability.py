"""Compute selection stability from archived Table 10 records or fresh RQ1 cases."""

import argparse
import csv
import itertools
import json
import os
import statistics
import tempfile
from collections import Counter
from pathlib import Path


HISTORICAL_ROOT = Path("/root/autodl-tmp/fast30_final")
MAIN = Path("results/alpha001_full30_20261009")
SCALED = Path("results/scaled_evaluator_full_vs_selected_20261010/cases")
DATASETS = ("BankFull", "OnlineShoppers", "StudentsAcademic", "Optdigits", "SoftwareQuality")
SEEDS = tuple(f"S{index:02d}" for index in range(1, 31))


def archive_path(root, reference):
    path = Path(reference)
    if path.is_relative_to(HISTORICAL_ROOT):
        return root / path.relative_to(HISTORICAL_ROOT)
    return root / path


def manifest_records(root, datasets):
    requested = set(datasets) - {"Optdigits"}
    records = {}
    if requested:
        path = root / MAIN / "official_main_results_15x30_manifest.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = row["dataset"], row["seed_id"]
                if key[0] in requested:
                    if key in records:
                        raise ValueError(f"duplicate manifest row for {key[0]} {key[1]}")
                    records[key] = archive_path(root, row["record_json"])
    return records


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def selection_for_seed(root, records, dataset, seed, case_root, scaled_root):
    if case_root is not None:
        record_path = case_root / dataset / seed / "record.json"
        evaluation_path = (
            scaled_root / dataset / seed / "selected_scaled.json"
            if scaled_root is not None
            else None
        )
    elif dataset == "Optdigits":
        record_path = root / MAIN / dataset / seed / "record.json"
        evaluation_path = record_path.with_name("selected_scaled.json")
    else:
        record_path = records[dataset, seed]
        evaluation_path = root / SCALED / dataset / seed / "selected_scaled.json"

    record = read_json(record_path)
    reference_cache = (
        record.get("source_policy", {}).get("p1_cache") if case_root is None else None
    )
    cache_path = (
        archive_path(root, reference_cache)
        if reference_cache
        else record_path.with_name("p1_cache.json")
    )
    cache = read_json(cache_path)
    selection = record["selection"]
    final = selection["final_features"]
    universe = set(cache["feature_names"])
    stage_sets = (
        set(selection["p1_high_contrib_features"]),
        set(selection["p2_features"]),
        set(final),
    )

    if case_root is not None and record["status"] != "complete":
        raise ValueError(f"{dataset} {seed}: incomplete selection record")
    if record["split_hash"] != cache["split_hash"]:
        raise ValueError(f"{dataset} {seed}: selection and P1 split hashes differ")
    if evaluation_path is not None:
        evaluation = read_json(evaluation_path)
        if record["split_hash"] != evaluation["split_hash"]:
            raise ValueError(f"{dataset} {seed}: selection and CNN split hashes differ")
        if final != evaluation["features"] or len(final) != evaluation["dimension"]:
            raise ValueError(f"{dataset} {seed}: CNN final feature order or dimension differs")
    if len(final) != len(stage_sets[2]) or len(universe) != selection["raw_dimension"]:
        raise ValueError(f"{dataset} {seed}: feature universe or final subset has an invalid dimension")
    if any(not group <= universe for group in stage_sets) or not stage_sets[1] <= stage_sets[2]:
        raise ValueError(f"{dataset} {seed}: selection contains inconsistent stage subsets")
    return universe, stage_sets[2]


def summarize(root, records, dataset, seeds, case_root=None, scaled_root=None):
    cases = [
        selection_for_seed(root, records, dataset, seed, case_root, scaled_root)
        for seed in seeds
    ]
    universe = cases[0][0]
    if any(reference != universe for reference, _ in cases):
        raise ValueError(f"{dataset}: feature universe changed between seeds")

    subsets = [subset for _, subset in cases]
    dimensions = [len(subset) for subset in subsets]
    reductions = [100 * (len(universe) - size) / len(universe) for size in dimensions]
    jaccards = [
        len(left & right) / len(left | right) if left or right else 1.0
        for left, right in itertools.combinations(subsets, 2)
    ]
    counts = Counter(feature for subset in subsets for feature in subset)
    return {
        "dataset": dataset,
        "n_seeds": len(subsets),
        "pair_count": len(jaccards),
        "original_dimension": len(universe),
        "dimension_mean": statistics.mean(dimensions),
        "dimension_sd": statistics.stdev(dimensions),
        "reduction_percent_mean": statistics.mean(reductions),
        "reduction_percent_sd": statistics.stdev(reductions),
        "pairwise_jaccard_mean": statistics.mean(jaccards),
        "pairwise_jaccard_sd": statistics.stdev(jaccards) if len(jaccards) > 1 else None,
        "consensus_count": sum(count == len(subsets) for count in counts.values()),
        "ever_selected_count": len(counts),
    }


def write_summary(output, rows):
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False
    )
    temporary = Path(stream.name)
    try:
        with stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--source-root",
        type=Path,
        help="archive root (default: FAST30_SOURCE_ROOT or the original AutoDL root)",
    )
    source.add_argument(
        "--case-root",
        type=Path,
        help="fresh RQ1 root containing DATASET/Sxx/record.json and p1_cache.json",
    )
    parser.add_argument(
        "--scaled-root",
        type=Path,
        help="optional fresh evaluator root containing DATASET/Sxx/selected_scaled.json",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        choices=SEEDS,
        help="fresh run seed IDs (default: all thirty, at least two required)",
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, required=True)
    parser.add_argument("--output", type=Path, required=True, help="summary CSV outside the archive")
    args = parser.parse_args()
    if args.case_root is None and (args.seeds is not None or args.scaled_root is not None):
        parser.error("--seeds and --scaled-root require --case-root")
    source_root = args.source_root or Path(os.environ.get("FAST30_SOURCE_ROOT", HISTORICAL_ROOT))
    root = source_root.expanduser().resolve()
    case_root = args.case_root.expanduser().resolve() if args.case_root else None
    scaled_root = args.scaled_root.expanduser().resolve() if args.scaled_root else None
    seeds = args.seeds if args.seeds is not None else SEEDS
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        parser.error("--seeds requires at least two distinct seeds, with no duplicates")
    output = args.output.expanduser().resolve()
    input_roots = (root,) if case_root is None else (case_root, scaled_root)
    if any(path is not None and output.is_relative_to(path) for path in input_roots):
        parser.error("--output must be outside the input roots")
    try:
        records = manifest_records(root, args.datasets) if case_root is None else {}
        rows = [
            summarize(root, records, ds, seeds, case_root, scaled_root)
            for ds in dict.fromkeys(args.datasets)
        ]
        write_summary(output, rows)
    except (KeyError, OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
