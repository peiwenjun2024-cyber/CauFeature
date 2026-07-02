import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


METRIC_COLUMNS = [
    "selected_dim",
    "reduction_pct",
    "original_accuracy",
    "selected_accuracy",
    "accuracy_delta_pct_points",
    "original_f1",
    "selected_f1",
    "f1_delta_pct_points",
    "total_time_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CauFeature per-seed CSV results.")
    parser.add_argument("--input", default="results/caufeature_runs.csv", help="Input per-run CSV.")
    parser.add_argument("--output", default="results/caufeature_summary.csv", help="Output summary CSV.")
    return parser.parse_args()


def fmt(values: List[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.6g}"
    return f"{mean(values):.6g} +/- {stdev(values):.6g}"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    with input_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            groups[row["dataset"]].append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["dataset", "runs"] + METRIC_COLUMNS
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, rows in sorted(groups.items()):
            summary = {"dataset": dataset, "runs": len(rows)}
            for column in METRIC_COLUMNS:
                values = [float(row[column]) for row in rows]
                summary[column] = fmt(values)
            writer.writerow(summary)

    print(f"Wrote summary for {len(groups)} datasets to {output_path}")


if __name__ == "__main__":
    main()
