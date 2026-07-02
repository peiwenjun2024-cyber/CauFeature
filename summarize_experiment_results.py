import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Tuple


DEFAULT_METRICS = [
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
    parser = argparse.ArgumentParser(description="Summarize experiment CSV results by dataset and method.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input per-run CSV files.")
    parser.add_argument("--output", default="results/experiment_summary.csv", help="Output summary CSV.")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS, help="Numeric metrics to summarize.")
    parser.add_argument(
        "--group-by-setting",
        action="store_true",
        help="Keep different experiment settings as separate rows instead of merging by dataset/method only.",
    )
    return parser.parse_args()


def fmt(values: List[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.4f}"
    return f"{mean(values):.4f} ± {stdev(values):.4f}"


def main() -> None:
    args = parse_args()
    groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)

    for input_name in args.inputs:
        input_path = Path(input_name)
        if not input_path.exists():
            raise SystemExit(
                f"Missing input CSV: {input_path}. Run the corresponding experiment script first."
            )
        with input_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                method = row.get("method") or "CauFeature"
                setting = row.get("experiment_setting") or row.get("selection_rule") or ""
                row["_setting"] = setting
                group_setting = setting if args.group_by_setting else ""
                groups[(row["dataset"], method, group_setting)].append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["dataset", "method", "runs", "setting"] + args.metrics
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for (dataset, method, _), rows in sorted(groups.items()):
            summary = {"dataset": dataset, "method": method, "runs": len(rows)}
            settings = sorted({row.get("_setting", "") for row in rows if row.get("_setting", "")})
            summary["setting"] = " | ".join(settings)
            for metric in args.metrics:
                values = [float(row[metric]) for row in rows if row.get(metric) not in (None, "")]
                summary[metric] = fmt(values) if values else ""
            writer.writerow(summary)

    print(f"Wrote summary for {len(groups)} dataset/method groups to {output_path}")


if __name__ == "__main__":
    main()
