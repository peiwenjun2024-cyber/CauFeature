import os
import sys
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PACKAGE/'scripts'))
from isolated_worker import required_output
SOURCE=Path(os.environ.get('FAST30_SOURCE_ROOT','/root/autodl-tmp/fast30_final')).resolve()
import csv
import os
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = required_output("RQ5_OUTPUT_ROOT",forbidden=(SOURCE/"results",))
FILES = sorted((ROOT / "cases").glob("*.json"))
STRATEGIES = ("default", "quantile", "uniform")
TIE = 1e-7


def stats(rows, rng):
    by_seed = defaultdict(list)
    feature_results = []
    for r in rows:
        e = [r["reference_js"] - r[f"{name}_js"] for name in STRATEGIES]
        assert min(e) >= -1e-15
        feature_results.append(e)
        by_seed[r["dataset"], r["seed"]].append(e)
    assert by_seed
    errors = np.array(feature_results)
    cases = defaultdict(list)
    for (dataset, _), group in by_seed.items():
        cases[dataset].append(np.mean(group, axis=0))
    # Within dataset, average features within the same seed before inference.
    # Pooled strata give each available dataset the same weight.
    blocks = [np.array(items) for items in cases.values()]
    seed_means = np.mean([block.mean(axis=0) for block in blocks], axis=0)
    boot = []
    for block in blocks:
        ids = rng.integers(0, len(block), size=(10000, len(block)))
        boot.append(block[ids].mean(axis=1))
    boot = np.mean(boot, axis=0)

    def paired(other):
        delta = errors[:, other] - errors[:, 0]
        contrast = boot[:, other] - boot[:, 0]
        return dict(delta=float(seed_means[other] - seed_means[0]),
                    ci95=np.quantile(contrast, [.025, .975]).tolist(),
                    wins=int(np.sum(delta > TIE)), ties=int(np.sum(abs(delta) <= TIE)),
                    losses=int(np.sum(delta < -TIE)))

    return dict(n_feature_seed=len(rows), n_dataset_seed=len(by_seed),
        actual_points_min=min(r["actual_points"] for r in rows),
        actual_points_median=float(np.median([r["actual_points"] for r in rows])),
        actual_points_max=max(r["actual_points"] for r in rows),
        mean_error=dict(zip(STRATEGIES, seed_means.tolist())),
        median_error=dict(zip(STRATEGIES, np.median(errors, axis=0).tolist())),
        p90_error=dict(zip(STRATEGIES, np.quantile(errors, .9, axis=0).tolist())),
        vs_quantile=paired(1), vs_uniform=paired(2))


records = [json.loads(path.read_text()) for path in FILES]
rows = [r for case in records for r in case["rows"]]
if not rows:
    raise SystemExit("No data yet")
rng = np.random.default_rng(20261114)
complete = len(FILES) == 150
if complete:
    assert len(rows) == 510, len(rows)
    assert Counter(r["branch"] for r in rows) == {"quantile": 367, "uniform": 143}
    assert len({(r["dataset"], r["seed"], r["feature"]) for r in rows}) == 510
for r in rows:
    assert r["default_js"] == r[r["branch"] + "_js"]
    assert r["reference_js"] >= r["coarse_reference_js"]
    assert r["reference_js"] >= max(r["default_js"], r["uniform_js"], r["quantile_js"])
diff = [r["reference_js"] - max(r["coarse_reference_js"], r["native_js"]) for r in rows]
parity = [abs(r["native_js"] - r["archived_js"]) for r in rows]
summary = dict(complete=complete, n_cases=len(FILES),
    cases_by_dataset=dict(Counter(r["dataset"] for r in records)),
    continuous_feature_seed_count=len(rows),
    branch_counts=dict(Counter(r["branch"] for r in rows)),
    split_and_model_checked=True,
    dense_points=[min(r["dense_points"] for r in rows), max(r["dense_points"] for r in rows)],
    dense_refinement=dict(mean=float(np.mean(diff)), p95=float(np.quantile(diff,.95)),
                          maximum=max(diff), n_over_1e7=sum(x > 1e-7 for x in diff),
                          n_over_1e6=sum(x > 1e-6 for x in diff)),
    archived_native_difference=dict(median=float(np.median(parity)), p95=float(np.quantile(parity,.95)),
                                    maximum=max(parity), n_over_1e6=sum(x > 1e-6 for x in parity)),
    by_dataset={}, by_dataset_and_branch={}, pooled_by_branch={})
for dataset in sorted({r["dataset"] for r in rows}):
    z = [r for r in rows if r["dataset"] == dataset]
    summary["by_dataset"][dataset] = stats(z, rng)
    for branch in sorted({r["branch"] for r in z}):
        summary["by_dataset_and_branch"][dataset + "/" + branch] = stats(
            [r for r in z if r["branch"] == branch], rng)
for branch in sorted({r["branch"] for r in rows}):
    summary["pooled_by_branch"][branch] = stats([r for r in rows if r["branch"] == branch], rng)
summary["pooled_macro_dataset"] = stats(rows, rng)
if complete:
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2))
    with (ROOT / "per_feature_seed.csv").open("w", newline="") as out:
        header = list(rows[0]) + ["default_error", "quantile_error", "uniform_error"]
        writer = csv.DictWriter(out, header)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row, **{f"{s}_error": row["reference_js"] - row[f"{s}_js"]
                                         for s in STRATEGIES}))
print(json.dumps(summary, separators=(",", ":")))
