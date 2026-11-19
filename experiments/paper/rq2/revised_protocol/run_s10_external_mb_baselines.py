"""Run PPFS, BAMB-Z, and EAMB-Z on the existing fixed S10 splits.

Only the three external methods are executed.  Historical CauFeature,
FeatureX, SHAP, and LIME records are never recomputed.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
RESULTS = CODE.parent / "results_revised_20260910"
RAW = RESULTS / "raw" / "external_mb_baselines_s10_20260917"
DERIVED = RESULTS / "derived"
FULL_RAW = RESULTS / "raw" / "rq1_s10" / "full"
DATASETS = ("MagicGamma", "BrainMethod", "SoftwareQuality")
METHODS = ("PPFS", "BAMB-Z", "EAMB-Z")
CAUSAL = CODE / "third_party" / "CausalLearner" / "source" / "Causal_Learner"
EAMB = CODE / "third_party" / "EAMB" / "EAMB-code (matlab)"
BAMB_FILE = CAUSAL / "alg_MB" / "BAMB" / "BAMB_Z.m"
EAMB_FILE = EAMB / "EAMB.m"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def git_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def octave_quote(path: Path) -> str:
    return str(path).replace("'", "''")


def source_provenance() -> dict[str, Any]:
    return {
        "PyImpetus_version": importlib.metadata.version("PyImpetus"),
        "CausalLearner_commit": git_commit(CODE / "third_party" / "CausalLearner"),
        "EAMB_commit": git_commit(CODE / "third_party" / "EAMB"),
        "octave_binary": shutil.which("octave"),
        "bamb_entry": str(BAMB_FILE.relative_to(CODE)),
        "eamb_entry": str(EAMB_FILE.relative_to(CODE)),
        "octave_compatibility": (
            "In-memory distchck scalar-parameter shim only; upstream algorithm files unchanged."
        ),
    }


def load_full_split(dataset: str, seed_id: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    record_path = FULL_RAW / dataset / seed_id / "record.json"
    full = read_json(record_path)
    split = pd.read_csv(full["split_file"])
    parts = {
        name: split.loc[split["partition"] == name, "row_index"].to_numpy(dtype=int)
        for name in ("train", "validation", "test")
    }
    for name, indices in parts.items():
        expected = full[f"{name}_index_hash"]
        observed = __import__("hashlib").sha256(np.sort(np.asarray(indices, dtype=np.int64)).tobytes()).hexdigest()
        if observed != expected:
            raise RuntimeError(f"{dataset}/{seed_id}: {name} index hash mismatch")
    return full, parts["train"], parts["validation"], parts["test"]


def selector_data(dataset: str, seed_id: str) -> tuple[dict[str, Any], Path, Any, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from revised_protocol.fixed_split_evaluator import _encode_frame

    full, train, validation, test = load_full_split(dataset, seed_id)
    data_path = CODE / "dataset" / f"{dataset}.csv"
    x, y = _encode_frame(data_path, None)
    selector_idx = np.sort(np.unique(np.r_[train, validation]))
    if np.intersect1d(selector_idx, test).size:
        raise RuntimeError("test observations reached feature selection")
    return full, data_path, x, y, train, validation, test


def ppfs_select(x: Any, y: np.ndarray, indices: np.ndarray, seed: int, args: argparse.Namespace) -> list[str]:
    from PyImpetus import PPIMBC
    from sklearn.tree import DecisionTreeClassifier

    model = PPIMBC(
        model=DecisionTreeClassifier(random_state=seed),
        p_val_thresh=0.05,
        num_simul=args.ppfs_num_simul,
        simul_size=0.2,
        simul_type=1,
        sig_test_type="non-parametric",
        cv=0,
        random_state=seed,
        n_jobs=args.ppfs_n_jobs,
        verbose=0,
    )
    model.fit(x.iloc[indices].reset_index(drop=True), y[indices])
    return [str(column) for column in model.MB]


def octave_select(method: str, x: Any, y: np.ndarray, indices: np.ndarray, args: argparse.Namespace) -> list[str]:
    values = x.iloc[indices].to_numpy(dtype=float)
    data = np.column_stack((values, y[indices].astype(float)))
    with tempfile.TemporaryDirectory(prefix="external_mb_", dir=RAW) as temp:
        temp_path = Path(temp)
        input_csv, output_csv = temp_path / "input.csv", temp_path / "selected.csv"
        np.savetxt(input_csv, data, delimiter=",")
        common = CAUSAL / "common"
        shim = "function [e,a,b,c]=distchck(n,a,b,c); e=0; endfunction;"
        if method == "BAMB-Z":
            call = f"[mb,~,~]=BAMB_Z(D,p,0.05,rows(D),p,{args.bamb_max_k});"
            paths = (CAUSAL / "alg_MB" / "BAMB", common)
        else:
            call = "[mb,~]=EAMB(D,'con',p,0.05,0.05);"
            paths = (EAMB, EAMB / "common_fuc")
        addpath = "".join(f"addpath('{octave_quote(path)}');" for path in paths)
        program = (
            f"pkg load statistics;{shim}{addpath}"
            f"D=csvread('{octave_quote(input_csv)}');p=columns(D);{call}"
            f"csvwrite('{octave_quote(output_csv)}',mb(:));"
        )
        run = subprocess.run(
            ["octave", "--no-gui", "--quiet", "--eval", program],
            capture_output=True, text=True, timeout=args.octave_timeout,
        )
        if run.returncode:
            raise RuntimeError(f"{method} failed: {run.stderr[-1500:]}")
        if not output_csv.exists() or output_csv.stat().st_size == 0:
            return []
        selected = np.atleast_1d(np.loadtxt(output_csv, dtype=int))
    valid = sorted({int(index) for index in selected if 1 <= int(index) <= x.shape[1]})
    return x.columns[np.asarray(valid, dtype=int) - 1].astype(str).tolist()


def write_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run_one(dataset: str, seed_id: str, method: str, args: argparse.Namespace) -> None:
    output = RAW / method / dataset / seed_id / "record.json"
    if output.exists() and read_json(output).get("status") == "complete":
        print(f"SKIP {dataset} {seed_id} {method}", flush=True)
        return
    full, data_path, x, y, train, validation, test = selector_data(dataset, seed_id)
    selector_idx = np.sort(np.unique(np.r_[train, validation]))
    started = time.perf_counter()
    try:
        if method == "PPFS":
            selected = ppfs_select(x, y, selector_idx, int(seed_id[1:]), args)
        else:
            selected = octave_select(method, x, y, selector_idx, args)
        selected = list(dict.fromkeys(selected))
        if not selected:
            raise RuntimeError("native selector returned an empty Markov blanket")
        from revised_protocol.fixed_split_evaluator import evaluate_fixed_split
        metrics = evaluate_fixed_split(data_path, train, validation, test, int(seed_id[1:]), selected)
        record = {
            "status": "complete", "dataset": dataset, "seed_id": seed_id, "method": method,
            "matching_type": "native_markov_blanket_fixed_split",
            "split_hash": full["split_hash"], "full_reference_run_id": full["run_id"],
            "selected_features": selected, "actual_selected_dimension": len(selected),
            "original_dimension": int(x.shape[1]), "selection_seconds": time.perf_counter() - started,
            "metrics": metrics, "source_provenance": source_provenance(),
            "selector_parameters": {
                "PPFS": {"num_simul": args.ppfs_num_simul, "p_val_thresh": 0.05, "cv": 0},
                "BAMB-Z": {"alpha": 0.05, "maxK": args.bamb_max_k},
                "EAMB-Z": {"alpha": 0.05, "k_or": 0.05},
            }[method],
        }
    except Exception as exc:
        record = {
            "status": "failed", "dataset": dataset, "seed_id": seed_id, "method": method,
            "error_type": type(exc).__name__, "error": str(exc),
            "split_hash": full["split_hash"], "source_provenance": source_provenance(),
        }
    write_record(output, record)
    print(f"{record['status'].upper()} {dataset} {seed_id} {method}: {output}", flush=True)


def static_check(args: argparse.Namespace) -> None:
    checks = {
        "octave": shutil.which("octave") is not None,
        "bamb_source": BAMB_FILE.is_file(),
        "eamb_source": EAMB_FILE.is_file(),
        "raw_is_separate": "matched_comparison_s10" not in str(RAW),
        "methods_are_new_only": set(args.methods).issubset(METHODS),
    }
    for dataset in args.datasets:
        for seed_id in args.seeds:
            full, train, validation, test = load_full_split(dataset, seed_id)
            checks[f"{dataset}_{seed_id}_fixed_split"] = bool(
                full["split_hash"] and not np.intersect1d(np.r_[train, validation], test).size
            )
    if not all(checks.values()):
        raise RuntimeError(json.dumps(checks, indent=2))
    DERIVED.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "EXTERNAL_MB_BASELINES_STATIC_OK", "checks": checks,
        "source_provenance": source_provenance(), "datasets": list(args.datasets),
        "seeds": list(args.seeds), "methods": list(args.methods),
    }
    (DERIVED / "external_mb_baselines_s10_20260917_static_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(report["status"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--seeds", nargs="+", default=[f"S{i:02d}" for i in range(1, 11)])
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--static-check", action="store_true")
    parser.add_argument("--ppfs-num-simul", type=int, default=5)
    parser.add_argument("--ppfs-n-jobs", type=int, default=-1)
    parser.add_argument("--bamb-max-k", type=int, default=3)
    parser.add_argument("--octave-timeout", type=int, default=1800)
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    if args.static_check:
        static_check(args)
        return
    for dataset in args.datasets:
        for seed_id in args.seeds:
            for method in args.methods:
                run_one(dataset, seed_id, method, args)


if __name__ == "__main__":
    main()
