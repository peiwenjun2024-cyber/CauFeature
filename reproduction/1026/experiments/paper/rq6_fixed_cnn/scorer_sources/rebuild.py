#!/usr/bin/env python3
"""Rebuild the Table 9 P1 scorer inputs using a fresh RQ1 case."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[3]
DATASETS = ("BankFull", "OnlineShoppers", "StudentsAcademic", "Optdigits", "SoftwareQuality")
SEEDS = tuple(f"S{seed:02d}" for seed in range(1, 31))
SCORERS = ("CNN", "MLP", "RF")
SOURCES = ("common.py", "prepare.py", "worker.py", "exact_scoring.py", "align_scores.py")
HISTORICAL_ROOT = Path("/root/autodl-tmp/fast30_final")

sys.path.insert(1, str(PACKAGE / "scripts"))
from isolated_worker import finish_worker


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def locked_runtime(dataset):
    lock = read_json(HERE.parent / "protocol.lock.json")["code_hashes"]
    for name in SOURCES:
        historical = HISTORICAL_ROOT / "code_p1_model_sensitivity_20261023" / name
        if sha(HERE / name) != lock[str(historical)]:
            raise ValueError(f"Historical scorer source failed checksum: {name}")
    frozen = HISTORICAL_ROOT / "code_ablation_five_20261028" / "frozen_runtime"
    if dataset == "Optdigits":
        runtime = PACKAGE / "02_runtime_variants" / "Optdigits"
        historical_runtime = HISTORICAL_ROOT / "code_optdigits_30_20261011"
    else:
        runtime = PACKAGE / "06_ablation" / "frozen_runtime"
        historical_runtime = frozen
    for name in ("model_train_v6.py", "high_contribution.py"):
        expected = lock.get(str(historical_runtime / name), lock[str(frozen / name)])
        if not (runtime / name).is_file():
            raise ValueError(f"Restore the frozen runtime before running Table 9: {runtime}")
        if sha(runtime / name) != expected:
            raise ValueError(f"Historical runtime failed checksum: {runtime / name}")
    return runtime


def fresh_p1(case, dataset, seed):
    path = case / "p1_cache.json"
    p1 = read_json(path)
    if (p1["dataset"] not in (dataset, f"{dataset}.csv")
            or p1["seed_id"] != seed or p1["status"] != "complete"):
        raise ValueError(f"P1 cache does not match {dataset}/{seed}: {path}")
    for key in ("selector_input", "selector_model"):
        artifact = Path(p1[key]).resolve(strict=True)
        if not artifact.is_relative_to(case):
            raise ValueError(f"P1 {key} is outside the supplied fresh case: {artifact}")
    if sha(Path(p1["selector_model"])) != p1["selector_model_sha256"]:
        raise ValueError(f"P1 model checksum changed: {path}")
    return path, p1


def run_phase(args, runtime, p1_path):
    # These exact source files import their historical common module by name.
    # Patch only its case locations before importing the historical stages.
    import common

    common.OUT = args.study_case.parents[2]
    common.case_dir = lambda dataset, seed: args.study_case
    common.source = lambda dataset, seed: p1_path
    common.source_code = lambda dataset: runtime
    (common.OUT / "logs").mkdir(parents=True, exist_ok=True)

    if args.phase == "prepare":
        from prepare import prepare_case
        prepare_case(args.dataset, args.seed)
    elif args.phase.startswith("align-"):
        from align_scores import align
        align(args.dataset, args.seed, args.phase.removeprefix("align-"))
    else:
        import worker
        if args.phase == "MLP":
            worker.neural_case(args.dataset, args.seed)
        elif args.phase == "RF":
            worker.rf_case(args.dataset, args.seed)
        else:
            tf = worker.init_gpu()
            meta, arrays = worker.validate_shared(args.study_case)
            with arrays:
                worker.cnn_case(args.dataset, args.seed, tf, meta, arrays)


def check_shared(p1_path, p1, dest):
    shared = read_json(dest / "shared.json")
    if (shared["source_p1_sha256"] != sha(p1_path)
            or shared["selector_input_sha256"] != sha(Path(p1["selector_input"]))
            or shared["shared_npz_sha256"] != sha(dest / "shared.npz")):
        raise ValueError(f"Existing scorer inputs differ from the fresh case: {dest}")
    return shared


def check_models(dest, scorers, shared, p1):
    for scorer in scorers:
        record = read_json(dest / f"{scorer}.json")
        if (record["status"] != "complete" or record["model"] != scorer
                or record["shared_npz_sha256"] != shared["shared_npz_sha256"]):
            raise ValueError(f"{scorer} does not match the rebuilt shared inputs: {dest}")
        if scorer == "CNN":
            if not record["frozen_score_reused"] or record["frozen_model_sha256"] != p1["selector_model_sha256"]:
                raise ValueError(f"CNN record does not match the refreshed P1 checkpoint: {dest}")
        else:
            model = dest / ("MLP.keras" if scorer == "MLP" else "RF.joblib")
            if not record["numeric_precision_alignment"]["completed"] or sha(model) != record["model_sha256"]:
                raise ValueError(f"{scorer} checkpoint or precision alignment is incomplete: {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("seed", choices=SEEDS)
    parser.add_argument("--core-case", type=Path, required=True)
    parser.add_argument("--study-case", type=Path, required=True)
    parser.add_argument("--scorers", nargs="+", choices=SCORERS, default=SCORERS)
    parser.add_argument("--phase", choices=("prepare", "CNN", "MLP", "RF", "align-MLP", "align-RF"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.core_case = args.core_case.resolve()
    args.study_case = args.study_case.resolve()
    if args.study_case.parts[-3:] != ("cases", args.dataset, args.seed):
        parser.error("--study-case must end in cases/DATASET/SEED")
    study = args.study_case.parents[2]
    if any(study.is_relative_to(path) or path.is_relative_to(study)
           for path in (PACKAGE, args.core_case, HISTORICAL_ROOT)):
        parser.error("The scorer study directory must be separate from the input and package trees.")

    # Set these before importing NumPy, scikit-learn or TensorFlow, as in the study.
    env = os.environ.copy()
    env.update({name: "1" for name in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "PYTHONDONTWRITEBYTECODE",
    )})
    runtime = locked_runtime(args.dataset)
    p1_path, p1 = fresh_p1(args.core_case, args.dataset, args.seed)
    if args.phase:
        os.environ.update(env)
        run_phase(args, runtime, p1_path)
        finish_worker(0)

    def phase(name):
        subprocess.run([sys.executable, str(Path(__file__).resolve()), args.dataset, args.seed,
                        "--core-case", str(args.core_case), "--study-case", str(args.study_case),
                        "--phase", name], env=env, check=True)

    scorers = tuple(dict.fromkeys(("CNN", *args.scorers)))
    phase("prepare")
    shared = check_shared(p1_path, p1, args.study_case)
    phase("MLP" if "MLP" in scorers else "CNN")
    if "RF" in scorers:
        phase("RF")
    for scorer in scorers:
        if scorer != "CNN":
            phase(f"align-{scorer}")
    check_models(args.study_case, scorers, shared, p1)
    print(f"REBUILT {args.dataset}/{args.seed} P1 scorer inputs ({', '.join(scorers)})", flush=True)


if __name__ == "__main__":
    main()
