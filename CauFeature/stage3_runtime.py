from __future__ import annotations
import argparse, fcntl, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MAN = Path(os.environ.get("FAST30_MANIFEST", ROOT / "manifest" / "split_manifest.jsonl"))
DATASET_DIR = Path(os.environ.get("FAST30_DATASET_DIR", str(ROOT / "code_revised_20260910" / "dataset")))
from runtime_provenance import (
    METHOD_VERSION, DEFAULT_OUTPUT_NAME, atomic_json, code_provenance,
    require_cache_signature, sha256_file,
)
OUT = Path(os.environ.get("FAST30_OUTPUT_ROOT", str(ROOT / "results" / DEFAULT_OUTPUT_NAME)))
BASELINE_CACHE = Path(os.environ.get("FAST30_BASELINE_CACHE", ROOT / "results" / "shared_full_baseline_cache_profilefix"))

import shared_globals
from model_train_v6 import load_and_preprocess_data
from feature_perturbation import CausalFeaturePerturbation
from p3_recovery import ResidualInteractionRecovery
from fixed_split_evaluator import evaluate_fixed_split

def rec_for(ds, sid):
    hits = [
        json.loads(x) for x in MAN.read_text().splitlines()
        if x.strip() and json.loads(x)["dataset"] == ds and json.loads(x)["split_seed_id"] == sid
    ]
    if len(hits) != 1:
        raise RuntimeError(f"manifest match={len(hits)}")
    return hits[0]

def split_idx(r):
    f = pd.read_csv(r["split_file"])
    return [f.loc[f.partition == x, "row_index"].to_numpy(int) for x in ["train", "validation", "test"]]

def baseline_cache_path(dataset, split_hash):
    code_dir = Path(__file__).resolve().parent
    provenance = code_provenance(code_dir, [
        "stage3.py", "fixed_split_evaluator.py", "model_train_v6.py", "shared_globals.py",
        "runtime_provenance.py",
    ])
    return (
        BASELINE_CACHE
        / Path(dataset).stem
        / f"{split_hash}_{provenance['signature'][:20]}.json"
    ), provenance

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split-seed-id", required=True)
    a = ap.parse_args()

    r = rec_for(a.dataset, a.split_seed_id)
    seed = int(r["split_seed"])
    data = DATASET_DIR / a.dataset
    split_file = Path(r["split_file"])
    if sha256_file(data) != r.get("sha256"):
        raise RuntimeError(f"Dataset hash mismatch for {data}")
    if sha256_file(split_file) != r.get("split_hash"):
        raise RuntimeError(f"Split hash mismatch for {split_file}")
    rd = OUT / Path(a.dataset).stem / a.split_seed_id
    out = rd / "record.json"
    code_dir = Path(__file__).resolve().parent
    stage3_provenance = code_provenance(code_dir, [
        "stage3.py", "stage3_runtime.py", "p3_recovery.py", "feature_perturbation.py",
        "fixed_split_evaluator.py", "model_train_v6.py", "shared_globals.py",
        "runtime_provenance.py", "feature_semantic_metadata.json",
    ])
    if out.exists():
        cached_record = json.load(open(out))
        if cached_record.get("method_version") != METHOD_VERSION:
            raise RuntimeError("Final record belongs to a different method version")
        if cached_record.get("stage3_code_signature") != stage3_provenance["signature"]:
            raise RuntimeError("Final record was produced by different Stage3/evaluator code")
        print("FINAL_EXISTS_VALID", out)
        return

    tf = __import__("tensorflow")
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    p1_path = rd / "p1_cache.json"
    p2_path = rd / "p2_cache.json"
    p1 = json.load(open(p1_path))
    p2 = json.load(open(p2_path))
    p1_cache_sha256 = sha256_file(p1_path)
    p2_cache_sha256 = sha256_file(p2_path)
    if p1.get("method_version") != METHOD_VERSION or p2.get("method_version") != METHOD_VERSION:
        raise RuntimeError("P1/P2 cache method version mismatch")
    current_p1 = code_provenance(code_dir, [
        "stage1.py", "stage1_runtime.py", "model_train_v6.py", "feature_perturbation.py",
        "p1_reliability.py", "high_contribution.py", "shared_globals.py",
        "runtime_provenance.py", "feature_semantic_metadata.json",
    ])
    current_p2 = code_provenance(code_dir, [
        "stage2.py", "stage2_runtime.py", "p2_parallel_pc.py", "runtime_provenance.py",
    ])
    require_cache_signature(p1, current_p1["signature"], "P1")
    require_cache_signature(p2, current_p2["signature"], "P2")
    if (
        p2.get("input_p1_code_signature") != p1.get("code_signature")
        or p2.get("input_p1_cache_sha256") != p1_cache_sha256
    ):
        raise RuntimeError("P2 cache input P1 artifact mismatch")
    if p1.get("split_hash") != r["split_hash"] or p2.get("split_hash") != r["split_hash"]:
        raise RuntimeError("P1/P2 split hash does not match current manifest")
    shared_globals.init()
    shared_globals.random_seed = seed

    model_path = Path(p1["selector_model"])
    if not model_path.exists():
        raise RuntimeError(f"P1 selector model missing: {model_path}")
    if p1.get("selector_model_sha256") != sha256_file(model_path):
        raise RuntimeError("P1 selector model hash mismatch")

    started = time.perf_counter()
    load_and_preprocess_data(p1["selector_input"], None)
    context_restore_sec = time.perf_counter() - started
    started = time.perf_counter()
    model = tf.keras.models.load_model(str(model_path), compile=False)
    model_load_sec = time.perf_counter() - started
    shared_globals.model = model

    if list(shared_globals.feature_names) != list(p1["feature_names"]):
        raise RuntimeError("feature order mismatch")
    con = [float(p1["contribution_scores"][n]) for n in shared_globals.feature_names]
    shared_globals.con_list = con
    shared_globals.p1_contribution_reliability = {
        k: float(v["reliability"]) for k, v in p1["reliability_records"].items()
    }
    shared_globals.p1_reliable_score = {k: float(v) for k, v in p1["reliable_scores"].items()}
    shared_globals.high_contrib_features = list(p1["high_contrib_features"])

    perturb = CausalFeaturePerturbation(model, shared_globals.data, dataset_name=a.dataset)
    if p1.get("perturbation_classes") != perturb.perturbation_classes:
        raise RuntimeError("P1 perturbation classes do not match current semantic metadata")
    if p1.get("perturbation_policy") != perturb.perturbation_policy:
        raise RuntimeError("P1 perturbation policy does not match current semantic metadata")
    if p1.get("semantic_types") != perturb.semantic_types:
        raise RuntimeError("P1 semantic-type metadata does not match current feature metadata")
    if p1.get("baseline_policy") != perturb.baseline_policy:
        raise RuntimeError("P1 baseline policy does not match current feature metadata")
    if p1.get("semantic_metadata_sha256") != sha256_file(perturb.semantic_metadata_path):
        raise RuntimeError("P1 semantic metadata hash mismatch")
    cached_baseline = p1.get("baseline_values", {})
    for i, name in enumerate(shared_globals.feature_names):
        if name not in cached_baseline:
            raise RuntimeError(f"P1 cache missing baseline value for {name}")
        if not np.isclose(
            float(cached_baseline[name]),
            float(perturb.baseline_np[0, i]),
            rtol=1e-6,
            atol=1e-7,
        ):
            raise RuntimeError(f"P1 baseline mismatch for feature {name}")
    cached_values = p1.get("perturbation_values")
    if not cached_values:
        raise RuntimeError("P1 cache lacks perturbation_values; rerun Stage1 with semantic-perturbation code")
    perturb.perturb_values = {
        name: list(cached_values[name]) for name in shared_globals.feature_names
    }
    perturb.con_list = con
    shared_globals.x_list = [perturb.perturb_values[n] for n in shared_globals.feature_names]
    shared_globals.feature_values = perturb.feature_values
    shared_globals.perturbation_instance = perturb
    shared_globals.filtered_features = list(p2["p2_features"])

    started = time.perf_counter()
    p3_info = ResidualInteractionRecovery(candidate_top_k=200).run()
    p3_sec = time.perf_counter() - started
    final = list(shared_globals.filtered_features or [])
    recovered = sorted(set(final) - set(p2["p2_features"]))
    if not final:
        raise RuntimeError("empty final features")

    p3_out = rd / "p3_cache.json"
    atomic_json(p3_out, {
        "status": "complete", "method_version": METHOD_VERSION,
        "stage3_code_signature": stage3_provenance["signature"],
        "input_p1_code_signature": p1["code_signature"],
        "input_p2_code_signature": p2["code_signature"],
        "input_p1_cache_sha256": p1_cache_sha256,
        "input_p2_cache_sha256": p2_cache_sha256,
        "dataset": a.dataset, "seed_id": a.split_seed_id,
        "split_hash": r["split_hash"], "p2_features": p2["p2_features"],
        "final_features": final, "recovered_features": recovered,
        "candidate_pairs": p3_info["candidate_pairs"],
        "evaluated_pairs": p3_info["evaluated_pairs"],
        "eligible_residual_features": p3_info["eligible_residual_features"],
        "p3_sec": p3_sec,
    })

    tr, va, te = split_idx(r)
    full_features = list(p1["feature_names"])
    same_as_full = final == full_features

    started = time.perf_counter()
    metrics = evaluate_fixed_split(data, tr, va, te, seed, final)
    eval_sec = time.perf_counter() - started

    bp = rd / "baseline_record.json"
    if same_as_full:
        baseline_metrics = dict(metrics)
        b = {
            "status": "complete", "method": "full_feature_baseline",
            "dataset": a.dataset, "split_seed_id": a.split_seed_id,
            "split_seed": seed, "split_hash": r["split_hash"],
            "metrics": baseline_metrics, "timing_seconds": {"evaluator": 0.0},
            "reused_from_final": True, "reused_from_shared_cache": False,
        }
        atomic_json(bp, b)
    else:
        shared_bp, baseline_provenance = baseline_cache_path(a.dataset, r["split_hash"])
        lock_path = shared_bp.with_name(shared_bp.name + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as baseline_lock:
            fcntl.flock(baseline_lock.fileno(), fcntl.LOCK_EX)
            # Re-check only after obtaining the lock. Another runner may have
            # completed the exact same full-feature baseline while we waited.
            if shared_bp.exists():
                cached = json.load(open(shared_bp))
                if (
                    cached.get("split_hash") != r["split_hash"]
                    or cached.get("dataset_sha256") != r.get("sha256")
                    or int(cached.get("split_seed", -1)) != seed
                    or cached.get("code_signature") != baseline_provenance["signature"]
                ):
                    raise RuntimeError(f"shared baseline cache mismatch: {shared_bp}")
                baseline_metrics = cached["metrics"]
                baseline_sec = 0.0
                reused_shared = True
            else:
                started = time.perf_counter()
                baseline_metrics = evaluate_fixed_split(data, tr, va, te, seed, full_features)
                baseline_sec = time.perf_counter() - started
                atomic_json(shared_bp, {
                    "status": "complete", "method_version": METHOD_VERSION,
                    "code_signature": baseline_provenance["signature"],
                    "code_provenance": baseline_provenance["files"],
                    "dataset": a.dataset,
                    "dataset_sha256": r.get("sha256"), "split_hash": r["split_hash"],
                    "split_seed": seed, "metrics": baseline_metrics,
                })
                reused_shared = False
            fcntl.flock(baseline_lock.fileno(), fcntl.LOCK_UN)
        b = {
            "status": "complete", "method": "full_feature_baseline",
            "dataset": a.dataset, "split_seed_id": a.split_seed_id,
            "split_seed": seed, "split_hash": r["split_hash"],
            "metrics": baseline_metrics, "timing_seconds": {"evaluator": baseline_sec},
            "reused_from_final": False, "reused_from_shared_cache": reused_shared,
            "shared_cache": str(shared_bp),
        }
        atomic_json(bp, b)

    bm = b["metrics"]
    accuracy_pp = 0.0 if same_as_full else 100 * (float(metrics["accuracy"]) - float(bm["accuracy"]))
    f1_macro_pp = 0.0 if same_as_full else 100 * (float(metrics["f1_macro"]) - float(bm["f1_macro"]))
    code_dir = Path(__file__).resolve().parent

    payload = {
        "status": "complete",
        "method": METHOD_VERSION,
        "method_version": METHOD_VERSION,
        "stage3_code_signature": stage3_provenance["signature"],
        "input_cache_sha256": {
            "p1_cache.json": p1_cache_sha256,
            "p2_cache.json": p2_cache_sha256,
        },
        "dataset": a.dataset,
        "taxonomy": {
            "dataset": a.dataset, "n_samples": r.get("n_samples"),
            "n_features": r.get("n_features"), "separator": r.get("separator"),
        },
        "split_seed_id": a.split_seed_id, "split_seed": seed, "split_hash": r["split_hash"],
        "execution": {
            "cnn_deterministic": True,
            "p3_model_source": "p1_saved_model",
            "p3_perturbation_source": "p1_cache",
            "pc_workers": int(p2["workers"]),
            "perturbation_profile_fix": True,
            "semantic_baseline_version": "v4_frozen",
            "semantic_perturbation_version": "v6",
            "flat_code_directory": str(code_dir),
            "determinism_env": {
                "TF_DETERMINISTIC_OPS": os.getenv("TF_DETERMINISTIC_OPS"),
                "TF_CUDNN_DETERMINISTIC": os.getenv("TF_CUDNN_DETERMINISTIC"),
                "TF_ENABLE_ONEDNN_OPTS": os.getenv("TF_ENABLE_ONEDNN_OPTS"),
                "CUBLAS_WORKSPACE_CONFIG": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
                "PYTHONHASHSEED": os.getenv("PYTHONHASHSEED"),
            },
        },
        "selection": {
            "raw_dimension": int(p2["raw_dimension"]),
            "p1_high_contrib_features": p1["high_contrib_features"],
            "p2_dimension": int(p2["p2_dimension"]),
            "p2_features": p2["p2_features"],
            "p3_recovered_features": recovered,
            "final_dimension": len(final), "final_features": final,
            "frontier_edge_count": p2["frontier_edge_count"],
            "deep_candidate_edge_count": p2["deep_candidate_edge_count"],
            "adaptive_stopped": p2["adaptive_stopped"],
            "adaptive_stop_level": p2["adaptive_stop_level"],
            "target_path_count": p2["target_path_count"],
        },
        "timing_seconds": {
            **p1["timing"], "p2": p2["p2_sec"],
            "p3_model_load": model_load_sec,
            "p3_context_restore": context_restore_sec,
            "p3": p3_sec, "evaluator": eval_sec,
        },
        "pc": {"shallow": p2["shallow"], "deep": p2["deep"]},
        "metrics": metrics,
        "paired_full_feature_baseline": {
            "record": str(bp), "accuracy": float(bm["accuracy"]),
            "f1_macro": float(bm["f1_macro"]),
            "dimension": int(bm["selected_dimension"]),
            "reused_from_final": bool(same_as_full),
            "reused_from_shared_cache": bool(b.get("reused_from_shared_cache", False)),
        },
        "paired_delta": {
            "accuracy_pp": accuracy_pp, "f1_macro_pp": f1_macro_pp,
            "dimension_reduction_percent": 100 * (1 - len(final) / max(1, int(p2["raw_dimension"]))),
        },
        "provenance": {
            **p1.get("code_provenance", {}),
            **p2.get("code_provenance", {}),
            **stage3_provenance["files"],
            "runner.py": sha256_file(code_dir / "runner.py"),
            "stage1.py": sha256_file(code_dir / "stage1.py"),
            "stage2.py": sha256_file(code_dir / "stage2.py"),
            "stage3.py": sha256_file(code_dir / "stage3.py"),
        },
    }
    atomic_json(out, payload)
    print("FINAL_COMPLETE", json.dumps({
        "dataset": a.dataset, "seed": a.split_seed_id, "dim": len(final),
        "dAcc": accuracy_pp, "dF1": f1_macro_pp, "P3_added": len(recovered),
    }), flush=True)

if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
    except Exception as exc:
        print(f"STAGE3_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        os._exit(1)
