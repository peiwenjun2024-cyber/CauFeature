from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np, pandas as pd
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
MAN = Path(os.environ.get("FAST30_MANIFEST", ROOT / "manifest" / "split_manifest.jsonl"))
DATASET_DIR = Path(os.environ.get("FAST30_DATASET_DIR", str(ROOT / "code_revised_20260910" / "dataset")))
from runtime_provenance import METHOD_VERSION, DEFAULT_OUTPUT_NAME, atomic_json, code_provenance, require_cache_signature, sha256_file
OUT = Path(os.environ.get("FAST30_OUTPUT_ROOT", str(ROOT / "results" / DEFAULT_OUTPUT_NAME)))

import shared_globals
from model_train_v6 import train
from feature_perturbation import CausalFeaturePerturbation
from high_contribution import select_high_contribution_features
from p1_reliability import compute_reliability_batched

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
    rd.mkdir(parents=True, exist_ok=True)
    out = rd / "p1_cache.json"
    code_dir = Path(__file__).resolve().parent
    provenance = code_provenance(code_dir, [
        "stage1.py", "stage1_runtime.py", "model_train_v6.py", "feature_perturbation.py",
        "p1_reliability.py", "high_contribution.py", "shared_globals.py",
        "runtime_provenance.py", "feature_semantic_metadata.json",
    ])
    if out.exists():
        cached = json.load(open(out))
        require_cache_signature(cached, provenance["signature"], "P1")
        if (
            cached.get("dataset") != a.dataset
            or cached.get("seed_id") != a.split_seed_id
            or cached.get("split_hash") != r["split_hash"]
            or cached.get("dataset_sha256") != r["sha256"]
        ):
            raise RuntimeError("P1 cache dataset/split identity mismatch")
        model_path = Path(cached.get("selector_model", ""))
        if not model_path.exists():
            raise RuntimeError(f"P1 cache is complete but selector model is missing: {model_path}")
        if cached.get("selector_model_sha256") != sha256_file(model_path):
            raise RuntimeError("P1 selector model hash mismatch")
        print("P1_EXISTS_VALID", out)
        return

    tr, va, _ = split_idx(r)
    source = None
    last_decode_error = None
    for encoding in ("utf-8-sig", "gbk"):
        try:
            # Use the manifest separator rather than dialect inference. utf-8-sig
            # removes a possible BOM from the first feature name.
            source = pd.read_csv(data, sep=r["separator"], encoding=encoding)
            break
        except UnicodeDecodeError as exc:
            last_decode_error = exc
    if source is None:
        raise RuntimeError(f"Unable to decode dataset {data}") from last_decode_error
    if len(source) != int(r["n_samples"]) or len(source.columns) - 1 != int(r["n_features"]):
        raise RuntimeError(
            f"Dataset shape mismatch for {a.dataset}: "
            f"got={source.shape}, manifest=({r['n_samples']}, {int(r['n_features']) + 1})"
        )
    selector = np.concatenate([tr, va])
    selector_input = rd / "selector_input_train_validation.csv"
    source.iloc[selector].to_csv(selector_input, index=False)

    shared_globals.init()
    shared_globals.random_seed = seed
    started = time.perf_counter()
    model = train(str(selector_input), None)
    model_sec = time.perf_counter() - started

    model_path = rd / "p1_selector_model.keras"
    started = time.perf_counter()
    model.save(str(model_path))
    model_save_sec = time.perf_counter() - started

    perturb = CausalFeaturePerturbation(model, shared_globals.data, dataset_name=a.dataset)
    x_list = perturb.disturbance_range()
    perturb.perturb_values = {
        shared_globals.feature_names[i]: x_list[i]
        for i in range(len(shared_globals.feature_names))
    }

    started = time.perf_counter()
    _, con = perturb.calculate_contributions(x_list)
    contrib_sec = time.perf_counter() - started
    con = list(map(float, con))
    perturb.con_list = con
    shared_globals.con_list = con

    started = time.perf_counter()
    with tf.device("/CPU:0"):
        reliability_records, reliable_scores = compute_reliability_batched(
            perturb, con, n_bootstrap=4, sample_fraction=0.8, seed=seed
        )
    rel_sec = time.perf_counter() - started

    high, high_threshold, high_reason = select_high_contribution_features(
        list(shared_globals.feature_names), con
    )
    payload = {
        "status": "complete",
        "method_version": METHOD_VERSION,
        "code_signature": provenance["signature"],
        "code_provenance": provenance["files"],
        "dataset": a.dataset,
        "seed_id": a.split_seed_id,
        "seed": seed,
        "dataset_sha256": r["sha256"],
        "split_hash": r["split_hash"],
        "split_file": str(split_file),
        "selector_input": str(selector_input),
        "selector_model": str(model_path),
        "selector_model_sha256": sha256_file(model_path),
        "feature_names": list(shared_globals.feature_names),
        "target_name": shared_globals.target_name[0],
        "contribution_scores": {
            n: float(v) for n, v in zip(shared_globals.feature_names, con)
        },
        "reliability_records": reliability_records,
        "reliable_scores": {k: float(v) for k, v in reliable_scores.items()},
        "high_contrib_features": high,
        "high_contrib_threshold": float(high_threshold),
        "high_contrib_reason": high_reason,
        "perturbation_values": perturb.perturb_values,
        "perturbation_policy_version": "semantic_metadata_v6",
        "baseline_policy_version": "semantic_metadata_v4_frozen",
        "semantic_types": dict(perturb.semantic_types),
        "baseline_policy": dict(perturb.baseline_policy),
        "perturbation_classes": dict(perturb.perturbation_classes),
        "perturbation_policy": dict(perturb.perturbation_policy),
        "semantic_metadata_file": str(perturb.semantic_metadata_path),
        "semantic_metadata_sha256": sha256_file(perturb.semantic_metadata_path),
        "baseline_values": {
            name: float(perturb.baseline_np[0, i])
            for i, name in enumerate(shared_globals.feature_names)
        },
        "timing": {
            "selector_model_sec": model_sec,
            "selector_model_save_sec": model_save_sec,
            "p1_contribution_sec": contrib_sec,
            "p1_reliability_sec": rel_sec,
        },
        "outer_test_rows_used": False,
    }
    atomic_json(out, payload)
    print("P1_COMPLETE", a.dataset, a.split_seed_id, "high", len(high), "out", out, flush=True)

if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(0)
    except Exception as exc:
        print(f"STAGE1_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        os._exit(1)
