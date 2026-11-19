#!/usr/bin/env python3
import argparse, fcntl, gc, json, os, traceback
from pathlib import Path
from stage1 import run as stage1_run
from stage2 import run as stage2_run
from stage3 import run as stage3_run
from runtime_provenance import METHOD_VERSION, DEFAULT_OUTPUT_NAME, sha256_file

ROOT = Path('/root/autodl-tmp/fast30_final')
MANIFEST = ROOT / "manifest" / "split_manifest.jsonl"
OUTPUT = Path(os.environ.get(
    "FAST30_OUTPUT_ROOT",
    str(ROOT / "results" / DEFAULT_OUTPUT_NAME),
))
os.environ["OMP_NUM_THREADS"] = "4"

def load_manifest():
    with MANIFEST.open() as f:
        return [json.loads(x) for x in f if x.strip()]

def seeds():
    return [f"S{i:02d}" for i in range(1, 31)]

def done(dataset, seed):
    d = OUTPUT / Path(dataset).stem / seed
    record = d / "record.json"
    if not record.exists():
        return False
    try:
        record_payload = json.loads(record.read_text())
        if record_payload.get("method_version") != METHOD_VERSION:
            return False
        if str(record_payload.get("status", "")).lower() not in {
            "complete", "completed", "success", "done"
        }:
            return False

        # A completed record is reusable only when its exact source files and
        # upstream cache artifacts still match what produced it.
        code_dir = Path(__file__).resolve().parent
        for name, expected_hash in record_payload.get("provenance", {}).items():
            source = code_dir / name
            if not source.exists() or sha256_file(source) != expected_hash:
                return False
        for cache_name, expected_hash in record_payload.get("input_cache_sha256", {}).items():
            cache_path = d / cache_name
            if not cache_path.exists() or sha256_file(cache_path) != expected_hash:
                return False
    except Exception:
        return False
    status = d / "running_status.json"
    if status.exists():
        try:
            state = str(json.loads(status.read_text()).get("status", "")).lower()
            if state and state not in {"complete", "completed", "success", "done"}:
                return False
        except Exception:
            return False
    return True

def cleanup_gpu():
    try:
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
    except Exception:
        pass

def run_one(dataset, seed):
    seed_dir = OUTPUT / Path(dataset).stem / seed
    seed_dir.mkdir(parents=True, exist_ok=True)
    lock_path = seed_dir / ".run.lock"

    with lock_path.open("a+") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("SKIP_LOCKED", dataset, seed, flush=True)
            return "skip"

        if done(dataset, seed):
            print("SKIP_COMPLETE", dataset, seed, flush=True)
            return "skip"

        print("START", dataset, seed, flush=True)
        try:
            cleanup_gpu()
            stage1_run(dataset, seed, str(MANIFEST), str(OUTPUT))
            stage2_run(dataset, seed, str(MANIFEST), str(OUTPUT))
            stage3_run(dataset, seed, str(MANIFEST), str(OUTPUT))
            cleanup_gpu()
            print("COMPLETE", dataset, seed, flush=True)
            return "complete"
        except Exception as exc:
            (seed_dir / "failed.json").write_text(json.dumps({
                "dataset": dataset, "seed": seed, "method_version": METHOD_VERSION,
                "error": str(exc), "traceback": traceback.format_exc(),
            }, indent=2))
            (seed_dir / "running_status.json").write_text(json.dumps({
                "dataset": dataset, "seed": seed, "stage": "failed",
                "status": "failed", "error": str(exc),
            }, indent=2))
            cleanup_gpu()
            print("FAILED", dataset, seed, exc, flush=True)
            return "failed"
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--seed", default=None)
    parser.add_argument("--resume", action="store_true", help="Compatibility flag; valid caches are resumed automatically.")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    datasets = sorted({x["dataset"] for x in load_manifest()})
    if args.dataset:
        datasets = [args.dataset]
    selected_seeds = [f"S{int(args.seed):02d}"] if args.seed else seeds()
    summary = {"complete": 0, "skip": 0, "failed": 0}
    for dataset in datasets:
        for seed in selected_seeds:
            summary[run_one(dataset, seed)] += 1
    print("RUN_SUMMARY", json.dumps(summary), flush=True)

if __name__ == "__main__":
    main()
