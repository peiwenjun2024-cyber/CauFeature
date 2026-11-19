import json, os, subprocess, sys
from pathlib import Path
RUNTIME = Path(__file__).with_name("stage3_runtime.py")

def _tail(path, n=120):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except Exception:
        return ""

def run(dataset, seed, manifest, output_root):
    out = Path(output_root) / Path(dataset).stem / seed
    out.mkdir(parents=True, exist_ok=True)
    (out / "running_status.json").write_text(json.dumps(
        {"dataset": dataset, "seed": seed, "stage": "stage3", "status": "running"}
    ))
    env = os.environ.copy()
    seed_num = int(str(seed).lstrip("S"))
    env.update({
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "FAST30_OUTPUT_ROOT": str(output_root),
        "TF_DETERMINISTIC_OPS": "1",
        "TF_CUDNN_DETERMINISTIC": "1",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": str(seed_num),
    })
    log = out / "stage3.log"
    with log.open("a") as fh:
        r = subprocess.run(
            [sys.executable, str(RUNTIME), "--dataset", dataset, "--split-seed-id", seed],
            env=env, stdout=fh, stderr=subprocess.STDOUT, text=True,
        )
    if r.returncode != 0:
        raise RuntimeError(f"stage3 failed; log={log}\n{_tail(log)}")
    (out / "running_status.json").write_text(json.dumps(
        {"dataset": dataset, "seed": seed, "stage": "complete", "status": "complete"}
    ))
    return out
