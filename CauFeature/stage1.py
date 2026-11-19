import json, os, subprocess, sys
from pathlib import Path
RUNTIME = Path(__file__).with_name("stage1_runtime.py")

def _tail(path, n=120):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except Exception:
        return ""

def run(dataset, seed, manifest, output_root):
    sid = seed
    if isinstance(seed, int) or (isinstance(seed, str) and seed.isdigit()):
        sid = f"S{int(seed):02d}"
    out = Path(output_root) / Path(dataset).stem / sid
    out.mkdir(parents=True, exist_ok=True)
    (out / "running_status.json").write_text(json.dumps(
        {"dataset": dataset, "seed": sid, "stage": "stage1", "status": "running"}
    ))
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "FAST30_MANIFEST": str(manifest),
        "FAST30_OUTPUT_ROOT": str(output_root),
    })
    log = out / "stage1.log"
    with log.open("a") as fh:
        r = subprocess.run(
            [sys.executable, str(RUNTIME), "--dataset", dataset, "--split-seed-id", sid],
            env=env, stdout=fh, stderr=subprocess.STDOUT, text=True,
        )
    if r.returncode != 0:
        raise RuntimeError(f"stage1 failed; log={log}\n{_tail(log)}")
    return out
