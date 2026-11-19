import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RUNTIME = Path(__file__).with_name("stage2_runtime.py")


def _tail(path, n=120):
    try:
        return "\n".join(Path(path).read_text(errors="replace").splitlines()[-n:])
    except Exception:
        return ""


def _cleanup_process_group(pgid):
    """Remove orphan PC workers if the Stage2 parent exits unexpectedly."""
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        return
    time.sleep(0.2)
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run(dataset, seed, manifest, output_root):
    out = Path(output_root) / Path(dataset).stem / seed
    out.mkdir(parents=True, exist_ok=True)
    (out / "running_status.json").write_text(json.dumps(
        {"dataset": dataset, "seed": seed, "stage": "stage2", "status": "running"}
    ))
    env = os.environ.copy()
    # P2 already uses up to four worker processes. BLAS/OpenMP is single-threaded
    # inside each worker to avoid severe oversubscription.
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "FAST30_OUTPUT_ROOT": str(output_root),
    })
    log = out / "stage2.log"
    with log.open("a") as fh:
        proc = subprocess.Popen(
            [sys.executable, str(RUNTIME), "--dataset", dataset, "--split-seed-id", seed],
            env=env,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = proc.wait()
        finally:
            # With start_new_session=True, the runtime PID is also its process-group ID.
            # If ProcessPoolExecutor workers survived their parent, this removes them.
            _cleanup_process_group(proc.pid)
    if returncode != 0:
        raise RuntimeError(f"stage2 failed; log={log}\n{_tail(log)}")
    return out
