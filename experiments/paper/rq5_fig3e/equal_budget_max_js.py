import csv
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.update(CUDA_VISIBLE_DEVICES="-1", TF_CPP_MIN_LOG_LEVEL="2",
                  TF_ENABLE_ONEDNN_OPTS="0", TF_NUM_INTEROP_THREADS="1",
                  TF_NUM_INTRAOP_THREADS="1", OMP_NUM_THREADS="1",
                  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.special import rel_entr

PACKAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE / 'scripts'))
from isolated_worker import required_output
SOURCE = Path(os.environ.get('FAST30_SOURCE_ROOT', '/root/autodl-tmp/fast30_final')).resolve()
ROOT = Path(os.environ.get('RQ5_CASES_ROOT', SOURCE / 'results/sensitivity_five_20261028/cases')).resolve()
OUT = required_output('RQ5_OUTPUT_ROOT', forbidden=(SOURCE / 'results', Path('/root/autodl-fs/multibranch_ablation_20261113')))
OUT.mkdir(parents=True, exist_ok=True)
DATASETS = ("BankFull", "OnlineShoppers", "StudentsAcademic", "Optdigits", "SoftwareQuality")
POLICY = "continuous_profile_quantile_or_linspace"
COARSE, FINE = 1025, 4097


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def values(items):
    return np.unique(np.asarray(items, dtype=np.float32))


def js(pred, base):
    p = np.clip(pred, 1e-10, 1.).astype(np.float64)
    q = np.broadcast_to(np.clip(base, 1e-10, 1.).astype(np.float64), p.shape)
    m = (p + q) * .5
    p = p / p.sum(axis=1, keepdims=True)
    q = q / q.sum(axis=1, keepdims=True)
    m = m / m.sum(axis=1, keepdims=True)
    return np.maximum(.5 * (rel_entr(p, m) + rel_entr(q, m)).sum(axis=1), 0.)


def score(model, baseline, base_pred, index, grid):
    out = []
    for start in range(0, len(grid), 4096):
        block = grid[start:start + 4096]
        x = np.repeat(baseline, len(block), axis=0)
        x[:, index, 0] = block
        out.append(js(np.asarray(model(x, training=False)), base_pred))
    return np.concatenate(out)


def run_case(ds, seed):
    case = ROOT / ds / seed
    mf = load(case / "manifest.json")
    source = load(mf["p1"])
    assert sha(mf["p1"]) == mf["p1_sha256"]
    assert source["split_hash"] == mf["split_hash"]
    names = source["feature_names"]
    continuous = [f for f in names if source["perturbation_policy"][f] == POLICY]
    if not continuous:
        return {"dataset": ds, "seed": seed, "rows": [], "split_hash": mf["split_hash"]}

    old_q = load(case / "p1/continuous_policy__quantile_only.json")
    old_u = load(case / "p1/continuous_policy__linspace_only.json")
    assert old_q["source_p1_sha256"] == old_u["source_p1_sha256"] == mf["p1_sha256"]
    model_path = source["selector_model"]
    assert sha(model_path) == source["selector_model_sha256"] == mf["selector_model_sha256"]
    baseline = np.array([source["baseline_values"][name] for name in names], dtype=np.float32).reshape(1, -1, 1)
    assert np.isfinite(baseline).all()
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(model_path, compile=False)
    base_pred = np.asarray(model(baseline, training=False))[0]
    raw = pd.read_csv(source["selector_input"], usecols=continuous)
    p_fine = np.linspace(0., 1., FINE)
    stride = (FINE - 1) // (COARSE - 1)
    rows = []
    for name in continuous:
        index = names.index(name)
        observed = pd.to_numeric(raw[name], errors="coerce").dropna().to_numpy(dtype=float)
        observed = observed[np.isfinite(observed)]
        lo, hi = float(observed.min()), float(observed.max())
        grids = [values(p["perturbation_values"][name]) for p in (source, old_q, old_u)]
        native, q, old_uniform = grids
        assert len(old_uniform) == 101 and 2 <= len(q) <= 101, (ds, seed, name, list(map(len, grids)))
        assert all(g[0] == np.float32(lo) and g[-1] == np.float32(hi) for g in grids), (ds, seed, name, "bounds")
        same_q, same_u = np.array_equal(native, q), np.array_equal(native, old_uniform)
        assert same_q != same_u, (ds, seed, name, "branch mismatch")
        m = len(q)
        uniform = values(np.linspace(lo, hi, m))
        assert m == len(uniform) and q[0] == uniform[0] and q[-1] == uniform[-1]
        dense_u = np.linspace(lo, hi, FINE)
        dense_q = np.quantile(observed, p_fine)
        coarse = values(np.concatenate([dense_u[::stride], dense_q[::stride], q, uniform]))
        fine = values(np.concatenate([dense_u, dense_q, q, uniform, native]))
        scores = score(model, baseline, base_pred, index, fine)

        def maximum(grid):
            ii = np.searchsorted(fine, grid)
            assert np.array_equal(fine[ii], grid)
            return float(scores[ii].max())

        j_ref = float(scores.max())
        j_coarse = maximum(coarse)
        j_q, j_u, j_native = map(maximum, (q, uniform, native))
        archived = source["contribution_scores"][name]
        assert abs(j_native - archived) < 1e-3, (ds, seed, name, j_native, archived)
        use_quantile = same_q
        rows.append(dict(dataset=ds, seed=seed, feature=name,
            branch="quantile" if use_quantile else "uniform", actual_points=m,
            default_native_points=len(native), dense_points=len(fine), coarse_points=len(coarse),
            reference_js=j_ref, coarse_reference_js=j_coarse, quantile_js=j_q,
            uniform_js=j_u, default_js=j_q if use_quantile else j_u,
            native_js=j_native, archived_js=archived))
    return dict(dataset=ds, seed=seed, split_hash=mf["split_hash"],
                model_sha256=mf["selector_model_sha256"], p1_sha256=mf["p1_sha256"], rows=rows)


def main():
    keys = [(ds, f"S{i:02}") for ds in DATASETS for i in range(1, 31)]
    if len(sys.argv) == 3:
        keys = [(sys.argv[1], sys.argv[2])]
    run_dir = OUT / "cases"
    run_dir.mkdir(exist_ok=True)
    for ds, seed in keys:
        dest = run_dir / f"{ds}_{seed}.json"
        if not dest.exists():
            result = run_case(ds, seed)
            tmp = dest.with_suffix(".tmp")
            tmp.write_text(json.dumps(result, allow_nan=False))
            tmp.replace(dest)
            print(ds, seed, len(result["rows"]), flush=True)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
