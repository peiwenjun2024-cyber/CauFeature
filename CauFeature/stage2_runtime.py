from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
from runtime_provenance import METHOD_VERSION, DEFAULT_OUTPUT_NAME, atomic_json, code_provenance, require_cache_signature, sha256_file
OUT = Path(os.environ.get("FAST30_OUTPUT_ROOT", str(ROOT / "results" / DEFAULT_OUTPUT_NAME)))
import p2_parallel_pc as v22

def encode_frame(path):
    df = pd.read_csv(path).copy()
    x, y = df.iloc[:, :-1].copy(), df.iloc[:, -1].copy()
    le = LabelEncoder()
    for c in x.select_dtypes(exclude=["number"]).columns:
        x[c] = le.fit_transform(x[c])
    if x.isnull().any().any():
        x = x.fillna(x.mean())
    if pd.Series(y).isna().any():
        raise ValueError(f"Target {df.columns[-1]} contains missing values")
    # Classification labels are categorical even when stored as non-contiguous
    # numbers (PrecisionFog uses {0,1,3}); always map to 0..K-1.
    y = LabelEncoder().fit_transform(pd.Series(y))
    return pd.concat(
        [x, pd.Series(y, index=df.index, name=df.columns[-1])],
        axis=1,
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split-seed-id", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--alpha", type=float, choices=(0.01, 0.05), default=0.01)
    a = ap.parse_args()

    rd = OUT / Path(a.dataset).stem / a.split_seed_id
    p1p, out = rd / "p1_cache.json", rd / "p2_cache.json"
    p1 = json.load(open(p1p))
    p1_cache_sha256 = sha256_file(p1p)
    if p1.get("method_version") != METHOD_VERSION or not p1.get("code_signature"):
        raise RuntimeError("P1 cache is not from the current method version")

    code_dir = Path(__file__).resolve().parent
    provenance = code_provenance(code_dir, [
        "stage2.py", "stage2_runtime.py", "p2_parallel_pc.py", "runtime_provenance.py",
    ])
    if out.exists():
        cached = json.load(open(out))
        require_cache_signature(cached, provenance["signature"], "P2")
        if cached.get("pc_alpha", 0.01) != a.alpha:
            raise RuntimeError("P2 cache uses a different alpha; choose a fresh output directory")
        if (
            cached.get("input_p1_code_signature") != p1.get("code_signature")
            or cached.get("input_p1_cache_sha256") != p1_cache_sha256
        ):
            raise RuntimeError("P2 cache was built from a different P1 artifact")
        print("P2_EXISTS_VALID", out)
        return
    df = encode_frame(p1["selector_input"])
    raw, target = list(p1["feature_names"]), p1["target_name"]
    nunique = df[raw].nunique(dropna=False)
    constants = [n for n in raw if int(nunique[n]) <= 1]
    pc = [n for n in raw if n not in constants]
    names, ti = pc + [target], len(pc)

    mat = df[names].to_numpy(float)
    corr = np.corrcoef(mat, rowvar=False)
    if not np.isfinite(corr).all():
        raise RuntimeError("nonfinite correlation")
    n, d = len(df), len(names)
    reliable = np.zeros(d, float)
    rmap = p1["reliable_scores"]
    for i, name in enumerate(pc):
        reliable[i] = float(rmap.get(name, 0.0))
    cutoff = float(np.percentile(reliable[:ti], 75.0)) if ti else 0.0

    adj = np.ones((d, d), dtype=np.int8) - np.eye(d, dtype=np.int8)
    sep = [[[] for _ in range(d)] for _ in range(d)]
    evidence, shallow, margins = {}, {}, {}
    t0 = time.perf_counter()

    for level in range(4):
        q = time.perf_counter()
        st = v22.parallel_level(
            corr, n, adj, sep, evidence, level, a.alpha, a.workers,
            mode="global", record_margin=(level == 3)
        )
        shallow[str(level)] = {
            "any_testable": bool(st["any_testable"]),
            "ci_tests": int(st["ci_tests"]),
            "deletions": int(st["deletions"]),
            "execution": st["execution"],
            "sec": time.perf_counter() - q,
        }
        if level == 3:
            margins = st["margins"]
        print(f"PC shallow L{level} ci={st['ci_tests']} del={st['deletions']} mode={st['execution']}", flush=True)
        if not st["any_testable"]:
            break

    frontier = v22.build_frontier(adj, ti, reliable, cutoff, margins)
    deep, stopped, stop_level = {}, False, None
    for level in range(4, d - 1):
        q = time.perf_counter()
        st = v22.parallel_level(
            corr, n, adj, sep, evidence, level, a.alpha, a.workers,
            edge_candidates=frontier["candidate_edges"],
            mode="restricted",
            allowed_nodes=frontier["frontier_nodes"],
            rescue_k=min(level, 8),
        )
        deep[str(level)] = {
            "ci_tests": int(st["ci_tests"]),
            "local_tests": int(st["local_tests"]),
            "rescue_tests": int(st["rescue_tests"]),
            "deletions": int(st["deletions"]),
            "execution": st["execution"],
            "sec": time.perf_counter() - q,
            "rescue_k": min(level, 8),
        }
        print(f"PC deep L{level} ci={st['ci_tests']} del={st['deletions']} mode={st['execution']}", flush=True)
        if not st["any_testable"]:
            break
        stopped, stop_level = v22.should_adaptive_stop({int(k): v for k, v in deep.items()})
        if stopped:
            print("ADAPTIVE_STOP", stop_level, flush=True)
            break

    all_paths, task_paths = v22.task_paths_from_adj(adj, ti, reliable, cutoff)
    nodes = set()
    for path in task_paths:
        nodes.update(int(x) for x in path if int(x) != ti)
    p2set = {names[i] for i in nodes}
    p2 = [x for x in raw if x in p2set]
    if not p2:
        raise RuntimeError("empty P2")

    payload = {
        "status": "complete", "method_version": METHOD_VERSION,
        "code_signature": provenance["signature"],
        "code_provenance": provenance["files"],
        "input_p1_code_signature": p1["code_signature"],
        "input_p1_cache_sha256": p1_cache_sha256,
        "dataset": a.dataset, "seed_id": a.split_seed_id,
        "seed": int(p1["seed"]), "split_hash": p1["split_hash"],
        "workers": a.workers, "pc_alpha": a.alpha, "execution": "flat_v22_c_u25_adaptive_serial",
        "raw_dimension": len(raw), "constant_features": constants,
        "pc_dimension": len(pc), "cutoff": cutoff,
        "p2_features": p2, "p2_dimension": len(p2),
        "frontier_node_count": len(frontier["frontier_nodes"]),
        "frontier_edge_count": len(frontier["frontier_edges"]),
        "path_edge_count": len(frontier["path_edges"]),
        "uncertainty_edge_count": len(frontier["uncertainty_edges"]),
        "deep_candidate_edge_count": len(frontier["candidate_edges"]),
        "target_path_count": len(all_paths), "task_path_count": len(task_paths),
        "adaptive_stopped": bool(stopped), "adaptive_stop_level": stop_level,
        "shallow": shallow, "deep": deep, "p2_sec": time.perf_counter() - t0,
    }
    atomic_json(out, payload)
    print("P2_COMPLETE", a.dataset, a.split_seed_id, "dim", len(p2), "out", out, flush=True)

if __name__ == "__main__":
    main()
