"""Flat V22-C-U25 target-local PC-stable implementation.

Method semantics match the active fast30 Stage2:
- global PC-stable levels 0..3
- P1 reliable-score task frontier
- deep candidates = task-path edges + top-25% uncertain frontier chords
- frontier-local conditioning with conservative boundary rescue
- stop after two consecutive deep levels with zero deletions

Small workloads run serially to avoid multiprocessing spawn overhead.
"""
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from multiprocessing import get_context
import math
import numpy as np
from scipy.stats import t as student_t

CORR_REG = 1e-4
MAX_PATH_LENGTH = 3
U_FRAC = 0.25
ZERO_DELETE_PATIENCE = 2
SERIAL_PAIR_THRESHOLD = 128

_W_CORR = None
_W_SNAPSHOT = None
_W_N = None
_W_LEVEL = None
_W_ALPHA = None
_W_ALLOWED = None
_W_RESCUE_K = None

def _init_worker(corr, snapshot, sample_num, level, alpha, allowed_nodes, rescue_k):
    global _W_CORR, _W_SNAPSHOT, _W_N, _W_LEVEL, _W_ALPHA, _W_ALLOWED, _W_RESCUE_K
    _W_CORR = corr
    _W_SNAPSHOT = snapshot
    _W_N = int(sample_num)
    _W_LEVEL = int(level)
    _W_ALPHA = float(alpha)
    _W_ALLOWED = set(allowed_nodes or [])
    _W_RESCUE_K = int(rescue_k)

def _partial_corr_pvalue(corr_matrix, i, j, cond, sample_num):
    idx = [i, j] + list(cond)
    sub = np.asarray(corr_matrix[np.ix_(idx, idx)], dtype=float)
    sub = np.nan_to_num(sub, nan=0.0, posinf=1.0, neginf=-1.0)
    sub = sub + CORR_REG * np.eye(len(idx))
    precision = np.linalg.pinv(sub, rcond=1e-3)
    denom = float(np.sqrt(abs(precision[0, 0] * precision[1, 1])))
    rho = float(-precision[0, 1] / denom) if denom >= 1e-12 else 0.0
    rho = float(np.clip(rho, -0.999999, 0.999999))
    df = int(sample_num - len(cond) - 2)
    if df <= 0:
        return rho, 1.0
    t_stat = rho * np.sqrt(df / max(1e-12, 1.0 - rho * rho))
    return rho, float(2.0 * student_t.sf(abs(t_stat), df=df))

def _global_edge_job(pair):
    i, j = pair
    s, level = _W_SNAPSHOT, _W_LEVEL
    if s[i, j] == 0:
        return i, j, False, 0, None, None
    n = s.shape[0]
    nbr_i = [k for k in range(n) if s[i, k] and k != j]
    nbr_j = [k for k in range(n) if s[j, k] and k != i]
    pools = []
    if len(nbr_i) >= level:
        pools.append(nbr_i)
    if len(nbr_j) >= level and nbr_j != nbr_i:
        pools.append(nbr_j)
    if not pools:
        return i, j, False, 0, None, None
    seen, tests, max_p = set(), 0, -1.0
    for pool in pools:
        for cond in combinations(pool, level):
            if cond in seen:
                continue
            seen.add(cond)
            tests += 1
            rho, p = _partial_corr_pvalue(_W_CORR, i, j, cond, _W_N)
            max_p = max(max_p, float(p))
            if p >= _W_ALPHA:
                return i, j, True, tests, (tuple(cond), rho, p, "global"), max_p
    return i, j, True, tests, None, max_p

def _rank_boundary(i, j, candidates):
    return sorted(
        {int(k) for k in candidates if int(k) not in (i, j)},
        key=lambda k: (
            -(abs(float(_W_CORR[i, k])) + abs(float(_W_CORR[j, k]))),
            int(k),
        ),
    )

def _restricted_edge_job(pair):
    i, j = pair
    s, level = _W_SNAPSHOT, _W_LEVEL
    if s[i, j] == 0:
        return i, j, False, 0, 0, 0, None, None
    n = s.shape[0]
    global_i = [k for k in range(n) if s[i, k] and k != j]
    global_j = [k for k in range(n) if s[j, k] and k != i]
    local_i = [k for k in global_i if k in _W_ALLOWED]
    local_j = [k for k in global_j if k in _W_ALLOWED]
    seen = set()
    tests = local_tests = rescue_tests = 0
    max_p = -1.0
    any_testable = False

    pools = []
    if len(local_i) >= level:
        pools.append(local_i)
    if len(local_j) >= level and local_j != local_i:
        pools.append(local_j)
    if pools:
        any_testable = True
    for pool in pools:
        for cond in combinations(pool, level):
            if cond in seen:
                continue
            seen.add(cond)
            tests += 1
            local_tests += 1
            rho, p = _partial_corr_pvalue(_W_CORR, i, j, cond, _W_N)
            max_p = max(max_p, float(p))
            if p >= _W_ALPHA:
                return i, j, True, tests, local_tests, rescue_tests, (tuple(cond), rho, p, "local"), max_p

    boundary_i = [k for k in global_i if k not in _W_ALLOWED]
    boundary_j = [k for k in global_j if k not in _W_ALLOWED]
    aug_i = list(dict.fromkeys(local_i + _rank_boundary(i, j, boundary_i)[:_W_RESCUE_K]))
    aug_j = list(dict.fromkeys(local_j + _rank_boundary(i, j, boundary_j)[:_W_RESCUE_K]))
    pools = []
    if len(aug_i) >= level:
        pools.append(aug_i)
    if len(aug_j) >= level and aug_j != aug_i:
        pools.append(aug_j)
    if pools:
        any_testable = True
    for pool in pools:
        for cond in combinations(pool, level):
            if cond in seen:
                continue
            seen.add(cond)
            tests += 1
            rescue_tests += 1
            rho, p = _partial_corr_pvalue(_W_CORR, i, j, cond, _W_N)
            max_p = max(max_p, float(p))
            if p >= _W_ALPHA:
                return i, j, True, tests, local_tests, rescue_tests, (tuple(cond), rho, p, "rescue"), max_p
    return i, j, any_testable, tests, local_tests, rescue_tests, None, max_p

def parallel_level(
    corr, sample_num, adj, sep_sets, evidence, level, alpha, workers,
    edge_candidates=None, mode="global", allowed_nodes=None, rescue_k=0,
    record_margin=False,
):
    snapshot = adj.copy()
    pairs = (
        list(combinations(range(adj.shape[0]), 2))
        if edge_candidates is None
        else sorted({tuple(sorted((int(i), int(j)))) for i, j in edge_candidates})
    )
    job = _global_edge_job if mode == "global" else _restricted_edge_job
    initargs = (
        corr, snapshot, sample_num, level, alpha,
        tuple(sorted(allowed_nodes or [])), rescue_k,
    )

    # Spawn overhead dominates low-dimensional data. Serial execution is numerically
    # identical because each level already uses a frozen PC-stable snapshot.
    if workers <= 1 or len(pairs) < SERIAL_PAIR_THRESHOLD:
        _init_worker(*initargs)
        results = [job(pair) for pair in pairs]
        execution = "serial"
    else:
        chunksize = max(1, len(pairs) // max(1, workers * 16))
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=get_context("spawn"),
            initializer=_init_worker,
            initargs=initargs,
        ) as ex:
            results = list(ex.map(job, pairs, chunksize=chunksize))
        execution = "process"

    ci_tests = local_tests = rescue_tests = 0
    any_testable = False
    margins, deletions = {}, []
    for row in results:
        if mode == "global":
            i, j, testable, nt, deletion, max_p = row
            nl = nr = 0
        else:
            i, j, testable, nt, nl, nr, deletion, max_p = row
        any_testable = any_testable or bool(testable)
        ci_tests += int(nt)
        local_tests += int(nl)
        rescue_tests += int(nr)
        if record_margin and deletion is None and max_p is not None:
            margins[(int(i), int(j))] = float(max_p)
        if deletion is not None:
            cond, rho, p_value, deletion_mode = deletion
            deletions.append((i, j, cond, rho, p_value, deletion_mode))

    for i, j, cond, rho, p_value, deletion_mode in deletions:
        if adj[i, j] == 0:
            continue
        adj[i, j] = adj[j, i] = 0
        sep_sets[i][j] = sep_sets[j][i] = list(cond)
        evidence[(i, j)] = {
            "conditioning_set": list(cond),
            "rho": float(rho),
            "p_value": float(p_value),
            "level": int(level),
            "mode": deletion_mode,
        }
    return {
        "any_testable": bool(any_testable),
        "ci_tests": int(ci_tests),
        "local_tests": int(local_tests),
        "rescue_tests": int(rescue_tests),
        "deletions": int(len(deletions)),
        "margins": margins,
        "execution": execution,
    }

def should_adaptive_stop(deep_stats, patience=ZERO_DELETE_PATIENCE):
    if len(deep_stats) < patience:
        return False, None
    levels = sorted(int(k) for k in deep_stats)
    recent = levels[-patience:]
    if all(int(deep_stats[k]["deletions"]) == 0 for k in recent):
        return True, recent[-1]
    return False, None

def find_target_skeleton_paths(skeleton, target_idx, max_length=MAX_PATH_LENGTH):
    n = skeleton.shape[0]
    out = set()
    for source in range(n):
        if source == target_idx:
            continue
        stack = [(source, [source])]
        while stack:
            current, path = stack.pop()
            edges = len(path) - 1
            if current == target_idx:
                if 1 <= edges <= max_length:
                    out.add(tuple(path))
                continue
            if edges >= max_length:
                continue
            for nxt in range(n):
                if nxt == current or nxt in path:
                    continue
                if skeleton[current, nxt] == 1:
                    stack.append((nxt, path + [nxt]))
    return [list(path) for path in sorted(out)]

def _path_edge_union(paths):
    edges = set()
    for path in paths:
        for a, b in zip(path[:-1], path[1:]):
            edges.add(tuple(sorted((int(a), int(b)))))
    return edges

def task_paths_from_adj(adj, target_idx, reliable_scores, cutoff):
    paths = find_target_skeleton_paths(adj, target_idx, MAX_PATH_LENGTH)
    task = [p for p in paths if len(p) == 2 or reliable_scores[p[0]] >= cutoff]
    return paths, task

def build_frontier(adj, target_idx, reliable_scores, cutoff, level3_margins):
    shallow_paths, task_paths = task_paths_from_adj(adj, target_idx, reliable_scores, cutoff)
    frontier_nodes = {target_idx}
    for path in task_paths:
        frontier_nodes.update(int(x) for x in path)
    frontier_edges = {
        (i, j)
        for i, j in combinations(sorted(frontier_nodes), 2)
        if adj[i, j] != 0
    }
    path_edges = {e for e in _path_edge_union(task_paths) if adj[e[0], e[1]] != 0}
    skipped = sorted(frontier_edges - path_edges)
    ranked = sorted(
        [(float(level3_margins.get(e, -1.0)), e) for e in skipped],
        key=lambda x: (-x[0], x[1][0], x[1][1]),
    )
    budget = int(math.ceil(U_FRAC * len(skipped))) if skipped else 0
    uncertainty_edges = {e for _, e in ranked[:budget]}
    return {
        "shallow_paths": shallow_paths,
        "task_paths": task_paths,
        "frontier_nodes": frontier_nodes,
        "frontier_edges": frontier_edges,
        "path_edges": path_edges,
        "skipped_edges": set(skipped),
        "uncertainty_edges": uncertainty_edges,
        "candidate_edges": path_edges | uncertainty_edges,
        "margin_ranking": ranked,
        "shallow_edge_count": int(np.sum(np.triu(adj, 1))),
    }
