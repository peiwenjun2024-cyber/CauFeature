import numpy as np
import pandas as pd
import shared_globals

def _js_divergence(p, q):
    from scipy.stats import entropy
    p = np.clip(np.asarray(p, dtype=float), 1e-10, 1.0)
    q = np.clip(np.asarray(q, dtype=float), 1e-10, 1.0)
    m = 0.5 * (p + q)
    return float(0.5 * (entropy(p, m) + entropy(q, m)))

def _predict(model, batch):
    pred = model(batch, training=False)
    return pred.numpy() if hasattr(pred, "numpy") else np.asarray(pred)

def compute_reliability_batched(
    perturb, original_con, n_bootstrap=4, sample_fraction=0.8, seed=401, chunk_size=4096
):
    """Same reliability statistic, with perturbations shared across large inference batches."""
    names = list(shared_globals.feature_names)
    x = shared_globals.feature_data
    rng = np.random.default_rng(seed)
    baselines = [perturb.baseline_np[0].astype(np.float32).copy()]
    sample_n = max(2, int(round(len(x) * sample_fraction)))
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), size=sample_n)
        # Use the same semantic baseline policy as the original P1 reference:
        # categorical/discrete -> mode, continuous -> mean.
        baseline = perturb.reference_baseline_vector(idx)
        bad = ~np.isfinite(baseline)
        baseline[bad] = baselines[0][bad]
        baselines.append(baseline.astype(np.float32))

    score_matrix = np.zeros((len(baselines), len(names)), dtype=float)
    score_matrix[0, :] = np.asarray(original_con, dtype=float)

    for b_idx, baseline in enumerate(baselines[1:], start=1):
        base_pred = _predict(
            perturb.model, baseline.reshape(1, -1, 1).astype(np.float32)
        )[0]
        maxima = np.zeros(len(names), dtype=float)
        jobs = []

        def flush(batch_jobs):
            if not batch_jobs:
                return
            batch = np.repeat(baseline.reshape(1, -1), len(batch_jobs), axis=0)
            for row, (j, value) in enumerate(batch_jobs):
                batch[row, j] = value
            batch = batch.reshape(-1, len(names), 1).astype(np.float32)
            preds = _predict(perturb.model, batch)
            for (j, _), pred in zip(batch_jobs, preds):
                maxima[j] = max(maxima[j], _js_divergence(pred, base_pred))

        for j, name in enumerate(names):
            for value in perturb.perturb_values.get(name, []):
                try:
                    jobs.append((j, float(value)))
                except (TypeError, ValueError):
                    continue
                if len(jobs) >= chunk_size:
                    flush(jobs)
                    jobs.clear()
        flush(jobs)
        score_matrix[b_idx, :] = maxima

    eps = 1e-10
    reliability, records = {}, {}
    original = np.asarray(original_con, dtype=float)
    for j, name in enumerate(names):
        scores = score_matrix[:, j]
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        rel_disp = float(1.4826 * mad / (abs(median) + eps))
        rel = float(1.0 / (1.0 + rel_disp))
        reliability[name] = rel
        records[name] = {
            "original_contribution": float(original[j]),
            "median_contribution": median,
            "mad_contribution": mad,
            "robust_relative_dispersion": rel_disp,
            "reliability": rel,
            "baseline_scores": [float(v) for v in scores],
        }
    max_con = max(float(v) for v in original) if len(original) else 0.0
    denom = max(max_con, eps)
    reliable_scores = {
        name: float(original[j] / denom) * reliability[name]
        for j, name in enumerate(names)
    }
    shared_globals.p1_contribution_reliability = reliability
    shared_globals.p1_reliable_score = reliable_scores
    return records, reliable_scores
