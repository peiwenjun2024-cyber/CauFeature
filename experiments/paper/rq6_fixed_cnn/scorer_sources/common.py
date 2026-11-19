"""Isolated P1 scoring-model experiment; frozen main artifacts are read-only."""
from pathlib import Path
import hashlib, json, os, platform
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'results/p1_model_sensitivity_20261023'
DATASETS = ('BankFull', 'OnlineShoppers', 'StudentsAcademic', 'Optdigits', 'SoftwareQuality')
MODELS = ('CNN', 'MLP', 'RF')
PROTOCOL = 'p1-model-sensitivity-v1-unscaled-20261023'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def array_sha(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(a.tobytes())
    return h.hexdigest()

def save_json(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(OUT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    os.replace(tmp, path)

def load(path): return json.loads(Path(path).read_text())

def source(ds, sid):
    assert ds in DATASETS
    base = ROOT / ('results/alpha001_full30_20261009' if ds == 'Optdigits'
                   else 'results/fast30_semanticperturb_flat_v6_20260930')
    return base / ds / sid / 'p1_cache.json'

def case_dir(ds, sid): return OUT / 'cases' / ds / sid

def source_code(ds):
    return ROOT / ('code_optdigits_30_20261011' if ds == 'Optdigits'
                   else 'code_flat_profilefix_20260929')

def model_metrics(y, prob):
    from sklearn.metrics import accuracy_score, f1_score, log_loss
    pred = np.argmax(prob, axis=1)
    return dict(acc=float(accuracy_score(y, pred)),
                f_measure=float(f1_score(y, pred, average='macro', zero_division=0)),
                log_loss=float(log_loss(y, prob, labels=list(range(prob.shape[1])))),
                mean_confidence=float(np.max(prob, axis=1).mean()))

def compute_scores(predict, shared, meta, batch_size=8192, reference_context=None):
    """Same max-JS as production; vectorized entropy after clipping/normalizing."""
    from scipy.special import rel_entr
    refs, names = shared['references'], meta['feature_names']
    supports = meta['perturbation_values']
    scores = np.zeros((5, len(names)), dtype=np.float64)
    queries = 0
    for bi, ref in enumerate(refs):
        def infer(x):
            if reference_context is None:return predict(x)
            with reference_context(bi):return predict(x)
        base = np.clip(infer(ref[None, :])[0].astype(float), 1e-10, 1.)
        base /= base.sum()
        jobs, indices = [], []
        def flush():
            nonlocal queries
            if not jobs: return
            x = np.repeat(ref[None, :], len(jobs), axis=0).astype(np.float32)
            jj = np.asarray(indices)
            x[np.arange(len(jobs)), jj] = np.asarray(jobs, dtype=np.float32)
            q = np.clip(np.asarray(infer(x), dtype=float), 1e-10, 1.)
            if q.ndim != 2 or not np.isfinite(q).all():
                raise RuntimeError('Invalid class probability array')
            q /= q.sum(axis=1, keepdims=True)
            mid = (q + base) * .5
            js = .5 * (rel_entr(q, mid).sum(axis=1) + rel_entr(base, mid).sum(axis=1))
            # Production initialized maxima at zero, including numerical negatives.
            np.maximum.at(scores[bi], jj, js)
            queries += len(jobs)
            jobs.clear(); indices.clear()
        for j, n in enumerate(names):
            for v in supports[n]:
                jobs.append(float(v)); indices.append(j)
                if len(jobs) >= batch_size: flush()
        flush()
    return scores, queries

def evidence(scores, names, nonconstant, select_high):
    original = scores[0]
    med = np.median(scores, axis=0)
    mad = np.median(np.abs(scores-med), axis=0)
    dispersion = 1.4826 * mad / (np.abs(med) + 1e-10)
    r = 1 / (1 + dispersion)
    s = original / max(float(original.max()), 1e-10) * r
    h, t, reason = select_high(names, original)
    valid = np.asarray(nonconstant, dtype=bool)
    cutoff = float(np.percentile(s[valid], 75)) if valid.any() else 0.
    a = [n for j, n in enumerate(names) if valid[j] and s[j] >= cutoff]
    unique, counts = np.unique(original, return_counts=True)
    d = len(names)
    return dict(contribution_scores=dict(zip(names, map(float, original))),
                reliability=dict(zip(names, map(float, r))),
                reliable_scores=dict(zip(names, map(float, s))),
                baseline_scores=scores.tolist(),
                high_contrib_features=h, high_contrib_threshold=float(t),
                high_contrib_reason=reason, p2_gate_features=a,
                p2_gate_cutoff=cutoff,
                zero_contribution_fraction=float(np.mean(original == 0)),
                distinct_contribution_values=int(len(unique)),
                tied_pair_fraction=float(np.sum(counts*(counts-1))/(d*(d-1))) if d>1 else 0.,
                all_zero_contributions=bool(np.all(original == 0)))

def import_function(path, name, namespace):
    """Execute the actual production function without importing TF in CPU jobs."""
    import ast
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]
