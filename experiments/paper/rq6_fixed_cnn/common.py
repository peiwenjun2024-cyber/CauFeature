"""No ML imports or work at import time. All writes confined to the new campaign."""
from pathlib import Path
import hashlib
import json
import os
import sys
import uuid
from functools import lru_cache

ROOT = Path('/root/autodl-tmp/fast30_final')
CODE = Path(__file__).resolve().parent
OUT = ROOT / 'results/unscaled_scoring_cnn_eval_20261110'
STUDY = ROOT / 'results/p1_model_sensitivity_20261023'
ABL = ROOT / 'code_ablation_five_20261028'
SENS = ROOT / 'code_sensitivity_five_20261028'
BASE = ABL / 'frozen_runtime'
DS = ('BankFull', 'OnlineShoppers', 'StudentsAcademic', 'Optdigits', 'SoftwareQuality')
KINDS = ('CNN', 'MLP', 'RF')
PROTOCOL = 'unscaled-scorers-main-cnn-reference-cnn-evaluation-v2-20261110'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def atomic(path, value, root=OUT):
    p = Path(path)
    if not p.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f'Write outside new campaign: {p}')
    p.parent.mkdir(parents=True, exist_ok=True)
    q = p.with_name(p.name + '.' + uuid.uuid4().hex + '.tmp')
    q.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    q.replace(p)


def save(path, value):
    atomic(path, value)


def case(ds, sid):
    if ds not in DS or sid not in [f'S{i:02}' for i in range(1, 31)]:
        raise ValueError('Invalid dataset/seed')
    return OUT / 'cases' / ds / sid


def artifact(ds, sid, stage, key):
    return case(ds, sid) / stage / (key + '.json')


@lru_cache(maxsize=1)
def lock():
    z = load(CODE / 'protocol.lock.json')
    if z['protocol'] != PROTOCOL:
        raise RuntimeError('Wrong protocol lock')
    return z


def validate_code():
    z = lock()
    for p, digest in z['code_hashes'].items():
        if sha(p) != digest:
            raise RuntimeError(f'Frozen code changed: {p}')
    import importlib.metadata as metadata
    for name, version in z['packages'].items():
        if metadata.version(name) != version:
            raise RuntimeError(f'Package version changed: {name}')
    if sys.version.split()[0] != z['python_version']:
        raise RuntimeError('Python version changed')
    return z


def verify_runtime_modules():
    """Called after imports during an explicitly launched job, never in static checks."""
    expected = ('model_train_v6', 'feature_perturbation', 'p3_recovery',
                'fixed_split_evaluator', 'shared_globals', 'p2_parallel_pc')
    for name in expected:
        if name not in sys.modules:
            continue
        actual = Path(sys.modules[name].__file__).resolve()
        wanted = (BASE / (name + '.py')).resolve()
        if actual != wanted or sha(actual) != lock()['code_hashes'][str(wanted)]:
            raise RuntimeError(f'Unexpected runtime module: {name}: {actual}')


def source_case(ds, sid):
    return lock()['cases'][ds + '/' + sid]


def validate_case(ds, sid):
    z = source_case(ds, sid)
    for p, digest in z['input_hashes'].items():
        if sha(p) != digest:
            raise RuntimeError(f'Frozen input changed: {p}')
    return z


def manifest(ds, sid):
    return load(source_case(ds, sid)['manifest'])


def p1(ds, sid):
    return load(manifest(ds, sid)['p1'])


def getp1(ds, sid, key):
    return load(artifact(ds, sid, 'p1', key))


def stamp(ds, sid, key, features):
    return dict(protocol=PROTOCOL, dataset=ds, seed_id=sid, kind=key,
                features=list(features), dimension=len(features),
                split_hash=manifest(ds, sid)['split_hash'])


@lru_cache(maxsize=1)
def jobs():
    """Stable stage-level DAG; constructing it neither reads models nor writes files."""
    out = []
    for ds in DS:
        for n in range(1, 31):
            sid = f'S{n:02}'
            prep = f'prepare/{ds}/{sid}'
            out.append(dict(id=prep, args=['prepare', ds, sid], deps=[],
                            dest=str(case(ds, sid) / 'manifest.json')))
    # All five original-CNN gates must pass before any alternative selection starts.
    gates = [f'parity/{ds}/S01' for ds in DS]
    for ds in DS:
        out.append(dict(id=f'parity/{ds}/S01', args=['parity', ds, 'S01'],
                        deps=[f'prepare/{ds}/S01'],
                        dest=str(artifact(ds, 'S01', 'parity', 'CNN'))))
    for ds in DS:
        for n in range(1, 31):
            sid = f'S{n:02}'
            for key in ('MLP', 'RF'):
                p = f'p2/{ds}/{sid}/{key}'
                q = f'recover/{ds}/{sid}/{key}'
                out.append(dict(id=p, args=['p2', ds, sid, key],
                                deps=[f'prepare/{ds}/{sid}'] + gates,
                                dest=str(artifact(ds, sid, 'p2', key))))
                out.append(dict(id=q, args=['recover', ds, sid, key], deps=[p],
                                dest=str(artifact(ds, sid, 'selection', key))))
            for key in ('FULL', 'CNN', 'MLP', 'RF'):
                deps = [f'prepare/{ds}/{sid}'] + gates if key in ('FULL', 'CNN') else [f'recover/{ds}/{sid}/{key}']
                out.append(dict(id=f'evaluate/{ds}/{sid}/{key}',
                                args=['evaluate', ds, sid, key], deps=deps,
                                dest=str(artifact(ds, sid, 'evaluation', key))))
    return out


def receipt_path(job):
    return OUT / 'receipts' / (job['id'] + '.json')


@lru_cache(maxsize=8192)
def _cached_sha(path, modified_ns, size):
    return sha(path)


def checked_sha(path):
    p = Path(path)
    st = p.stat()
    return _cached_sha(str(p), st.st_mtime_ns, st.st_size)


def fingerprint(job):
    by_id = {j['id']: j for j in jobs()}
    deps = {k: checked_sha(by_id[k]['dest']) for k in job['deps']}
    return hashlib.sha256(canonical(dict(job=job, protocol_lock=checked_sha(CODE / 'protocol.lock.json'), dependencies=deps)).encode()).hexdigest()


def complete(job):
    try:
        r = load(receipt_path(job))
        return (r['fingerprint'] == fingerprint(job) and r['output_sha256'] == checked_sha(job['dest'])
                and all(checked_sha(p) == h for p, h in r['artifacts'].items()))
    except (OSError, KeyError, ValueError):
        return False


def bundle(job):
    paths = [Path(job['dest'])]
    mode, ds, sid, *key = job['args']
    if mode == 'prepare':
        paths += [artifact(ds, sid, 'p1', k) for k in KINDS]
        paths += [artifact(ds, sid, 'p2', 'default'), artifact(ds, sid, 'selection', 'CNN'),
                  case(ds, sid) / 'shallow/alpha_0.01.pkl']
    elif mode == 'parity':
        paths += [artifact(ds, sid, 'p2', 'default_probe')]
    elif mode == 'evaluate' and load(job['dest']).get('reuse') is False:
        paths += [Path(job['dest']).with_suffix(s) for s in ('.keras', '.history.json', '.probabilities.npz')]
    return {str(p): sha(p) for p in paths}
