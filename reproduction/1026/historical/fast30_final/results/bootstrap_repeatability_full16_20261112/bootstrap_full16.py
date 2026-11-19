"""Extend the archived bootstrap reference bank without retraining the models."""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
root = Path('/root/autodl-tmp/fast30_final')
sys.path.insert(0, str(root / 'code_sensitivity_five_20261028'))
import sens_worker as worker
import ablation_gpu
import numpy as np
from scipy.stats import entropy
import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)
ablation_gpu.tf_setup = lambda: tf
output = Path(__file__).resolve().parent
datasets = ('BankFull', 'OnlineShoppers', 'StudentsAcademic', 'Optdigits', 'SoftwareQuality')
permutation_rng = np.random.default_rng(20261112)
partitions = np.array([permutation_rng.permutation(16) + 1 for _ in range(1024)])

def save(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(content, indent=2, allow_nan=False))
    temp.replace(path)

def perturbation_jobs(supports, names):
    jobs = [(i, value) for i, name in enumerate(names) for value in supports[name]]
    return np.array([i for i, _ in jobs], dtype=int), np.array([value for _, value in jobs], dtype=np.float32)

def reference_scores(model, baseline, jobs):
    baseline = np.asarray(baseline, dtype=np.float32).reshape(1, -1)
    q = np.clip(np.asarray(model(baseline.reshape(1, -1, 1), training=False), dtype=float)[0], 1e-10, 1)
    indices, values = jobs
    maxima = np.zeros(baseline.shape[1])
    for start in range(0, len(indices), 4096):
        features = indices[start:start+4096]
        batch = np.repeat(baseline, len(features), axis=0)
        batch[np.arange(len(features)), features] = values[start:start+4096]
        p = np.clip(np.asarray(model(batch[:, :, None], training=False), dtype=float), 1e-10, 1)
        middle = (p + q) / 2
        js = (entropy(p, middle, axis=1) + entropy(q[None, :], middle, axis=1)) / 2
        np.maximum.at(maxima, features, js)
    return maxima

def compute_seed(dataset, sid):
    dest = output / 'references' / dataset / f'{sid}.json'
    if dest.exists():
        result = worker.load(dest)
        assert len(result['scores']) == 17
        return result
    case = worker.case(dataset, sid)
    manifest = worker.load(case / 'manifest.json')
    assert worker.sha(manifest['p1']) == manifest['p1_sha256']
    archived = worker.load(case / 'p1/bootstrap__8.json')
    tf.keras.backend.clear_session()
    _, sg, model_context, frozen = worker.setup_p1_context(dataset, sid, p=worker.load(manifest['p1']))
    names = frozen['feature_names']
    assert names == archived['feature_names']
    assert archived['source_p1_sha256'] == manifest['p1_sha256']
    supports = frozen['perturbation_values']
    assert supports == archived['perturbation_values']
    previous = np.array([archived['reliability_records'][name]['baseline_scores'] for name in names]).T
    assert previous.shape == (9, len(names))
    assert np.allclose(previous[:5], np.array([frozen['reliability_records'][name]['baseline_scores'] for name in names]).T, rtol=1e-5, atol=1e-7)
    assert np.allclose(previous[0], [frozen['contribution_scores'][name] for name in names], rtol=1e-8, atol=1e-10)
    jobs = perturbation_jobs(supports, names)
    rng = np.random.default_rng(int(frozen['seed']))
    n_rows = len(sg.feature_data)
    count = max(2, int(round(n_rows * 0.8)))
    extra = []
    hashes = []
    parity_error = []
    with tf.device('/CPU:0'):
        for ref in range(1, 17):
            indices = rng.integers(0, n_rows, size=count)
            if 1 < ref <= 8 and sid != 'S01':
                continue
            baseline = model_context.reference_baseline_vector(indices)
            bad = ~np.isfinite(baseline)
            baseline[bad] = model_context.baseline_np[0][bad]
            values = reference_scores(model_context.model, baseline, jobs)
            assert np.isfinite(values).all()
            if ref <= 8:
                assert np.allclose(values, previous[ref], rtol=1e-5, atol=1e-7), (dataset, sid, ref, float(np.max(abs(values - previous[ref]))))
                parity_error.append(float(np.max(abs(values - previous[ref]))))
            else:
                extra.append(values)
                hashes.append(hashlib.sha256(indices.tobytes()).hexdigest())
    assert len(extra) == 8 and len(set(hashes)) == 8
    result = {'dataset': dataset, 'seed': sid, 'split_hash': manifest['split_hash'],
              'source_p1_sha256': manifest['p1_sha256'], 'model_sha256': frozen['selector_model_sha256'],
              'feature_names': names, 'new_reference_index_sha256': hashes,
              'max_historical_score_error': max(parity_error), 'scores': np.vstack([previous, extra]).tolist()}
    save(dest, result)
    return result

def weights(values):
    median = np.median(values, axis=1)
    mad = np.median(np.abs(values - median[:, None, :]), axis=1)
    return 1 / (1 + 1.4826 * mad / (np.abs(median) + 1e-10))

def summarize():
    cases = []
    for dataset in datasets:
        for index in range(1, 31):
            sid = f'S{index:02d}'
            record = worker.load(output / 'references' / dataset / f'{sid}.json')
            a = np.asarray(record['scores'])
            assert a.shape == (17, len(record['feature_names']))
            values = {}
            for b in (2, 4, 8):
                disjoint = np.column_stack([partitions[:, :b], partitions[:, 8:8+b]])
                assert all(len(set(row)) == 2*b for row in disjoint)
                original = np.zeros((len(partitions), 1), dtype=int)
                x = weights(a[np.concatenate([original, disjoint[:, :b]], axis=1)])
                y = weights(a[np.concatenate([original, disjoint[:, b:]], axis=1)])
                values[str(b)] = float(np.abs(x - y).mean())
            cases.append({'dataset': dataset, 'seed': sid, 'differences': values})
        print('SUMMARY_DATASET', dataset, flush=True)
    def means(records):
        return {b: float(np.mean([row['differences'][b] for row in records])) for b in ('2','4','8')}
    means_by_dataset = {dataset:means([r for r in cases if r['dataset']==dataset]) for dataset in datasets}
    counts = {f'{b}_below_{a}': sum(row['differences'][b] < row['differences'][a] for row in cases)
              for a,b in [('2','4'),('4','8'),('2','8')]}
    result = {'datasets': means_by_dataset, 'overall': means(cases), 'runs': len(cases),
              'references': 16, 'new_references_per_run': 8, 'partition_draws_per_run': len(partitions),
              'partition_rng_seed': 20261112, 'lower_weight_difference_counts': counts}
    save(output / 'per_seed_repeatability.json', cases)
    save(output / 'summary.json', result)
    print('RESULT',json.dumps(result),flush=True)

if __name__ == '__main__':
    clock = time.monotonic()
    total = 0
    for index in range(1,31):
        sid = f'S{index:02d}'
        for dataset in datasets:
            record = compute_seed(dataset, sid)
            total += 1
            if index==1 or total%10==0:
                print('COMPLETE',total,dataset,sid,'old_score_max_error',record['max_historical_score_error'],'elapsed',round(time.monotonic()-clock,1),flush=True)
    summarize()

