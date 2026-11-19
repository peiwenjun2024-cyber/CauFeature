"""Explicitly gated worker. Importing this file starts no experiment."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.update(TF_CPP_MIN_LOG_LEVEL='2', TF_ENABLE_ONEDNN_OPTS='0', TF_DETERMINISTIC_OPS='1')
import ast
import contextlib
import fcntl
import io
import pickle
import resource
import random
import sys
import time
import traceback
import uuid
from common import *

CONFIGS = {k: dict(stage='P1', group='scoring_model', value=k) for k in KINDS}
TF = None


def init_tf():
    global TF
    if TF is None:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
        gpu = tf.config.list_physical_devices('GPU')
        if not gpu:
            raise RuntimeError('GPU unavailable; refusing implicit CPU inference/training')
        tf.config.set_logical_device_configuration(gpu[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        tf.config.experimental.enable_op_determinism()
        TF = tf
    return TF


def prepare(ds, sid):
    import numpy as np
    m = manifest(ds, sid)
    original = p1(ds, sid)
    shared = STUDY / 'cases' / ds / sid
    meta = load(shared / 'shared.json')
    assert meta['input_scaling'] == 'none; identical encoded unscaled input for all models'
    assert meta['split_hash'] == m['split_hash'] and meta['source_p1_sha256'] == m['p1_sha256']
    for kind in KINDS:
        r = load(shared / (kind + '.json'))
        assert r['status'] == 'complete' and r['outer_test_rows_used'] is False
        assert r['feature_names'] == original['feature_names']
        assert r['shared_npz_sha256'] == meta['shared_npz_sha256']
        z = dict(original)
        fields = ('contribution_scores', 'reliable_scores', 'high_contrib_features', 'high_contrib_threshold', 'high_contrib_reason')
        if kind == 'CNN':
            assert all(r[k] == original[k] for k in fields)
            modelpath = Path(original['selector_model'])
            digest = original['selector_model_sha256']
        else:
            modelpath = shared / (kind + ('.joblib' if kind == 'RF' else '.keras'))
            digest = r['model_sha256']
            z.update({k: r[k] for k in fields})
            # Replace inherited reliability records too; never expose stale CNN evidence.
            matrix = np.asarray(r['baseline_scores'], float)
            med = np.median(matrix, axis=0)
            mad = np.median(np.abs(matrix - med), axis=0)
            rel = 1 / (1 + 1.4826 * mad / (np.abs(med) + 1e-10))
            z['reliability_records'] = {
                n: dict(original_contribution=float(matrix[0, j]), median_contribution=float(med[j]),
                        mad_contribution=float(mad[j]), reliability=float(rel[j]),
                        robust_relative_dispersion=float(1 / rel[j] - 1), baseline_scores=matrix[:, j].tolist())
                for j, n in enumerate(z['feature_names'])}
        assert sha(modelpath) == digest
        z.update(protocol=PROTOCOL, scoring_model=kind, selector_model=str(modelpath),
                 selector_model_sha256=digest, scorer_model_path=str(modelpath), scorer_model_sha256=digest,
                 model_record=str(shared / (kind + '.json')), model_record_sha256=sha(shared / (kind + '.json')),
                 baseline_scores=r['baseline_scores'], shared_source=str(shared / 'shared.json'),
                 input_scaling='none', p1_retrained=False)
        save(artifact(ds, sid, 'p1', kind), z)
    shallow = ROOT / 'results/sensitivity_five_20261028/cases' / ds / sid / 'shallow/alpha_0.01.pkl'
    x = pickle.loads(shallow.read_bytes())
    assert x['p1_sha256'] == m['p1_sha256']
    dest = case(ds, sid) / 'shallow/alpha_0.01.pkl'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        import shutil
        shutil.copyfile(shallow, dest)
    assert sha(dest) == sha(shallow)
    default = load(ROOT / 'results/sensitivity_five_20261028/cases' / ds / sid / 'p2/default.json')
    main = load(m['record'])['selection']
    active_p2 = load(source_case(ds, sid)['main_p2'])
    assert default['p2_features'] == active_p2['p2_features'] == main['p2_features']
    assert set(default['deep']) == set(active_p2['deep'])
    for level in default['deep']:
        for field in ('ci_tests', 'deletions', 'local_tests', 'rescue_tests'):
            assert default['deep'][level][field] == active_p2['deep'][level][field]
    save(artifact(ds, sid, 'p2', 'default'), default)
    z = stamp(ds, sid, 'CNN', main['final_features'])
    z.update(p2_features=main['p2_features'], reuse='frozen_main_experiment',
             source_record=m['record'], source_record_sha256=m['record_sha256'])
    save(artifact(ds, sid, 'selection', 'CNN'), z)
    save(case(ds, sid) / 'manifest.json', dict(m, protocol=PROTOCOL,
         study_source=str(shared), protocol_lock_sha256=sha(CODE / 'protocol.lock.json')))


def p2(ds, sid, key, force=False):
    import numpy as np
    import sens_restricted_kernel
    verify_runtime_modules()
    sens_restricted_kernel.save = save
    source = SENS / 'sens_worker.py'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'p2')
    ns = dict(globals(), np=np)
    ns['cpuclock'] = lambda: time.process_time() + resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime + resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
    ns['p2'](ds, sid, key, force=force)


def recover_result(ds, sid, kind, core):
    import numpy as np
    p = getp1(ds, sid, kind)
    eligible = set(p['high_contrib_features']) - set(core)
    if not eligible:
        return dict(core_features=list(core), eligible_residual_features=[], recovered_features=[],
                    candidate_pairs=0, evaluated_pairs=0, final_features=list(core), model_not_loaded=True)
    tf = init_tf()
    import shared_globals as sg
    from model_train_v6 import load_and_preprocess_data
    from feature_perturbation import CausalFeaturePerturbation
    from p3_recovery import ResidualInteractionRecovery
    verify_runtime_modules()
    tf.keras.utils.set_random_seed(int(p['seed']))
    sg.init()
    sg.random_seed = int(p['seed'])
    load_and_preprocess_data(p['selector_input'], None)
    assert sg.feature_names == p['feature_names']
    if kind == 'RF':
        import joblib
        model = joblib.load(p['scorer_model_path'])
        assert list(model.classes_) == list(range(len(load(STUDY / 'cases' / ds / sid / 'shared.json')['target_classes'])))
        model.n_jobs = 1
    else:
        model = tf.keras.models.load_model(p['scorer_model_path'], compile=False)

    class RawProbabilityAdapter:
        def __call__(self, x, training=False):
            a = np.asarray(x).reshape(-1, len(p['feature_names']))
            if kind == 'RF':
                out = model.predict_proba(a)
            else:
                a = a.astype(np.float32)
                out = model(a[:, :, None] if kind == 'CNN' else a, training=False).numpy()
            assert out.ndim == 2 and np.isfinite(out).all()
            return tf.convert_to_tensor(out)

        def predict(self, x, **kwargs):
            return self(x).numpy()

    # Keep the main CNN's native predict()/call() paths, batch handling and dtype.
    # Only non-CNN scorers need an interface adapter; it never scales raw inputs.
    adapter = model if kind == 'CNN' else RawProbabilityAdapter()
    sg.model = adapter
    c = CausalFeaturePerturbation(adapter, sg.data, dataset_name=ds + '.csv',
                                semantic_metadata_path=p['semantic_metadata_file'])
    assert sha(c.semantic_metadata_path) == p['semantic_metadata_sha256']
    for field in ('semantic_types', 'baseline_policy', 'perturbation_classes', 'perturbation_policy'):
        assert getattr(c, field) == p[field], ('Semantic policy changed', field)
    assert all(abs(float(c.baseline_np[0, j]) - p['baseline_values'][n]) < 1e-6 for j, n in enumerate(p['feature_names']))
    c.perturb_values = {n: list(p['perturbation_values'][n]) for n in p['feature_names']}
    c.con_list = [p['contribution_scores'][n] for n in p['feature_names']]
    # The historical P3 starts from the post-context RNG state, not a new seed reset.
    # Diagnostic contribution replay must leave that state untouched.
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        _, con = c.calculate_contributions([c.perturb_values[n] for n in p['feature_names']])
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)
    parity = dict(rtol=1e-5, atol=1e-7, max_absolute_error=float(np.max(np.abs(np.asarray(con) - c.con_list))))
    assert np.allclose(con, c.con_list, rtol=1e-5, atol=1e-7), ('P1/P3 scorer mismatch', ds, sid, kind, parity)
    sg.con_list = c.con_list
    sg.high_contrib_features = p['high_contrib_features']
    sg.p1_reliable_score = p['reliable_scores']
    sg.p1_contribution_reliability = {n: float(v['reliability']) for n, v in p['reliability_records'].items()}
    sg.perturbation_instance = c
    sg.x_list = [c.perturb_values[n] for n in p['feature_names']]
    sg.feature_values = c.feature_values
    sg.filtered_features = list(core)
    start = time.perf_counter()
    result = ResidualInteractionRecovery(candidate_top_k=200).run()
    result.update(adapter_score_parity=parity, input_scaling='none',
                  scorer_model_path=p['scorer_model_path'], scorer_model_sha256=p['scorer_model_sha256'],
                  pair_loop_seconds=time.perf_counter() - start)
    return result


def recover(ds, sid, key):
    core = load(artifact(ds, sid, 'p2', key))['p2_features']
    r = recover_result(ds, sid, key, core)
    z = stamp(ds, sid, key, r['final_features'])
    z.update(p2_features=core, p3=r, p1_source_sha256=sha(artifact(ds, sid, 'p1', key)),
             p2_source_sha256=sha(artifact(ds, sid, 'p2', key)))
    save(artifact(ds, sid, 'selection', key), z)


def parity(ds, sid):
    p2(ds, sid, 'CNN', force=True)
    q = load(artifact(ds, sid, 'p2', 'default_probe'))
    reference = load(artifact(ds, sid, 'selection', 'CNN'))
    assert q['p2_features'] == reference['p2_features']
    r = recover_result(ds, sid, 'CNN', q['p2_features'])
    assert r['final_features'] == reference['features'], ('Main CNN final-subset parity', ds, sid)
    save(artifact(ds, sid, 'parity', 'CNN'), dict(protocol=PROTOCOL, passed=True,
         p2_source_sha256=sha(artifact(ds, sid, 'p2', 'default_probe')), p3=r,
         main_final_features=reference['features'], source_record_sha256=reference['source_record_sha256']))


def evaluate(ds, sid, key):
    m = manifest(ds, sid)
    features = p1(ds, sid)['feature_names'] if key == 'FULL' else load(artifact(ds, sid, 'selection', key))['features']
    # Only main evaluator records and receipt-verified outputs of THIS campaign are eligible.
    if key == 'CNN':
        candidates = [(Path(m['selected_eval']), 'frozen_main_selected')]
    elif key == 'FULL':
        candidates = [(Path(m['full_eval']), 'frozen_main_full')]
    else:
        candidates = [(Path(m['full_eval']), 'frozen_main_full'), (Path(m['selected_eval']), 'frozen_main_selected')]
    for j in jobs():
        if j['args'][:3] == ['evaluate', ds, sid] and j['args'][3] != key and complete(j):
            candidates.append((Path(j['dest']), 'verified_campaign_same_evaluator'))
    for path, reason in candidates:
        r = load(path)
        if r['features'] == features and r['split_hash'] == m['split_hash']:
            z = stamp(ds, sid, key, features)
            z.update(evaluator='CNN', accuracy=r['accuracy'], f1_macro=r['f1_macro'], reuse=reason,
                     source_evaluation=str(path), source_evaluation_sha256=sha(path),
                     evaluation_source_sha256=sha(ROOT / 'code_sensitivity_v6_20261008/run_scaled_evaluator_full_vs_selected.py'))
            save(artifact(ds, sid, 'evaluation', key), z)
            return
    assert key in ('MLP', 'RF'), 'Main reference must reuse its exact evaluation'
    tf = init_tf()
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    from fixed_split_evaluator import _prepare_fixed_splits, _one_hot
    from model_train_v6 import smote_enn_resample, build_model, train_model
    import shared_globals
    verify_runtime_modules()
    source = ROOT / 'code_sensitivity_v6_20261008/run_scaled_evaluator_full_vs_selected.py'
    node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'scaled_eval')
    dest = artifact(ds, sid, 'evaluation', key)
    held_out = {}

    def recording_splits(*args, **kwargs):
        result = _prepare_fixed_splits(*args, **kwargs)
        held_out['y_true'] = np.asarray(result[5])
        return result

    def recording_train(*args, **kwargs):
        model, history = train_model(*args, **kwargs)
        model.save(dest.with_suffix('.keras'))
        save(dest.with_suffix('.history.json'), {k: list(map(float, v)) for k, v in history.history.items()})

        class RecordingModel:
            def predict(self, x, **kw):
                prob = model.predict(x, **kw)
                np.savez_compressed(dest.with_suffix('.probabilities.npz'), probabilities=prob,
                                    predicted=np.argmax(prob, axis=1), y_true=held_out['y_true'])
                return prob
        return RecordingModel(), history

    ns = dict(globals())
    ns.update(locals(), OLD=Path(m['p1']).parents[2], DATA=Path(m['data']).parent,
              train_model=recording_train, _prepare_fixed_splits=recording_splits)
    assert ns['DATA'] / (ds + '.csv') == Path(m['data'])
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), ns)
    dest.parent.mkdir(parents=True, exist_ok=True)
    z = ns['scaled_eval'](ds, sid, features, 'selected')
    assert z['features'] == features and z['split_hash'] == m['split_hash']
    z.update(protocol=PROTOCOL, kind=key, evaluator='CNN', reuse=False,
             evaluation_source=str(source), evaluation_source_sha256=sha(source),
             trained_model_sha256=sha(dest.with_suffix('.keras')),
             probabilities_sha256=sha(dest.with_suffix('.probabilities.npz')))
    save(dest, z)


def main():
    if len(sys.argv) < 3 or sys.argv[1] != '--execute-job':
        raise SystemExit('No work started. Worker requires --execute-job and a planned job ID.')
    job = next((j for j in jobs() if j['id'] == sys.argv[2]), None)
    if job is None:
        raise SystemExit('Unknown job ID')
    validate_code()
    if not all(complete(next(j for j in jobs() if j['id'] == dep)) for dep in job['deps']):
        raise RuntimeError('Dependencies incomplete or changed')
    path = OUT / 'job_locks' / (job['id'] + '.lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        if complete(job):
            return 0
        mode, ds, sid, *rest = job['args']
        attempt = OUT / 'attempts' / job['id'] / (time.strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex)
        for previous in attempt.parent.glob('*/status.json'):
            if load(previous).get('state') == 'running':
                save(previous.parent / 'interruption_detected.json', dict(detected_unix=time.time(),
                     reason='New attempt acquired same job lock without a valid completion receipt'))
        attempt.mkdir(parents=True)
        save(attempt / 'status.json', dict(job=job, pid=os.getpid(), state='running', started_unix=time.time()))
        try:
            validate_case(ds, sid)
            destination = Path(job['dest'])
            previous_outputs = list(destination.parent.glob(destination.stem + '.*'))
            previous_outputs.append(receipt_path(job))
            for path in previous_outputs:
                if path.exists():
                    import shutil
                    shutil.copy2(path, attempt / ('prior-' + path.parent.name + '-' + path.name))
            # Do not allow PYTHONPATH or a prior working directory to shadow frozen modules.
            runtime_paths = [str(BASE), str(ABL), str(SENS)]
            sys.path[:] = runtime_paths + [p for p in sys.path if p not in runtime_paths]
            os.chdir(attempt)
            with (attempt / 'stdout.log').open('w') as out, (attempt / 'stderr.log').open('w') as err:
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    globals()[mode](ds, sid, *rest)
            result = load(job['dest'])
            if result.get('protocol') != PROTOCOL:
                raise RuntimeError('Output protocol mismatch')
            save(receipt_path(job), dict(job=job['id'], fingerprint=fingerprint(job),
                 output_sha256=sha(job['dest']), artifacts=bundle(job), completed_unix=time.time(), attempt=str(attempt)))
            save(attempt / 'status.json', dict(job=job, state='complete', finished_unix=time.time()))
            return 0
        except BaseException:
            save(attempt / 'status.json', dict(job=job, state='failed', traceback=traceback.format_exc(), finished_unix=time.time()))
            raise


if __name__ == '__main__':
    try:
        status = main()
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
