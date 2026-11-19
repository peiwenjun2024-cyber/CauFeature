"""Prepare Table 9 inputs from a fresh RQ1 case and regenerated scorer study."""

import ast
from pathlib import Path
import pickle


def checked_file(worker, path, expected):
    path = Path(path).resolve(strict=True)
    if worker.sha(path) != expected:
        raise ValueError(f'Input SHA-256 differs from its case record: {path}')
    return path


def build_manifest(worker, dataset, seed_id, core, scaled, data_file):
    p1_path = core / 'p1_cache.json'
    record_path = core / 'record.json'
    full_path = scaled / 'full_scaled.json'
    selected_path = scaled / 'selected_scaled.json'
    p1 = worker.load(p1_path)
    record = worker.load(record_path)
    full, selected = map(worker.load, (full_path, selected_path))
    split_hash = p1['split_hash']
    if (
        p1['dataset'] != f'{dataset}.csv' or p1['seed_id'] != seed_id
        or p1['status'] != 'complete' or p1['outer_test_rows_used'] is not False
        or record['status'] != 'complete' or record['split_seed_id'] != seed_id
        or any(obj['split_hash'] != split_hash for obj in (record, full, selected))
        or (full['dataset'], full['seed_id'], full['mode']) != (dataset, seed_id, 'full')
        or (selected['dataset'], selected['seed_id'], selected['mode']) != (dataset, seed_id, 'selected')
        or full['features'] != p1['feature_names']
        or selected['features'] != record['selection']['final_features']
    ):
        raise ValueError(f'Inconsistent fresh RQ1 inputs: {dataset}/{seed_id}')
    data = checked_file(worker, data_file, p1['dataset_sha256'])
    split = checked_file(worker, p1['split_file'], split_hash)
    metadata = checked_file(worker, p1['semantic_metadata_file'], p1['semantic_metadata_sha256'])
    model = checked_file(worker, p1['selector_model'], p1['selector_model_sha256'])
    if model.parent != core or Path(p1['selector_input']).resolve(strict=True).parent != core:
        raise ValueError(f'The CNN selector and development CSV must belong to {core}')
    return dict(
        dataset=dataset, seed_id=seed_id, split_hash=split_hash, record=str(record_path),
        record_sha256=worker.sha(record_path), p1=str(p1_path), p1_sha256=worker.sha(p1_path),
        data=str(data), data_sha256=p1['dataset_sha256'], full_eval=str(full_path),
        selected_eval=str(selected_path), full_eval_sha256=worker.sha(full_path),
        selected_eval_sha256=worker.sha(selected_path), selector_model_sha256=p1['selector_model_sha256'],
        split_file=str(split), metadata_file=str(metadata), metadata_sha256=p1['semantic_metadata_sha256'],
    )


def rebuild_shallow(worker, dataset, seed_id):
    """Use the sensitivity campaign's original shallow PC and audit kernel."""
    import numpy as np

    source = worker.SENS / 'sens_worker.py'
    names = {'cpuclock', 'shallow'}
    tree = ast.parse(source.read_text(encoding='utf-8'))
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in functions} != names:
        raise ValueError(f'Cannot find shallow PC entrypoint: {source}')
    namespace = dict(vars(worker), np=np)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
    namespace['shallow'](dataset, seed_id, .01)


def prepare_case(worker, dataset, seed_id, core_case, scaled_case, study_case, output_case, data_file, scorers):
    core = Path(core_case).resolve(strict=True)
    scaled = Path(scaled_case).resolve(strict=True)
    study = Path(study_case).resolve(strict=True)
    output_case = Path(output_case).resolve()
    if output_case != worker.case(dataset, seed_id).resolve() or study != (
        worker.STUDY / 'cases' / dataset / seed_id
    ).resolve():
        raise ValueError('Fresh case and study paths do not match the worker output layout')
    manifest = build_manifest(worker, dataset, seed_id, core, scaled, data_file)
    p2_path = core / 'p2_cache.json'
    p2 = worker.load(p2_path)
    expected_alpha = .01 if dataset == 'Optdigits' else None
    if (
        p2.get('pc_alpha', expected_alpha) != .01
        or p2['status'] != 'complete' or p2['seed_id'] != seed_id
        or p2['split_hash'] != manifest['split_hash']
        or p2['input_p1_cache_sha256'] != manifest['p1_sha256']
        or p2['p2_features'] != worker.load(manifest['record'])['selection']['p2_features']
    ):
        raise ValueError(f'Expected a complete matching α=0.01 P2 cache: {p2_path}')
    worker.save(output_case / 'manifest.json', manifest)
    rebuild_shallow(worker, dataset, seed_id)
    shallow = output_case / 'shallow' / 'alpha_0.01.pkl'
    with shallow.open('rb') as stream:
        summary = pickle.load(stream)['summary']
    if summary != {
        level: {key: numbers[key] for key in ('ci_tests', 'deletions', 'any_testable')}
        for level, numbers in p2['shallow'].items()
    }:
        raise ValueError(f'Regenerated shallow PC counters differ from fresh RQ1: {dataset}/{seed_id}')

    # Give the frozen worker the same paths it expects, local to this new run.
    local_sensitivity = worker.ROOT / 'results' / 'sensitivity_five_20261028' / 'cases' / dataset / seed_id
    local_shallow = local_sensitivity / 'shallow' / 'alpha_0.01.pkl'
    local_shallow.parent.mkdir(parents=True, exist_ok=True)
    if not local_shallow.is_symlink() and not local_shallow.exists():
        local_shallow.symlink_to(shallow)
    if local_shallow.resolve(strict=True) != shallow.resolve(strict=True):
        raise ValueError(f'Unexpected shallow PC source in fresh output: {local_shallow}')
    default = dict(p2, **worker.stamp(dataset, seed_id, 'default', p2['p2_features']))
    worker.save(local_sensitivity / 'p2' / 'default.json', default)

    source_case, kinds = worker.source_case, worker.KINDS
    try:
        worker.source_case = lambda _dataset, _seed: {'main_p2': str(p2_path)}
        worker.KINDS = ('CNN', *scorers)
        worker.prepare(dataset, seed_id)
    finally:
        worker.source_case, worker.KINDS = source_case, kinds
