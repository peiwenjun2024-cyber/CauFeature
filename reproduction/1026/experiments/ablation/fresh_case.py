"""Validate one locally built RQ1 case for the downstream experiment workers."""
import ast
import hashlib
import json
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_case_input(destination, data):
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, allow_nan=False).encode()
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError(f'Existing file describes different inputs: {destination}')
        return
    with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        stream.write(payload)
    Path(stream.name).replace(destination)


def install_manifest(source, fresh_root, output_case, dataset, seed):
    origin = (fresh_root / dataset / seed).resolve()
    if not origin.is_relative_to(fresh_root):
        raise ValueError(f'Fresh case escapes the RQ1 output root: {origin}')
    if output_case.resolve().is_relative_to(fresh_root):
        raise ValueError('The downstream output must be outside FAST30_FRESH_CASES_ROOT')
    p1_path, record_path = origin / 'p1_cache.json', origin / 'record.json'
    p1, record = (json.loads(path.read_text()) for path in (p1_path, record_path))
    if (p1['dataset'], p1['seed_id'], p1['status']) != (dataset + '.csv', seed, 'complete'):
        raise ValueError(f'P1 cache has the wrong case or status: {p1_path}')
    if (record['dataset'], record['split_seed_id'], record['status']) != (dataset + '.csv', seed, 'complete'):
        raise ValueError(f'RQ1 record has the wrong case or status: {record_path}')
    p1_hash = digest(p1_path)
    if record['input_cache_sha256']['p1_cache.json'] != p1_hash:
        raise ValueError(f'RQ1 record does not reference this P1 cache: {record_path}')
    if p1['split_hash'] != record['split_hash'] or int(p1['seed']) != int(seed[1:]):
        raise ValueError(f'RQ1 record and P1 cache do not agree on their split: {origin}')
    data_dir = ('pilot_optdigits_30_20261011/dataset' if dataset == 'Optdigits'
                else 'code_revised_20260910/dataset')
    data = source / data_dir / (dataset + '.csv')
    inputs = (
        (data, p1['dataset_sha256']),
        (Path(p1['split_file']), p1['split_hash']),
        (Path(p1['selector_model']), p1['selector_model_sha256']),
        (Path(p1['semantic_metadata_file']), p1['semantic_metadata_sha256']),
    )
    for path, expected in inputs:
        if digest(path) != expected:
            raise ValueError(f'Fresh RQ1 input has the wrong SHA-256: {path}')
    for key in ('selector_input', 'selector_model'):
        path = Path(p1[key]).resolve()
        if not path.is_relative_to(origin) or not path.is_file():
            raise ValueError(f'P1 {key} must be a file in this fresh case: {path}')
    manifest = dict(
        dataset=dataset, seed_id=seed, split_hash=p1['split_hash'],
        record=str(record_path), record_sha256=digest(record_path),
        p1=str(p1_path), p1_sha256=p1_hash, data=str(data), data_sha256=p1['dataset_sha256'],
        selector_model_sha256=p1['selector_model_sha256'], split_file=p1['split_file'],
        metadata_file=p1['semantic_metadata_file'], metadata_sha256=p1['semantic_metadata_sha256'],
        input_origin='locally_generated_rq1',
    )
    write_case_input(output_case / 'manifest.json', manifest)
    return manifest


def evaluate_fresh(common, dataset, seed, kind, protocol):
    """Run the archived evaluator function against the locally generated case."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    selection = common.load(common.selection_path(dataset, seed, kind))
    manifest = common.manifest(dataset, seed)
    if selection['split_hash'] != manifest['split_hash'] or not selection['features']:
        raise ValueError('Fresh selection does not match the case split')
    sys.path.insert(0, str(common.BASE))
    tf = common.tf_setup()
    from fixed_split_evaluator import _one_hot, _prepare_fixed_splits
    from model_train_v6 import build_model, smote_enn_resample, train_model
    import shared_globals

    source = Path(__file__).resolve().parents[1] / 'paper/rq6_fixed_cnn/historical/frozen_evaluator.py'
    node = next(n for n in ast.parse(source.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'scaled_eval')
    scope = dict(globals(), **locals(), OLD=Path(manifest['p1']).parents[2],
                 DATA=Path(manifest['data']).parent)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    result = scope['scaled_eval'](dataset, seed, selection['features'], 'selected')
    if result['features'] != selection['features'] or result['split_hash'] != manifest['split_hash']:
        raise ValueError('Fresh CNN evaluation changed feature order or split')
    result.update(kind=kind, protocol=protocol, evaluation_source=str(source),
                  evaluation_source_sha256=digest(source), reuse='new_standardized_evaluation')
    common.save(common.evaluation_path(dataset, seed, kind), result)
