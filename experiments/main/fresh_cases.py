"""Resolve the inputs of a locally run CauFeature case."""

import hashlib
import json
import re
from pathlib import Path


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def load_fresh_case(data_root, cases_root, dataset, seed_id):
    if not re.fullmatch(r'[A-Za-z0-9_]+', dataset) or not re.fullmatch(r'S(?:0[1-9]|[12][0-9]|30)', seed_id):
        raise ValueError(f'Invalid dataset or seed: {dataset}/{seed_id}')

    case = cases_root / dataset / seed_id
    with (case / 'record.json').open(encoding='utf-8') as stream:
        record = json.load(stream)
    with (case / 'p1_cache.json').open(encoding='utf-8') as stream:
        p1 = json.load(stream)
    if record['status'] != 'complete' or p1['status'] != 'complete':
        raise ValueError(f'Run CauFeature through P3 first: {case}')
    for recorded_name in (record['dataset'], p1['dataset']):
        if recorded_name not in (dataset, f'{dataset}.csv'):
            raise ValueError(f'Unexpected dataset in {case}: {recorded_name}')
    if p1['seed_id'] != seed_id or record['split_seed_id'] != seed_id:
        raise ValueError(f'Unexpected seed in {case}: {seed_id}')
    if record['split_hash'] != p1['split_hash']:
        raise ValueError(f'Split mismatch between record and P1 cache: {case}')

    if dataset == 'Optdigits':
        dataset_dir = data_root / 'pilot_optdigits_30_20261011' / 'dataset'
    else:
        dataset_dir = data_root / 'code_revised_20260910' / 'dataset'
    data = dataset_dir / f'{dataset}.csv'
    split = Path(p1['split_file'])
    for path, expected in ((data, p1['dataset_sha256']), (split, p1['split_hash'])):
        if sha256(path) != expected:
            raise ValueError(f'Input SHA-256 differs from the P1 cache: {path}')
    return data, p1, record
