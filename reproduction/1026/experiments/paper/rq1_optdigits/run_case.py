#!/usr/bin/env python3
"""Run the frozen Optdigits selector and Table 3 evaluator for one seed."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


PACKAGE = Path(__file__).resolve().parents[3]
RUNTIME = PACKAGE / '02_runtime_variants' / 'Optdigits'
REGISTRY = PACKAGE / 'provenance' / 'rq1-inputs.csv'
DEFAULT_SOURCE = '/root/autodl-tmp/fast30_final'


def verify_hash(path, expected):
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != expected:
        raise ValueError(f'Input SHA-256 mismatch for {path}: expected {expected}, got {digest}')


def case_inputs(seed, data_root):
    with REGISTRY.open(newline='', encoding='utf-8') as stream:
        row, = (r for r in csv.DictReader(stream) if r['dataset'] == 'Optdigits' and r['seed_id'] == seed)
    data = data_root / row['data_path']
    split = (data_root / row['split_path']).resolve()
    verify_hash(data, row['data_sha256'])
    verify_hash(split, row['split_sha256'])
    manifest = {
        'dataset': 'Optdigits.csv',
        'separator': row['separator'],
        'n_samples': int(row['n_samples']),
        'n_features': int(row['n_features']),
        'sha256': row['data_sha256'],
        'split_seed_id': seed,
        'split_seed': int(row['seed']),
        'split_file': str(split),
        'split_hash': row['split_sha256'],
        **{f'n_{part}': int(row[f'n_{part}']) for part in ('train', 'validation', 'test')},
    }
    return data, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', choices=[f'S{i:02d}' for i in range(1, 31)], required=True)
    parser.add_argument('--data-root', type=Path, help='Root of the input paths in provenance/rq1-inputs.csv')
    parser.add_argument('--output', type=Path, required=True, help='Output root outside the package and source tree')
    args = parser.parse_args()

    source = Path(os.environ.get('FAST30_SOURCE_ROOT', DEFAULT_SOURCE)).expanduser().resolve()
    data_root = (args.data_root or source).expanduser().resolve()
    output = args.output.expanduser().resolve()
    for protected in (PACKAGE, source, data_root, Path(DEFAULT_SOURCE)):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            parser.error(f'--output must not overlap {protected}')
    if not (RUNTIME / 'stage1_runtime.py').is_file():
        parser.error('Run python scripts/restore_archives.py to restore the Optdigits runtime first.')

    data, manifest = case_inputs(args.seed, data_root)
    case = (output / 'Optdigits' / args.seed).resolve()
    if not case.is_relative_to(output):
        parser.error(f'Case path escapes --output: {case}')
    case.mkdir(parents=True, exist_ok=False)
    manifest_path = case / 'split_manifest.jsonl'
    manifest_path.write_text(json.dumps(manifest) + '\n', encoding='utf-8')

    env = os.environ.copy()
    env.update({
        'FAST30_DATASET_DIR': str(data.parent),
        'FAST30_OUTPUT_ROOT': str(output),
        'OMP_NUM_THREADS': '4',
        'OPENBLAS_NUM_THREADS': '1',
        'MKL_NUM_THREADS': '1',
        'NUMEXPR_NUM_THREADS': '1',
        'PYTHONDONTWRITEBYTECODE': '1',
    })
    env.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
    worker = Path(__file__).with_name('worker.py').resolve()
    for stage in ('stage1', 'stage2', 'stage3', 'scaled'):
        log = case / f'{stage}.log'
        command = [sys.executable, '-u', str(worker), stage, str(manifest_path), str(data), str(case)]
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, env=env, cwd=case, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(f'Optdigits {args.seed} {stage} failed ({result.returncode}), see {log}')
        print(f'Optdigits {args.seed} {stage} complete', flush=True)


if __name__ == '__main__':
    main()
