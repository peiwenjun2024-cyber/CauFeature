#!/usr/bin/env python3
"""Verify the archived inputs, regenerate fixed splits, and build a local RQ2 manifest."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHODS = ('SHAP', 'LIME', 'FeatureX', 'PPFS', 'BAMB-Z', 'EAMB-Z')
PARTITIONS = ('train', 'validation', 'test')


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def require_hash(observed, expected, name):
    if observed != expected:
        raise ValueError(f'{name}: SHA-256 {observed}, expected {expected}')


def read_labels(path):
    with path.open(encoding='utf-8', newline='') as source:
        return [int(row['target']) for row in csv.DictReader(source)]


def create_split(row, labels):
    from sklearn.model_selection import train_test_split

    n_rows = sum(int(row[f'n_{part}']) for part in PARTITIONS)
    seed = int(row['seed_id'][1:])
    if row['dataset'] == 'Optdigits':
        dev, test = train_test_split(range(n_rows), test_size=0.2, random_state=seed, stratify=labels)
        train, validation = train_test_split(
            dev, test_size=0.2, random_state=2 if seed == 1 else seed + 1000,
            stratify=[labels[index] for index in dev],
        )
    else:
        dev, test = train_test_split(range(n_rows), test_size=0.2, random_state=seed)
        train, validation = train_test_split(dev, test_size=0.25, random_state=seed)

    records = [(index, part) for part, indices in zip(PARTITIONS, (train, validation, test))
               for index in indices]
    if row['dataset'] == 'Optdigits':
        records.sort()
    buffer = io.StringIO(newline='')
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(('row_index', 'partition'))
    writer.writerows(records)
    return buffer.getvalue().encode('utf-8')


def verify_inputs(rows, data_root, check_regeneration, write_missing):
    checked_datasets = set()
    labels = None
    for row in rows:
        data = data_root / row['data_path']
        split = data_root / row['split_path']
        if data not in checked_datasets:
            require_hash(sha256(data), row['data_sha256'], data)
            checked_datasets.add(data)
            if row['dataset'] == 'Optdigits' and (check_regeneration or write_missing):
                labels = read_labels(data)

        if check_regeneration or (write_missing and not split.exists()):
            generated = create_split(row, labels)
            require_hash(hashlib.sha256(generated).hexdigest(), row['split_sha256'], f'generated {split}')
            if write_missing and not split.exists():
                split.parent.mkdir(parents=True, exist_ok=True)
                with split.open('xb') as destination:
                    destination.write(generated)

        require_hash(sha256(split), row['split_sha256'], split)


def write_rq2_manifest(destination, rows, data_root):
    records = [
        {
            'dataset': row['dataset'], 'seed_id': row['seed_id'], 'method': method,
            'data_path': str(data_root / row['data_path']),
            'split_path': str(data_root / row['split_path']),
            'split_file_sha256': row['split_sha256'],
            'reference_split_hash': row['split_sha256'],
            'selection_policy': 'native_method_no_budget_matching',
        }
        for row in rows for method in METHODS
    ]
    manifest = {'total': len(records), 'datasets': list(dict.fromkeys(row['dataset'] for row in rows)),
                'methods': METHODS, 'rows': records}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as output:
        json.dump(manifest, output, indent=2)
        output.write('\n')


def write_rq1_manifest(destination, rows, data_root):
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8') as output:
        for row in rows:
            entry = {
                'dataset': Path(row['data_path']).name,
                'separator': row['separator'],
                'n_samples': int(row['n_samples']),
                'n_features': int(row['n_features']),
                'sha256': row['data_sha256'],
                'split_seed_id': row['seed_id'],
                'split_seed': int(row['seed']),
                'split_file': str(data_root / row['split_path']),
                'split_hash': row['split_sha256'],
                **{f'n_{part}': int(row[f'n_{part}']) for part in PARTITIONS},
            }
            output.write(json.dumps(entry) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--experiment', choices=('rq1', 'rq2'), default='rq2')
    parser.add_argument('--dataset', help='Check a single dataset.')
    parser.add_argument('--seed', help='Check a single archived seed ID, such as S01.')
    parser.add_argument('--check-regeneration', action='store_true', help='Recompute each split hash in memory.')
    parser.add_argument('--write-missing-splits', action='store_true', help='Only create missing splits whose hash matches.')
    parser.add_argument('--rq1-manifest', type=Path, help='Create a fresh manifest for the common RQ1 pipeline.')
    parser.add_argument('--rq2-manifest', type=Path, help='Create a fresh manifest with local input paths.')
    args = parser.parse_args()
    if args.rq1_manifest and args.experiment != 'rq1':
        parser.error('--rq1-manifest requires --experiment rq1')
    if args.rq2_manifest and args.experiment != 'rq2':
        parser.error('--rq2-manifest requires --experiment rq2')

    registry = ROOT / 'provenance' / f'{args.experiment}-inputs.csv'
    with registry.open(encoding='utf-8-sig', newline='') as source:
        rows = [row for row in csv.DictReader(source)
                if (not args.dataset or row['dataset'] == args.dataset)
                and (not args.seed or row['seed_id'] == args.seed)]
    if not rows:
        parser.error('No archived cases match the requested dataset and seed.')
    if args.rq1_manifest and any(row['dataset'] == 'Optdigits' for row in rows):
        parser.error('Select a non-Optdigits dataset for --rq1-manifest. The Optdigits launcher builds its own manifest.')

    data_root = args.data_root.resolve(strict=True)
    verify_inputs(rows, data_root, args.check_regeneration, args.write_missing_splits)
    if args.rq1_manifest:
        write_rq1_manifest(args.rq1_manifest, rows, data_root)
    if args.rq2_manifest:
        write_rq2_manifest(args.rq2_manifest, rows, data_root)
    print(f'Verified {len(rows)} dataset/split pairs across {len({row["dataset"] for row in rows})} datasets.')


if __name__ == '__main__':
    main()
