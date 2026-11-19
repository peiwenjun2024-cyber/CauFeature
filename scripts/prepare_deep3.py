#!/usr/bin/env python3
"""Generate one Deep3 dataset, split, truth record, and local manifest."""
import argparse
import csv
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from prepare_inputs import PARTITIONS, sha256


PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE / 'historical/fast30_final/RQ4_EXPERIMENT_HUB_20261014/configs'
sys.path.insert(0, str(CONFIG))
from e1_deep3_extension_20261015 import DIMS, KINDS, SEEDS, generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dimension', type=int, choices=DIMS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory for the generated case')
    args = parser.parse_args()

    dim, seed = args.dimension, args.seed
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    stem = f'{KINDS[0]}_n5000_d{dim}_s{seed:02d}'
    data = output / f'{stem}.csv'
    split = output / f'{stem}.split.csv'
    names, x, y, truth = generate(n=5000, d=dim, seed=seed)
    frame = pd.DataFrame(x, columns=names)
    frame['target'] = y
    frame.to_csv(data, index=False, lineterminator='\n')
    development, test = train_test_split(range(len(frame)), test_size=.2, random_state=seed)
    train, validation = train_test_split(development, test_size=.25, random_state=seed)
    with split.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(('row_index', 'partition'))
        for part, indices in zip(PARTITIONS, (train, validation, test)):
            writer.writerows((i, part) for i in indices)

    truth.update(dataset=data.name, dataset_sha256=sha256(data), split_sha256=sha256(split))
    (output / f'{stem}.truth.json').write_text(json.dumps(truth, indent=2) + '\n', encoding='utf-8')
    manifest = dict(dataset=data.name, separator=',', n_samples=len(frame), n_features=dim,
                    sha256=truth['dataset_sha256'], split_seed_id=f'S{seed:02d}',
                    split_seed=seed, split_file=str(split), split_hash=truth['split_sha256'],
                    synthetic_profile='RQ4V3_E1_DEEP3', mechanism=KINDS[0], generation_seed=seed)
    (output / 'synthetic_manifest.jsonl').write_text(json.dumps(manifest) + '\n', encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
