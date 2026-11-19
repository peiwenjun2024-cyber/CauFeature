#!/usr/bin/env python3
"""Build Fig. 3(e) case inputs from a locally generated RQ1 case."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PACKAGE = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(PACKAGE / 'scripts'), str(PACKAGE / 'experiments/ablation')]
from fresh_case import install_manifest, write_case_input
from isolated_worker import required_output

CONTINUOUS = 'continuous_profile_quantile_or_linspace'
DATASETS = ('BankFull', 'OnlineShoppers', 'StudentsAcademic', 'Optdigits', 'SoftwareQuality')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=DATASETS)
    parser.add_argument('seed', choices=tuple(f'S{i:02}' for i in range(1, 31)))
    args = parser.parse_args()
    for name in ('FAST30_SOURCE_ROOT', 'FAST30_FRESH_CASES_ROOT'):
        if not os.environ.get(name):
            parser.error(f'Set {name} before preparing a fresh case')
    source = Path(os.environ['FAST30_SOURCE_ROOT']).expanduser().resolve()
    fresh_root = Path(os.environ['FAST30_FRESH_CASES_ROOT']).expanduser().resolve()
    cases_root = required_output('RQ5_CASES_ROOT', forbidden=(source, fresh_root, PACKAGE))
    case = cases_root / args.dataset / args.seed
    if not case.resolve().is_relative_to(cases_root):
        parser.error(f'Case directory escapes RQ5_CASES_ROOT: {case}')
    manifest = install_manifest(source, fresh_root, case, args.dataset, args.seed)
    p1 = json.loads(Path(manifest['p1']).read_text())
    continuous = [name for name in p1['feature_names'] if p1['perturbation_policy'][name] == CONTINUOUS]
    grids = {strategy: dict(p1['perturbation_values']) for strategy in ('quantile_only', 'linspace_only')}
    if continuous:
        development = pd.read_csv(p1['selector_input'], usecols=continuous)
        for name in continuous:
            values = pd.to_numeric(development[name], errors='coerce').dropna().to_numpy(float)
            if not len(values) or not np.isfinite(values).all():
                raise ValueError(f'Continuous feature has no finite development support: {name}')
            quantiles = np.percentile(values, np.linspace(0, 100, 101))
            uniform = np.linspace(float(values.min()), float(values.max()), 101)
            grids['quantile_only'][name] = np.unique(quantiles).tolist()
            grids['linspace_only'][name] = np.unique(uniform).tolist()
    for strategy, support in grids.items():
        write_case_input(case / 'p1' / f'continuous_policy__{strategy}.json',
                         dict(source_p1_sha256=manifest['p1_sha256'], perturbation_values=support))
    print('FIG3E_SUPPORTS_COMPLETE', args.dataset, args.seed, case)


if __name__ == '__main__':
    main()
