#!/usr/bin/env python3
"""Check the documented one-case runs against the external historical records."""
import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


ONLINE = Path('OnlineShoppers/S01')
THYROID = Path('ThyroidDiff/S04')
OPT = Path('Optdigits/S01')
BANK = Path('BankFull/S01')
SCORES = ('accuracy', 'f1_macro')
CNN = ('dataset', 'seed_id', 'features', 'dimension', 'split_hash',
       'resampled_train_n', 'scaler_mean', 'scaler_scale', 'preprocessing',
       'epochs_ran', 'best_epoch')
TIMING_KEYS = {'sec', 'seconds', 'cpu_seconds', 'wall_seconds', 'train_seconds',
               'test_seconds', 'train_test_seconds', 'replay_seconds', 'pair_loop_seconds',
               'selection_seconds', 'evaluation_seconds', 'shallow_wall_seconds',
               'shallow_cpu_seconds', 'deep_wall_seconds', 'deep_cpu_seconds'}
DEFAULT_JS = Path('/root/autodl-fs/multibranch_ablation_20261113/equal_budget_max_js_20261114')


def load(path):
    with path.open(encoding='utf-8') as f:
        return json.load(f)


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def without_timing(value):
    if isinstance(value, dict):
        return {k: without_timing(v) for k, v in value.items() if k not in TIMING_KEYS}
    if isinstance(value, list):
        return [without_timing(v) for v in value]
    return value


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Verification:
    def __init__(self, output, source, js_source):
        self.output = output
        self.source = source
        self.js_source = js_source
        self.failures = []
        self.score_differences = []

    def check(self, description, observed, original):
        if without_timing(observed) != without_timing(original):
            self.failures.append(description)

    def pair(self, name, new, old, keys=(), scores=False):
        a, b = load(new), load(old)
        for key in keys:
            self.check(f'{name}.{key}', a[key], b[key])
        if scores:
            for key in SCORES:
                if a[key] != b[key]:
                    self.score_differences.append(
                        f'{name}.{key} {a[key]:.10g} vs {b[key]:.10g}'
                    )


def rq1(v):
    original = v.source / 'results/scaled_evaluator_full_vs_selected_20261010/cases' / THYROID
    for mode in ('full', 'selected'):
        name = mode + '_scaled.json'
        v.pair(mode, v.output / 'rq1/cases' / THYROID / name, original / name, CNN, True)


def rq1_core(v):
    p = 'record.json'
    a = load(v.output / 'rq1-core' / THYROID / p)
    b = load(v.source / 'results/alpha001_full30_20261009' / THYROID / p)
    v.check('complete', a['status'], 'complete')
    v.check('split_hash', a['split_hash'], b['split_hash'])
    for key in ('p1_high_contrib_features', 'p2_features', 'p3_recovered_features',
                'final_features', 'target_path_count'):
        v.check(key, a['selection'][key], b['selection'][key])


def optdigits(v):
    new = v.output / 'rq1-optdigits' / OPT
    old = v.source / 'results/alpha001_full30_20261009' / OPT
    record, reference = (load(root / 'record.json') for root in (new, old))
    for key in ('status', 'dataset', 'split_seed_id', 'split_seed', 'split_hash', 'selection'):
        v.check(f'record.{key}', record[key], reference[key])
    for stage, keys in {
        'p1': ('split_hash', 'high_contrib_features', 'feature_names',
               'contribution_scores', 'reliable_scores'),
        'p2': ('split_hash', 'p2_features', 'target_path_count', 'task_path_count',
               'frontier_edge_count', 'adaptive_stopped', 'deep'),
        'p3': ('split_hash', 'final_features', 'recovered_features',
               'candidate_pairs', 'evaluated_pairs'),
    }.items():
        file = stage + '_cache.json'
        v.pair(stage, new / file, old / file, keys)
    for mode in ('full', 'selected'):
        name = mode + '_scaled.json'
        v.pair(mode, new / name, old / name,
               ('dataset', 'seed_id', 'mode', 'dimension', 'features', 'split_hash',
                'best_epoch', 'scaler_fit', 'resampling'), True)


def rq2(v):
    output = v.output / 'rq2'
    formal = v.source / '2026_peiwenjun/statistics/rq2/identity_reuse_20261023/records'
    historical = v.source / 'explainable_5ds_s30_20261012/results'
    selections = sorted((output / 'results').glob('*/OnlineShoppers/S01/selection.json'))
    if not any(selection.parents[2].name == 'PPFS' for selection in selections):
        raise ValueError('missing RQ2 OnlineShoppers S01 PPFS selection')
    entries = [json.loads(line) for line in (output / 'exit_audit.jsonl').read_text().splitlines()]
    for selection in selections:
        method = selection.parents[2].name
        rel = Path(method) / ONLINE
        v.pair(method + '.selection', selection, historical / rel / 'selection.json',
               ('status', 'features', 'selected_dimension', 'split_file_sha256'))
        v.pair(method, selection.with_name('scaled_eval.json'), formal / rel / 'scaled_eval.json',
               ('status', 'selected_dimension', 'selected_features', 'split_file_sha256'), True)
        last_exit = {x['stage']: (x['exit_code'], x['result_valid']) for x in entries
                     if (x['method'], x['dataset'], x['seed_id']) == (method, 'OnlineShoppers', 'S01')}
        v.check(method + '.worker_exits', last_exit, {'select': (0, True), 'evaluate': (0, True)})


def rq3(v):
    case = Path('MI') / ONLINE
    new = v.output / 'rq3' / case
    old = v.source / 'results/matched_baselines_5ds_s30_trainonly_20261011' / case
    v.check('selection', load(new / 'selection.json'), load(old / 'selection.json'))
    v.pair('MI CNN', new / 'scaled_eval.json', old / 'scaled_eval.json',
           ('split_hash', 'selected_features', 'budget_k'), True)


def rq4(v):
    new = v.output / 'rq4/cases' / ONLINE
    old = v.source / 'results/ablation_five_20261028/cases' / ONLINE
    name = 'p3_all_pairs.json'
    v.pair('P3', new / 'selection' / name, old / 'selection' / name,
           ('features', 'split_hash', 'candidate_pairs', 'evaluated_pairs', 'pair_details'))
    v.pair('CNN', new / 'evaluation' / name, old / 'evaluation' / name, CNN, True)


def deep3(v):
    old = v.source / 'RQ4_EXPERIMENT_HUB_20261014/reports/paper_s4_validation_20261015'
    for name in ('verified_per_seed.csv', 'verified_summary.json'):
        v.check(name, sha(v.output / 'rq4-deep3' / name), sha(old / name))


def rq5_alpha(v):
    new = v.output / 'rq5-alpha/cases' / BANK
    old = v.source / 'results/sensitivity_five_20261028/cases' / BANK
    name = 'alpha__0p005.json'
    v.pair('P2', new / 'p2' / name, old / 'p2' / name,
           ('p2_features', 'split_hash', 'task_paths', 'all_target_paths', 'shallow', 'deep'))
    v.pair('selection', new / 'selection' / name, old / 'selection' / name,
           ('features', 'dimension', 'split_hash', 'p3'))
    v.pair('CNN', new / 'evaluation' / name, old / 'evaluation' / name, CNN, True)


def rq5_e(v):
    case = Path('cases/OnlineShoppers_S01.json')
    new = load(v.output / 'rq5-e' / case)
    original = load(v.js_source / case)
    for key in ('split_hash', 'p1_sha256', 'model_sha256'):
        v.check(key, new[key], original[key])
    v.check('feature count', len(new['rows']), len(original['rows']))
    for item, ref in zip(new['rows'], original['rows']):
        v.check('row fields', set(item), set(ref))
        for key, value in item.items():
            if key.endswith('_js'):
                if not math.isclose(value, ref[key], rel_tol=0, abs_tol=1e-7):
                    v.failures.append(f'JS {key}')
            else:
                v.check(key, value, ref[key])
    panel = load(v.output / 'rq5-e/fig3e_data.json')
    summary = load(v.js_source / 'summary.json')
    v.check('panel dataset coverage', set(panel), set(summary['cases_by_dataset']))
    for ds, info in panel.items():
        counts = {'default': 0, 'uniform': 0}
        cases = v.source / 'results/sensitivity_five_20261028/cases' / ds
        count = summary['cases_by_dataset'][ds]
        for number in range(1, count + 1):
            path = cases / f'S{number:02d}'
            manifest = load(path / 'manifest.json')
            for key, record in [('default', load(Path(manifest['p1']))),
                                ('uniform', load(path / 'p1/continuous_policy__linspace_only.json'))]:
                counts[key] += sum(len(set(grid)) for grid in record['perturbation_values'].values())
        v.check(ds + ' point counts', info['native_total_points'], counts)
        v.check(ds + ' archived case count', info['n'], count)
        v.check(ds + ' mean error', info['equal_budget_mean_error'],
                summary['by_dataset'].get(ds, {}).get('mean_error'))
    png = v.output / 'rq5-e/rq5_continuous_policy_default_vs_uniform.png'
    v.check('PNG signature', png.read_bytes()[:8], bytes.fromhex('89504e470d0a1a0a'))


def rq6(v):
    new = v.output / 'rq6/cases' / ONLINE
    old = v.source / 'results/unscaled_scoring_cnn_eval_20261110/cases' / ONLINE
    v.pair('RF evaluation', new / 'evaluation/RF.json', old / 'evaluation/RF.json', (*CNN, 'reuse'), True)
    for phase in ('p1', 'p2'):
        v.check(phase, load(new / phase / 'RF.json'), load(old / phase / 'RF.json'))
    selections = []
    for label, root in (('replay', new), ('historical', old)):
        selection = load(root / 'selection/RF.json')
        # The referenced P2 JSON includes time measurements, which change its digest.
        v.check(label + ' P2 digest', selection.pop('p2_source_sha256'), sha(root / 'p2/RF.json'))
        selections.append(selection)
    v.check('selection', *selections)
    import numpy as np
    with np.load(new / 'evaluation/RF.probabilities.npz') as a, np.load(old / 'evaluation/RF.probabilities.npz') as b:
        v.check('prediction keys', set(a.files), set(b.files))
        for key in set(a.files) & set(b.files):
            v.check(key, np.array_equal(a[key], b[key]), True)


def stability(v):
    observed = table(v.output / 'rq6-stability.csv')
    v.check('requested datasets', [r['dataset'] for r in observed], ['OnlineShoppers'])
    for row in observed:
        ds = row['dataset']
        archive = v.source / '2026_peiwenjun/statistics/rq6/stage_stability.csv'
        ref = next(r for r in table(archive) if r['dataset'] == ds and r['phase'] == 'Final')
        for key in ('n_seeds', 'pair_count', 'consensus_count', 'ever_selected_count'):
            v.check(key, int(row[key]), int(ref[key]))
        main_path = v.source / '2026_peiwenjun/revision_optdigits_main_20261012/figures/data/rq1_main_scaled_15.csv'
        main = next(r for r in table(main_path) if r['dataset'] == ds)
        v.check('original_dimension', int(row['original_dimension']), int(main['original_dimension']))
        expected = {
            'dimension_mean': ref['dimension_mean'], 'dimension_sd': ref['dimension_sd'],
            'pairwise_jaccard_mean': ref['jaccard_mean'], 'pairwise_jaccard_sd': ref['jaccard_sd'],
            'reduction_percent_mean': main['reduction_mean'], 'reduction_percent_sd': main['reduction_sd'],
        }
        for key, value in expected.items():
            v.check(key, math.isclose(float(row[key]), float(value), rel_tol=0, abs_tol=1e-12), True)


def rq7(v):
    new = v.output / 'rq7'
    old = v.source / 'results/rq7_onlineshoppers_s01_20261110'
    keys = ('split_hash', 'p1', 'p2', 'p3', 'final', 'main_metrics')
    v.pair('evidence', new / 'evidence.json', old / 'evidence.json', keys)
    a, b = (load(path / 'evidence.json')['sources'] for path in (new, old))
    v.check('source hashes', {k: x['sha256'] for k, x in a.items()}, {k: x['sha256'] for k, x in b.items()})
    figures = sorted(old.glob('fig[34]_onlineshoppers_s01_*.png'))
    v.check('historical figure count', len(figures), 4)
    for f in figures:
        v.check(f.name, sha(new / f.name), sha(f))


def rq8(v):
    old = v.source / '2026_peiwenjun/statistics/rq678_audit_20261024/runtime_per_seed.csv'
    actual = table(v.output / 'rq8.csv')
    v.check('dataset', [row['Dataset'] for row in actual], ['SoftwareQuality'])
    for row in actual:
        ref = next(x for x in table(old) if x['dataset'] == row['Dataset'] and x['seed'] == 'S01')
        values = [float(ref[k]) for k in ('selector_fit_s', 'p1_s', 'p2_s', 'p3_s')]
        v.check('n', int(row['N']), 1)
        for key, expected in zip(('Selector', 'P1', 'P2', 'P3', 'Overall'), (*values, sum(values))):
            v.check(key, float(row[key + '_mean_s']), expected)
            v.check(key + ' sd', float(row[key + '_sd_s']), 0.0)


CHECKS = {'rq1': rq1, 'rq1-core': rq1_core, 'optdigits': optdigits, 'rq2': rq2,
          'rq3': rq3, 'rq4': rq4, 'rq4-deep3': deep3, 'rq5-alpha': rq5_alpha,
          'rq5-e': rq5_e, 'rq6': rq6, 'rq6-stability': stability, 'rq7': rq7, 'rq8': rq8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Root containing the documented smoke outputs')
    parser.add_argument('--source', type=Path, default=Path(os.environ.get('FAST30_SOURCE_ROOT', '/root/autodl-tmp/fast30_final')))
    parser.add_argument('--js-source', type=Path, default=DEFAULT_JS)
    parser.add_argument('--checks', nargs='+', choices=CHECKS, default=tuple(CHECKS))
    args = parser.parse_args()

    invalid = False
    score_difference = False
    for name in dict.fromkeys(args.checks):
        v = Verification(args.output, args.source, args.js_source)
        try:
            receipt = args.output / f'{name}.rc'
            if receipt.is_file():
                v.check('process exit', int(receipt.read_text().strip()), 0)
            CHECKS[name](v)
        except (OSError, KeyError, ValueError, TypeError, StopIteration) as error:
            v.failures.append(str(error))
        if v.failures:
            invalid = True
            print(f'{name} INVALID ' + ', '.join(v.failures))
        elif v.score_differences:
            score_difference = True
            print(f'{name} SCORE_DIFFERENCE ' + ', '.join(v.score_differences))
        else:
            print(f'{name} MATCH')
    return 1 if invalid else 2 if score_difference else 0


if __name__ == '__main__':
    raise SystemExit(main())
