"""Build Fig. 3(e) from frozen perturbation supports and equal-budget results."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE / 'scripts'))
from isolated_worker import required_output

DATASETS = ('BankFull', 'OnlineShoppers', 'StudentsAcademic', 'Optdigits', 'SoftwareQuality')


def load(path):
    with path.open() as stream:
        return json.load(stream)


def point_totals(cases):
    counts = {}
    for ds in DATASETS:
        counts[ds] = Counter()
        for i in range(1, 31):
            case = cases / ds / f'S{i:02}'
            manifest = load(case / 'manifest.json')
            original = load(Path(manifest['p1']))
            uniform = load(case / 'p1' / 'continuous_policy__linspace_only.json')
            if original['split_hash'] != manifest['split_hash'] or uniform['source_p1_sha256'] != manifest['p1_sha256']:
                raise ValueError(f'Inconsistent frozen inputs: {case}')
            for strategy, data in [('default', original), ('uniform', uniform)]:
                counts[ds][strategy] += sum(len(set(grid)) for grid in data['perturbation_values'].values())
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, help='Explicit read-only equal-budget summary; default: summary.json in output root')
    args = parser.parse_args()
    source = Path(os.environ.get('FAST30_SOURCE_ROOT', '/root/autodl-tmp/fast30_final')).resolve()
    output = required_output('RQ5_OUTPUT_ROOT', forbidden=(source / 'results',))
    output.mkdir(parents=True, exist_ok=True)
    cases = Path(os.environ.get('RQ5_CASES_ROOT', source / 'results/sensitivity_five_20261028/cases')).resolve()
    counts = point_totals(cases)
    summary = load(args.summary or output / 'summary.json')
    if not summary['complete'] or summary['cases_by_dataset'] != dict.fromkeys(DATASETS, 30):
        raise ValueError('The complete 150-case equal-budget summary is required for the paper panel')
    stats = summary['by_dataset']
    combined = {ds: dict(native_total_points=counts[ds], n=30,
                        equal_budget_mean_error=stats.get(ds, {}).get('mean_error'))
                for ds in DATASETS}
    (output / 'fig3e_data.json').write_text(json.dumps(combined, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt

    fig, axis = plt.subplots(figsize=(7.2, 2.65), layout='constrained')
    errors = axis.twinx()
    for i, ds in enumerate(DATASETS):
        xs = (i - .19, i + .19)
        styles = ('default', 'uniform')
        axis.bar(xs, [counts[ds][k] / 30 for k in styles], .23, color=['#99b7d4', '#e0c6b7'])
        if ds in stats:
            errors.plot(xs, [stats[ds]['mean_error'][k] for k in styles], 'o-',
                        color='#214d7b', linewidth=1, markersize=3)
        axis.text(i, -.16, ds, transform=axis.get_xaxis_transform(), ha='center', fontsize=8)
    axis.set_xticks([i + off for i in range(len(DATASETS)) for off in (-.19, .19)],
                    ['Default', 'Uniform'] * len(DATASETS), fontsize=7)
    axis.set_ylabel('Total perturbation points per run')
    errors.set_ylabel('Max. JS error')
    errors.set_yscale('log')
    axis.set_ylim(bottom=0)
    fig.savefig(output / 'rq5_continuous_policy_default_vs_uniform.png', dpi=600)


if __name__ == '__main__':
    main()
