#!/usr/bin/env python3
"""Regenerate one RQ6 Table 9 case from a local RQ1 public-data run."""

import argparse
import csv
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys

import common


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[2]
FROZEN = PACKAGE / '06_ablation/frozen_runtime'
ABLATION = PACKAGE / 'experiments/ablation'
SENSITIVITY = PACKAGE / 'historical/fast30_final/code_sensitivity_five_20261028'
REGISTRY = PACKAGE / 'provenance/rq1-inputs.csv'
ARCHIVE = Path('/root/autodl-tmp/fast30_final')
REGENERATED = common.PROTOCOL + '-regenerated'
SEEDS = tuple(f'S{i:02d}' for i in range(1, 31))
MODULES = (
    'model_train_v6', 'feature_perturbation', 'p3_recovery',
    'fixed_split_evaluator', 'shared_globals', 'p2_parallel_pc', 'stage2_runtime',
)

sys.path.insert(0, str(PACKAGE / 'scripts'))
from isolated_worker import finish_worker


def source_paths(args):
    root = args.run_root
    core = root / 'core' / args.dataset / args.seed
    scaled = core if args.dataset == 'Optdigits' else (
        root / 'scaled' / args.dataset / args.seed / 'cases' / args.dataset / args.seed
    )
    with REGISTRY.open(encoding='utf-8-sig', newline='') as stream:
        row, = (row for row in csv.DictReader(stream)
                if (row['dataset'], row['seed_id']) == (args.dataset, args.seed))
    data = root / 'inputs' / row['data_path']
    split = root / 'inputs' / row['split_path']
    p1 = common.load(core / 'p1_cache.json')
    if Path(p1['split_file']).resolve() != split.resolve():
        raise ValueError(f'P1 must use the generated split under {root / "inputs"}')
    for file, field, recorded in (
        (data, 'data_sha256', p1['dataset_sha256']), (split, 'split_sha256', p1['split_hash']),
    ):
        if common.sha(file) != row[field] or recorded != row[field]:
            raise ValueError(f'Data or split does not match the public case registry: {file}')
    return core, scaled, data


def verify_runtime():
    lock = common.lock()
    if sys.version.split()[0] != lock['python_version']:
        raise RuntimeError('Python version differs from the Table 9 protocol lock')
    for package, expected in lock['packages'].items():
        if metadata.version(package) != expected:
            raise RuntimeError(f'Package version differs from the Table 9 lock: {package}')
    pairs = {
        HERE / name: ARCHIVE / 'code_unscaled_scoring_cnn_eval_20261110' / name
        for name in ('common.py', 'worker.py')
    }
    pairs.update({FROZEN / (name + '.py'): common.BASE / (name + '.py') for name in MODULES})
    pairs[HERE / 'historical/frozen_evaluator.py'] = (
        ARCHIVE / 'code_sensitivity_v6_20261008/run_scaled_evaluator_full_vs_selected.py'
    )
    for name in ('ablation_common.py', 'ablation_batched_pc.py', 'ablation_batched_pc_deep.py'):
        pairs[ABLATION / name] = common.ABL / name
    for name in ('sens_common.py', 'sens_worker.py', 'sens_restricted_kernel.py'):
        pairs[SENSITIVITY / name] = common.SENS / name
    for release, original in pairs.items():
        if not release.is_file() or common.sha(release) != lock['code_hashes'][str(original)]:
            raise RuntimeError(f'Frozen source mismatch at {release}. Verify the ZIP and restore its sources.')


def run_phase(args, phase, scorer):
    common.OUT = args.output
    common.STUDY = args.output / 'scorer_study'
    common.PROTOCOL = REGENERATED
    common.manifest = lambda ds, sid: common.load(common.case(ds, sid) / 'manifest.json')
    common.save = lambda path, value: common.atomic(path, value, root=args.output)
    sys.path[:0] = [str(path) for path in (FROZEN, ABLATION, SENSITIVITY)]
    import worker

    if phase in ('prepare', 'parity', 'p2'):
        # The frozen sensitivity import prepends its original AutoDL code path.
        # Bind its ablation dependencies to the verified, bundled copies first.
        import ablation_common
        import ablation_batched_pc
        import ablation_batched_pc_deep
        import sens_restricted_kernel

        sys.path[:] = [path for path in sys.path if not Path(path).is_relative_to(ARCHIVE)]
        for module, directory in (
            (ablation_common, ABLATION), (ablation_batched_pc, ABLATION),
            (ablation_batched_pc_deep, ABLATION), (sens_restricted_kernel, SENSITIVITY),
        ):
            if Path(module.__file__).resolve().parent != directory:
                raise RuntimeError(f'Unexpected frozen dependency: {module.__file__}')

    worker.ROOT = args.output / 'source_view'
    worker.SENS = SENSITIVITY

    def verify_loaded():
        for name in MODULES:
            if name in sys.modules and Path(sys.modules[name].__file__).resolve() != FROZEN / (name + '.py'):
                raise RuntimeError(f'Study phase imported an unexpected {name} module')

    worker.verify_runtime_modules = verify_loaded
    if phase == 'prepare':
        from fresh_prepare import prepare_case

        core, scaled, data = source_paths(args)
        study = common.STUDY / 'cases' / args.dataset / args.seed
        prepare_case(worker, args.dataset, args.seed, core, scaled, study,
                     common.case(args.dataset, args.seed), data, tuple(args.scorers))
    elif phase == 'parity':
        worker.parity(args.dataset, args.seed)
    elif phase == 'evaluate':
        from fresh_evaluation import evaluate

        evaluate(worker, args.dataset, args.seed, scorer)
    else:
        getattr(worker, phase)(args.dataset, args.seed, scorer)
    verify_loaded()
    print('COMPLETE', args.dataset, args.seed, phase, scorer, flush=True)


def run_case(args):
    core, _, _ = source_paths(args)
    verify_runtime()
    case = args.output / 'cases' / args.dataset / args.seed
    case.mkdir(parents=True, exist_ok=False)
    study = args.output / 'scorer_study' / 'cases' / args.dataset / args.seed
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    rebuild = HERE / 'scorer_sources/rebuild.py'
    subprocess.run(
        [sys.executable, str(rebuild), args.dataset, args.seed,
         '--core-case', str(core), '--study-case', str(study), '--scorers', 'CNN', *args.scorers],
        env=env, check=True,
    )
    phases = [('prepare', 'CNN'), ('parity', 'CNN'), ('evaluate', 'FULL'), ('evaluate', 'CNN')]
    for scorer in args.scorers:
        phases.extend((phase, scorer) for phase in ('p2', 'recover', 'evaluate'))
    for phase, scorer in phases:
        work = args.output / 'work' / args.dataset / args.seed / scorer / phase
        work.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), args.dataset, args.seed,
             '--run-root', str(args.run_root), '--output', str(args.output),
             '--scorers', *args.scorers, '--phase', phase, scorer],
            cwd=work, env=env, check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=common.DS)
    parser.add_argument('seed', choices=SEEDS)
    parser.add_argument('--run-root', type=Path, required=True, help='Workspace of the public RQ1 run')
    parser.add_argument('--output', type=Path, required=True, help='Workspace for the regenerated Table 9 cases')
    parser.add_argument('--scorers', nargs='+', choices=('MLP', 'RF'), default=('MLP', 'RF'),
                        help='Alternative scorers to regenerate (default: both)')
    parser.add_argument('--phase', nargs=2, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if len(set(args.scorers)) != len(args.scorers):
        parser.error('--scorers needs distinct scorer names')
    args.run_root = args.run_root.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    protected = (PACKAGE, ARCHIVE, *(args.run_root / part for part in ('core', 'inputs', 'scaled')))
    if any(path.is_relative_to(args.output) or args.output.is_relative_to(path) for path in protected):
        parser.error('--output must be separate from the package, historical root, and RQ1 input and output paths')
    if args.phase:
        phase, scorer = args.phase
        if phase not in ('prepare', 'parity', 'p2', 'recover', 'evaluate') or scorer not in (
            'FULL', 'CNN', *args.scorers
        ):
            parser.error('invalid study phase')
        run_phase(args, phase, scorer)
        finish_worker(0)
    try:
        run_case(args)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'{args.dataset}/{args.seed}: {exc}\n')


if __name__ == '__main__':
    main()
