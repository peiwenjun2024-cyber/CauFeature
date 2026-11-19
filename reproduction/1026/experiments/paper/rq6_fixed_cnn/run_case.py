"""Replay one fixed-split scoring case using the archived campaign implementation."""
import argparse
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PACKAGE / 'scripts'))
from isolated_worker import finish_worker, required_output

MODES = ('prepare', 'p2', 'recover', 'evaluate')
HISTORICAL = 'code_unscaled_scoring_cnn_eval_20261110'


def run_phase(args, source, output):
    import common

    original_code = source / HISTORICAL
    common.OUT = output
    common.save = lambda path, value: common.atomic(path, value, root=output)
    lock = common.lock()
    if sys.version.split()[0] != lock['python_version']:
        raise RuntimeError('Python version differs from the frozen study')
    for name, version in lock['packages'].items():
        if metadata.version(name) != version:
            raise RuntimeError(f'Package version differs from the frozen study: {name}')
    common.validate_case(args.dataset, args.seed)
    frozen = PACKAGE / '06_ablation' / 'frozen_runtime'
    sensitivity = PACKAGE / 'historical/fast30_final/code_sensitivity_five_20261028'
    for name in ('common.py', 'worker.py'):
        local = Path(__file__).resolve().with_name(name)
        if common.sha(local) != lock['code_hashes'][str(original_code / name)]:
            raise RuntimeError(f'Release copy differs from original: {name}')
    roots = (frozen, PACKAGE / 'experiments' / 'ablation', sensitivity)
    required_sources = {
        roots[1] / 'ablation_common.py': common.ABL / 'ablation_common.py',
        roots[1] / 'ablation_batched_pc_deep.py': common.ABL / 'ablation_batched_pc_deep.py',
        sensitivity / 'sens_common.py': common.SENS / 'sens_common.py',
        sensitivity / 'sens_worker.py': common.SENS / 'sens_worker.py',
        sensitivity / 'sens_restricted_kernel.py': common.SENS / 'sens_restricted_kernel.py',
        Path(__file__).with_name('historical') / 'frozen_evaluator.py':
            source / 'code_sensitivity_v6_20261008/run_scaled_evaluator_full_vs_selected.py',
    }
    for copy, original in required_sources.items():
        if common.sha(copy) != lock['code_hashes'][str(original)]:
            raise RuntimeError(f'Restored runtime differs from original: {copy}')
    sys.path[:0] = list(map(str, roots))
    # Preload the bundled adapter so sens_common cannot resolve a different copy.
    import ablation_common
    import ablation_batched_pc_deep
    import worker
    worker.SENS = sensitivity
    # Exact historical worker resolves the evaluator and inputs under one root.
    # A links-only view keeps that convention without exposing historical outputs.
    view = output / 'source_view'
    evaluator_dir = view / 'code_sensitivity_v6_20261008'
    evaluator_dir.mkdir(parents=True, exist_ok=True)
    links = {view / 'results': source / 'results',
             evaluator_dir / 'run_scaled_evaluator_full_vs_selected.py':
                 Path(__file__).resolve().with_name('historical') / 'frozen_evaluator.py'}
    for link, target in links.items():
        if not link.is_symlink() and not link.exists():
            link.symlink_to(target)
        if link.resolve() != target.resolve():
            raise RuntimeError(f'Unexpected source link: {link}')
    worker.ROOT = view

    # These are source inputs. Worker writes are confined to common.OUT.
    def verify_loaded():
        for name in ('model_train_v6', 'feature_perturbation', 'p3_recovery',
                     'fixed_split_evaluator', 'shared_globals', 'p2_parallel_pc'):
            mod = sys.modules.get(name)
            if mod is None:
                continue
            seen = Path(mod.__file__).resolve()
            expected = frozen / (name + '.py')
            original = common.BASE / (name + '.py')
            if seen != expected.resolve() or common.sha(seen) != lock['code_hashes'][str(original)]:
                raise RuntimeError(f'Unexpected runtime module: {seen}')
    worker.verify_runtime_modules = verify_loaded
    work = output / 'work' / args.dataset / args.seed / args.scorer / args.phase
    work.mkdir(parents=True, exist_ok=True)
    os.chdir(work)
    case_args = [args.dataset, args.seed]
    if args.phase != 'prepare':
        case_args.append(args.scorer)
    getattr(worker, args.phase)(*case_args)
    stage = {'p2': 'p2', 'recover': 'selection', 'evaluate': 'evaluation'}
    path = (common.case(args.dataset, args.seed) / 'manifest.json'
            if args.phase == 'prepare'
            else common.artifact(args.dataset, args.seed, stage[args.phase], args.scorer))
    evidence = common.load(path)
    if evidence.get('protocol') != common.PROTOCOL:
        raise RuntimeError(f'Incomplete result: {path}')
    verify_loaded()
    print(f'COMPLETE {args.phase} {path}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('dataset')
    ap.add_argument('seed')
    ap.add_argument('--scorer', choices=('FULL', 'CNN', 'MLP', 'RF'), default='MLP')
    ap.add_argument('--phase', choices=MODES, help=argparse.SUPPRESS)
    args = ap.parse_args()
    source = Path(os.environ.get('FAST30_SOURCE_ROOT', '/root/autodl-tmp/fast30_final')).resolve()
    # The historical protocol lock binds campaign data to absolute paths.
    if source != Path('/root/autodl-tmp/fast30_final'):
        ap.error('the original RQ6 protocol lock is bound to /root/autodl-tmp/fast30_final')
    output = required_output('RQ6_OUTPUT_ROOT', forbidden=(source / 'results',))
    if args.phase:
        run_phase(args, source, output)
        finish_worker(0)
    phases = ('prepare', 'evaluate') if args.scorer in ('FULL', 'CNN') else MODES
    for phase in phases:
        cmd = [sys.executable, str(Path(__file__).resolve()), args.dataset,
               args.seed, '--scorer', args.scorer, '--phase', phase]
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
