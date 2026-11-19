"""Run a frozen ablation worker in an isolated output tree."""
import argparse
import os
from pathlib import Path
import re
import runpy
import sys
from tempfile import NamedTemporaryFile

import ablation_common as common

CODE = Path(__file__).resolve().parent
PACKAGE = CODE.parents[1]
sys.path.insert(0, str(PACKAGE / 'scripts'))
from isolated_worker import finish_worker, required_output
from fresh_case import evaluate_fresh, install_manifest

CPU_STAGES = ('shallow', 'deep', 'deepcheck')
GPU_STAGES = ('generic', 'recover', 'policy', 'evaluate')


def prepare_manifest(source, dataset, seed):
    archived = source / 'results/ablation_five_20261028/cases' / dataset / seed / 'manifest.json'
    if not archived.is_file():
        raise SystemExit(f'Missing original per-case ablation manifest: {archived}')
    expected = archived.read_bytes()
    destination = common.case(dataset, seed) / 'manifest.json'
    if destination.is_file():
        if destination.read_bytes() != expected:
            raise SystemExit(f'Ablation manifest does not match the original: {destination}')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        stream.write(expected)
    Path(stream.name).replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare',) + CPU_STAGES + GPU_STAGES)
    parser.add_argument('dataset', choices=common.DS)
    parser.add_argument('seed', choices=tuple(f'S{i:02}' for i in range(1, 31)), metavar='SEED')
    parser.add_argument('variant', nargs='?')
    args = parser.parse_args()
    requires_variant = args.stage not in ('prepare', 'shallow', 'generic')
    if requires_variant != (args.variant is not None):
        parser.error(f'{args.stage} {"requires" if requires_variant else "does not take"} a variant')
    if args.variant and not re.fullmatch('[a-z][a-z0-9_]*', args.variant):
        parser.error('variant must use lowercase letters, digits and underscores')

    source = Path(os.environ.get('FAST30_SOURCE_ROOT', common.ROOT)).resolve()
    fresh = os.environ.get('FAST30_FRESH_CASES_ROOT')
    fresh_root = Path(fresh).expanduser().resolve() if fresh else None
    output = required_output('RQ4_OUTPUT_ROOT', forbidden=(
        (source / 'results', fresh_root) if fresh_root else (source / 'results',)
    ))
    runtime = PACKAGE / '06_ablation/frozen_runtime'
    if not (runtime / 'p2_parallel_pc.py').is_file():
        raise SystemExit('Restore the frozen runtime with python scripts/restore_archives.py.')
    common.ROOT, common.OUT, common.CODE, common.BASE = source, output, CODE, runtime
    case = common.case(args.dataset, args.seed)
    if fresh_root:
        install_manifest(source, fresh_root, case, args.dataset, args.seed)
    else:
        prepare_manifest(source, args.dataset, args.seed)
    if args.stage == 'prepare':
        print(f'PREPARED {case / "manifest.json"}')
        return
    if fresh_root and args.stage == 'evaluate':
        dest = common.evaluation_path(args.dataset, args.seed, args.variant)
        if not dest.is_file():
            scratch = case / 'scratch'
            scratch.mkdir(exist_ok=True)
            os.chdir(scratch)
            evaluate_fresh(common, args.dataset, args.seed, args.variant, 'ablation-five-frozen-20261028')
        print('DONE evaluate', args.dataset, args.seed, args.variant)
        finish_worker(0)
    worker = CODE / ('ablation_cpu.py' if args.stage in CPU_STAGES else 'ablation_gpu.py')
    sys.argv = [str(worker), args.stage, args.dataset, args.seed]
    if args.variant:
        sys.argv.append(args.variant)
    runpy.run_path(str(worker), run_name='__main__')


if __name__ == '__main__':
    main()
