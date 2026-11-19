#!/usr/bin/env python3
"""Run one Figure 3(c) significance-level case with the frozen sensitivity workers."""
import argparse
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import NamedTemporaryFile
import traceback


ENTRY = Path(__file__).resolve()
PACKAGE = ENTRY.parents[3]
SENSITIVITY_CODE = PACKAGE / 'historical/fast30_final/code_sensitivity_five_20261028'
ABLATION_CODE = PACKAGE / 'experiments/ablation'
FROZEN_RUNTIME = PACKAGE / '06_ablation/frozen_runtime'
DATASETS = ('BankFull', 'OnlineShoppers', 'StudentsAcademic', 'Optdigits', 'SoftwareQuality')
PHASES = ('shallow', 'p2', 'recover', 'evaluate')


def digest(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def copy_archive_file(source, destination):
    payload = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != payload:
            raise SystemExit(f'Conflicting input in the output directory: {destination}')
        return
    with NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
        stream.write(payload)
    Path(stream.name).replace(destination)


def prepare_case(archive, case, dataset, seed):
    path = archive / 'manifest.json'
    if not path.is_file():
        raise SystemExit(f'Missing sensitivity case manifest: {path}')
    manifest = json.loads(path.read_text())
    if (manifest['dataset'], manifest['seed_id']) != (dataset, seed):
        raise SystemExit(f'Case manifest key does not match {dataset}/{seed}: {path}')
    for field, sha_field in (('data', 'data_sha256'), ('p1', 'p1_sha256'), ('split_file', 'split_hash')):
        input_path = Path(manifest[field])
        if not input_path.is_file():
            raise SystemExit(f'Missing archived {field} input: {input_path}')
        if digest(input_path) != manifest[sha_field]:
            raise SystemExit(f'Archived {field} SHA-256 does not match the case manifest: {input_path}')
    p1 = json.loads(Path(manifest['p1']).read_text())
    if not Path(p1['selector_input']).is_file():
        raise SystemExit(f"Missing archived selector development CSV: {p1['selector_input']}")
    copy_archive_file(path, case / 'manifest.json')


def run_phase(phase, dataset, seed, alpha, source, output, archive, case, fresh):
    # sens_common sets release-independent defaults at import. Apply the scope
    # before loading the workers that import those defaults by value.
    sys.path[:0] = [str(SENSITIVITY_CODE), str(ABLATION_CODE)]
    ablation = importlib.import_module('ablation_common')
    sensitivity = importlib.import_module('sens_common')
    sys.path.remove(str(sensitivity.ABL))
    ablation.ROOT = sensitivity.ROOT = source
    ablation.OUT = sensitivity.OUT = output
    ablation.BASE = sensitivity.BASE = FROZEN_RUNTIME
    ablation.CODE = sensitivity.CODE = SENSITIVITY_CODE
    sensitivity.ABL = ABLATION_CODE
    sensitivity.OLDOUT = source / 'results/ablation_five_20261028'
    worker = importlib.import_module('sens_worker')
    key = sensitivity.cfgkey('alpha', alpha)
    scratch = case / 'scratch' / key / phase
    scratch.mkdir(parents=True, exist_ok=True)
    os.chdir(scratch)

    if phase == 'shallow':
        worker.shallow(dataset, seed, alpha)
    elif phase == 'p2' and key == 'default':
        baseline = case / 'p2/default.json'
        reference = archive / 'p2/default.json'
        if fresh:
            # The probe uses only ordered P2 features and deep-stage counts.
            control = ablation.main_record(dataset, seed)
            reference = scratch / 'rq1_default_reference.json'
            ablation.save(reference, dict(p2_features=control['selection']['p2_features'],
                                          deep=control['pc']['deep']))
        if not reference.is_file():
            raise SystemExit(f'Missing default P2 comparator: {reference}')
        original_artifact = worker.artifact

        def probe_artifact(ds, sid, stage, variant):
            if (ds, sid, stage, variant) == (dataset, seed, 'p2', 'default'):
                return reference
            return original_artifact(ds, sid, stage, variant)

        worker.artifact = probe_artifact
        worker.p2(dataset, seed, key, force=True)
        (case / 'p2/default_probe.json').replace(baseline)
    elif phase == 'evaluate' and fresh:
        from fresh_case import evaluate_fresh
        evaluate_fresh(ablation, dataset, seed, key, sensitivity.PROTOCOL)
    else:
        getattr(worker, phase)(dataset, seed, key)
    print('FIG3C_STAGE_COMPLETE', dataset, seed, alpha, phase, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=DATASETS)
    parser.add_argument('seed', choices=tuple(f'S{n:02d}' for n in range(1, 31)))
    parser.add_argument('--alpha', type=float, choices=(.005, .01, .05), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true', help='Write the case manifest without computing P2 or CNN')
    parser.add_argument('--phase', choices=PHASES, help=argparse.SUPPRESS)
    args = parser.parse_args()

    source = Path(os.environ.get('FAST30_SOURCE_ROOT', '/root/autodl-tmp/fast30_final')).resolve()
    fresh = os.environ.get('FAST30_FRESH_CASES_ROOT')
    fresh_root = Path(fresh).expanduser().resolve() if fresh else None
    output = args.output.expanduser().resolve()
    for protected in (source, PACKAGE, *((fresh_root,) if fresh_root else ())):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            parser.error(f'--output must not overlap {protected}')
    if not (FROZEN_RUNTIME / 'p2_parallel_pc.py').is_file():
        parser.error('Restore the frozen ablation runtime with python scripts/restore_archives.py.')

    archive = source / 'results/sensitivity_five_20261028/cases' / args.dataset / args.seed
    case = output / 'cases' / args.dataset / args.seed
    if not case.resolve().is_relative_to(output):
        parser.error('Case output resolves outside --output.')
    if args.phase:
        status = 0
        try:
            run_phase(args.phase, args.dataset, args.seed, args.alpha, source, output, archive, case,
                      fresh=bool(fresh_root))
        except Exception:
            traceback.print_exc()
            status = 1
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(status)

    case.mkdir(parents=True, exist_ok=True)
    with (case / '.rq5-alpha.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if fresh_root:
            sys.path.insert(0, str(ABLATION_CODE))
            from fresh_case import install_manifest
            install_manifest(source, fresh_root, case, args.dataset, args.seed)
        else:
            prepare_case(archive, case, args.dataset, args.seed)
        if args.prepare_only:
            print(f'PREPARED {case / "manifest.json"}')
            return
        for phase in PHASES:
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            if phase in ('shallow', 'p2'):
                environment['CUDA_VISIBLE_DEVICES'] = '-1'
            result = subprocess.run([sys.executable, str(ENTRY), *sys.argv[1:], '--phase', phase], env=environment)
            if result.returncode:
                raise SystemExit(f'Figure 3(c) phase {phase} failed with status {result.returncode}.')
    print('FIG3C_CASE_COMPLETE', args.dataset, args.seed, args.alpha, flush=True)


if __name__ == '__main__':
    main()
