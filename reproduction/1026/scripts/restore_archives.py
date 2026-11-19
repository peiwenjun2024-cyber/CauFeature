#!/usr/bin/env python3
"""Restore the byte-exact runtime snapshots beside the release entrypoints."""
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    'ablation_frozen_runtime.zip': '06_ablation/frozen_runtime',
    'optdigits_runtime_snapshot.zip': '02_runtime_variants/Optdigits',
}


def restore(name, destination):
    allowed = (ROOT / destination).resolve()
    if not allowed.is_relative_to(ROOT):
        raise ValueError(f'Archive destination escapes the release directory: {allowed}')
    with ZipFile(ROOT / 'source_archives' / name) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            target = (ROOT / member.filename).resolve()
            if not target.is_relative_to(allowed):
                raise ValueError(f'Unexpected archive member: {member.filename}')
            data = archive.read(member)
            if target.exists():
                if target.read_bytes() != data:
                    raise ValueError(f'Restored file differs from the snapshot: {target}')
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as output:
                output.write(data)
    print(f'Restored {destination} from {name}')


if __name__ == '__main__':
    for name, destination in ARCHIVES.items():
        restore(name, destination)
