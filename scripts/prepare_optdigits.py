#!/usr/bin/env python3
"""Recreate the archived project CSV from the original UCI Optdigits files."""
import argparse
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd


SOURCES = {
    'optdigits.tra': 'e1b683cc211604fe8fd8c4417e6a69f31380e0c61d4af22e93cc21e9257ffedd',
    'optdigits.tes': '6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8',
}
EXPECTED_CSV = 'de50f12059166079f5e50acc226c189732fc39842dd46c5617ad62034d1ff140'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('uci_zip', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()

    pieces = []
    with ZipFile(args.uci_zip) as archive:
        for file_name, expected_hash in SOURCES.items():
            content = archive.read(file_name)
            if hashlib.sha256(content).hexdigest() != expected_hash:
                raise ValueError(f'Unexpected SHA-256 for {file_name}')
            pieces.append(np.loadtxt(io.BytesIO(content), delimiter=',', dtype=int))

    data = np.vstack(pieces)
    frame = pd.DataFrame(data[:, :-1].astype(float), columns=[f'F{i:03d}' for i in range(1, 65)])
    frame['target'] = data[:, -1]
    content = frame.to_csv(index=False, lineterminator='\n').encode('utf-8')
    if hashlib.sha256(content).hexdigest() != EXPECTED_CSV:
        raise ValueError('Recreated Optdigits.csv differs from the historical data file')
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with args.destination.open('xb') as output:
        output.write(content)
    print(args.destination)


if __name__ == '__main__':
    main()
