#!/usr/bin/env python3
"""Build the project CSV for a supported public dataset release."""
import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import io
from pathlib import Path
from zipfile import BadZipFile, ZipFile, is_zipfile


ROOT = Path(__file__).resolve().parents[1]
MEMBERS = {
    'BankFull': 'bank-full.csv',
    'BrainMethod': 'BM.csv',
    'CustomerChurn': 'churn.csv',
    'DiabetesHealth_sample10': 'diabetes_binary_health_indicators_BRFSS2015.csv',
    'DriedBean': 'DryBeanDataset/Dry_Bean_Dataset.arff',
    'HeartDisease': 'heart.csv',
    'MagicGamma': 'magic04.data',
    'MentalStress': 'StressLevelDataset.csv',
    'OnlineShoppers': 'online_shoppers_intention.csv',
    'SmokeAlarm': 'smoke_detection_iot.csv',
    'SoftwareQuality': 'TUANDROMD.csv',
    'StudentsAcademic': 'data.csv',
    'ThyroidDiff': 'Thyroid_Diff.csv',
}
MAGIC_HEADER = b'fLength,fWidth,fSize,fConc,fConc1,fAsym,fM3Long,fM3Trans,fAlpha,fDist,class\r\n'
ADULT_HEADER = (
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
    'hours-per-week', 'native-country', 'income',
)
PROJECT_ID_COLUMNS = {'BrainMethod': 'NAME', 'CustomerChurn': 'ID'}


def read_public(source, dataset):
    if not is_zipfile(source):
        return source.read_bytes()
    with ZipFile(source) as archive:
        if dataset == 'BankFull' and 'bank.zip' in archive.namelist():
            with ZipFile(io.BytesIO(archive.read('bank.zip'))) as inner:
                return inner.read(MEMBERS[dataset])
        if dataset == 'SmokeAlarm' and 'final.csv' in archive.namelist():
            return archive.read('final.csv')
        if dataset == 'DiabetesHealth_sample10' and 'data.csv' in archive.namelist():
            return archive.read('data.csv')
        return archive.read(MEMBERS[dataset])


def bean_value(value):
    number = Decimal(value)
    places = max(0, 10 - len(str(int(number))))
    formatted = format(number.quantize(Decimal('0.1') ** places, rounding=ROUND_HALF_UP), 'f')
    return formatted.rstrip('0').rstrip('.') if '.' in formatted else formatted


def dried_bean(data):
    columns, rows = [], []
    reading_data = False
    for line in data.decode('utf-8-sig').splitlines():
        line = line.strip()
        if not line or line.startswith('%'):
            continue
        if line.upper() == '@DATA':
            reading_data = True
        elif line.upper().startswith('@ATTRIBUTE '):
            columns.append(line.split()[1])
        elif reading_data:
            fields = line.split(',')
            rows.append(','.join([*(bean_value(value) for value in fields[:-1]), fields[-1]]))
    return ('\r\n'.join([','.join(columns), *rows]) + '\r\n').encode('utf-8-sig')


def adult_income(source):
    target = io.StringIO(newline='')
    writer = csv.writer(target, lineterminator='\r\n')
    writer.writerow(ADULT_HEADER)
    with ZipFile(source) as archive:
        for member, skip in (('adult.data', 3), ('adult.test', 1)):
            raw = archive.read(member).decode('ascii')
            clean = []
            for fields in csv.reader(io.StringIO(raw, newline='')):
                fields = [field.strip() for field in fields]
                if len(fields) == len(ADULT_HEADER) and '?' not in fields:
                    fields[-1] = fields[-1].removesuffix('.')
                    clean.append(fields)
            writer.writerows(clean[skip:])
    return target.getvalue().encode('ascii')


def diabetes_sample(data):
    import numpy as np

    rows = csv.reader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    header = next(rows)
    columns = [name for name in header if name not in ('ID', 'Diabetes_binary')]
    columns.append('Diabetes_binary')
    positions = [header.index(name) for name in columns]
    rows = list(rows)
    order = np.random.RandomState(20260924).permutation(len(rows))
    sample = order[:round(.1 * len(rows))]

    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow(columns)
    writer.writerows([rows[index][i].removesuffix('.0') for i in positions] for index in sample)
    return output.getvalue().encode('ascii')


def normalize(dataset, data):
    if dataset == 'DiabetesHealth_sample10':
        return diabetes_sample(data)
    if dataset == 'DriedBean':
        return dried_bean(data)
    if dataset == 'MagicGamma':
        return MAGIC_HEADER + b'\r\n'.join(data.splitlines()) + b'\r\n'
    if dataset not in (*PROJECT_ID_COLUMNS, 'BankFull', 'MentalStress', 'SmokeAlarm', 'SoftwareQuality'):
        return data
    source = io.StringIO(data.decode('utf-8-sig'), newline='')
    target = io.StringIO(newline='')
    writer = csv.writer(target, lineterminator='\r\n')
    if dataset in PROJECT_ID_COLUMNS:
        rows = csv.reader(source)
        header = next(rows)
        omitted = header.index(PROJECT_ID_COLUMNS[dataset])
        keep = [index for index in range(len(header)) if index != omitted]
        writer.writerow(header[index] for index in keep)
        for row in rows:
            writer.writerow(row[index] for index in keep)
    elif dataset == 'BankFull':
        writer.writerows(csv.reader(source, delimiter=';'))
    elif dataset == 'MentalStress':
        rows = csv.reader(source)
        header = next(rows)
        label = header.index('stress_level')
        writer.writerow(header)
        for row in rows:
            if row[label] == '2':
                row[label] = '1'
            writer.writerow(row)
    elif dataset == 'SmokeAlarm':
        rows = csv.reader(source)
        header = next(rows)
        columns = [name for name in header if name not in ('', 'UTC', 'Fire Alarm')] + ['Fire Alarm']
        keep = [header.index(name) for name in columns]
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row[index].removesuffix('.0') for index in keep)
    else:
        rows = csv.reader(source)
        # Preserve the project's historical TUANDROMD column spellings.
        writer.writerow(name.replace('goodware', '0') for name in next(rows))
        target_label = {'goodware': '0', 'malware': '1'}
        for row in rows:
            if any(row):
                writer.writerow([*row[:-1], target_label[row[-1]]])
    return target.getvalue().encode('utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=sorted((*MEMBERS, 'Income')))
    parser.add_argument('source', type=Path, help='Public ZIP or its listed dataset file')
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    registry = ROOT / 'provenance/rq1-inputs.csv'
    with registry.open(encoding='utf-8-sig', newline='') as file:
        entry = next(row for row in csv.DictReader(file) if row['dataset'] == args.dataset)
    destination = args.data_root / entry['data_path']
    try:
        if args.dataset == 'Income':
            data = adult_income(args.source)
        else:
            data = normalize(args.dataset, read_public(args.source, args.dataset))
        actual = hashlib.sha256(data).hexdigest()
        if actual != entry['data_sha256']:
            parser.error(f'Generated {args.dataset} SHA-256 {actual}; expected {entry["data_sha256"]}')
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as file:
            file.write(data)
    except (OSError, KeyError, ValueError, BadZipFile) as error:
        parser.error(str(error))
    print(destination)


if __name__ == '__main__':
    main()
