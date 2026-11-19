"""Internal isolated worker for the frozen Optdigits and scaled CNN stages."""

import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


RUNTIME = Path(__file__).resolve().parents[3] / '02_runtime_variants' / 'Optdigits'


def run_stage(stage, manifest_path, data, case, manifest):
    module = importlib.import_module(stage + '_runtime')
    if stage in ('stage1', 'stage3'):
        module.MAN = manifest_path
        module.DATASET_DIR = data.parent
    if stage == 'stage3':
        module.BASELINE_CACHE = case / 'shared_full_baseline_cache_profilefix'
    sys.argv = [module.__file__, '--dataset', data.name, '--split-seed-id', manifest['split_seed_id']]
    if stage == 'stage2':
        sys.argv += ['--workers', '8']
    module.main()


def run_scaled(data, case, manifest):
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    from fixed_split_evaluator import _one_hot, _prepare_fixed_splits
    from model_train_v6 import build_model, smote_enn_resample, train_model
    from runtime_provenance import atomic_json
    import shared_globals

    record = json.loads((case / 'record.json').read_text(encoding='utf-8'))
    if record['split_hash'] != manifest['split_hash']:
        raise ValueError('The Optdigits selection uses a different split.')

    split = pd.read_csv(manifest['split_file'])
    indices = [split.loc[split.partition == p, 'row_index'].to_numpy(int) for p in ('train', 'validation', 'test')]
    seed = manifest['split_seed']
    for mode, features in [('full', None), ('selected', record['selection']['final_features'])]:
        x_train, x_val, x_test, y_train, y_val, y_test, meta = _prepare_fixed_splits(data, *indices, features)
        scaler = StandardScaler()
        fit = scaler.fit_transform(x_train)
        validation = scaler.transform(x_val)
        test = scaler.transform(x_test)
        try:
            samples, labels = smote_enn_resample(pd.DataFrame(fit, columns=x_train.columns), y_train, seed)
        except ValueError:
            samples, labels = fit, y_train
        samples, labels = np.asarray(samples), np.asarray(labels)
        samples = samples.reshape(len(samples), samples.shape[1], 1)
        validation = validation.reshape(len(validation), validation.shape[1], 1)
        test = test.reshape(len(test), test.shape[1], 1)

        classes = int(meta['target_class_count'])
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        shared_globals.init()
        shared_globals.random_seed = seed
        shared_globals.sample_num = len(samples)
        started = time.time()
        model = build_model(samples.shape[1:], classes, len(samples))
        model, history = train_model(
            model, samples, _one_hot(labels, classes), validation, _one_hot(y_val, classes), len(samples),
        )
        predictions = np.argmax(model.predict(test, verbose=0), axis=1)
        result = {
            'dataset': 'Optdigits', 'seed_id': manifest['split_seed_id'], 'mode': mode,
            'dimension': len(x_train.columns), 'features': list(x_train.columns),
            'accuracy': float(accuracy_score(y_test, predictions)),
            'f1_macro': float(f1_score(y_test, predictions, average='macro', zero_division=0)),
            'train_test_seconds': round(time.time() - started, 2),
            'best_epoch': int(np.argmin(history.history['val_loss']) + 1),
            'scaler_fit': 'train_only', 'resampling': 'SMOTEENN train only',
            'split_hash': manifest['split_hash'],
        }
        atomic_json(case / f'{mode}_scaled.json', result)
        tf.keras.backend.clear_session()


def main():
    stage, manifest_path, data, case = sys.argv[1:]
    manifest_path, data, case = map(Path, (manifest_path, data, case))
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    sys.path.insert(0, str(RUNTIME))
    if stage == 'scaled':
        run_scaled(data, case, manifest)
    else:
        run_stage(stage, manifest_path, data, case, manifest)


if __name__ == '__main__':
    status = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
