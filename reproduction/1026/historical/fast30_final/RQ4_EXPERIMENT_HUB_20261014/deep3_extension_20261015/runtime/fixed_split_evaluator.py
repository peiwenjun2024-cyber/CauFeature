"""CNN evaluation on externally fixed train/validation/test row indices.

Preprocessing is fit on the training partition only. Validation/test rows never
participate in category-code discovery, numeric imputation, or target encoding.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import shared_globals
from model_train_v6 import build_model, read_tabular_data, smote_enn_resample, train_model


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _one_hot(values: np.ndarray, categories: int) -> np.ndarray:
    encoder = OneHotEncoder(
        categories=[np.arange(categories)],
        sparse_output=False,
        handle_unknown="ignore",
    )
    return encoder.fit_transform(np.asarray(values).reshape(-1, 1))


def _json_scalar(value):
    return value.item() if hasattr(value, "item") else value


def _prepare_fixed_splits(
    data_path: Path,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    selected_features: list[str] | None,
):
    """Split raw rows first, then fit all preprocessing on train only."""
    frame = read_tabular_data(data_path).copy()
    target = frame.columns[-1]
    if selected_features is None:
        selected_features = list(frame.columns[:-1])
    selected_features = list(selected_features)
    missing = sorted(set(selected_features) - set(frame.columns))
    if missing:
        raise KeyError(f"Selected features absent from {data_path.name}: {missing}")

    x_raw = frame[selected_features].copy()
    y_raw = frame[target].copy()
    if y_raw.isna().any():
        raise ValueError(f"Target {target} contains missing values")

    x_train = x_raw.iloc[train_indices].reset_index(drop=True).copy()
    x_validation = x_raw.iloc[validation_indices].reset_index(drop=True).copy()
    x_test = x_raw.iloc[test_indices].reset_index(drop=True).copy()

    unknown_validation = {}
    unknown_test = {}
    for column in selected_features:
        train_col = x_train[column]

        if pd.api.types.is_bool_dtype(train_col):
            x_train[column] = train_col.astype(np.int8)
            x_validation[column] = x_validation[column].astype(np.int8)
            x_test[column] = x_test[column].astype(np.int8)
            continue

        if pd.api.types.is_numeric_dtype(train_col):
            x_train[column] = pd.to_numeric(x_train[column], errors="coerce")
            x_validation[column] = pd.to_numeric(x_validation[column], errors="coerce")
            x_test[column] = pd.to_numeric(x_test[column], errors="coerce")
            median = x_train[column].median()
            if not np.isfinite(median):
                median = 0.0
            x_train[column] = x_train[column].fillna(median)
            x_validation[column] = x_validation[column].fillna(median)
            x_test[column] = x_test[column].fillna(median)
            continue

        # Nominal feature: fit codes on train only. Held-out unseen categories
        # receive -1 and are never used to alter the training mapping.
        train_s = x_train[column].astype("string").fillna("__MISSING__")
        val_s = x_validation[column].astype("string").fillna("__MISSING__")
        test_s = x_test[column].astype("string").fillna("__MISSING__")
        encoder = LabelEncoder()
        encoder.fit(train_s)
        mapping = {str(value): int(i) for i, value in enumerate(encoder.classes_)}

        def transform(series):
            return series.map(mapping).fillna(-1).astype(np.int64)

        x_train[column] = transform(train_s)
        x_validation[column] = transform(val_s)
        x_test[column] = transform(test_s)
        unknown_validation[column] = int((~val_s.isin(mapping)).sum())
        unknown_test[column] = int((~test_s.isin(mapping)).sum())

    # Targets are categorical even when stored as numeric labels (e.g. {0,1,3}).
    # Fit on train only and map to contiguous 0..K-1.
    target_encoder = LabelEncoder()
    y_train_raw = y_raw.iloc[train_indices].reset_index(drop=True)
    y_validation_raw = y_raw.iloc[validation_indices].reset_index(drop=True)
    y_test_raw = y_raw.iloc[test_indices].reset_index(drop=True)
    target_encoder.fit(y_train_raw)
    try:
        y_train = target_encoder.transform(y_train_raw)
        y_validation = target_encoder.transform(y_validation_raw)
        y_test = target_encoder.transform(y_test_raw)
    except ValueError as exc:
        raise RuntimeError(
            "Validation/test contains a target class absent from the training partition"
        ) from exc

    metadata = {
        "preprocessing_scope": "train_only",
        "target_classes_original": [_json_scalar(v) for v in target_encoder.classes_],
        "target_class_count": int(len(target_encoder.classes_)),
        "unknown_validation_categories": {
            k: v for k, v in unknown_validation.items() if v
        },
        "unknown_test_categories": {k: v for k, v in unknown_test.items() if v},
    }
    return (
        x_train,
        x_validation,
        x_test,
        np.asarray(y_train),
        np.asarray(y_validation),
        np.asarray(y_test),
        metadata,
    )


def evaluate_fixed_split(
    data_path: Path,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
    selected_features: list[str] | None = None,
) -> dict:
    """Train the CNN on fixed train rows and score only the fixed test rows."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)

    (
        x_train_raw,
        x_validation,
        x_test,
        y_train_raw,
        y_validation,
        y_test,
        preprocessing,
    ) = _prepare_fixed_splits(
        data_path,
        train_indices,
        validation_indices,
        test_indices,
        selected_features,
    )

    try:
        x_train, y_train = smote_enn_resample(x_train_raw, y_train_raw, seed)
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)
    except ValueError:
        x_train = x_train_raw.to_numpy()
        y_train = y_train_raw

    classes = int(preprocessing["target_class_count"])
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_validation = x_validation.to_numpy().reshape(
        x_validation.shape[0], x_validation.shape[1], 1
    )
    x_test = x_test.to_numpy().reshape(x_test.shape[0], x_test.shape[1], 1)
    y_train_encoded = _one_hot(y_train, classes)
    y_validation_encoded = _one_hot(y_validation, classes)

    shared_globals.init()
    shared_globals.random_seed = seed
    shared_globals.sample_num = x_train.shape[0]
    started = time.perf_counter()
    model = build_model(
        input_shape=x_train.shape[1:],
        num_classes=classes,
        sample_num=x_train.shape[0],
    )
    model, _ = train_model(
        model,
        x_train,
        y_train_encoded,
        x_validation,
        y_validation_encoded,
        x_train.shape[0],
    )
    train_seconds = time.perf_counter() - started

    started = time.perf_counter()
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    test_seconds = time.perf_counter() - started
    result = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "train_seconds": float(train_seconds),
        "test_seconds": float(test_seconds),
        "selected_dimension": int(x_train.shape[1]),
        "evaluator_model": "CNN",
        "model_architecture_hash": _hash_text(model.to_json()),
        "preprocessing": preprocessing,
    }
    tf.keras.backend.clear_session()
    return result
