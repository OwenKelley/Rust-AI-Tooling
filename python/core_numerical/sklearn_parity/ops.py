"""scikit-learn parity ops (Python reference)."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler

from core_numerical.numpy_parity.rng import seeded_uniform


def _make_x(n: int, d: int, seed: int) -> np.ndarray:
    return seeded_uniform((n, d), seed, -1.0, 1.0)


def _make_docs(n: int, seed: int) -> list[str]:
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "cat", "bird"]
    state = seed | 1
    docs = []
    for i in range(n):
        parts = []
        for _ in range(3 + (i % 3)):
            state = (state * 6364136223846793005 + 1) & 0xFFFFFFFFFFFFFFFF
            parts.append(words[state % len(words)])
        docs.append(" ".join(parts))
    return docs


def _checksum_array(a: np.ndarray) -> float:
    flat = np.asarray(a, dtype=np.float64).ravel()
    return float(flat.size + np.nansum(flat))


def _checksum_f64(v) -> float:
    a = np.asarray(v, dtype=np.float64).ravel()
    return float(a.size + np.nansum(a))


def _checksum_i64(v) -> float:
    a = np.asarray(v, dtype=np.int64).ravel()
    return float(a.size + a.sum())


def prepare(op: str, size: int, seed: int) -> tuple[Any, Callable[[], Any]]:
    n = max(size, 16)
    d = 3

    def thunk():
        if op == "standard_scaler":
            x = _make_x(n, d, seed)
            return StandardScaler().fit_transform(x)
        if op == "label_encoder":
            labels = [["a", "b", "c"][i % 3] for i in range(n)]
            return LabelEncoder().fit_transform(labels)
        if op == "train_test_split":
            x = _make_x(n, d, seed)
            y = np.arange(n, dtype=np.float64)
            return train_test_split(x, y, test_size=0.25, shuffle=False)
        if op == "linear_regression":
            x = _make_x(n, d, seed)
            y = 1.0 + 2.0 * x[:, 0] + 3.0 * x[:, 1] - 0.5 * x[:, 2]
            lr = LinearRegression().fit(x, y)
            pred = lr.predict(x)
            return (lr.coef_, lr.intercept_, pred)
        if op == "logistic_regression":
            x = _make_x(n, d, seed)
            y = ((x[:, 0] + x[:, 1]) > 0.0).astype(np.float64)
            lr = LogisticRegression(
                solver="lbfgs",
                max_iter=800,
                penalty=None,
                random_state=seed,
            ).fit(x, y)
            pred = lr.predict(x)
            # Accuracy only (GD vs lbfgs coefs differ).
            return accuracy_score(y, pred)
        if op == "knn_classify":
            x = _make_x(n, d, seed)
            y = np.array([i % 3 for i in range(n)], dtype=np.int64)
            return KNeighborsClassifier(n_neighbors=3, algorithm="brute").fit(x, y).predict(x)
        if op == "knn_regress":
            x = _make_x(n, d, seed)
            y = x[:, 0] + 0.1 * np.arange(n)
            return KNeighborsRegressor(n_neighbors=3, algorithm="brute").fit(x, y).predict(x)
        if op == "kmeans":
            x = _make_x(n, d, seed)
            init = x[:3].copy()
            km = KMeans(
                n_clusters=3,
                init=init,
                n_init=1,
                random_state=seed,
                algorithm="lloyd",
            ).fit(x)
            return float(km.inertia_)
        if op == "metrics_class":
            y_true = np.array([i % 2 for i in range(n)], dtype=np.int64)
            y_pred = y_true.copy()
            y_pred[::5] = 1 - y_pred[::5]
            return (
                accuracy_score(y_true, y_pred)
                + precision_score(y_true, y_pred, pos_label=1, zero_division=0)
                + recall_score(y_true, y_pred, pos_label=1, zero_division=0)
                + f1_score(y_true, y_pred, pos_label=1, zero_division=0)
            )
        if op == "metrics_reg":
            y_true = np.arange(n, dtype=np.float64) * 0.1
            y_pred = y_true + 0.05
            return (
                mean_squared_error(y_true, y_pred)
                + mean_absolute_error(y_true, y_pred)
                + r2_score(y_true, y_pred)
            )
        if op == "hashing_vectorizer":
            docs = _make_docs(n, seed)
            return (
                HashingVectorizer(n_features=64, alternate_sign=True, norm=None, binary=False)
                .transform(docs)
                .toarray()
            )
        if op == "count_vectorizer":
            docs = _make_docs(n, seed)
            return CountVectorizer().fit_transform(docs).toarray()
        raise ValueError(f"unknown op: {op}")

    return thunk(), thunk


def checksum(value: Any) -> float:
    if isinstance(value, (float, int, np.floating, np.integer)):
        return float(value)
    if isinstance(value, (tuple, list)):
        if len(value) == 3:
            coef, intercept, pred = value
            return _checksum_f64(coef) + float(intercept) + _checksum_f64(pred)
        if len(value) == 4:
            xtr, xte, ytr, yte = value
            return (
                _checksum_array(xtr)
                + _checksum_array(xte)
                + _checksum_f64(ytr)
                + _checksum_f64(yte)
            )
        return sum(checksum(v) for v in value)
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.integer):
            return _checksum_i64(value)
        return _checksum_array(value)
    raise TypeError(type(value))


def run_op(op: str, size: int, seed: int) -> float:
    result, _ = prepare(op, size, seed)
    return checksum(result)
