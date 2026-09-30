"""Shared scoring/CV-splitting helpers used across agents, fusion, and search.
"""
import numpy as np


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def confusion_matrix(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def precision_recall_f1(tp, tn, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _stratified_kfold_indices(y, k: int = 5, seed: int = 0):
    """Manual stratified k-fold split (no sklearn) - each fold gets a
    proportional share of each class, in random order."""
    y = np.asarray(y).ravel()
    rng = np.random.default_rng(seed)
    fold_ids = [[] for _ in range(k)]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        for i, sample_idx in enumerate(idx):
            fold_ids[i % k].append(sample_idx)
    return [np.array(sorted(f)) for f in fold_ids]


