"""Correctness tests for metrics.py.

Run from the project root with:
    python -m unittest discover -s tests
or just:
    python -m unittest tests.test_metrics
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared import accuracy, confusion_matrix, precision_recall_f1, _stratified_kfold_indices


class TestAccuracy(unittest.TestCase):
    def test_perfect(self):
        y = np.array([0, 1, 1, 0, 1])
        self.assertEqual(accuracy(y, y), 1.0)

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        self.assertEqual(accuracy(y_true, y_pred), 0.0)

    def test_known_value(self):
        # 3 correct out of 4 -> 0.75
        y_true = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 0])
        self.assertAlmostEqual(accuracy(y_true, y_pred), 0.75)


class TestConfusionMatrix(unittest.TestCase):
    def test_hand_computed(self):
        # y_true: 1 1 1 0 0 0
        # y_pred: 1 1 0 0 0 1
        # TP=2 (pos 0,1), FN=1 (pos 2), TN=2 (pos 3,4), FP=1 (pos 5)
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 1])
        tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
        self.assertEqual((tp, tn, fp, fn), (2, 2, 1, 1))

    def test_counts_sum_to_n(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=50)
        y_pred = rng.integers(0, 2, size=50)
        tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
        self.assertEqual(tp + tn + fp + fn, 50)


class TestPrecisionRecallF1(unittest.TestCase):
    def test_hand_computed(self):
        # tp=2, fp=1, fn=1 -> precision=2/3, recall=2/3, f1=2/3
        precision, recall, f1 = precision_recall_f1(tp=2, tn=2, fp=1, fn=1)
        self.assertAlmostEqual(precision, 2 / 3)
        self.assertAlmostEqual(recall, 2 / 3)
        self.assertAlmostEqual(f1, 2 / 3)

    def test_zero_division_guards(self):
        # tp=0, fp=0 -> precision should not raise (defined as 0.0), same for recall
        precision, recall, f1 = precision_recall_f1(tp=0, tn=5, fp=0, fn=0)
        self.assertEqual(precision, 0.0)
        self.assertEqual(recall, 0.0)
        self.assertEqual(f1, 0.0)


class TestStratifiedKFold(unittest.TestCase):
    def test_folds_partition_all_indices_exactly_once(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, size=97)
        folds = _stratified_kfold_indices(y, k=5, seed=0)
        all_idx = np.concatenate(folds)
        self.assertEqual(len(all_idx), len(y))
        self.assertEqual(len(np.unique(all_idx)), len(y))  # no duplicates, none missing

    def test_class_proportions_preserved_per_fold(self):
        # 80 class-0, 20 class-1 -> each of 5 folds should get ~16 class-0, ~4 class-1
        y = np.array([0] * 80 + [1] * 20)
        folds = _stratified_kfold_indices(y, k=5, seed=0)
        for fold in folds:
            n_pos = np.sum(y[fold] == 1)
            n_neg = np.sum(y[fold] == 0)
            # stratified split should keep each fold within 1 sample of the
            # exact proportional share (20/5=4 positives, 80/5=16 negatives)
            self.assertLessEqual(abs(n_pos - 4), 1)
            self.assertLessEqual(abs(n_neg - 16), 1)


if __name__ == "__main__":
    unittest.main()
