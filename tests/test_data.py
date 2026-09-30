"""Tests for data.py: dataset loading, splitting, and fairness diagnostics.

The OULAD tests build a small synthetic OULAD-shaped folder on the fly
(matching the real schema: studentInfo.csv, studentRegistration.csv,
assessments.csv, studentAssessment.csv, vle.csv, studentVle.csv) rather
than requiring the real ~10M-row dataset to be present. This is also a
standing regression test for the pandas-3.0 string-dtype bug that broke
this exact pipeline before (get_dummies silently skipping non-numeric
columns) - if that regresses, test_oulad_loads_with_no_leftover_strings
will fail with the same "could not convert string to float" symptom
instead of surfacing only when someone tries it against real data.

Run from the project root with:
    python -m unittest discover -s tests
or just:
    python -m unittest tests.test_data
"""
import os
import sys
import shutil
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared import (
    load_and_preprocess, train_test_split, group_train_test_split,
    fairness_report, make_synthetic_dataset,
)


def _build_fake_oulad(root, n_students=60, seed=0):
    """Writes a small OULAD-shaped set of CSVs into `root`, matching the
    real dataset's column names and relationships closely enough to
    exercise the real loading/aggregation code path end to end."""
    rng = np.random.default_rng(seed)
    student_ids = np.arange(1000, 1000 + n_students)
    modules = rng.choice(["AAA", "BBB", "CCC"], size=n_students)
    presentations = rng.choice(["2013J", "2014B"], size=n_students)

    student_info = pd.DataFrame({
        "code_module": modules, "code_presentation": presentations, "id_student": student_ids,
        "gender": rng.choice(["M", "F"], size=n_students),
        "region": rng.choice(["East Anglian Region", "Scotland"], size=n_students),
        "highest_education": rng.choice(["HE Qualification", "A Level or Equivalent"], size=n_students),
        "imd_band": rng.choice(["0-10%", "10-20%", "90-100%"], size=n_students),
        "age_band": rng.choice(["0-35", "35-55"], size=n_students),
        "num_of_prev_attempts": rng.integers(0, 3, n_students),
        "studied_credits": rng.integers(30, 120, n_students),
        "disability": rng.choice(["Y", "N"], size=n_students),
        "final_result": rng.choice(["Pass", "Fail", "Withdrawn", "Distinction"], size=n_students),
    })
    student_info.to_csv(os.path.join(root, "studentInfo.csv"), index=False)

    pd.DataFrame({
        "code_module": modules, "code_presentation": presentations, "id_student": student_ids,
        "date_registration": rng.integers(-60, -1, n_students),
        "date_unregistration": [np.nan] * n_students,
    }).to_csv(os.path.join(root, "studentRegistration.csv"), index=False)

    assessments = pd.DataFrame({
        "code_module": ["AAA", "AAA", "BBB", "BBB", "CCC", "CCC"],
        "code_presentation": ["2013J", "2013J", "2014B", "2014B", "2013J", "2013J"],
        "id_assessment": [1, 2, 3, 4, 5, 6],
        "assessment_type": ["TMA"] * 6,
        "date": [10, 20, 10, 20, 10, 20],
        "weight": [50, 50, 50, 50, 50, 50],
    })
    assessments.to_csv(os.path.join(root, "assessments.csv"), index=False)

    rows = []
    for sid, mod, pres in zip(student_ids, modules, presentations):
        aids = assessments[(assessments.code_module == mod) &
                            (assessments.code_presentation == pres)]["id_assessment"].tolist()
        for aid in aids:
            rows.append((aid, sid, int(rng.integers(1, 25)), int(rng.integers(0, 2)), int(rng.integers(30, 100))))
    pd.DataFrame(rows, columns=["id_assessment", "id_student", "date_submitted", "is_banked", "score"]
                 ).to_csv(os.path.join(root, "studentAssessment.csv"), index=False)

    vle = pd.DataFrame({
        "id_site": [100, 101, 102, 103],
        "code_module": ["AAA", "AAA", "BBB", "CCC"],
        "code_presentation": ["2013J", "2013J", "2014B", "2013J"],
        "activity_type": ["resource", "forumng", "oucontent", "quiz"],
    })
    vle.to_csv(os.path.join(root, "vle.csv"), index=False)

    vle_rows = []
    for sid, mod, pres in zip(student_ids, modules, presentations):
        sites = vle[(vle.code_module == mod) & (vle.code_presentation == pres)]["id_site"].tolist()
        for _ in range(15):
            if not sites:
                continue
            vle_rows.append((mod, pres, sid, rng.choice(sites), int(rng.integers(0, 27)), int(rng.integers(1, 5))))
    pd.DataFrame(vle_rows, columns=["code_module", "code_presentation", "id_student",
                                     "id_site", "date", "sum_click"]
                 ).to_csv(os.path.join(root, "studentVle.csv"), index=False)


def _build_fake_uci(path, n=100, seed=0):
    """Writes a semicolon-separated UCI-Student-Performance-shaped CSV with
    exactly 33 columns (the real dataset's column count, which
    load_and_preprocess validates against)."""
    rng = np.random.default_rng(seed)
    cols = {
        "school": rng.choice(["GP", "MS"], size=n), "sex": rng.choice(["F", "M"], size=n),
        "age": rng.integers(15, 22, n), "address": rng.choice(["U", "R"], size=n),
        "famsize": rng.choice(["LE3", "GT3"], size=n), "Pstatus": rng.choice(["T", "A"], size=n),
        "Medu": rng.integers(0, 5, n), "Fedu": rng.integers(0, 5, n),
        "Mjob": rng.choice(["teacher", "other"], size=n), "Fjob": rng.choice(["teacher", "other"], size=n),
        "reason": rng.choice(["home", "course"], size=n), "guardian": rng.choice(["mother", "father"], size=n),
        "traveltime": rng.integers(1, 5, n), "studytime": rng.integers(1, 5, n),
        "failures": rng.integers(0, 4, n), "schoolsup": rng.choice(["yes", "no"], size=n),
        "famsup": rng.choice(["yes", "no"], size=n), "paid": rng.choice(["yes", "no"], size=n),
        "activities": rng.choice(["yes", "no"], size=n), "nursery": rng.choice(["yes", "no"], size=n),
        "higher": rng.choice(["yes", "no"], size=n), "internet": rng.choice(["yes", "no"], size=n),
        "romantic": rng.choice(["yes", "no"], size=n), "famrel": rng.integers(1, 6, n),
        "freetime": rng.integers(1, 6, n), "goout": rng.integers(1, 6, n),
        "Dalc": rng.integers(1, 6, n), "Walc": rng.integers(1, 6, n),
        "health": rng.integers(1, 6, n), "absences": rng.integers(0, 30, n),
        "G1": rng.integers(0, 21, n), "G2": rng.integers(0, 21, n), "G3": rng.integers(0, 21, n),
    }
    assert len(cols) == 33, f"fixture must have exactly 33 columns, has {len(cols)}"
    pd.DataFrame(cols).to_csv(path, sep=";", index=False)


class TestOuladLoading(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        _build_fake_oulad(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_oulad_loads_with_no_leftover_strings(self):
        """Regression test for the pandas-3.0 dtype bug: get_dummies must
        actually one-hot-encode every non-numeric column. If it silently
        skips one (as it did when the check was `dtype == object`, which
        pandas 3.0 no longer uses for plain string columns), the final
        .astype(np.float64) would raise here."""
        X, y, header, feature_groups, protected, groups = load_and_preprocess(
            self.tmpdir, dataset_type="oulad")
        self.assertEqual(X.dtype, np.float64)
        self.assertEqual(X.shape[0], 60)
        self.assertEqual(y.shape[0], 60)
        self.assertTrue(set(np.unique(y)).issubset({0.0, 1.0}))
        self.assertFalse(np.isnan(X).any())

    def test_oulad_auto_mode_also_detects_the_folder(self):
        X, y, *_ = load_and_preprocess(self.tmpdir, dataset_type="auto")
        self.assertEqual(X.shape[0], 60)

    def test_oulad_missing_file_raises_clear_error(self):
        os.remove(os.path.join(self.tmpdir, "studentVle.csv"))
        with self.assertRaises(FileNotFoundError) as ctx:
            load_and_preprocess(self.tmpdir, dataset_type="oulad")
        self.assertIn("studentVle.csv", str(ctx.exception))

    def test_oulad_missing_required_column_raises_clear_error(self):
        """Regression test for the schema-validation added after this
        pipeline previously failed deep inside pandas with a bare KeyError
        for a wrong-schema file - now it should fail fast, before any
        merging/aggregation, naming both the file and the missing column."""
        path = os.path.join(self.tmpdir, "studentInfo.csv")
        df = pd.read_csv(path).drop(columns=["final_result"])
        df.to_csv(path, index=False)
        with self.assertRaises(ValueError) as ctx:
            load_and_preprocess(self.tmpdir, dataset_type="oulad")
        msg = str(ctx.exception)
        self.assertIn("studentInfo.csv", msg)
        self.assertIn("final_result", msg)

    def test_oulad_zip_path_is_rejected_with_clear_message(self):
        """The GUI only ever offers a folder picker for OULAD (never a zip
        file), and load_and_preprocess enforces that explicitly - this
        locks that contract in so it doesn't silently change."""
        zip_path = os.path.join(self.tmpdir, "..", "fake_oulad.zip")
        zip_path = os.path.abspath(zip_path)
        with zipfile.ZipFile(zip_path, "w") as zf:
            for fname in os.listdir(self.tmpdir):
                zf.write(os.path.join(self.tmpdir, fname), arcname=fname)
        try:
            with self.assertRaises(ValueError):
                load_and_preprocess(zip_path, dataset_type="oulad")
        finally:
            os.remove(zip_path)

    def test_groups_are_student_ids_for_leakage_safe_splitting(self):
        X, y, header, feature_groups, protected, groups = load_and_preprocess(
            self.tmpdir, dataset_type="oulad")
        self.assertIsNotNone(groups)
        self.assertEqual(len(groups), len(y))


class TestUCILoading(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.csv_path = os.path.join(self.tmpdir, "student-mat.csv")
        _build_fake_uci(self.csv_path)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_uci_loads_correctly(self):
        X, y, header, feature_groups, protected, groups = load_and_preprocess(
            self.csv_path, dataset_type="uci")
        self.assertEqual(X.dtype, np.float64)
        self.assertEqual(X.shape[0], 100)
        self.assertTrue(set(np.unique(y)).issubset({0.0, 1.0}))
        self.assertIn("sex", protected)
        self.assertIn("school", protected)

    def test_uci_wrong_column_count_raises(self):
        bad_path = os.path.join(self.tmpdir, "bad.csv")
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(bad_path, sep=";", index=False)
        with self.assertRaises(ValueError):
            load_and_preprocess(bad_path, dataset_type="uci", expected_n_columns=33)

    def test_uci_missing_target_column_raises(self):
        path = os.path.join(self.tmpdir, "no_target.csv")
        df = pd.read_csv(self.csv_path, sep=";").drop(columns=["G3"])
        df["extra_col_to_keep_33"] = 0  # keep column count at 33
        df.to_csv(path, sep=";", index=False)
        with self.assertRaises(ValueError):
            load_and_preprocess(path, dataset_type="uci", target_col="G3")


class TestSplitting(unittest.TestCase):
    def test_train_test_split_shapes(self):
        X, y = make_synthetic_dataset(n_samples=200)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_frac=0.3, seed=0)
        self.assertEqual(len(X_train) + len(X_test), 200)
        self.assertEqual(len(X_test), 60)

    def test_train_test_split_is_deterministic_given_seed(self):
        X, y = make_synthetic_dataset(n_samples=100)
        a = train_test_split(X, y, seed=5)
        b = train_test_split(X, y, seed=5)
        for arr_a, arr_b in zip(a, b):
            np.testing.assert_array_equal(arr_a, arr_b)

    def test_group_split_never_splits_a_group_across_train_and_test(self):
        rng = np.random.default_rng(0)
        n = 300
        # 60 groups, ~5 rows each - the thing that matters is that a group
        # id never appears on both sides of the split.
        groups = rng.integers(0, 60, size=n)
        X = rng.normal(size=(n, 3))
        y = rng.integers(0, 2, size=n)
        X_train, X_test, y_train, y_test, train_idx, test_idx = group_train_test_split(
            X, y, groups, test_frac=0.25, seed=0)
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        self.assertEqual(len(train_groups & test_groups), 0)
        self.assertEqual(len(train_idx) + len(test_idx), n)


class TestFairnessReport(unittest.TestCase):
    def test_hand_computed_group_metrics(self):
        # Group A: 4 samples, all correctly predicted (2 TP, 2 TN)
        # Group B: 4 samples, model predicts everyone as class 0 (misses both actual positives)
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 0, 0, 0, 0])
        groups = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])
        df = fairness_report(y_true, y_pred, {"grp": groups})

        row_a = df[(df["attribute"] == "grp") & (df["group"] == "A")].iloc[0]
        row_b = df[(df["attribute"] == "grp") & (df["group"] == "B")].iloc[0]

        self.assertAlmostEqual(row_a["accuracy"], 1.0)
        self.assertAlmostEqual(row_a["recall_tpr"], 1.0)
        self.assertAlmostEqual(row_b["recall_tpr"], 0.0)  # both actual positives missed
        self.assertAlmostEqual(row_b["accuracy"], 0.5)

        gap_row = df[(df["attribute"] == "grp") & (df["group"] == "GAP(max-min)")].iloc[0]
        self.assertAlmostEqual(gap_row["recall_tpr"], 1.0)  # 1.0 - 0.0

    def test_empty_protected_attributes_returns_empty_frame(self):
        df = fairness_report(np.array([1, 0]), np.array([1, 0]), {})
        self.assertTrue(df.empty)


if __name__ == "__main__":
    unittest.main()
