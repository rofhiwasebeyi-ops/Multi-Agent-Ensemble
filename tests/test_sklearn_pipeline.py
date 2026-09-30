"""Tests for sklearn_pipeline.py - the existing-library pipeline required
alongside the from-scratch one (agents.py/fusion.py). These are
integration-level tests, not algorithmic ones: scikit-learn's own
implementations are already thoroughly tested upstream, so what matters
here is that this project's wiring around them (registry, subsampling
wrapper, stacking fusion, output shape) behaves correctly and produces
results comparable to the from-scratch pipeline's output format.

Run from the project root with:
    python -m unittest discover -s tests
or just:
    python -m unittest tests.test_sklearn_pipeline
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shared import make_synthetic_dataset, train_test_split
from scratch_pipeline import AGENT_REGISTRY
from sklearn_pipeline import (
    SKLEARN_AGENT_REGISTRY, Subsampled, FromScratchAdapter,
    run_sklearn_single, run_sklearn_ensemble_pipeline,
)


class TestSklearnAgentRegistry(unittest.TestCase):
    def test_all_registered_models_fit_and_predict(self):
        X, y = make_synthetic_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        for name in SKLEARN_AGENT_REGISTRY:
            result = run_sklearn_single(name, X_train, y_train, X_test, y_test,
                                         random_state=0, n_estimators=5)
            self.assertEqual(len(result["y_pred"]), len(y_test))
            self.assertTrue(set(np.unique(result["y_pred"])).issubset({0, 1}))
            proba = result["predict_proba"](X_test)
            self.assertEqual(proba.shape, (len(y_test),))
            self.assertTrue(np.all((proba >= 0) & (proba <= 1)))

    def test_unknown_agent_name_raises(self):
        X, y = make_synthetic_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        with self.assertRaises(ValueError):
            run_sklearn_single("NotARealModel", X_train, y_train, X_test, y_test)

    def test_reasonable_accuracy_on_easy_synthetic_data(self):
        rng = np.random.default_rng(0)
        # n=1000 rather than 300: AIS's radius calibration (a CV search
        # over self-distance quantiles) is genuinely noisier on very small
        # samples - at n=300/225-train this test was flaky (0.55-0.72
        # across seeds, occasionally dropping below any reasonable
        # threshold) even after fixing the calibration grid's range (see
        # AISAgent._select_radius_supervised/_select_radius_cv docstrings
        # for that fix). More data stabilizes it, same as it would for any
        # model relying on a data-driven threshold.
        n = 1000
        X = rng.normal(size=(n, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        split = int(n * 0.75)
        X_train, X_test, y_train, y_test = X[:split], X[split:], y[:split], y[split:]
        # MLP with early_stopping can land a bit lower than the linear/tree
        # models purely from training variance. AIS is a genuinely weaker
        # standalone learner by design (negative-selection/novelty-detection
        # framing, not a direct discriminative classifier) - its bar here is
        # "clearly better than chance on easy data", not "matches the other
        # four".
        thresholds = {"MLP": 0.75, "AIS": 0.65}
        for name in SKLEARN_AGENT_REGISTRY:
            result = run_sklearn_single(name, X_train, y_train, X_test, y_test,
                                         random_state=0, n_estimators=8)
            acc = np.mean(result["y_pred"] == y_test)
            self.assertGreater(acc, thresholds.get(name, 0.8),
                                f"{name} scored too low on trivially separable data")


class TestSubsampledWrapper(unittest.TestCase):
    """Regression tests for a real OOM/timeout hit: scikit-learn's SVC with
    an RBF kernel does not scale to OULAD's real ~24k-row training set
    (nor does it complete in a practical time once StackingClassifier's
    internal cv folds mean it gets refit several times over). Subsampled
    caps training set size while leaving predict/predict_proba able to
    accept inputs of any size."""

    def test_large_n_is_capped(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(0)
        n = 5000
        X = rng.normal(size=(n, 4))
        y = (X[:, 0] > 0).astype(int)

        wrapped = Subsampled(estimator=LogisticRegression(), max_samples=200, random_state=0)
        wrapped.fit(X, y)
        preds = wrapped.predict(X)  # predict on the FULL original n
        self.assertEqual(len(preds), n)
        proba = wrapped.predict_proba(X)
        self.assertEqual(proba.shape[0], n)

    def test_small_n_is_not_affected(self):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.default_rng(0)
        X = rng.normal(size=(50, 3))
        y = (X[:, 0] > 0).astype(int)
        wrapped = Subsampled(estimator=LogisticRegression(), max_samples=4000, random_state=0)
        wrapped.fit(X, y)
        acc = np.mean(wrapped.predict(X) == y)
        self.assertGreater(acc, 0.8)

    def test_reproducible_given_seed(self):
        from sklearn.tree import DecisionTreeClassifier
        rng = np.random.default_rng(0)
        n = 1000
        X = rng.normal(size=(n, 3))
        y = (X[:, 0] > 0).astype(int)

        a = Subsampled(estimator=DecisionTreeClassifier(random_state=0), max_samples=100, random_state=42)
        a.fit(X, y)
        b = Subsampled(estimator=DecisionTreeClassifier(random_state=0), max_samples=100, random_state=42)
        b.fit(X, y)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))

    def test_clones_correctly_inside_stacking_classifier(self):
        """StackingClassifier internally clones each base estimator several
        times (once per cv fold, plus a final refit) - this only works if
        Subsampled correctly exposes get_params/set_params via
        BaseEstimator, which is what subclassing it (rather than hand-
        rolling the sklearn estimator interface) is for."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import StackingClassifier
        rng = np.random.default_rng(0)
        n = 300
        X = rng.normal(size=(n, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        wrapped = Subsampled(estimator=LogisticRegression(), max_samples=100, random_state=0)
        stack = StackingClassifier(
            estimators=[("wrapped", wrapped), ("plain", LogisticRegression())],
            final_estimator=LogisticRegression(), cv=3)
        stack.fit(X, y)  # must not raise
        preds = stack.predict(X)
        self.assertEqual(len(preds), n)


class TestFromScratchAdapter(unittest.TestCase):
    """Tests for the one deliberate exception to 'this pipeline is all
    library code': the AIS slot reuses the project's own from-scratch
    AISAgent via FromScratchAdapter, since scikit-learn has no negative-
    selection/AIS equivalent. These check the adapter's plumbing
    (fit/predict/predict_proba shape, BaseEstimator compliance, genuine
    hyperparameter reuse) - AISAgent's own algorithmic correctness is
    already covered in tests/test_agents.py."""

    def test_reuses_the_exact_same_hyperparameters_as_the_from_scratch_registry(self):
        """This must be genuine reuse of AGENT_REGISTRY["AIS"]'s config,
        not a separately-tuned copy - otherwise the two pipelines' AIS
        components wouldn't actually be comparable."""
        factory, kwargs = SKLEARN_AGENT_REGISTRY["AIS"]
        adapter = factory(**kwargs)
        self.assertEqual(adapter.agent_kwargs, AGENT_REGISTRY["AIS"][1])

    def test_predict_proba_shape_and_range(self):
        rng = np.random.default_rng(0)
        n = 150
        X = rng.normal(size=(n, 4))
        y = (X[:, 0] > 0).astype(int)
        adapter = FromScratchAdapter(agent_factory=AGENT_REGISTRY["AIS"][0],
                                      agent_kwargs=AGENT_REGISTRY["AIS"][1],
                                      n_estimators=5, random_state=0)
        adapter.fit(X, y)
        proba = adapter.predict_proba(X)
        self.assertEqual(proba.shape, (n, 2))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
        self.assertTrue(np.all((proba >= 0) & (proba <= 1)))

    def test_reproducible_given_seed(self):
        rng = np.random.default_rng(0)
        n = 150
        X = rng.normal(size=(n, 4))
        y = (X[:, 0] > 0).astype(int)
        a = FromScratchAdapter(agent_factory=AGENT_REGISTRY["AIS"][0],
                                agent_kwargs=AGENT_REGISTRY["AIS"][1], n_estimators=5, random_state=7)
        a.fit(X, y)
        b = FromScratchAdapter(agent_factory=AGENT_REGISTRY["AIS"][0],
                                agent_kwargs=AGENT_REGISTRY["AIS"][1], n_estimators=5, random_state=7)
        b.fit(X, y)
        np.testing.assert_array_equal(a.predict(X), b.predict(X))

    def test_clones_correctly_inside_stacking_classifier(self):
        """Same concern as Subsampled's equivalent test: StackingClassifier
        clones each base estimator internally, which requires correct
        get_params/set_params (via subclassing BaseEstimator)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import StackingClassifier
        rng = np.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 4))
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        adapter = FromScratchAdapter(agent_factory=AGENT_REGISTRY["AIS"][0],
                                      agent_kwargs=AGENT_REGISTRY["AIS"][1], n_estimators=3, random_state=0)
        stack = StackingClassifier(
            estimators=[("ais", adapter), ("lr", LogisticRegression())],
            final_estimator=LogisticRegression(), cv=3)
        stack.fit(X, y)  # must not raise
        preds = stack.predict(X)
        self.assertEqual(len(preds), n)


class TestSklearnEnsemblePipeline(unittest.TestCase):
    def test_output_shape_matches_from_scratch_pipeline_contract(self):
        """The from-scratch run_full_ensemble_pipeline and this sklearn
        equivalent are meant to be directly comparable - both should
        report y_pred, per_agent_test_acc, and fusion_weights so a caller
        (main.py, the GUI) can display/log them identically."""
        X, y = make_synthetic_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        result = run_sklearn_ensemble_pipeline(X_train, y_train, X_test, y_test,
                                                 n_estimators=5, random_state=0)
        for key in ("y_pred", "per_agent_test_acc", "fusion_weights", "agent_names", "predict_proba"):
            self.assertIn(key, result)
        self.assertEqual(set(result["per_agent_test_acc"].keys()), set(SKLEARN_AGENT_REGISTRY.keys()))
        self.assertEqual(set(result["fusion_weights"].keys()), set(SKLEARN_AGENT_REGISTRY.keys()))
        self.assertEqual(len(result["y_pred"]), len(y_test))

    def test_stacking_is_at_least_competitive_with_best_base_learner(self):
        """Not a strict requirement (stacking can occasionally underperform
        the single best base learner), but on clean, easy synthetic data it
        should be in the same ballpark - a large gap would suggest the
        meta-learner or cv wiring is broken, not just unlucky."""
        rng = np.random.default_rng(0)
        n = 400
        X = rng.normal(size=(n, 5))
        y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
        X_train, X_test, y_train, y_test = X[:300], X[300:], y[:300], y[300:]

        result = run_sklearn_ensemble_pipeline(X_train, y_train, X_test, y_test,
                                                 n_estimators=5, random_state=0)
        stack_acc = np.mean(result["y_pred"] == y_test)
        best_base_acc = max(result["per_agent_test_acc"].values())
        self.assertGreater(stack_acc, best_base_acc - 0.1)

    def test_reproducible_given_seed(self):
        X, y = make_synthetic_dataset()
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        a = run_sklearn_ensemble_pipeline(X_train, y_train, X_test, y_test, n_estimators=5, random_state=3)
        b = run_sklearn_ensemble_pipeline(X_train, y_train, X_test, y_test, n_estimators=5, random_state=3)
        np.testing.assert_array_equal(a["y_pred"], b["y_pred"])


if __name__ == "__main__":
    unittest.main()
