"""SHAP-based explainability wrapper. Works against any predictProba-style
callable, so it's equally at home wrapping a single agent, a BaggingEnsemble,
or the full MASController's fused prediction.
"""
import numpy as np
import shap


class ExplainabilityEngine:
    """
    Model-agnostic explainability via SHAP's KernelExplainer.
    """

    @staticmethod
    def _make_explainer(predict_proba_fn, X_background, background_size=50, seed=0):
        X_background = np.asarray(X_background, dtype=np.float64)
        bg = shap.sample(X_background, min(background_size, len(X_background)),
                          random_state=seed)
        return shap.KernelExplainer(predict_proba_fn, bg)

    @staticmethod
    def global_importance(predict_proba_fn, X, feature_names=None,
                           X_background=None, background_size=50, max_samples=50,
                           nsamples=100, seed=0):
        """
        Mean |SHAP value| per feature over a sample of X - the SHAP analogue
        of a global feature-importance ranking.
        """
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        names = feature_names or [f"f{i}" for i in range(n_features)]

        X_background = X if X_background is None else X_background
        explainer = ExplainabilityEngine._make_explainer(
            predict_proba_fn, X_background, background_size, seed)

        X_eval = X if n_samples <= max_samples else \
            X[rng.choice(n_samples, max_samples, replace=False)]

        shap_values = explainer.shap_values(X_eval, nsamples=nsamples, silent=True)
        shap_values = np.asarray(shap_values)

        mean_abs = np.abs(shap_values).mean(axis=0)
        pairs = list(zip(names, mean_abs))
        pairs.sort(key=lambda t: -abs(t[1]))
        return pairs

    @staticmethod
    def global_importance_and_explanation(predict_proba_fn, X, feature_names=None,
                                           X_background=None, background_size=50, max_samples=50,
                                           nsamples=100, seed=0):
        """
        Same computation as global_importance(), but also returns a
        shap.Explanation object over the evaluated sample so it can be
        rendered with SHAP's own matplotlib plots (shap.plots.bar,
        shap.plots.beeswarm) instead of hand-rolled bar charts.
        """
        rng = np.random.default_rng(seed)
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        names = feature_names or [f"f{i}" for i in range(n_features)]

        X_background = X if X_background is None else X_background
        explainer = ExplainabilityEngine._make_explainer(
            predict_proba_fn, X_background, background_size, seed)

        X_eval = X if n_samples <= max_samples else \
            X[rng.choice(n_samples, max_samples, replace=False)]

        shap_values = explainer.shap_values(X_eval, nsamples=nsamples, silent=True)
        shap_values = np.asarray(shap_values)

        mean_abs = np.abs(shap_values).mean(axis=0)
        pairs = list(zip(names, mean_abs))
        pairs.sort(key=lambda t: -abs(t[1]))

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = np.ravel(base_value)[0]
        explanation = shap.Explanation(
            values=shap_values,
            base_values=np.full(len(X_eval), base_value),
            data=X_eval,
            feature_names=names,
        )
        return pairs, explanation

    @staticmethod
    def explain_instance(predict_proba_fn, x, X_background, feature_names=None,
                          background_size=50, nsamples=100, seed=0):
        """
        Per-feature SHAP values for a single instance x - how much each
        feature pushed this specific student's prediction up or down
        relative to the background distribution.
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        n_features = len(x)
        names = feature_names or [f"f{i}" for i in range(n_features)]

        explainer = ExplainabilityEngine._make_explainer(
            predict_proba_fn, X_background, background_size, seed)

        shap_values = explainer.shap_values(x.reshape(1, -1), nsamples=nsamples, silent=True)
        contributions = np.asarray(shap_values).ravel()
        baseline_proba = float(predict_proba_fn(x.reshape(1, -1))[0])

        pairs = list(zip(names, contributions))
        pairs.sort(key=lambda t: -abs(t[1]))
        return pairs, baseline_proba

    @staticmethod
    def explain_instance_and_explanation(predict_proba_fn, x, X_background, feature_names=None,
                                         background_size=50, nsamples=100, seed=0):
        """
        Same computation as explain_instance(), but also returns a
        shap.Explanation object for this single instance so it can be
        rendered with shap.plots.waterfall (a matplotlib plot).
        """
        x = np.asarray(x, dtype=np.float64).ravel()
        n_features = len(x)
        names = feature_names or [f"f{i}" for i in range(n_features)]

        explainer = ExplainabilityEngine._make_explainer(
            predict_proba_fn, X_background, background_size, seed)

        shap_values = explainer.shap_values(x.reshape(1, -1), nsamples=nsamples, silent=True)
        contributions = np.asarray(shap_values).ravel()
        baseline_proba = float(predict_proba_fn(x.reshape(1, -1))[0])

        pairs = list(zip(names, contributions))
        pairs.sort(key=lambda t: -abs(t[1]))

        base_value = explainer.expected_value
        if isinstance(base_value, (list, np.ndarray)):
            base_value = np.ravel(base_value)[0]
        explanation = shap.Explanation(
            values=contributions,
            base_values=base_value,
            data=x,
            feature_names=names,
        )
        return pairs, baseline_proba, explanation

    @staticmethod
    def format_bar_chart(items, max_width: int = 24, value_fmt: str = "{:+.4f}"):
        if not items:
            return []
        max_abs = max(abs(v) for _, v in items) or 1e-9
        lines = []
        for name, v in items:
            bar_len = max(1, int(round(abs(v) / max_abs * max_width))) if v != 0 else 0
            bar = "#" * bar_len
            lines.append(f"  {name:<24} {bar:<{max_width}} {value_fmt.format(v)}")
        return lines


