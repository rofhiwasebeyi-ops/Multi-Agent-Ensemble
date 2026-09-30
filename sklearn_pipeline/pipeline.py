"""The library-based pipeline, built with scikit-learn 
"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scratch_pipeline import AISAgent, BaggingEnsemble, AGENT_REGISTRY


class Subsampled(BaseEstimator, ClassifierMixin):
    """Wraps an estimator and trains it on at most max_samples rows (a
    reproducible random subsample); predict/predict_proba still accept
    inputs of any size."""

    def __init__(self, estimator=None, max_samples=4000, random_state=0):
        self.estimator = estimator
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        if len(X) > self.max_samples:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(X), size=self.max_samples, replace=False)
            X, y = X[idx], y[idx]
        self.estimator_ = clone(self.estimator).fit(X, y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class FromScratchAdapter(BaseEstimator, ClassifierMixin):
    """Adapts a from-scratch BaseAgent-style model
    """

    def __init__(self, agent_factory=None, agent_kwargs=None, n_estimators=10, random_state=0):
        self.agent_factory = agent_factory
        self.agent_kwargs = agent_kwargs
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.model_ = BaggingEnsemble(base_agent_factory=self.agent_factory,
                                       n_estimators=self.n_estimators, bootstrap=True,
                                       random_state=self.random_state,
                                       agent_kwargs=dict(self.agent_kwargs or {}))
        self.model_.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X))

    def predict_proba(self, X):
        p1 = self.model_.predictProba(np.asarray(X))
        return np.column_stack([1.0 - p1, p1])


def _make_pipeline(model):
    """Wraps a model that benefits from feature scaling (everything except
    RandomForest) in a StandardScaler."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


SKLEARN_AGENT_REGISTRY = {
    "LogisticRegression": (lambda **kw: _make_pipeline(LogisticRegression(**kw)),
                            dict(max_iter=1000)),
    "SVM": (lambda **kw: Subsampled(estimator=_make_pipeline(SVC(kernel="rbf", probability=True)),
                                     max_samples=4000, random_state=kw.get("random_state", 0)),
             dict()),
    "RandomForest": (lambda **kw: RandomForestClassifier(**kw),
                      dict(n_estimators=200, max_depth=8, min_samples_leaf=2)),
    "MLP": (lambda **kw: _make_pipeline(MLPClassifier(**kw)),
             dict(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True)),
    # AIS has no library equivalent - reuses the from-scratch AISAgent with
    # the SAME hyperparameters as AGENT_REGISTRY["AIS"] 
    "AIS": (lambda **kw: FromScratchAdapter(agent_factory=AISAgent,
                                             agent_kwargs=AGENT_REGISTRY["AIS"][1],
                                             n_estimators=kw.get("n_estimators", 10),
                                             random_state=kw.get("random_state", 0)),
             dict()),
}


def run_sklearn_single(agent_name, X_train, y_train, X_test, y_test, random_state=0, n_estimators=10):
    """Fits ONE scikit-learn model (no ensembling) """
    if agent_name not in SKLEARN_AGENT_REGISTRY:
        raise ValueError(f"Unknown sklearn agent '{agent_name}'. "
                          f"Available: {list(SKLEARN_AGENT_REGISTRY.keys())}")
    factory, kwargs = SKLEARN_AGENT_REGISTRY[agent_name]
    model = factory(**dict(kwargs))
    if "random_state" in model.get_params():
        model.set_params(random_state=random_state)
    if isinstance(model, FromScratchAdapter):
        model.set_params(n_estimators=n_estimators)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "model": model,
        "model_name": f"sklearn {agent_name}",
        "y_pred": y_pred,
        "predict_proba": lambda X: model.predict_proba(X)[:, 1],
    }


def run_sklearn_ensemble_pipeline(X_train, y_train, X_test, y_test, feature_names=None,
                                    cv=5, n_estimators=10, random_state=0):
    """Trains all five SKLEARN_AGENT_REGISTRY models and fuses them with
    scikit-learn's StackingClassifier (a logistic-regression meta-learner
    trained on out-of-fold predictions via internal cv-fold splitting).
    """
    estimators = []
    for name, (factory, kwargs) in SKLEARN_AGENT_REGISTRY.items():
        kwargs = dict(kwargs)
        model = factory(**kwargs)
        if "random_state" in model.get_params():
            model.set_params(random_state=random_state)
        if isinstance(model, FromScratchAdapter):
            model.set_params(n_estimators=n_estimators)
        estimators.append((name, model))

    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        cv=cv,
        stack_method="predict_proba",
        n_jobs=None,
    )
    stack.fit(X_train, y_train)
    y_pred = stack.predict(X_test)

    per_agent_test_acc = {}
    for name, fitted_model in zip([n for n, _ in estimators], stack.estimators_):
        preds = fitted_model.predict(X_test)
        per_agent_test_acc[name] = float(np.mean(preds == y_test))

    # StackingClassifier's final_estimator_ is a LogisticRegression whose
    # coefficients weight each base estimator's P(y=1) output - the direct
    # analogue of the from-scratch FusionLayer's learned per-agent weights.
    meta = stack.final_estimator_
    fusion_weights = {name: float(w) for name, w in zip([n for n, _ in estimators], meta.coef_.ravel())}

    return {
        "model": stack,
        "model_name": f"scikit-learn StackingClassifier ({len(estimators)} base learners)",
        "y_pred": y_pred,
        "per_agent_test_acc": per_agent_test_acc,
        "fusion_weights": fusion_weights,
        "agent_names": [n for n, _ in estimators],
        "predict_proba": lambda X: stack.predict_proba(X)[:, 1],
    }
