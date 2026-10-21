"""The library-based pipeline, built with scikit-learn - required alongside
the from-scratch pipeline in agents.py/fusion.py so the project has the two
distinct pipelines called for: one using an existing library, one built by
hand. Both are trained and evaluated on the exact same data (same
load_and_preprocess, same train/test split, same metrics.py functions), so
results are directly comparable rather than apples-to-oranges.

Scope note on AIS: scikit-learn has no built-in negative-selection /
artificial-immune-system algorithm - the closest concepts (IsolationForest,
OneClassSVM) are unsupervised novelty detectors with a different training
contract (predict "outlier"/"inlier" from unlabeled data, not P(y=1) from
labeled data) and would be an apples-to-oranges substitute, not a real
equivalent. Rather than omit that agent type or force in an unrelated
detector for the sake of a matching slot, this pipeline reuses the
project's OWN from-scratch AISAgent (via FromScratchAdapter below) for
that one component, with the exact same hyperparameters as the from-
scratch pipeline's AGENT_REGISTRY entry - a genuine like-for-like reuse,
not a separately-tuned one. This is the one deliberate, disclosed
exception to "everything here comes from a library": four of five agents
are real library implementations; AIS is not, because no library
equivalent exists.
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
    inputs of any size. Mirrors the same tractability mitigation used in
    the from-scratch SVMAgent (see agents.py's max_train_samples): even
    scikit-learn's optimized libsvm-based SVC does not scale to OULAD's
    real ~24k-row training set with an RBF kernel in a practical amount of
    time - and StackingClassifier's internal cv folds mean it would be
    refit several times over on top of that. Subclassing BaseEstimator
    (rather than hand-writing get_params/set_params) is what lets this
    drop straight into StackingClassifier's estimator list and survive its
    internal clone() calls correctly."""

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
    """Adapts a from-scratch BaseAgent-style model - fit(X, y) / predict(X)
    / predictProba(X) returning a 1D P(y=1) array, no get_params/set_params
    - to scikit-learn's estimator interface (predict_proba returning a 2D
    [P(y=0), P(y=1)] array, get_params/set_params via BaseEstimator) so it
    can sit in StackingClassifier's estimator list alongside genuine
    library models. Used specifically for AIS (see module docstring).
    Wraps the agent in this project's own BaggingEnsemble internally,
    matching how the from-scratch full-ensemble pipeline always bags AIS
    too - a standalone (unbagged) AIS is noticeably less stable.
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
    RandomForest) in a StandardScaler - the from-scratch agents each do
    their own internal z-scoring (see LogisticRegressionAgent/SVMAgent's
    self._mu/self._sigma), so this keeps the comparison fair rather than
    handicapping the library pipeline with unscaled inputs."""
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
    # the SAME hyperparameters as AGENT_REGISTRY["AIS"] (genuine reuse, not
    # a separately-tuned config), bagged internally. n_estimators here
    # controls that internal bagging, overridden per-call from the caller's
    # own --n-estimators (see run_sklearn_ensemble_pipeline/run_sklearn_single)
    # rather than hardcoded, since StackingClassifier's cv folds mean this
    # gets refit several times over - it's the most expensive slot here.
    "AIS": (lambda **kw: FromScratchAdapter(agent_factory=AISAgent,
                                             agent_kwargs=AGENT_REGISTRY["AIS"][1],
                                             n_estimators=kw.get("n_estimators", 10),
                                             random_state=kw.get("random_state", 0)),
             dict()),
}


SKLEARN_PARAM_GRIDS = {
    "LogisticRegression": ({"model__C": [0.01, 0.1, 1.0, 10.0]}, 10000),
    "SVM": ({"estimator__model__C": [0.5, 1.0, 4.0], "estimator__model__gamma": ["scale", 0.01]}, 1500),
    "RandomForest": ({"max_depth": [6, 8, 12], "min_samples_leaf": [1, 3]}, 3000),
    "MLP": ({"model__hidden_layer_sizes": [(32,), (64, 32), (32, 16)], "model__alpha": [1e-4, 1e-2]}, 3000),
}


def tune_sklearn_agents(estimators, X, y, cv=3, random_state=0, verbose=False):
    """GridSearchCV on each library model (on a row subsample, since SVC and
    MLP get slow) and set the winning parameters on it in place. AIS has no
    library equivalent, so it is tuned with the from-scratch
    hyperparameter_search and its kwargs are passed to the adapter. Returns
    {name: {"params": ..., "cv_acc": ...}} for reporting."""
    from sklearn.model_selection import GridSearchCV
    from scratch_pipeline.search import hyperparameter_search
    X = np.asarray(X); y = np.asarray(y).ravel()
    rng = np.random.default_rng(random_state)
    best = {}
    for name, model in estimators:
        if name == "AIS":
            _, b = hyperparameter_search("AIS", X, y, k_folds=cv, seed=random_state, verbose=False)
            model.set_params(agent_kwargs=b["agent_kwargs"])
            best[name] = {"params": b["params"], "cv_acc": b["mean_acc"]}
            continue
        grid, cap = SKLEARN_PARAM_GRIDS[name]
        idx = rng.choice(len(y), min(len(y), cap), replace=False)
        gs = GridSearchCV(model, grid, cv=cv, scoring="accuracy", refit=False, n_jobs=None)
        gs.fit(X[idx], y[idx])
        model.set_params(**gs.best_params_)
        best[name] = {"params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in gs.best_params_.items()},
                      "cv_acc": float(gs.best_score_)}
        if verbose:
            print(f"  tuned {name}: {best[name]}")
    return best


def run_sklearn_single(agent_name, X_train, y_train, X_test, y_test, random_state=0, n_estimators=10):
    """Fits ONE scikit-learn model (no ensembling) - the library-pipeline
    equivalent of main.py's --agent-mode single. Returns a dict shaped like
    the from-scratch single-agent path so both can be reported identically.
    n_estimators only affects the AIS slot's internal bagging (see
    FromScratchAdapter) - it's ignored by every other model here (note
    RandomForestClassifier has its OWN, unrelated n_estimators already
    fixed in SKLEARN_AGENT_REGISTRY; this parameter must never override
    that by accident, hence the isinstance check below rather than a
    blanket "if the model has this param" check)."""
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
                                    cv=5, n_estimators=10, random_state=0, tune=False):
    """Trains all five SKLEARN_AGENT_REGISTRY models and fuses them with
    scikit-learn's StackingClassifier (a logistic-regression meta-learner
    trained on out-of-fold predictions via internal cv-fold splitting) -
    the library-pipeline equivalent of MASController/FusionLayer. The cv
    parameter plays the same anti-leakage role our from-scratch fusion
    layer's OOB-based training does: the meta-learner never sees a base
    model's prediction on a row that base model was trained on.

    n_estimators only controls the AIS slot's internal bagging (see
    FromScratchAdapter / SKLEARN_AGENT_REGISTRY) - note this means AIS gets
    refit n_estimators times PER cv fold PLUS once more for the final
    refit, so it's by far the most expensive slot here; keep this modest
    for a quick run.

    Returns a dict shaped to match run_full_ensemble_pipeline's keys
    (y_pred, per_agent_test_acc, fusion_weights, agent_names) so the two
    pipelines' results can be displayed/compared identically.
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

    best_configs = tune_sklearn_agents(estimators, X_train, y_train, random_state=random_state) if tune else {}

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
    agent_cols, agent_proba_cols = [], []
    for name, fitted_model in zip([n for n, _ in estimators], stack.estimators_):
        preds = fitted_model.predict(X_test)
        per_agent_test_acc[name] = float(np.mean(preds == y_test))
        agent_cols.append(np.asarray(preds).astype(int))
        agent_proba_cols.append(fitted_model.predict_proba(X_test)[:, 1])

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
        "best_configs": best_configs,
        "agent_test_pred": np.column_stack(agent_cols),
        "agent_test_proba": np.column_stack(agent_proba_cols),
        "predict_proba": lambda X: stack.predict_proba(X)[:, 1],
    }
