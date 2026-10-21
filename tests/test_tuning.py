import numpy as np
import pytest

from scratch_pipeline import (AGENT_REGISTRY, MASController, hyperparameter_search, DEFAULT_GRIDS)
from shared import make_synthetic_dataset, report


@pytest.fixture(scope="module")
def data():
    return make_synthetic_dataset()


@pytest.mark.parametrize("name", ["LogisticRegression", "SVM", "AIS"])
def test_hyperparameter_search_leaderboard(data, name):
    X, y = data
    lb, best = hyperparameter_search(name, X, y, verbose=False)
    assert len(lb) == len(DEFAULT_GRIDS[name])
    accs = [e["mean_acc"] for e in lb]
    assert accs == sorted(accs, reverse=True) and best is lb[0]
    # best kwargs are the registry base kwargs overridden by the winning params
    for k, v in best["params"].items():
        assert best["agent_kwargs"][k] == v
    cls = AGENT_REGISTRY[name][0]
    cls(**best["agent_kwargs"])   # constructible


def test_search_is_deterministic_and_subsamples(data):
    X, y = data
    a = hyperparameter_search("LogisticRegression", X, y, max_rows=150, verbose=False)[1]
    b = hyperparameter_search("LogisticRegression", X, y, max_rows=150, verbose=False)[1]
    assert a["params"] == b["params"] and a["mean_acc"] == b["mean_acc"] and a["n_rows"] == 150


def test_custom_grid_and_unknown_agent(data):
    X, y = data
    lb, best = hyperparameter_search("LogisticRegression", X, y, grid=[dict(l2=1e-3)], verbose=False)
    assert len(lb) == 1
    with pytest.raises(KeyError):
        hyperparameter_search("Nope", X, y, verbose=False)


def test_controller_auto_tune_all(data):
    X, y = data
    reg = {k: (c, dict(kw)) for k, (c, kw) in AGENT_REGISTRY.items()}
    reg["MLP"][1]["n_epochs"] = 15
    ctrl = MASController(agent_registry=reg, use_bagging=False, auto_tune_all=True,
                         mlp_search_kwargs=dict(architectures=[(8,)], k_folds=3, n_epochs=15),
                         rf_search_kwargs=dict(configs=[dict(max_depth=6, min_samples_split=4,
                                                             min_samples_leaf=1, class_weight=None)],
                                               final_n_estimators=5))
    res = ctrl.run_pipeline(X[:220], y[:220], X[220:], y[220:])
    assert set(res["best_configs"]) == {"LogisticRegression", "SVM", "AIS"}
    assert res["best_mlp_config"] is not None and res["best_rf_config"] is not None
    # the trained LR agent really uses the tuned hyperparameters
    best_l2 = res["best_configs"]["LogisticRegression"]["params"]["l2"]
    assert ctrl.agents["LogisticRegression"].l2 == best_l2
    # registry passed in by the caller is not mutated
    assert reg["LogisticRegression"][1] == AGENT_REGISTRY["LogisticRegression"][1]
    r = {"best_configs": res["best_configs"]}
    assert any("SVM search" in l for l in report.tuning_lines(r))


def test_sklearn_tuning(data):
    from sklearn_pipeline import run_sklearn_ensemble_pipeline
    X, y = data
    res = run_sklearn_ensemble_pipeline(X[:240], y[:240], X[240:], y[240:], cv=3, n_estimators=2, tune=True)
    assert set(res["best_configs"]) == {"LogisticRegression", "SVM", "RandomForest", "MLP", "AIS"}
    assert res["best_configs"]["RandomForest"]["cv_acc"] > 0.5
