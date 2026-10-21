import numpy as np
import pandas as pd
import pytest

from shared import agreement_report, cohen_kappa, fleiss_kappa, interpret_agreement
from shared import report, visuals
from scratch_pipeline import AGENT_REGISTRY, MASController
from shared import make_synthetic_dataset


def test_kappa_identical_and_known_value():
    a = np.array([1, 1, 0, 0, 1, 0])
    assert cohen_kappa(a, a) == pytest.approx(1.0)
    # Known 2x2 example: po=0.7, pe=0.5 -> kappa=0.4
    x = np.array([1] * 25 + [1] * 10 + [0] * 5 + [0] * 10 + [1] * 0 + [0] * 0)
    a = np.array([1] * 20 + [1] * 5 + [0] * 5 + [0] * 20)
    b = np.array([1] * 20 + [0] * 5 + [1] * 5 + [0] * 20)
    assert cohen_kappa(a, b) == pytest.approx(0.6)  # po=.8, pe=.5


def test_fleiss_perfect_and_chance():
    P = np.tile(np.array([[0], [1], [0], [1]]), (1, 3))
    assert fleiss_kappa(P) == pytest.approx(1.0)
    rng = np.random.default_rng(0)
    R = rng.integers(0, 2, (4000, 4))
    assert abs(fleiss_kappa(R)) < 0.05


def test_agreement_report_hand_example():
    y = np.array([1, 1, 0, 0])
    P = np.array([[1, 1, 0],    # all right
                  [1, 0, 0],    # A right
                  [0, 0, 1],    # A,B right
                  [1, 1, 1]])   # all wrong
    fused = np.array([1, 1, 0, 1])
    r = agreement_report(["A", "B", "C"], P, y, fused)
    assert r["oracle_accuracy"] == pytest.approx(0.75)
    assert r["all_wrong_rate"] == pytest.approx(0.25)
    assert r["unanimous_rate"] == pytest.approx(0.25)   # only the last row is unanimous
    assert r["pairwise_agreement"][0][1] == pytest.approx(0.75)
    assert r["ensemble_accuracy"] == pytest.approx(0.75)
    assert r["fusion_vs_agent_agreement"]["A"] == pytest.approx(1.0)  # A == fused on every row
    # symmetric matrices, unit diagonal
    for k in ("pairwise_agreement", "pairwise_kappa"):
        M = np.array(r[k]); assert np.allclose(M, M.T); assert np.allclose(np.diag(M), 1)
    assert all(isinstance(l, str) for l in interpret_agreement(r))


def test_agreement_identical_agents_has_kappa_one():
    y = np.random.default_rng(1).integers(0, 2, 200)
    p = np.random.default_rng(2).integers(0, 2, 200)
    r = agreement_report(["a", "b"], np.column_stack([p, p]), y, p,
                         agent_proba=np.column_stack([p * 1.0, p * 1.0]) + np.random.default_rng(3).normal(0, .1, (200, 2)))
    assert r["mean_pairwise_kappa"] == pytest.approx(1.0)
    assert r["mean_disagreement"] == 0
    assert r["mean_probability_correlation"] > 0.5


@pytest.fixture(scope="module")
def fitted():
    X, y = make_synthetic_dataset()
    reg = {k: (c, dict(kw)) for k, (c, kw) in AGENT_REGISTRY.items()}
    reg["MLP"][1]["n_epochs"] = 20
    ctrl = MASController(agent_registry=reg, use_bagging=False, fusion_type="gating")
    res = ctrl.run_pipeline(X[:200], y[:200], X[200:], y[200:])
    return X, y, ctrl, res


def test_controller_exposes_agent_predictions(fitted):
    X, y, ctrl, res = fitted
    assert res["agent_test_pred"].shape == (len(y) - 200, 5)
    assert set(np.unique(res["agent_test_pred"])) <= {0, 1}
    for k, n in enumerate(res["agent_names"]):
        assert np.mean(res["agent_test_pred"][:, k] == y[200:]) == pytest.approx(res["per_agent_test_acc"][n])
    r = agreement_report(res["agent_names"], res["agent_test_pred"], y[200:], res["y_pred"],
                         res["agent_test_proba"])
    assert r["ensemble_accuracy"] == pytest.approx(np.mean(res["y_pred"] == y[200:]))


def test_report_tables(fitted):
    X, y, ctrl, res = fitted
    ag = agreement_report(res["agent_names"], res["agent_test_pred"], y[200:], res["y_pred"],
                          res["agent_test_proba"])
    t = report.agreement_tables(ag)
    assert len(t["agreement"][1]) == 5 and len(t["agreement"][0]) == 6
    r = {"accuracy": .8, "precision": .7, "recall": .6, "f1": .65, "tp": 6, "fn": 4, "fp": 3, "tn": 7,
         "test_size": 20, "per_agent_test_acc": {"A": .7, "B": .75}, "fusion_weights": {"A": .1, "B": .9},
         "agreement": None}
    s, cards = report.headline(r)
    assert "6 of 10" in s and len(cards) == 4
    cols, rows = report.agent_rows(r)
    assert rows[1][0].endswith("★")
    fr = pd.DataFrame({"attribute": ["g", "g", "g"], "group": ["a", "b", "GAP(max-min)"],
                       "recall_tpr": [.9, .5, .4], "fpr": [.1, .2, .1], "positive_rate": [.5, .4, .1]})
    cols, rows, gaps = report.fairness_tables(fr)
    assert gaps == [2] and "g" in report.fairness_note(fr)


def test_mlp_diagram(fitted):
    X, y, ctrl, res = fitted
    mlp = ctrl.agents["MLP"]
    fig = visuals.plot_mlp_network(mlp, [f"f{i}" for i in range(X.shape[1])])
    assert len(fig.axes) == 1
    # untrained network is rejected
    from scratch_pipeline.agents import MLPAgent
    with pytest.raises(ValueError):
        visuals.plot_mlp_network(MLPAgent())


def test_tree_and_forest_diagrams(fitted):
    X, y, ctrl, res = fitted
    rf = ctrl.agents["RandomForest"]
    tree = rf.trees[0]
    n_nodes, n_leaves, depth = visuals.tree_stats(tree)
    assert n_nodes == 2 * n_leaves - 1 and depth <= rf.max_depth
    fig = visuals.plot_forest_tree(tree, [f"f{i}" for i in range(X.shape[1])], max_depth=2)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert any("<=" in t for t in texts)
    visuals.plot_forest_overview(rf)


def test_explain_report_helpers():
    gp = [("a", 0.4), ("b", 0.3), ("c", 0.2), ("d", 0.1)]
    cols, rows = report.explain_global_rows(gp, top=3)
    assert len(rows) == 3 and rows[0][3] == "40.0%" and len(cols) == 5
    ip = [("a", 0.2), ("b", -0.1), ("zzz", 0.05)]
    cols, rows = report.explain_instance_rows(ip, ["a", "b", "c"], np.array([1.0, 2.0, 3.0]))
    assert rows[0][2].startswith("raises") and rows[1][2].startswith("lowers") and rows[2][1] == "-"
    story = report.explain_narrative(gp, ip, 0.8)
    assert "a, b, c" in story[0] and "80%" in story[1]
    assert report.trust_rows([("x", .5)])[1] == [["x", "50.0%"]]
