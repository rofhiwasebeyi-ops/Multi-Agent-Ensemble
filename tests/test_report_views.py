"""Tk-free tests for the structured views behind the GUI's results tabs,
plus a Tk smoke test that is skipped when no display/tkinter is available."""
import numpy as np
import pandas as pd
import pytest

from shared import report as rv


def _fairness_df():
    rows = []
    for a, gs in {"education": [("none", .3, .2), ("HE", .85, .7)], "gender": [("F", .72, .4), ("M", .70, .42)]}.items():
        for g, t, f in gs:
            rows.append(dict(attribute=a, group=g, recall_tpr=t, fpr=f, positive_rate=(t + f) / 2))
        ts, fs = [x[1] for x in gs], [x[2] for x in gs]
        rows.append(dict(attribute=a, group="GAP(max-min)", recall_tpr=max(ts) - min(ts), fpr=max(fs) - min(fs), positive_rate=.1))
    return pd.DataFrame(rows)


def _results(with_fair=True):
    names = ["LR", "MLP"]
    return dict(model_name="M", feature_set="Full", n_features=3, train_size=10, test_size=8, time_sec=1.0,
                accuracy=.75, f1=.7, precision=.8, recall=.7, tp=3, fn=1, fp=1, tn=3, oob_score=float("nan"),
                mode="fusion", fusion_type="gating", per_agent_test_acc={"LR": .6, "MLP": .74},
                fusion_weights={"LR": .1, "MLP": .9},
                agreement=dict(agent_names=names, mean_pairwise_kappa=.4, oracle_accuracy=.9,
                               fusion_vs_agent_agreement={"LR": .7, "MLP": .95}),
                fairness_report=_fairness_df() if with_fair else None)


def test_severity_thresholds():
    assert rv.severity(.05) == ("small", "good")
    assert rv.severity(.15) == ("moderate", "warn")
    assert rv.severity(.5) == ("large", "bad")


def test_fairness_view_sorted_worst_first():
    summary, blocks = rv.fairness_view(_fairness_df())
    assert [s[0] for s in summary] == ["education", "gender"]
    assert summary[0][3] == ("large", "bad") and summary[1][3] == ("small", "good")
    assert [b[0] for b in blocks] == ["education", "gender"]
    assert len(blocks[0][3]) == 2          # gap row excluded from group rows


def test_agents_view_orders_by_accuracy_and_marks_best():
    cols, rows, wmax = rv.agents_view(_results())
    assert rows[0][0].startswith("MLP") and "★" in rows[0][0]
    assert rows[0][2][1] == pytest.approx(.74 - .75)
    assert wmax == pytest.approx(.9)


def test_key_findings_cover_fusion_agreement_fairness():
    tones = dict((t, x) for t, x in rv.key_findings(_results()))
    text = " ".join(tones.values())
    assert "Fusion" in text and "kappa" in text and "education" in text
    assert "not available" in " ".join(x for _, x in rv.key_findings(_results(False)))


def test_text_report_contains_all_sections():
    txt = rv.text_report(_results(), ["story line"])
    for needle in ("Confusion matrix", "Individual models", "Fairness", "education", "story line"):
        assert needle in txt


def test_explain_view_shapes():
    gp = [("a", .5), ("b", -.25)]
    v = rv.explain_view(gp, [("a", .2), ("b", -.1)], ["a", "b"], [1.0, 2.0], proba=.6, gate_pairs=[("MLP", .9)])
    assert v["global"][0][2] == pytest.approx(1.0)
    assert v["instance"][0][2] == "raises risk" and v["instance"][1][2] == "lowers risk"
    assert v["trust"] == [("MLP", "90%", .9)]


def test_widgets_smoke():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    import gui_widgets as gw
    page = gw.ScrollPage(root)
    page.pack(fill="both", expand=True)
    gw.heading(page, "h", "sub")
    gw.metric_cards(page, [("A", "1", "d")])
    gw.confusion_grid(page, 1, 2, 3, 4)
    gw.table(page, ["x", "y", "z", "w"], [["a", ("1", .5), ("+1", .1), ("t", "good")]],
             kinds=["text", "bar", "delta", "tag"])
    root.update()
    page.clear()
    root.destroy()
