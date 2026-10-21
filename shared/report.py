"""Turns a `last_results` dict into structured, display-ready sections so the
GUI (and any other front-end) can render readable tables and plain-language
explanations instead of a raw text dump. No Tk dependency - fully testable.
"""
import math


def _pct(x):
    return f"{x:.1%}"


def headline(r):
    """Plain-language verdict sentence + 4 metric cards."""
    cards = [
        ("Accuracy", f"{r['accuracy']:.1%}", "Share of all students classified correctly"),
        ("Precision", f"{r['precision']:.1%}", "Of students flagged at-risk, how many truly were"),
        ("Recall", f"{r['recall']:.1%}", "Of truly at-risk students, how many were caught"),
        ("F1 score", f"{r['f1']:.3f}", "Balance of precision and recall"),
    ]
    sentence = (f"On {r['test_size']:,} unseen students, the model caught {r['tp']:,} of "
                f"{r['tp'] + r['fn']:,} at-risk students ({r['recall']:.0%}) and raised "
                f"{r['fp']:,} false alarms.")
    return sentence, cards


def confusion_rows(r):
    return (["", "Predicted at-risk", "Predicted not at-risk"],
            [["Actually at-risk", f"{r['tp']:,}  (caught)", f"{r['fn']:,}  (missed)"],
             ["Actually not at-risk", f"{r['fp']:,}  (false alarm)", f"{r['tn']:,}  (correct)"]])


def agent_rows(r):
    """Per-agent table: accuracy, fusion weight, feature view, delta vs fused."""
    cols = ["Agent", "Test accuracy", "vs. ensemble", "Fusion weight", "Agrees with ensemble"]
    ov = r.get("agent_feature_overrides") or {}
    agree = (r.get("agreement") or {}).get("fusion_vs_agent_agreement", {})
    rows = []
    best = max(r["per_agent_test_acc"], key=r["per_agent_test_acc"].get)
    for name, acc in r["per_agent_test_acc"].items():
        w = r.get("fusion_weights", {}).get(name)
        rows.append([name + ("  ★" if name == best else ""), f"{acc:.4f}", f"{acc - r['accuracy']:+.4f}",
                     "-" if w is None else f"{w:+.4f}",
                     f"{agree[name]:.1%}" if name in agree else "-"])
    return cols, rows


def fusion_note(r):
    best = max(r["per_agent_test_acc"], key=r["per_agent_test_acc"].get)
    b = r["per_agent_test_acc"][best]
    lift = r["accuracy"] - b
    txt = (f"Best single agent: {best} ({b:.4f}). Ensemble: {r['accuracy']:.4f} "
           f"({'+' if lift >= 0 else ''}{lift:.4f}). ")
    txt += ("Fusion helped." if lift > 0.002 else "Fusion made little difference - "
            "the agents mostly make the same mistakes (see Agreement tab)." if lift > -0.002
            else "Fusion did worse than the best single agent.")
    if r.get("fusion_type") == "gating":
        txt += " Weights shown are MEAN gate weights; real weights vary per student."
    return txt


def agreement_tables(rep):
    """Returns dict of (cols, rows) tables from an agreement_report dict."""
    names = rep["agent_names"]
    def matrix(key, fmt="{:.2f}"):
        rows = [[names[i]] + [fmt.format(v) for v in rep[key][i]] for i in range(len(names))]
        return ["Agent"] + names, rows
    summary = [
        ["Mean pairwise agreement", _pct(rep["mean_pairwise_agreement"]), "How often two agents give the same label"],
        ["Mean Cohen's kappa", f"{rep['mean_pairwise_kappa']:.3f}", "Agreement beyond chance (1 = identical)"],
        ["Fleiss' kappa (all agents)", f"{rep['fleiss_kappa']:.3f}", "Chance-corrected agreement across all agents"],
        ["Mean disagreement", _pct(rep["mean_disagreement"]), "Pairs where exactly one agent is right"],
        ["Mean double-fault", _pct(rep["mean_double_fault"]), "Pairs where both agents are wrong"],
        ["Mean Q-statistic", f"{rep['mean_q_statistic']:.3f}", "+1 = right/wrong together, 0 = independent"],
        ["Unanimous votes", _pct(rep["unanimous_rate"]), "All agents give the same label"],
        ["Accuracy when unanimous", _pct(rep["unanimous_accuracy"]) if rep["unanimous_accuracy"] == rep["unanimous_accuracy"] else "n/a", ""],
        ["Ensemble acc. on split votes", _pct(rep["split_vote_accuracy"]) if rep["split_vote_accuracy"] == rep["split_vote_accuracy"] else "n/a", "Cases agents disagree on"],
        ["Majority-vote accuracy", _pct(rep["majority_vote_accuracy"]), "Simple unweighted vote, for comparison"],
        ["Oracle accuracy", _pct(rep["oracle_accuracy"]), "Upper bound: right if ANY agent is right"],
        ["Misclassified by every agent", _pct(rep["all_wrong_rate"]), "Hard cases fusion cannot fix"],
        ["Fusion rescued", _pct(rep["rescued_by_fusion"]), f"Ensemble right where {rep['best_agent']} was wrong"],
        ["Fusion broke", _pct(rep["broken_by_fusion"]), f"Ensemble wrong where {rep['best_agent']} was right"],
    ]
    if "mean_probability_correlation" in rep:
        summary.insert(5, ["Mean probability correlation", f"{rep['mean_probability_correlation']:.3f}",
                           "Pearson r between agents' predicted probabilities"])
    out = {"summary": (["Metric", "Value", "Meaning"], summary),
           "agreement": matrix("pairwise_agreement"),
           "kappa": matrix("pairwise_kappa")}
    if "probability_correlation" in rep:
        out["prob_corr"] = matrix("probability_correlation")
    return out


def fairness_tables(fr):
    """Rows for a fairness DataFrame; returns (cols, rows, gap_row_indices)."""
    cols = ["Attribute", "Group", "Recall (TPR)", "False-alarm rate (FPR)", "Flagged at-risk"]
    rows, gaps = [], []
    for _, row in fr.iterrows():
        is_gap = row["group"] == "GAP(max-min)"
        rows.append([row["attribute"], "Largest gap" if is_gap else row["group"],
                     f"{row['recall_tpr']:.3f}", f"{row['fpr']:.3f}", f"{row['positive_rate']:.3f}"])
        if is_gap:
            gaps.append(len(rows) - 1)
    return cols, rows, gaps


def fairness_note(fr):
    g = fr[fr["group"] == "GAP(max-min)"]
    if g.empty:
        return ""
    w = g.loc[g["recall_tpr"].idxmax()]
    return (f"Largest recall gap: {w['attribute']} ({w['recall_tpr']:.2f}). A gap near 0 means the model "
            f"treats groups alike; large gaps mean some groups' at-risk students are missed far more often.")


def tuning_lines(r):
    out = []
    m = r.get("best_mlp_config")
    if m:
        out.append(f"MLP architecture search: hidden layers {tuple(m['hidden_layer_sizes'])}, lr={m['lr']}, "
                   f"L2={m['l2']} (CV accuracy {m['mean_acc']:.4f}, F1 {m['mean_f1']:.4f}, {m['n_params']} parameters)")
    f = r.get("best_rf_config")
    if f:
        out.append(f"Random-forest search: depth {f['max_depth']}, min split {f['min_samples_split']}, "
                   f"min leaf {f['min_samples_leaf']}, class weight {f['class_weight']}, "
                   f"{f['n_estimators']} trees (CV accuracy {f['mean_acc']:.4f})")
    for name, b in (r.get("best_configs") or {}).items():
        ps = ", ".join(f"{k}={v}" for k, v in b["params"].items())
        out.append(f"{name} search: {ps} (CV accuracy {b['mean_acc']:.4f})")
    return out


# ---- explainability (SHAP) presentation ----
def explain_global_rows(pairs, top=10):
    """Rows: rank, feature, mean|SHAP|, share of total influence, bar."""
    total = sum(abs(v) for _, v in pairs) or 1e-12
    mx = max((abs(v) for _, v in pairs[:top]), default=1e-12) or 1e-12
    rows = []
    for i, (n, v) in enumerate(pairs[:top], 1):
        rows.append([i, n, f"{abs(v):.4f}", f"{abs(v) / total:.1%}", "█" * max(1, int(round(14 * abs(v) / mx)))])
    return ["#", "Feature", "Influence (mean |SHAP|)", "Share of total", ""], rows


def explain_instance_rows(pairs, feature_names=None, x=None, top=10):
    """Rows: feature, this student's (scaled) value, effect direction, contribution."""
    idx = {n: i for i, n in enumerate(feature_names)} if feature_names is not None else {}
    rows = []
    for n, v in pairs[:top]:
        val = f"{x[idx[n]]:.3g}" if x is not None and n in idx else "-"
        rows.append([n, val, "raises risk ▲" if v > 0 else "lowers risk ▼" if v < 0 else "none", f"{v:+.4f}"])
    return ["Feature", "Student's value (scaled)", "Effect on prediction", "Contribution"], rows


def explain_narrative(global_pairs, instance_pairs=None, proba=None, top=3):
    out = []
    if global_pairs:
        tot = sum(abs(v) for _, v in global_pairs) or 1e-12
        names = ", ".join(n for n, _ in global_pairs[:top])
        share = sum(abs(v) for _, v in global_pairs[:top]) / tot
        out.append(f"Across the test set, the predictions are driven mostly by {names} "
                   f"(together {share:.0%} of total feature influence).")
    if instance_pairs:
        up = [n for n, v in instance_pairs if v > 0][:2]
        down = [n for n, v in instance_pairs if v < 0][:2]
        s = "For the example student"
        if proba is not None:
            s += f" (predicted at-risk probability {proba:.0%})"
        s += ":"
        if up:
            s += f" {', '.join(up)} pushed the prediction towards at-risk"
        if down:
            s += ("; " if up else " ") + f"{', '.join(down)} pushed it away from at-risk"
        out.append(s + ".")
    return out


def trust_rows(pairs):
    return ["Agent", "Trust for this student"], [[n, f"{w:.1%}"] for n, w in pairs]


# ======================================================================
# Structured views used by the redesigned GUI (all Tk-free, all testable)
# ======================================================================
def severity(gap):
    """Fairness-gap severity -> (label, tone). Thresholds are rules of thumb."""
    if gap < 0.10:
        return "small", "good"
    if gap < 0.20:
        return "moderate", "warn"
    return "large", "bad"


def key_findings(r):
    """3-6 plain-language takeaways, each (tone, sentence), for the Overview."""
    out = []
    accs = r.get("per_agent_test_acc") or {}
    if accs:
        best = max(accs, key=accs.get)
        lift = r["accuracy"] - accs[best]
        if lift > 0.002:
            out.append(("good", f"Fusion helped: the ensemble ({r['accuracy']:.1%}) beats its best single agent, "
                                f"{best} ({accs[best]:.1%}), by {lift * 100:.1f} points."))
        elif lift > -0.002:
            out.append(("info", f"Fusion made little difference: the ensemble ({r['accuracy']:.1%}) is level with "
                                f"its best agent, {best} ({accs[best]:.1%})."))
        else:
            out.append(("warn", f"Fusion did worse than its best single agent: ensemble {r['accuracy']:.1%} vs "
                                f"{best} {accs[best]:.1%}."))
    ws = r.get("fusion_weights") or {}
    if ws and r.get("fusion_type") == "gating":
        top = max(ws, key=lambda k: abs(ws[k]))
        if ws[top] > 0.6:
            out.append(("warn", f"The gate leans heavily on one agent: {top} gets {ws[top]:.0%} of the average weight."))
    ag = r.get("agreement")
    if ag:
        k = ag["mean_pairwise_kappa"]
        word = ("almost identical" if k > 0.8 else "largely similar" if k > 0.6
                else "moderately different" if k > 0.4 else "quite diverse")
        out.append(("info", f"Agents are {word} (mean kappa {k:.2f}); an oracle that always picked the right agent "
                            f"would reach {ag['oracle_accuracy']:.1%}."))
    fr = r.get("fairness_report")
    if fr is not None and len(fr):
        g = fr[fr["group"] == "GAP(max-min)"]
        if len(g):
            w = g.loc[g["recall_tpr"].idxmax()]
            label, tone = severity(float(w["recall_tpr"]))
            out.append((tone, f"Largest fairness gap: {w['attribute']} - at-risk students in the best-served group "
                              f"are caught {w['recall_tpr'] * 100:.0f} points more often than in the worst-served "
                              f"group ({label})."))
    else:
        out.append(("muted", "Fairness is not available for this run (no protected attributes loaded)."))
    miss = r["fn"] / max(r["tp"] + r["fn"], 1)
    if miss > 0.25:
        out.append(("warn", f"{miss:.0%} of truly at-risk students are missed - consider a lower decision "
                            f"threshold if recall matters more than false alarms."))
    return out


def agents_view(r):
    """Rows for the Models table. Cells are (text, number) where a number drives a bar/delta."""
    accs = r["per_agent_test_acc"]
    best = max(accs, key=accs.get)
    ws = r.get("fusion_weights") or {}
    agree = (r.get("agreement") or {}).get("fusion_vs_agent_agreement", {})
    wmax = max((abs(v) for v in ws.values()), default=1.0) or 1.0
    cols = ["Agent", "Test accuracy", "vs. ensemble", "Fusion weight", "Same label as ensemble"]
    rows = []
    for name, acc in sorted(accs.items(), key=lambda kv: -kv[1]):
        d = acc - r["accuracy"]
        w = ws.get(name)
        rows.append([name + ("  ★" if name == best else ""), (f"{acc:.1%}", acc), (f"{d * 100:+.1f} pts", d),
                     (f"{w:.1%}" if w is not None else "-", w / wmax if w is not None else 0.0),
                     (f"{agree[name]:.0%}", agree[name]) if name in agree else "-"])
    return cols, rows, wmax


AGREEMENT_GROUPS = [
    ("How much the agents agree", [
        ("Mean pairwise agreement", "mean_pairwise_agreement", "pct", "How often two agents give the same label"),
        ("Mean Cohen's kappa", "mean_pairwise_kappa", "3", "Agreement beyond chance (1 = identical, 0 = chance)"),
        ("Fleiss' kappa", "fleiss_kappa", "3", "The same, measured across all agents at once"),
        ("Unanimous votes", "unanimous_rate", "pct", "Students all agents label identically"),
        ("Mean probability correlation", "mean_probability_correlation", "3", "How alike the agents' risk scores are"),
    ]),
    ("How differently they err", [
        ("Mean disagreement", "mean_disagreement", "pct", "Pairs where exactly one agent is right"),
        ("Mean double-fault", "mean_double_fault", "pct", "Pairs where both agents are wrong"),
        ("Mean Q-statistic", "mean_q_statistic", "3", "+1 = right and wrong together, 0 = independent"),
    ]),
    ("What fusion gains", [
        ("Majority-vote accuracy", "majority_vote_accuracy", "pct", "A plain unweighted vote, for comparison"),
        ("Oracle accuracy", "oracle_accuracy", "pct", "Upper bound: right if ANY agent is right"),
        ("Wrong for every agent", "all_wrong_rate", "pct", "Hard cases no fusion can fix"),
        ("Fusion rescued", "rescued_by_fusion", "pct", "Ensemble right where the best agent was wrong"),
        ("Fusion broke", "broken_by_fusion", "pct", "Ensemble wrong where the best agent was right"),
    ]),
]


def agreement_groups(rep):
    """[(group title, [[metric, (text, value), meaning], ...]), ...] - skips metrics the report lacks."""
    out = []
    for title, items in AGREEMENT_GROUPS:
        rows = []
        for label, key, fmt, meaning in items:
            if key not in rep:
                continue
            v = rep[key]
            if v != v:
                continue
            rows.append([label, f"{v:.1%}" if fmt == "pct" else f"{v:.3f}", meaning])
        out.append((title, rows))
    return out


def fairness_view(fr):
    """Returns (summary_rows, blocks).
    summary_rows: [attribute, (tpr_gap_text, gap), (fpr_gap_text, gap), (severity, tone)] sorted worst first
    blocks: [(attribute, tpr_gap, fpr_gap, [[group, (tpr, v), (fpr, v), (flagged, v)], ...])]"""
    gaps = fr[fr["group"] == "GAP(max-min)"].set_index("attribute")
    order = sorted(gaps.index, key=lambda a: -float(gaps.loc[a, "recall_tpr"]))
    summary, blocks = [], []
    for a in order:
        tg, fg = float(gaps.loc[a, "recall_tpr"]), float(gaps.loc[a, "fpr"])
        label, tone = severity(tg)
        summary.append([a, (f"{tg:.2f}", tg), (f"{fg:.2f}", fg), (label, tone)])
        sub = fr[(fr["attribute"] == a) & (fr["group"] != "GAP(max-min)")]
        rows = [[str(g.group), (f"{g.recall_tpr:.2f}", float(g.recall_tpr)), (f"{g.fpr:.2f}", float(g.fpr)),
                 (f"{g.positive_rate:.2f}", float(g.positive_rate))] for g in sub.itertuples()]
        blocks.append((a, tg, fg, rows))
    return summary, blocks


def _plain_table(cols, rows, indent="  "):
    rows = [[c[0] if isinstance(c, tuple) else str(c) for c in row] for row in rows]
    widths = [max(len(str(x)) for x in col) for col in zip(cols, *rows)]
    fmt = lambda row: indent + "  ".join(str(x).ljust(w) for x, w in zip(row, widths)).rstrip()
    return "\n".join([fmt(cols), indent + "  ".join("-" * w for w in widths)] + [fmt(r) for r in rows])


def text_report(r, explain_lines=None):
    """A clean plain-text version of everything shown in the GUI (used by Export Report)."""
    L = []
    L += [r["model_name"], "=" * len(r["model_name"]),
          f"{r.get('feature_set', 'Full features')} ({r.get('n_features', '?')} features) | "
          f"{r['train_size']:,} train / {r['test_size']:,} test | {r['time_sec']:.1f}s", ""]
    sentence, cards = headline(r)
    L += [sentence, ""]
    L += [f"  {t:<10}{v:>8}   {d}" for t, v, d in cards]
    L += ["", "Confusion matrix", _plain_table(*confusion_rows(r)), ""]
    if r.get("mode") == "fusion":
        cols, rows, _ = agents_view(r)
        L += ["Individual models", _plain_table(cols, rows), "", fusion_note(r), ""]
        for line in tuning_lines(r):
            L.append("  - " + line)
        ag = r.get("agreement")
        if ag:
            L += ["", "Agent agreement"]
            for title, rows in agreement_groups(ag):
                L += ["", title, _plain_table(["Metric", "Value", "Meaning"], rows)]
    fr = r.get("fairness_report")
    L += ["", "Fairness"]
    if fr is not None and len(fr):
        summary, blocks = fairness_view(fr)
        L += [fairness_note(fr), "", _plain_table(["Attribute", "Recall gap", "False-alarm gap", "Severity"], summary)]
        for a, tg, fg, rows in blocks:
            L += ["", f"{a}", _plain_table(["Group", "Recall", "False-alarm", "Flagged"], rows)]
    else:
        L.append("  Not available for this run (no protected attributes loaded).")
    if explain_lines:
        L += ["", "Explainability"] + ["  - " + x for x in explain_lines]
    return "\n".join(L) + "\n"


def explain_view(global_pairs, instance_pairs, feature_names=None, x=None, proba=None, gate_pairs=None, top=10):
    """Tk-free data for the Explain tab: dict with story, global, instance, trust."""
    total = sum(abs(v) for _, v in global_pairs) or 1e-12
    mx = max((abs(v) for _, v in global_pairs[:top]), default=1e-12) or 1e-12
    glob = [(n, f"{abs(v):.4f}", abs(v) / mx, f"{abs(v) / total:.1%}") for n, v in global_pairs[:top]]
    idx = {n: i for i, n in enumerate(feature_names)} if feature_names is not None else {}
    imax = max((abs(v) for _, v in instance_pairs[:top]), default=1e-12) or 1e-12
    inst = []
    for n, v in instance_pairs[:top]:
        val = f"{x[idx[n]]:.3g}" if x is not None and n in idx else "-"
        inst.append((n, val, "raises risk" if v > 0 else "lowers risk" if v < 0 else "none", f"{v:+.4f}", v / imax))
    trust = [(n, f"{w:.0%}", float(w)) for n, w in gate_pairs] if gate_pairs else None
    return {"story": explain_narrative(global_pairs, instance_pairs, proba), "global": glob,
            "instance": inst, "trust": trust}
