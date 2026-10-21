"""Ensemble / component-classifier agreement and diversity metrics.

Pure NumPy. Input is a matrix of per-agent hard predictions (n_samples x
n_agents, values 0/1), optionally per-agent probabilities, the true labels,
and the fused ensemble's predictions. Standard measures from Kuncheva &
Whitaker (2003), "Measures of diversity in classifier ensembles".
"""
import numpy as np


def _pair_counts(c1, c2):
    """Oracle-output contingency counts for two classifiers.
    c1, c2: boolean 'correct' vectors. Returns N11, N10, N01, N00 where the
    first digit is classifier 1 correct(1)/wrong(0)."""
    n11 = int(np.sum(c1 & c2))
    n10 = int(np.sum(c1 & ~c2))
    n01 = int(np.sum(~c1 & c2))
    n00 = int(np.sum(~c1 & ~c2))
    return n11, n10, n01, n00


def cohen_kappa(a, b):
    """Cohen's kappa between two label vectors (binary or multiclass)."""
    a, b = np.asarray(a), np.asarray(b)
    po = float(np.mean(a == b))
    pe = sum(float(np.mean(a == c)) * float(np.mean(b == c)) for c in np.unique(np.r_[a, b]))
    return 1.0 if pe >= 1.0 else (po - pe) / (1.0 - pe)


def fleiss_kappa(preds):
    """Fleiss' kappa for n_samples x n_raters hard labels."""
    preds = np.asarray(preds)
    n, m = preds.shape
    cats = np.unique(preds)
    counts = np.stack([(preds == c).sum(axis=1) for c in cats], axis=1)  # n x k
    p_i = (np.sum(counts ** 2, axis=1) - m) / (m * (m - 1))
    p_bar = float(np.mean(p_i))
    p_j = counts.sum(axis=0) / (n * m)
    pe = float(np.sum(p_j ** 2))
    return 1.0 if pe >= 1.0 else (p_bar - pe) / (1.0 - pe)


def agreement_report(agent_names, agent_preds, y_true, fused_pred, agent_proba=None):
    """Compute all agreement/diversity metrics. Returns a JSON-friendly dict."""
    P = np.asarray(agent_preds).astype(int)
    y = np.asarray(y_true).astype(int).ravel()
    f = np.asarray(fused_pred).astype(int).ravel()
    n, m = P.shape
    names = list(agent_names)
    correct = (P == y[:, None])

    pair_agree = np.eye(m)
    pair_kappa = np.eye(m)
    pair_q = np.eye(m)
    pair_disagree = np.zeros((m, m))
    pair_double_fault = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            if i == j:
                pair_double_fault[i, j] = float(np.mean(~correct[:, i]))
                continue
            pair_agree[i, j] = float(np.mean(P[:, i] == P[:, j]))
            pair_kappa[i, j] = cohen_kappa(P[:, i], P[:, j])
            n11, n10, n01, n00 = _pair_counts(correct[:, i], correct[:, j])
            den = n11 * n00 + n01 * n10
            pair_q[i, j] = (n11 * n00 - n01 * n10) / den if den else 0.0
            pair_disagree[i, j] = (n01 + n10) / n
            pair_double_fault[i, j] = n00 / n

    iu = np.triu_indices(m, 1)
    votes = P.sum(axis=1)
    unanimous = (votes == 0) | (votes == m)
    majority = np.where(votes * 2 > m, 1, 0)
    n_correct = correct.sum(axis=1)

    rep = {
        "agent_names": names,
        "pairwise_agreement": pair_agree.tolist(),
        "pairwise_kappa": pair_kappa.tolist(),
        "pairwise_q_statistic": pair_q.tolist(),
        "pairwise_disagreement": pair_disagree.tolist(),
        "pairwise_double_fault": pair_double_fault.tolist(),
        "mean_pairwise_agreement": float(pair_agree[iu].mean()) if m > 1 else 1.0,
        "mean_pairwise_kappa": float(pair_kappa[iu].mean()) if m > 1 else 1.0,
        "mean_q_statistic": float(pair_q[iu].mean()) if m > 1 else 1.0,
        "mean_disagreement": float(pair_disagree[iu].mean()) if m > 1 else 0.0,
        "mean_double_fault": float(pair_double_fault[iu].mean()) if m > 1 else 0.0,
        "fleiss_kappa": float(fleiss_kappa(P)) if m > 1 else 1.0,
        "unanimous_rate": float(unanimous.mean()),
        "unanimous_accuracy": float(np.mean(majority[unanimous] == y[unanimous])) if unanimous.any() else float("nan"),
        "split_vote_accuracy": float(np.mean(f[~unanimous] == y[~unanimous])) if (~unanimous).any() else float("nan"),
        "majority_vote_accuracy": float(np.mean(majority == y)),
        "oracle_accuracy": float(np.mean(n_correct > 0)),
        "all_wrong_rate": float(np.mean(n_correct == 0)),
        "all_correct_rate": float(np.mean(n_correct == m)),
        "ensemble_accuracy": float(np.mean(f == y)),
        "fusion_vs_agent_agreement": {n_: float(np.mean(f == P[:, k])) for k, n_ in enumerate(names)},
        "fusion_vs_majority_agreement": float(np.mean(f == majority)),
        # rescued: ensemble right while the best single agent is wrong, and vice versa
        "rescued_by_fusion": None,
        "broken_by_fusion": None,
    }
    best = int(np.argmax(correct.mean(axis=0)))
    rep["best_agent"] = names[best]
    rep["rescued_by_fusion"] = float(np.mean((f == y) & ~correct[:, best]))
    rep["broken_by_fusion"] = float(np.mean((f != y) & correct[:, best]))

    if agent_proba is not None:
        Pr = np.asarray(agent_proba, dtype=float)
        with np.errstate(invalid="ignore"):
            c = np.corrcoef(Pr, rowvar=False)
        c = np.nan_to_num(c, nan=0.0)
        rep["probability_correlation"] = c.tolist()
        rep["mean_probability_correlation"] = float(c[iu].mean()) if m > 1 else 1.0
    return rep


def interpret(rep):
    """Plain-language one-liners about the agreement report."""
    out = []
    k = rep["mean_pairwise_kappa"]
    word = ("almost identical" if k > 0.8 else "largely similar" if k > 0.6
            else "moderately different" if k > 0.4 else "quite diverse")
    out.append(f"Agents' predictions are {word} (mean Cohen's kappa {k:.2f}); "
               f"they agree on {rep['mean_pairwise_agreement']:.0%} of students on average.")
    out.append(f"All agents agree on {rep['unanimous_rate']:.0%} of students; on the remaining "
               f"split-vote cases the ensemble is right {rep['split_vote_accuracy']:.0%} of the time.")
    out.append(f"An oracle that always picked the right agent would reach {rep['oracle_accuracy']:.1%}; "
               f"{rep['all_wrong_rate']:.1%} of students are misclassified by every agent.")
    out.append(f"Fusion fixed {rep['rescued_by_fusion']:.1%} of students the best single agent "
               f"({rep['best_agent']}) got wrong, but broke {rep['broken_by_fusion']:.1%} it got right.")
    if rep["mean_q_statistic"] > 0.9:
        out.append("High Q-statistic: agents tend to be right and wrong together, which limits how much fusion can add.")
    return out
