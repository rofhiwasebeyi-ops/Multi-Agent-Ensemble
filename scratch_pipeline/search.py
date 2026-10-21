"""K-fold CV architecture/hyperparameter search for MLPAgent and
RandomForestAgent, used both standalone (GUI buttons) and by
MASController.train() when auto_tune_mlp/auto_tune_rf is set.
"""
import numpy as np

from .agents import (MLPAgent, RandomForestAgent, LogisticRegressionAgent, SVMAgent,
                     AISAgent, AGENT_REGISTRY)
from shared import accuracy, confusion_matrix, precision_recall_f1, _stratified_kfold_indices


def mlp_architecture_search(X, y, architectures=None, lr_choices=(0.05,), l2_choices=(1e-4,),
                             n_epochs=150, batch_size=32, k_folds=5, seed=0, verbose=True):
    """
    Grid search over MLP hidden-layer ARCHITECTURES (depth and width per
    layer), optionally crossed with learning rate / L2 choices, scored by
    stratified k-fold cross-validation (mean accuracy and F1 across folds).

    This searches actual network topology - not just a single fixed layer's
    width - since MLPAgent now accepts arbitrary hidden_layer_sizes tuples.
    Entirely NumPy-only: cross-validation splitting is done by
    _stratified_kfold_indices() rather than sklearn.

    Returns
    -------
    leaderboard : list of dicts, sorted best-first by mean CV accuracy.
        Each dict has: hidden_layer_sizes, lr, l2, mean_acc, std_acc, mean_f1, n_params
    best_config : dict, the top entry of the leaderboard (ready to unpack
        into MLPAgent(**best_config['agent_kwargs']))
    """
    if architectures is None:
        architectures = [
            (8,), (16,), (32,), (64,),
            (16, 8), (32, 16), (64, 32),
            (32, 16, 8),
        ]

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    folds = _stratified_kfold_indices(y, k=k_folds, seed=seed)
    n_features = X.shape[1]

    leaderboard = []
    total = len(architectures) * len(lr_choices) * len(l2_choices)
    tried = 0

    for arch in architectures:
        for lr in lr_choices:
            for l2 in l2_choices:
                tried += 1
                fold_accs, fold_f1s = [], []
                for fold_i in range(k_folds):
                    val_idx = folds[fold_i]
                    train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != fold_i])

                    model = MLPAgent(hidden_layer_sizes=arch, lr=lr, l2=l2,
                                      n_epochs=n_epochs, batch_size=batch_size, seed=seed,
                                      track_loss=False)
                    model.fit(X[train_idx], y[train_idx])

                    y_pred = model.predict(X[val_idx])
                    fold_accs.append(accuracy(y[val_idx], y_pred))
                    tp, tn, fp, fn = confusion_matrix(y[val_idx], y_pred)
                    _, _, f1 = precision_recall_f1(tp, tn, fp, fn)
                    fold_f1s.append(f1)

                layer_sizes = [n_features] + list(arch) + [1]
                n_params = sum(layer_sizes[i] * layer_sizes[i + 1] + layer_sizes[i + 1]
                                for i in range(len(layer_sizes) - 1))

                entry = {
                    "hidden_layer_sizes": arch, "lr": lr, "l2": l2,
                    "mean_acc": float(np.mean(fold_accs)), "std_acc": float(np.std(fold_accs)),
                    "mean_f1": float(np.mean(fold_f1s)), "n_params": n_params,
                    "agent_kwargs": dict(hidden_layer_sizes=arch, lr=lr, l2=l2,
                                          n_epochs=n_epochs, batch_size=batch_size),
                }
                leaderboard.append(entry)

                if verbose:
                    print(f"  [{tried}/{total}] arch={arch!s:<14} lr={lr:<6} l2={l2:<8} "
                          f"-> acc={entry['mean_acc']:.4f}±{entry['std_acc']:.4f}  "
                          f"f1={entry['mean_f1']:.4f}  params={n_params}")

    leaderboard.sort(key=lambda e: (-e["mean_acc"], -e["mean_f1"]))
    best_config = leaderboard[0]
    return leaderboard, best_config


def rf_architecture_search(X, y, configs=None, search_n_estimators=15, final_n_estimators=50,
                            k_folds=3, seed=0, verbose=True):
    """
    Grid search over RandomForestAgent hyperparameters (tree depth,
    min_samples_split/min_samples_leaf, and class_weight), scored by
    stratified k-fold cross-validation - the same pattern as
    mlp_architecture_search(), just over RandomForest's knobs instead of
    MLP's layer sizes. Entirely NumPy-only, using _stratified_kfold_indices.

    n_estimators (forest size) is deliberately NOT part of the search grid:
    unlike depth/leaf-size/class-weight, more trees almost always helps or
    is neutral (it reduces bagging variance, it doesn't change what any
    single tree can overfit to), so it doesn't need cross-validating the
    way genuine bias/variance tradeoffs do. Every candidate is scored with a
    small search_n_estimators (fast), then the winning combination is
    returned with n_estimators bumped up to final_n_estimators for the
    actual deployed model. Without this split, a full grid at deployment-
    size forests (100-200+ trees) x k_folds becomes a multi-minute wait for
    what's meant to be an interactive "Auto-tune RandomForest" button.

    Returns
    -------
    leaderboard : list of dicts, sorted best-first by mean CV accuracy
        (scored at search_n_estimators trees).
    best_config : dict, the top entry of the leaderboard (ready to unpack
        into RandomForestAgent(**best_config['agent_kwargs']), which uses
        final_n_estimators trees).
    """
    if configs is None:
        configs = [
            dict(max_depth=6, min_samples_split=4, min_samples_leaf=1, class_weight=None),
            dict(max_depth=8, min_samples_split=4, min_samples_leaf=1, class_weight=None),
            dict(max_depth=10, min_samples_split=6, min_samples_leaf=3, class_weight=None),
            dict(max_depth=8, min_samples_split=4, min_samples_leaf=1, class_weight="balanced"),
            dict(max_depth=10, min_samples_split=6, min_samples_leaf=3, class_weight="balanced"),
            dict(max_depth=12, min_samples_split=2, min_samples_leaf=1, class_weight="balanced"),
        ]

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    folds = _stratified_kfold_indices(y, k=k_folds, seed=seed)

    leaderboard = []
    total = len(configs)

    for tried, cfg in enumerate(configs, start=1):
        fold_accs, fold_f1s = [], []
        for fold_i in range(k_folds):
            val_idx = folds[fold_i]
            train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != fold_i])

            model = RandomForestAgent(n_estimators=search_n_estimators, random_state=seed, **cfg)
            model.fit(X[train_idx], y[train_idx])

            y_pred = model.predict(X[val_idx])
            fold_accs.append(accuracy(y[val_idx], y_pred))
            tp, tn, fp, fn = confusion_matrix(y[val_idx], y_pred)
            _, _, f1 = precision_recall_f1(tp, tn, fp, fn)
            fold_f1s.append(f1)

        entry = dict(cfg)
        entry["n_estimators"] = final_n_estimators
        entry["mean_acc"] = float(np.mean(fold_accs))
        entry["std_acc"] = float(np.std(fold_accs))
        entry["mean_f1"] = float(np.mean(fold_f1s))
        entry["agent_kwargs"] = dict(cfg, n_estimators=final_n_estimators)
        leaderboard.append(entry)

        if verbose:
            print(f"  [{tried}/{total}] depth={cfg['max_depth']:<4} "
                  f"mss={cfg['min_samples_split']:<3} msl={cfg['min_samples_leaf']:<3} "
                  f"cw={str(cfg['class_weight']):<9} -> acc={entry['mean_acc']:.4f}±{entry['std_acc']:.4f}  "
                  f"f1={entry['mean_f1']:.4f}")

    leaderboard.sort(key=lambda e: (-e["mean_acc"], -e["mean_f1"]))
    best_config = leaderboard[0]
    return leaderboard, best_config




# ---------------------------------------------------------------------------
# Generic CV hyperparameter search (used for LogisticRegression / SVM / AIS)
# ---------------------------------------------------------------------------
DEFAULT_GRIDS = {
    "LogisticRegression": [dict(lr=lr, l2=l2) for lr in (0.05, 0.15) for l2 in (1e-4, 1e-2, 1e-1)],
    "SVM": [dict(kernel="rbf", C=C) for C in (0.5, 1.0, 4.0, 10.0)] + [dict(kernel="linear", C=1.0)],
    "AIS": [dict(n_detectors=n, flip_prob=fp) for n in (100, 200, 300) for fp in (0.04, 0.12)],
}
# Rows used while SEARCHING (the final model is still trained on all rows its
# own caps allow). Kernel SVM and AIS are O(n^2), so they search on a subsample.
DEFAULT_SEARCH_ROWS = {"LogisticRegression": 20000, "SVM": 1200, "AIS": 1500}
_AGENT_CLASSES = {"LogisticRegression": LogisticRegressionAgent, "SVM": SVMAgent, "AIS": AISAgent}


def hyperparameter_search(agent_name, X, y, grid=None, base_kwargs=None, k_folds=3,
                          max_rows=None, seed=0, verbose=True):
    """K-fold CV grid search for any registered agent. `grid` is a list of
    kwarg-override dicts applied on top of `base_kwargs` (default: the
    agent's AGENT_REGISTRY kwargs). Returns (leaderboard, best) where each
    entry has params, mean_acc, std_acc, mean_f1, agent_kwargs (ready to
    unpack into the agent class), sorted best-first by accuracy then F1."""
    cls = _AGENT_CLASSES.get(agent_name) or AGENT_REGISTRY[agent_name][0]
    base = dict(AGENT_REGISTRY[agent_name][1] if base_kwargs is None else base_kwargs)
    grid = list(grid if grid is not None else DEFAULT_GRIDS[agent_name])
    max_rows = max_rows or DEFAULT_SEARCH_ROWS.get(agent_name, 5000)

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if len(y) > max_rows:
        take = np.random.default_rng(seed).choice(len(y), max_rows, replace=False)
        X, y = X[take], y[take]
    folds = _stratified_kfold_indices(y, k=k_folds, seed=seed)

    leaderboard = []
    for tried, params in enumerate(grid, start=1):
        kwargs = dict(base, **params)
        accs, f1s = [], []
        for i in range(k_folds):
            val = folds[i]
            tr = np.concatenate([folds[j] for j in range(k_folds) if j != i])
            model = cls(**kwargs)
            model.fit(X[tr], y[tr])
            pred = model.predict(X[val])
            accs.append(accuracy(y[val], pred))
            _, _, f1 = precision_recall_f1(*confusion_matrix(y[val], pred))
            f1s.append(f1)
        entry = {"agent": agent_name, "params": dict(params),
                 "mean_acc": float(np.mean(accs)), "std_acc": float(np.std(accs)),
                 "mean_f1": float(np.mean(f1s)), "agent_kwargs": kwargs, "n_rows": len(y)}
        leaderboard.append(entry)
        if verbose:
            print(f"  [{tried}/{len(grid)}] {agent_name} {params} -> "
                  f"acc={entry['mean_acc']:.4f}±{entry['std_acc']:.4f} f1={entry['mean_f1']:.4f}")
    leaderboard.sort(key=lambda e: (-e["mean_acc"], -e["mean_f1"]))
    return leaderboard, leaderboard[0]
