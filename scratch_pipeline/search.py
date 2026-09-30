"""K-fold CV architecture/hyperparameter search for MLPAgent and
RandomForestAgent, used both standalone (GUI buttons) and by
MASController.train() when auto_tune_mlp/auto_tune_rf is set.
"""
import numpy as np

from .agents import MLPAgent, RandomForestAgent
from shared import accuracy, confusion_matrix, precision_recall_f1, _stratified_kfold_indices


def mlp_architecture_search(X, y, architectures=None, lr_choices=(0.05,), l2_choices=(1e-4,),
                             n_epochs=150, batch_size=32, k_folds=5, seed=0, verbose=True):
    """
    Grid search over MLP hidden-layer ARCHITECTURES (depth and width per
    layer), optionally crossed with learning rate / L2 choices, scored by
    stratified k-fold cross-validation (mean accuracy and F1 across folds).
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


