"""Fusion layers (static logistic stacker and context-aware gating network)
and MASController, which trains the full heterogeneous agent ensemble,
fuses their predictions, and exposes SHAP-based explanations.
"""
import numpy as np

from .agents import AGENT_REGISTRY, BaggingEnsemble
from .search import mlp_architecture_search, rf_architecture_search, hyperparameter_search
from shared import _stratified_kfold_indices, accuracy, ExplainabilityEngine


class FusionLayer:
    """
    Static fusion: learns ONE fixed weight vector over the agents for the
    whole dataset. Simple, fast baseline to compare GatingFusionLayer against.

    The agents' predictions are almost always highly correlated with each
    other (they're all trained on the same features/target), so this is a
    textbook multicollinear regression setting: with weak L2, the fitted
    weights become unstable and can swing to large, sign-flipped values that
    fit training-set noise rather than genuine relative agent quality (e.g.
    a weak agent getting a confidently NEGATIVE weight, or a mediocre agent
    outweighing the strongest standalone one) - which is exactly what lets
    the fused model underperform the best single agent on held-out data.
    l2='auto' (the default) fixes this by cross-validating the L2 strength
    on a small grid rather than using one fixed, usually-too-weak value.
    """

    def __init__(self, lr: float = 0.2, n_iters: int = 3000, l2="auto", tol: float = 1e-8,
                 l2_candidates=(1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0),
                 cv_folds: int = 5, search_n_iters: int = 800):
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2                      # "auto" or a fixed float
        self.tol = tol
        self.l2_candidates = l2_candidates
        self.cv_folds = cv_folds
        self.search_n_iters = search_n_iters  # fewer iters during the CV search, for speed
        self.weights = None   # (n_agents,)
        self.bias = 0.0
        self._fitted = False
        self.loss_history = []       # for learning-curve visualization
        self.loss_steps = []
        self.loss_step_label = "iteration"
        # AIS-style calibration curve (score vs. candidate L2), populated
        # only when l2='auto'; picked up automatically by the same
        # Learning Curves plumbing that already renders AISAgent's
        # calibration_history, so no extra UI wiring is needed.
        self.calibration_history = None

    # uses_context=False tells MASController this fusion layer only needs
    # the stacking matrix (agent_preds), not the raw student features.
    uses_context = False

    @staticmethod
    def _sigmoid(z):
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out

    def _fit_weights(self, agent_preds, y, l2, n_iters, track_loss=True):
        """Core gradient-descent fit at a FIXED l2 - used both for the final
        fit and, with a smaller n_iters, inside the CV search below."""
        n_samples, n_agents = agent_preds.shape
        weights = np.full(n_agents, 1.0 / n_agents)
        bias = 0.0
        prev_loss = np.inf
        loss_history, loss_steps = [], []

        for i in range(n_iters):
            z = agent_preds @ weights + bias
            p = self._sigmoid(z)
            error = p - y

            grad_w = (agent_preds.T @ error) / n_samples + l2 * weights
            grad_b = np.mean(error)

            weights -= self.lr * grad_w
            bias -= self.lr * grad_b

            if track_loss and (i % 100 == 0 or i == n_iters - 1):
                eps = 1e-12
                loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                loss += 0.5 * l2 * np.sum(weights ** 2)
                loss_history.append(float(loss))
                loss_steps.append(i)
                if abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss

        return weights, bias, loss_history, loss_steps

    def _select_l2_cv(self, agent_preds, y, seed=0):
        """K-fold CV over l2_candidates, scored by mean validation accuracy.
        Returns (best_l2, candidates_tried, mean_acc_per_candidate)."""
        n_samples = agent_preds.shape[0]
        folds = _stratified_kfold_indices(y, k=self.cv_folds, seed=seed)

        best_l2, best_acc = self.l2_candidates[0], -1.0
        candidates_tried, scores = [], []
        for l2 in self.l2_candidates:
            fold_accs = []
            for i in range(self.cv_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(self.cv_folds) if j != i])
                w, b, _, _ = self._fit_weights(agent_preds[train_idx], y[train_idx],
                                                l2, self.search_n_iters, track_loss=False)
                p_val = self._sigmoid(agent_preds[val_idx] @ w + b)
                pred_val = (p_val >= 0.5).astype(int)
                fold_accs.append(float(np.mean(pred_val == y[val_idx])))
            mean_acc = float(np.mean(fold_accs))
            candidates_tried.append(float(l2))
            scores.append(mean_acc)
            if mean_acc > best_acc:
                best_acc, best_l2 = mean_acc, l2
        return best_l2, np.array(candidates_tried), np.array(scores)

    def fit(self, agent_preds: np.ndarray, y: np.ndarray) -> None:
        agent_preds = np.asarray(agent_preds, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if self.l2 == "auto":
            chosen_l2, candidates, scores = self._select_l2_cv(agent_preds, y)
            self.calibration_history = {"x": candidates, "y": scores, "chosen_x": chosen_l2,
                                         "x_label": "candidate L2", "y_label": "mean CV accuracy",
                                         "x_log": True}
        else:
            chosen_l2 = self.l2
            self.calibration_history = None

        self.weights, self.bias, self.loss_history, self.loss_steps = self._fit_weights(
            agent_preds, y, chosen_l2, self.n_iters, track_loss=True)
        self.l2 = chosen_l2  # so get_weights()/reporting reflect what was actually used
        self._fitted = True

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("FusionLayer must be fit() before predicting.")

    def predictProba(self, agent_preds: np.ndarray) -> np.ndarray:
        self._check_fitted()
        agent_preds = np.asarray(agent_preds, dtype=np.float64)
        z = agent_preds @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, agent_preds: np.ndarray) -> np.ndarray:
        return (self.predictProba(agent_preds) >= 0.5).astype(int)

    def get_weights(self) -> np.ndarray:
        return self.weights

    def get_name(self) -> str:
        return "FusionLayer (logistic stacker)"


# GatingFusionLayer (context-aware, mixture-of-experts style intelligent fusion)
class GatingFusionLayer:
    """
    Learned, PER-INSTANCE alternative to FusionLayer's single fixed weight
    vector. A small softmax gating network (one hidden ReLU layer) looks at
    BOTH each agent's prediction AND the student's own (standardised) feature
    vector, plus a few derived meta-features (inter-agent disagreement,
    confidence margin, spread), and outputs a fresh set of trust weights for
    every student - i.e. it can learn context-dependent rules like "trust
    AIS less for students with many active categorical risk flags" rather
    than one static blend for the whole dataset:

        gate_input   = [Xs | standardised(agent_preds) | entropy | margin | std_dev]
        gate_weights = softmax( ReLU(gate_input @ W1 + b1) @ W2 + b2 )
        final_prob   = sum(gate_weights * agent_preds, axis=1)   # RAW agent_preds, not standardised

    Trained via manual backprop (BCE loss) on the training features X and
    the same stacking matrix FusionLayer uses.

    Two stability fixes, both addressing the same failure mode: with weak,
    fixed L2 (the old default), the gate can collapse onto ALWAYS trusting
    whichever agent happens to look best on training data - a training-set
    shortcut that doesn't generalize (e.g. an agent whose raw predictProba
    is more saturated/near-binary than the others gets an outsized say in
    ITS OWN routing decision purely from that scale, independent of whether
    its actual predictions are more trustworthy):
      1. l2='auto' (new default) cross-validates L2 strength, the same fix
         already applied to FusionLayer's identical instability.
      2. agent_preds are z-scored before entering the gate's INPUT layer
         (not the final blending step, which still uses raw probabilities -
         only the routing DECISION is desensitised to any one agent's
         output scale).
    """

    def __init__(self, hidden_size: int = 16, lr: float = 0.05, n_epochs: int = 300,
                 batch_size: int = 32, l2="auto", seed: int = 42,
                 l2_candidates=(1e-4, 1e-3, 1e-2, 0.03, 0.1, 0.3, 1.0),
                 cv_folds: int = 3, search_n_epochs: int = 80):
        self.hidden_size = hidden_size
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2 = l2                      # "auto" or a fixed float
        self.seed = seed
        self.l2_candidates = l2_candidates
        self.cv_folds = cv_folds
        self.search_n_epochs = search_n_epochs  # fewer epochs during the CV search, for speed

        self.n_agents = None
        self.n_features = None
        self.W1 = self.b1 = self.W2 = self.b2 = None
        self._mu = None      # feature standardisation, fit on training X
        self._sigma = None
        self._agent_mu = None      # agent_preds standardisation, fit on training agent_preds -
        self._agent_sigma = None   # used only for the gate's INPUT, not the final blend
        self.loss_history = []       # for learning-curve visualization
        self.loss_steps = []
        self.loss_step_label = "epoch"
        # AIS/FusionLayer-style calibration curve (score vs. candidate L2),
        # populated only when l2='auto'; picked up automatically by the same
        # Learning Curves plumbing as the others.
        self.calibration_history = None
        self.mean_weights_ = None   # (n_agents,) mean gate weight over training set
        self.gate_weight_std_ = None
        self._fitted = False

    # uses_context=True tells MASController this fusion layer needs the raw
    # student feature matrix X in addition to the stacking matrix.
    uses_context = True

    @staticmethod
    def _agent_meta_features(agent_preds: np.ndarray) -> np.ndarray:
        """[entropy | mean_margin | std_across_agents] derived from RAW
        agent_preds (kept in probability space - these are meant to measure
        genuine inter-agent disagreement, not standardised)."""
        eps = 1e-9
        p = np.clip(agent_preds, eps, 1 - eps)
        mean_p = p.mean(axis=1, keepdims=True)
        entropy = -(mean_p * np.log(mean_p) + (1 - mean_p) * np.log(1 - mean_p))
        margin = np.abs(p - 0.5).mean(axis=1, keepdims=True)
        std_dev = p.std(axis=1, keepdims=True)
        return np.hstack([entropy, margin, std_dev])

    def _build_gate_input(self, X: np.ndarray, agent_preds: np.ndarray) -> np.ndarray:
        Xs = (X - self._mu) / self._sigma
        agent_preds_s = (agent_preds - self._agent_mu) / self._agent_sigma
        meta = self._agent_meta_features(agent_preds)
        return np.hstack([Xs, agent_preds_s, meta])

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def _forward(self, X, agent_preds):
        gate_input = self._build_gate_input(X, agent_preds)
        z1 = gate_input @ self.W1 + self.b1
        a1 = self._relu(z1)
        z2 = a1 @ self.W2 + self.b2
        gate_weights = self._softmax(z2)
        final_prob = np.sum(gate_weights * agent_preds, axis=1)  # RAW agent_preds for the blend
        cache = (gate_input, z1, a1, gate_weights, agent_preds)
        return final_prob, gate_weights, cache

    def _backward(self, y_true, final_prob, cache):
        gate_input, z1, a1, gate_weights, agent_preds = cache
        n = y_true.shape[0]
        eps = 1e-9
        p = np.clip(final_prob, eps, 1 - eps)

        dL_dp = (p - y_true) / (p * (1 - p) + eps)
        dL_dg = dL_dp[:, None] * agent_preds

        sum_term = np.sum(gate_weights * dL_dg, axis=1, keepdims=True)
        dL_dz2 = gate_weights * (dL_dg - sum_term)   # softmax vector-Jacobian product

        dW2 = a1.T @ dL_dz2 / n + self.l2 * self.W2
        db2 = dL_dz2.mean(axis=0)

        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * (z1 > 0)                     # ReLU grad

        dW1 = gate_input.T @ dL_dz1 / n + self.l2 * self.W1
        db1 = dL_dz1.mean(axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def _select_l2_cv(self, X, agent_preds, y, seed=0):
        """K-fold CV over l2_candidates, scored by mean validation accuracy -
        same pattern as FusionLayer._select_l2_cv. Trains a fresh, fully
        independent GatingFusionLayer per fold/candidate (fewer epochs, for
        speed) rather than mutating self, so this can't leak into the final
        fit's weights. Returns (best_l2, candidates_tried, mean_acc_per_candidate)."""
        folds = _stratified_kfold_indices(y, k=self.cv_folds, seed=seed)
        best_l2, best_acc = self.l2_candidates[0], -1.0
        candidates_tried, scores = [], []

        for l2 in self.l2_candidates:
            fold_accs = []
            for i in range(self.cv_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(self.cv_folds) if j != i])
                probe = GatingFusionLayer(hidden_size=self.hidden_size, lr=self.lr,
                                           n_epochs=self.search_n_epochs, batch_size=self.batch_size,
                                           l2=l2, seed=self.seed)
                probe.fit(X[train_idx], agent_preds[train_idx], y[train_idx])
                pred_val = probe.predict(X[val_idx], agent_preds[val_idx])
                fold_accs.append(float(np.mean(pred_val == y[val_idx])))
            mean_acc = float(np.mean(fold_accs))
            candidates_tried.append(float(l2))
            scores.append(mean_acc)
            if mean_acc > best_acc:
                best_acc, best_l2 = mean_acc, l2
        return best_l2, np.array(candidates_tried), np.array(scores)

    def fit(self, X: np.ndarray, agent_preds: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)
        agent_preds = np.asarray(agent_preds, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_agents = agent_preds.shape
        self.n_agents = n_agents
        self.n_features = X.shape[1]

        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0

        self._agent_mu = agent_preds.mean(axis=0)
        self._agent_sigma = agent_preds.std(axis=0)
        self._agent_sigma[self._agent_sigma == 0] = 1.0

        if self.l2 == "auto":
            chosen_l2, candidates, scores = self._select_l2_cv(X, agent_preds, y)
            self.calibration_history = {"x": candidates, "y": scores, "chosen_x": chosen_l2,
                                         "x_label": "candidate L2", "y_label": "mean CV accuracy",
                                         "x_log": True}
            self.l2 = chosen_l2
        else:
            self.calibration_history = None

        n_gate_inputs = self.n_features + n_agents + 3  # + entropy, margin, std_dev
        self.W1 = rng.normal(0, np.sqrt(2.0 / n_gate_inputs), size=(n_gate_inputs, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = rng.normal(0, np.sqrt(2.0 / self.hidden_size), size=(self.hidden_size, n_agents))
        self.b2 = np.zeros(n_agents)
        self.loss_history = []
        self.loss_steps = []

        for epoch in range(self.n_epochs):
            idx = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                b = idx[start:start + self.batch_size]
                final_prob, _, cache = self._forward(X[b], agent_preds[b])
                self._backward(y[b], final_prob, cache)

            if epoch % 25 == 0 or epoch == self.n_epochs - 1:
                full_prob, _, _ = self._forward(X, agent_preds)
                eps = 1e-12
                loss = -np.mean(y * np.log(full_prob + eps) + (1 - y) * np.log(1 - full_prob + eps))
                self.loss_history.append(float(loss))
                self.loss_steps.append(epoch)

        # Cache a global summary of gate behaviour for reporting/comparison
        # against FusionLayer's static weights (real weights vary per instance).
        _, gate_weights_full, _ = self._forward(X, agent_preds)
        self.mean_weights_ = gate_weights_full.mean(axis=0)
        self.gate_weight_std_ = gate_weights_full.std(axis=0)

        self._fitted = True

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("GatingFusionLayer must be fit() before predicting.")

    def predictProba(self, X: np.ndarray, agent_preds: np.ndarray) -> np.ndarray:
        self._check_fitted()
        final_prob, _, _ = self._forward(np.asarray(X, dtype=np.float64),
                                          np.asarray(agent_preds, dtype=np.float64))
        return final_prob

    def predict(self, X: np.ndarray, agent_preds: np.ndarray) -> np.ndarray:
        return (self.predictProba(X, agent_preds) >= 0.5).astype(int)

    def predict_with_gate_weights(self, X: np.ndarray, agent_preds: np.ndarray):
        """Like predictProba, but also returns the (n_samples, n_agents) gate
        weight matrix — the per-instance trust breakdown used for explainability."""
        self._check_fitted()
        final_prob, gate_weights, _ = self._forward(np.asarray(X, dtype=np.float64),
                                                      np.asarray(agent_preds, dtype=np.float64))
        return final_prob, gate_weights

    def get_weights(self) -> np.ndarray:
        """Mean gate weight per agent over the training set. This lets
        MASController.get_agent_trust() report something comparable to
        FusionLayer's fixed weights, while predict_with_gate_weights()
        remains the source of truth for any individual instance."""
        self._check_fitted()
        return self.mean_weights_

    def get_name(self) -> str:
        return "GatingFusionLayer (context-aware, per-instance gating network)"




class MASController:

    def __init__(self, agent_registry: dict = None, use_bagging: bool = True,
                 n_estimators: int = 10, random_state: int = 7,
                 fusion_type: str = "logistic", fusion_kwargs: dict = None,
                 auto_tune_mlp: bool = False, mlp_search_kwargs: dict = None,
                 auto_tune_rf: bool = False, rf_search_kwargs: dict = None,
                 agent_feature_overrides: dict = None,
                 auto_tune_all: bool = False, agent_search_kwargs: dict = None):
        """
        fusion_type: "logistic" -> FusionLayer (static weights, one blend for
                     the whole dataset). "gating" -> GatingFusionLayer
                     (per-instance weights, using both agent predictions AND
                     the student's own feature vector).
        fusion_kwargs: optional dict of constructor kwargs forwarded to
                     whichever fusion class is selected.
        auto_tune_mlp: if True, train() runs mlp_architecture_search() on the
                     training data first and swaps the MLP entry in the
                     agent registry for the best architecture found, before
                     training any agents. Adds real cost to train() (a full
                     k-fold search), so it's opt-in.
        mlp_search_kwargs: optional dict of kwargs forwarded to
                     mlp_architecture_search() (architectures, lr_choices,
                     l2_choices, n_epochs, k_folds, etc).
        auto_tune_rf: if True, train() runs rf_architecture_search() on the
                     training data first and swaps the RandomForest entry in
                     the agent registry for the best config found (tree
                     depth, min_samples_split/leaf, class_weight), before
                     training any agents. Same opt-in cost tradeoff as
                     auto_tune_mlp.
        rf_search_kwargs: optional dict of kwargs forwarded to
                     rf_architecture_search() (configs, search_n_estimators,
                     final_n_estimators, k_folds, etc).
        agent_feature_overrides: optional {agent_name: [col_idx, ...]} - lets
                     individual agents train/predict on a DIFFERENT column
                     subset of the same X than the rest of the ensemble,
                     rather than every agent seeing identical features.
                     Column indices are relative to the X passed into
                     train()/predictProba() (i.e. positions within whatever
                     feature_names apply to that call). Agents not listed
                     use the full X, same as before. Motivating case: AIS's
                     distance-based scoring degrades in high dimensions, so
                     it can be pointed at just the grade columns (e.g.
                     {"AIS": grade_positions}) while LogisticRegression/SVM/
                     RandomForest/MLP keep the full feature set - each
                     agent still just contributes a predictProba() column to
                     the same stacking matrix, so fusion is unaffected.
        """
        self.agent_registry = agent_registry or AGENT_REGISTRY
        self.use_bagging = use_bagging
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.fusion_type = fusion_type
        self.fusion_kwargs = fusion_kwargs or {}
        # auto_tune_all: CV-tune EVERY agent. MLP and RandomForest use their
        # dedicated searches; LogisticRegression, SVM and AIS use the generic
        # hyperparameter_search (each on its own feature view).
        # agent_search_kwargs: optional {agent_name: kwargs for hyperparameter_search}.
        self.auto_tune_all = auto_tune_all
        self.agent_search_kwargs = agent_search_kwargs or {}
        if auto_tune_all:
            auto_tune_mlp = auto_tune_rf = True
        self.auto_tune_mlp = auto_tune_mlp
        self.mlp_search_kwargs = mlp_search_kwargs or {}
        self.auto_tune_rf = auto_tune_rf
        self.rf_search_kwargs = rf_search_kwargs or {}
        self.agent_feature_overrides = agent_feature_overrides or {}

        self.agents = {}         # name -> fitted BaseAgent or BaggingEnsemble
        self.fusion = None       # set in train(), per fusion_type
        self.explainer = ExplainabilityEngine()
        self.feature_names = None
        self.mlp_search_leaderboard = None   # populated in train() if auto_tune_mlp
        self.best_mlp_config = None          # populated in train() if auto_tune_mlp
        self.rf_search_leaderboard = None    # populated in train() if auto_tune_rf
        self.best_rf_config = None           # populated in train() if auto_tune_rf
        self.best_configs = {}               # name -> best CV entry (LogisticRegression/SVM/AIS if auto_tune_all)
        self.search_leaderboards = {}
        self._fitted = False

    def _make_fusion(self):
        if self.fusion_type == "gating":
            return GatingFusionLayer(**self.fusion_kwargs)
        elif self.fusion_type == "logistic":
            return FusionLayer(**self.fusion_kwargs)
        else:
            raise ValueError(f"Unknown fusion_type '{self.fusion_type}'. "
                              f"Expected 'logistic' or 'gating'.")

    def _agent_view(self, X: np.ndarray, name: str) -> np.ndarray:
        """Returns the column subset of X that `name` should see, per
        agent_feature_overrides - or X unchanged if no override is set."""
        cols = self.agent_feature_overrides.get(name)
        return X[:, cols] if cols is not None else X

    def train(self, X: np.ndarray, y: np.ndarray, feature_names=None) -> None:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.feature_names = list(feature_names) if feature_names is not None \
            else [f"f{i}" for i in range(X.shape[1])]

        registry = dict(self.agent_registry)
        if self.auto_tune_mlp and "MLP" in registry:
            mlp_cls, _ = registry["MLP"]
            X_mlp = self._agent_view(X, "MLP")
            leaderboard, best = mlp_architecture_search(X_mlp, y, verbose=False, **self.mlp_search_kwargs)
            self.mlp_search_leaderboard = leaderboard
            self.best_mlp_config = best
            registry["MLP"] = (mlp_cls, best["agent_kwargs"])

        if self.auto_tune_rf and "RandomForest" in registry:
            rf_cls, _ = registry["RandomForest"]
            X_rf = self._agent_view(X, "RandomForest")
            rf_kwargs = dict(self.rf_search_kwargs)
            if self.use_bagging and "final_n_estimators" not in rf_kwargs:
                # RF is about to be wrapped by THIS controller's own
                # BaggingEnsemble (n_estimators members), so each RF member
                # doesn't need to be its own large forest - the outer
                # bagging already supplies that variance reduction. Without
                # this, auto-tuning's final_n_estimators (default 50) times
                # the outer n_estimators (often 10) means 500 trees just for
                # RandomForest, most of them redundant.
                rf_kwargs["final_n_estimators"] = 20
            leaderboard, best = rf_architecture_search(X_rf, y, verbose=False, **rf_kwargs)
            self.rf_search_leaderboard = leaderboard
            self.best_rf_config = best
            registry["RandomForest"] = (rf_cls, best["agent_kwargs"])

        if self.auto_tune_all:
            for name in ("LogisticRegression", "SVM", "AIS"):
                if name not in registry:
                    continue
                cls, kwargs = registry[name]
                leaderboard, best = hyperparameter_search(
                    name, self._agent_view(X, name), y, base_kwargs=kwargs, verbose=False,
                    **self.agent_search_kwargs.get(name, {}))
                self.search_leaderboards[name] = leaderboard
                self.best_configs[name] = best
                registry[name] = (cls, best["agent_kwargs"])

        self.agents = {}
        train_cols = []
        for name, (cls, kwargs) in registry.items():
            X_agent = self._agent_view(X, name)
            if self.use_bagging:
                model = BaggingEnsemble(base_agent_factory=cls, n_estimators=self.n_estimators,
                                         bootstrap=True, random_state=self.random_state, agent_kwargs=kwargs)
                model.fit(X_agent, y)
                proba = model.oob_proba_.copy()
                nan_mask = np.isnan(proba)
                if nan_mask.any():
                    proba[nan_mask] = model.predictProba(X_agent[nan_mask])
            else:
                model = cls(**kwargs)
                model.fit(X_agent, y)
                proba = model.predictProba(X_agent)

            self.agents[name] = model
            train_cols.append(proba)

        agent_preds = np.column_stack(train_cols)
        self.fusion = self._make_fusion()

        if getattr(self.fusion, "uses_context", False):
            self.fusion.fit(X, agent_preds, y)     # gating: needs student features too
        else:
            self.fusion.fit(agent_preds, y)        # logistic stacker: agent preds only

        self._fitted = True

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("MASController must be train()ed before predicting.")

    def _stack_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        cols = [self.agents[name].predictProba(self._agent_view(X, name))
                for name in self.agent_registry.keys()]
        return np.column_stack(cols)

    def predictProba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        agent_preds = self._stack_proba(X)
        if getattr(self.fusion, "uses_context", False):
            return self.fusion.predictProba(X, agent_preds)
        return self.fusion.predictProba(agent_preds)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predictProba(X) >= 0.5).astype(int)

    def run_pipeline(self, X_train, y_train, X_test, y_test, feature_names=None) -> dict:
        self.train(X_train, y_train, feature_names=feature_names)
        y_pred = self.predict(X_test)
        X_test = np.asarray(X_test, dtype=np.float64)
        per_agent_test_acc = {
            name: accuracy(y_test, self.agents[name].predict(self._agent_view(X_test, name)))
            for name in self.agent_registry.keys()
        }
        agent_test_proba = self._stack_proba(X_test)
        agent_test_pred = (agent_test_proba >= 0.5).astype(int)
        return {
            "agent_names": list(self.agent_registry.keys()),
            "agent_test_proba": agent_test_proba,
            "agent_test_pred": agent_test_pred,
            "per_agent_test_acc": per_agent_test_acc,
            "fusion_weights": self.get_agent_trust(),
            "fusion_type": self.fusion_type,
            "y_pred": y_pred,
            "best_mlp_config": self.best_mlp_config,
            "best_rf_config": self.best_rf_config,
            "best_configs": self.best_configs,
        }

    def get_agent_trust(self) -> dict:
        """For 'logistic': the single fixed weight vector. For 'gating': the
        MEAN gate weight per agent across the training set — a global summary
        only, since real weights vary per instance/student context
        (see get_instance_gate_weights)."""
        return dict(zip(self.agent_registry.keys(), self.fusion.get_weights()))

    def get_instance_gate_weights(self, X: np.ndarray) -> np.ndarray:
        """Per-instance agent trust weights, shape (n_samples, n_agents), in
        the same agent order as self.agent_registry.keys(). Only available
        when fusion_type='gating', since the weights are conditioned on each
        student's own feature vector."""
        if self.fusion_type != "gating":
            raise RuntimeError(
                "Instance-level gate weights are only available when fusion_type='gating'.")
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        agent_preds = self._stack_proba(X)
        _, gate_weights = self.fusion.predict_with_gate_weights(X, agent_preds)
        return gate_weights

    def explain_global(self, X, feature_names=None, background_size=50,
                        max_samples=50, nsamples=100, top_k: int = 10):
        """SHAP-based global feature importance (mean |SHAP value|) for the
        full fused ensemble prediction (self.predictProba)."""
        pairs = self.explainer.global_importance(
            self.predictProba, X, feature_names=self.feature_names,
            background_size=background_size, max_samples=max_samples, nsamples=nsamples)
        return pairs[:top_k]

    def explain_instance(self, x, X_background, background_size=50, nsamples=100, top_k: int = 10):
        """SHAP values for a single student against the fused ensemble prediction."""
        pairs, baseline = self.explainer.explain_instance(
            self.predictProba, x, X_background, feature_names=self.feature_names,
            background_size=background_size, nsamples=nsamples)
        return pairs[:top_k], baseline




def run_full_ensemble_pipeline(X_train, y_train, X_test, y_test,
                                use_bagging=True, n_estimators=10, agent_registry=None,
                                feature_names=None, fusion_type="logistic", fusion_kwargs=None,
                                auto_tune_mlp=False, mlp_search_kwargs=None,
                                agent_feature_overrides=None, auto_tune_rf=False,
                                rf_search_kwargs=None, auto_tune_all=False,
                                agent_search_kwargs=None):

    controller = MASController(agent_registry=agent_registry, use_bagging=use_bagging,
                                n_estimators=n_estimators, fusion_type=fusion_type,
                                fusion_kwargs=fusion_kwargs, auto_tune_mlp=auto_tune_mlp,
                                mlp_search_kwargs=mlp_search_kwargs,
                                auto_tune_rf=auto_tune_rf, rf_search_kwargs=rf_search_kwargs,
                                auto_tune_all=auto_tune_all, agent_search_kwargs=agent_search_kwargs,
                                agent_feature_overrides=agent_feature_overrides)
    result = controller.run_pipeline(X_train, y_train, X_test, y_test, feature_names=feature_names)
    result["controller"] = controller
    return result


