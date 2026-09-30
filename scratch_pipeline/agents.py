"""From-scratch (NumPy-only) binary-classification agents, plus the generic
BaggingEnsemble wrapper and the AGENT_REGISTRY used to construct the default
heterogeneous ensemble.

LogisticRegressionAgent, SVMAgent, RandomForestAgent, MLPAgent, and AISAgent
all implement BaseAgent. BaggingEnsemble wraps any of them (or any other
BaseAgent) in bootstrap aggregation with out-of-bag scoring.
"""
from abc import ABC, abstractmethod
import numpy as np

from shared import _stratified_kfold_indices


class BaseAgent(ABC):
    """Abstract base class for all binary-classification agents."""

    def __init__(self):
        self._fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def predictProba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError(f"{self.get_name()} must be fit() before predicting.")

    @staticmethod
    def _as_2d(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X


# LogisticRegressionAgent
class LogisticRegressionAgent(BaseAgent):

    def __init__(self, lr: float = 0.1, n_iters: int = 2000,
                 l2: float = 1e-3, tol: float = 1e-7, verbose: bool = False):
        super().__init__()
        self.lr = lr
        self.n_iters = n_iters
        self.l2 = l2
        self.tol = tol
        self.verbose = verbose
        self.weights = None
        self.bias = 0.0
        self._mu = None
        self._sigma = None
        self.loss_history = []       # for learning-curve visualization
        self.loss_steps = []
        self.loss_step_label = "iteration"

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = self._as_2d(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0
        Xs = (X - self._mu) / self._sigma

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        prev_loss = np.inf
        self.loss_history = []
        self.loss_steps = []

        for i in range(self.n_iters):
            z = Xs @ self.weights + self.bias
            p = self._sigmoid(z)
            error = p - y

            grad_w = (Xs.T @ error) / n_samples + self.l2 * self.weights
            grad_b = np.mean(error)

            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

            if i % 50 == 0 or i == self.n_iters - 1:
                eps = 1e-12
                loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                loss += 0.5 * self.l2 * np.sum(self.weights ** 2)
                self.loss_history.append(float(loss))
                self.loss_steps.append(i)
                if self.verbose:
                    print(f"[LR] iter {i} loss {loss:.6f}")
                if abs(prev_loss - loss) < self.tol:
                    break
                prev_loss = loss

        self._fitted = True

    def predictProba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        Xs = (X - self._mu) / self._sigma
        z = Xs @ self.weights + self.bias
        return self._sigmoid(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predictProba(X) >= 0.5).astype(int)

    def get_name(self) -> str:
        return "LogisticRegressionAgent"


# SVMAgent (SMO + Platt scaling)
class SVMAgent(BaseAgent):

    def __init__(self, C: float = 1.0, kernel: str = "rbf", gamma="scale",
                 tol: float = 1e-3, max_passes: int = 10, max_iter: int = 200,
                 platt_iters: int = 500, platt_lr: float = 0.05, random_state: int = None,
                 max_train_samples: int = 4000):
        super().__init__()
        self.C = C
        self.kernel = kernel
        self.gamma_param = gamma
        self.tol = tol
        self.max_passes = max_passes
        self.max_iter = max_iter
        self.platt_iters = platt_iters
        self.platt_lr = platt_lr
        self.random_state = random_state
        self.max_train_samples = max_train_samples
        self.subsampled_ = False  # set True in fit() if the cap actually kicked in

        self.alphas = None
        self.b = 0.0
        self.X_train = None
        self.y_train = None
        self.gamma_ = None
        self._A = 0.0
        self._B = 0.0
        self.loss_history = []       # Platt-scaling calibration loss, for learning-curve visualization
        self.loss_steps = []
        self.loss_step_label = "Platt calibration iteration"

    def _kernel_matrix(self, X1, X2):
        if self.kernel == "linear":
            return X1 @ X2.T
        sq1 = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        sq2 = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        sq_dists = np.maximum(sq1 + sq2 - 2 * X1 @ X2.T, 0)
        return np.exp(-self.gamma_ * sq_dists)

    def _decision_function(self, X):
        K = self._kernel_matrix(X, self.X_train)
        return (K @ (self.alphas * self.y_train)) + self.b

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        X = self._as_2d(X)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples_full = X.shape[0]
        if n_samples_full > self.max_train_samples:
            keep_idx = rng.choice(n_samples_full, size=self.max_train_samples, replace=False)
            X = X[keep_idx]
            y = y[keep_idx]
            self.subsampled_ = True
        else:
            self.subsampled_ = False

        y_pm = np.where(y > 0, 1.0, -1.0)

        n_samples, n_features = X.shape
        if self.gamma_param == "scale":
            var = X.var()
            self.gamma_ = 1.0 / (n_features * var) if var > 0 else 1.0
        else:
            self.gamma_ = float(self.gamma_param)

        self.X_train = X
        self.y_train = y_pm
        self.alphas = np.zeros(n_samples)
        self.b = 0.0
        K = self._kernel_matrix(X, X)

        passes = 0
        it = 0
        while passes < self.max_passes and it < self.max_iter:
            num_changed = 0
            for i in range(n_samples):
                f_i = np.sum(self.alphas * self.y_train * K[i]) + self.b
                Ei = f_i - self.y_train[i]

                if (self.y_train[i] * Ei < -self.tol and self.alphas[i] < self.C) or \
                   (self.y_train[i] * Ei > self.tol and self.alphas[i] > 0):

                    j = rng.choice([k for k in range(n_samples) if k != i])
                    f_j = np.sum(self.alphas * self.y_train * K[j]) + self.b
                    Ej = f_j - self.y_train[j]

                    alpha_i_old, alpha_j_old = self.alphas[i], self.alphas[j]

                    if self.y_train[i] != self.y_train[j]:
                        L = max(0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)
                    if L == H:
                        continue

                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    self.alphas[j] = alpha_j_old - (self.y_train[j] * (Ei - Ej)) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    if abs(self.alphas[j] - alpha_j_old) < 1e-7:
                        continue

                    self.alphas[i] = alpha_i_old + self.y_train[i] * self.y_train[j] * \
                        (alpha_j_old - self.alphas[j])

                    b1 = self.b - Ei - self.y_train[i] * (self.alphas[i] - alpha_i_old) * K[i, i] \
                        - self.y_train[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - Ej - self.y_train[i] * (self.alphas[i] - alpha_i_old) * K[i, j] \
                        - self.y_train[j] * (self.alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_changed += 1
            passes = passes + 1 if num_changed == 0 else 0
            it += 1

        self._fit_platt_scaling()
        self._fitted = True

    def _fit_platt_scaling(self):
        f = self._decision_function(self.X_train)
        y = (self.y_train > 0).astype(np.float64)
        A, B = 0.0, 0.0
        self.loss_history = []
        self.loss_steps = []
        record_every = max(1, self.platt_iters // 50)  # cap to ~50 points on the curve
        for i in range(self.platt_iters):
            p = 1.0 / (1.0 + np.exp(A * f + B))  # P(y=1 | f), Platt (1999)
            p = np.clip(p, 1e-12, 1 - 1e-12)
            grad_A = np.mean((p - y) * f)
            grad_B = np.mean(p - y)
            A += self.platt_lr * grad_A
            B += self.platt_lr * grad_B

            if i % record_every == 0 or i == self.platt_iters - 1:
                loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
                self.loss_history.append(float(loss))
                self.loss_steps.append(i)
        self._A, self._B = A, B

    def predictProba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        f = self._decision_function(X)
        return 1.0 / (1.0 + np.exp(self._A * f + self._B))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predictProba(X) >= 0.5).astype(int)

    def get_name(self) -> str:
        return "SVMAgent"


# RandomForestAgent (CART + bagging + feature subsampling)
class _Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class _CARTTree:
    def __init__(self, max_depth=8, min_samples_split=4, min_samples_leaf=1, max_features=None,
                 max_thresholds=32):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf   # previously unenforced - a common overfitting gap
        self.max_features = max_features
        self.max_thresholds = max_thresholds
        self.root = None

    @staticmethod
    def _gini(y, w=None):
        if len(y) == 0:
            return 0.0
        if w is None:
            p1 = np.mean(y)
        else:
            wsum = w.sum()
            p1 = np.sum(w * y) / wsum if wsum > 0 else 0.0
        return 1.0 - p1 ** 2 - (1 - p1) ** 2

    def _best_split(self, X, y, feature_idxs, w=None):
        n_samples = len(y)
        best_gain, best_feat, best_thr = 0.0, None, None
        parent_gini = self._gini(y, w)
        w_total = w.sum() if w is not None else n_samples

        for feat in feature_idxs:
            values = np.unique(X[:, feat])
            if len(values) <= 1:
                continue
            if len(values) > self.max_thresholds + 1:
                qs = np.linspace(0.0, 1.0, self.max_thresholds + 2)[1:-1]
                thresholds = np.unique(np.quantile(values, qs))
            else:
                thresholds = (values[:-1] + values[1:]) / 2.0
            for thr in thresholds:
                left_mask = X[:, feat] <= thr
                n_left = left_mask.sum()
                n_right = n_samples - n_left
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                w_left = w[left_mask] if w is not None else None
                w_right = w[~left_mask] if w is not None else None
                gini_left = self._gini(y[left_mask], w_left)
                gini_right = self._gini(y[~left_mask], w_right)
                if w is None:
                    weighted = (n_left / n_samples) * gini_left + (n_right / n_samples) * gini_right
                else:
                    weighted = (w_left.sum() / w_total) * gini_left + (w_right.sum() / w_total) * gini_right
                gain = parent_gini - weighted
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat, thr
        return best_feat, best_thr, best_gain

    @staticmethod
    def _leaf_value(y, w=None):
        if len(y) == 0:
            return 0.5
        if w is not None and w.sum() > 0:
            return float(np.sum(w * y) / w.sum())
        return float(np.mean(y))

    def _build(self, X, y, depth, w=None):
        n_samples, n_features = X.shape
        if depth >= self.max_depth or n_samples < self.min_samples_split or len(np.unique(y)) == 1:
            return _Node(value=self._leaf_value(y, w) if n_samples > 0 else 0.5)

        m = self.max_features or max(1, int(np.sqrt(n_features)))
        feature_idxs = self._rng.choice(n_features, min(m, n_features), replace=False)

        feat, thr, gain = self._best_split(X, y, feature_idxs, w)
        if feat is None or gain <= 0:
            return _Node(value=self._leaf_value(y, w))

        left_mask = X[:, feat] <= thr
        w_left = w[left_mask] if w is not None else None
        w_right = w[~left_mask] if w is not None else None
        left = self._build(X[left_mask], y[left_mask], depth + 1, w_left)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1, w_right)
        return _Node(feature=feat, threshold=thr, left=left, right=right)

    def fit(self, X, y, sample_weight=None, rng=None):
        self._rng = rng if rng is not None else np.random.default_rng()
        self.root = self._build(X, y, depth=0, w=sample_weight)

    def _predict_one(self, x, node):
        while node.value is None:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict_proba(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])


class RandomForestAgent(BaseAgent):

    def __init__(self, n_estimators: int = 50, max_depth: int = 8,
                 min_samples_split: int = 4, min_samples_leaf: int = 1,
                 max_features=None, class_weight=None, random_state: int = None,
                 max_thresholds: int = 32):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.class_weight = class_weight
        self.random_state = random_state
        self.max_thresholds = max_thresholds
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        X = self._as_2d(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples = X.shape[0]

        sample_weight = None
        if self.class_weight == "balanced":
            classes, counts = np.unique(y, return_counts=True)
            class_w = {c: n_samples / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
            sample_weight = np.array([class_w[v] for v in y])

        self.trees = []
        for _ in range(self.n_estimators):
            boot_idx = rng.integers(0, n_samples, n_samples)
            Xb, yb = X[boot_idx], y[boot_idx]
            wb = sample_weight[boot_idx] if sample_weight is not None else None
            tree = _CARTTree(max_depth=self.max_depth,
                              min_samples_split=self.min_samples_split,
                              min_samples_leaf=self.min_samples_leaf,
                              max_features=self.max_features,
                              max_thresholds=self.max_thresholds)
            tree.fit(Xb, yb, sample_weight=wb, rng=rng)
            self.trees.append(tree)

        self._fitted = True

    def predictProba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        preds = np.array([tree.predict_proba(X) for tree in self.trees])
        return preds.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predictProba(X) >= 0.5).astype(int)

    def get_name(self) -> str:
        return "RandomForestAgent"


# MLPAgent (arbitrary depth, manual backprop)
class MLPAgent(BaseAgent):
    """
    Fully-connected NumPy MLP with an arbitrary number of ReLU hidden layers,
    sized via hidden_layer_sizes (e.g. (16,) for one layer of 16 units,
    (32, 16) for two layers). 
    """

    def __init__(self, hidden_layer_sizes=(16,), lr: float = 0.05, n_epochs: int = 300,
                 batch_size: int = 32, l2: float = 1e-4, seed: int = None, verbose: bool = False,
                 hidden_size: int = None, track_loss: bool = True):
        super().__init__()
        # hidden_size kept as a backward-compatible alias for a single-layer net
        if hidden_size is not None:
            hidden_layer_sizes = (hidden_size,)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.seed = seed
        self.verbose = verbose
        self.track_loss = track_loss   # set False (e.g. during architecture search) to skip
        self.Ws = None   # list of weight matrices, one per layer (hidden layers + output)
        self.bs = None   # list of bias row-vectors, matching Ws
        self._mu = self._sigma = None
        self.loss_history = []       # for learning-curve visualization
        self.loss_steps = []
        self.loss_step_label = "epoch"

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_grad(z):
        return (z > 0).astype(np.float64)

    @staticmethod
    def _sigmoid(z):
        out = np.empty_like(z, dtype=np.float64)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        exp_z = np.exp(z[~pos])
        out[~pos] = exp_z / (1.0 + exp_z)
        return out

    def _init_params(self, n_features):
        rng = np.random.default_rng(self.seed)
        layer_sizes = [n_features] + list(self.hidden_layer_sizes) + [1]
        self.Ws, self.bs = [], []
        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            self.Ws.append(rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in))
            self.bs.append(np.zeros((1, fan_out)))

    def _forward(self, Xs):
        """Returns (zs, activations) for every layer; activations[0] is the input."""
        activations = [Xs]
        zs = []
        a = Xs
        n_layers = len(self.Ws)
        for i in range(n_layers):
            z = a @ self.Ws[i] + self.bs[i]
            zs.append(z)
            a = self._relu(z) if i < n_layers - 1 else self._sigmoid(z)  # ReLU hidden, sigmoid output
            activations.append(a)
        return zs, activations

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = self._as_2d(X)
        y = np.asarray(y, dtype=np.float64).ravel().reshape(-1, 1)
        n_samples, n_features = X.shape

        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0
        Xs = (X - self._mu) / self._sigma

        self._init_params(n_features)
        n_layers = len(self.Ws)
        rng = np.random.default_rng(self.seed)
        self.loss_history = []
        self.loss_steps = []
        record_every = max(1, self.n_epochs // 100)  # cap to ~100 points on the curve

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n_samples)
            Xs_shuf, y_shuf = Xs[perm], y[perm]

            for start in range(0, n_samples, self.batch_size):
                xb = Xs_shuf[start:start + self.batch_size]
                yb = y_shuf[start:start + self.batch_size]
                m = xb.shape[0]

                zs, activations = self._forward(xb)
                a_out = activations[-1]

                dWs = [None] * n_layers
                dbs = [None] * n_layers
                dz = (a_out - yb) / m  # output layer: sigmoid + BCE grad simplifies to (a - y)

                for i in reversed(range(n_layers)):
                    a_prev = activations[i]
                    dWs[i] = a_prev.T @ dz + self.l2 * self.Ws[i]
                    dbs[i] = dz.sum(axis=0, keepdims=True)
                    if i > 0:
                        da_prev = dz @ self.Ws[i].T
                        dz = da_prev * self._relu_grad(zs[i - 1])

                for i in range(n_layers):
                    self.Ws[i] -= self.lr * dWs[i]
                    self.bs[i] -= self.lr * dbs[i]

            if self.track_loss and (epoch % record_every == 0 or epoch == self.n_epochs - 1):
                _, activations = self._forward(Xs)
                p = activations[-1]
                eps = 1e-12
                loss = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
                self.loss_history.append(float(loss))
                self.loss_steps.append(epoch)
                if self.verbose and epoch % 50 == 0:
                    print(f"[MLP {self.hidden_layer_sizes}] epoch {epoch} loss {loss:.6f}")

        self._fitted = True

    def predictProba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        Xs = (X - self._mu) / self._sigma
        _, activations = self._forward(Xs)
        return activations[-1].ravel()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predictProba(X) >= 0.5).astype(int)

    def get_name(self) -> str:
        return f"MLPAgent{self.hidden_layer_sizes}"


# AISAgent (Negative Selection Algorithm)
class AISAgent(BaseAgent):
    """
    Negative Selection Algorithm (NSA). 
    """

    def __init__(self, n_detectors: int = 200, self_radius="auto",
                 max_tries_per_detector: int = 300, random_state: int = None,
                 detector_strategy: str = "mutation", radius_calibration: str = "cv",
                 k_folds: int = 5, flip_prob: float = 0.08, label_flip: bool = None):
        super().__init__()
        self.n_detectors = n_detectors
        self.self_radius = self_radius       # "auto" or a user-supplied float
        self.max_tries_per_detector = max_tries_per_detector
        self.random_state = random_state
        self.detector_strategy = detector_strategy   # "mutation" or "random"
        self.radius_calibration = radius_calibration  # "cv" or "insample"
        self.k_folds = k_folds
        self.flip_prob = flip_prob           # per-dim flip probability for one-hot dims when mutating
        self.label_flip = label_flip
        self.detectors = None
        self.self_samples_ = None
        self._resolved_radius = None         # the actual radius used, after auto-calibration
        self._min = self._max = None
        self._binary_mask = None             # which normalised dims are one-hot/binary (0/1 only)
        self._label_flipped = False          # set in fit() - see _detect_self_class
        self.calibration_history = None      # {'x','y','chosen_x','x_label','y_label'} or None

    @classmethod
    def precalibrate(cls, X, y, agent_kwargs=None):
        agent_kwargs = dict(agent_kwargs or {})
        if agent_kwargs.get("self_radius", "auto") != "auto":
            return {}

        probe = cls(**{**agent_kwargs, "self_radius": "auto"})
        X = probe._as_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.float64).ravel()
        n_features = X.shape[1]

        probe._binary_mask = np.array([
            set(np.unique(X[:, j])) <= {0.0, 1.0} for j in range(n_features)
        ])
        probe._min = X.min(axis=0)
        probe._max = X.max(axis=0)
        Xn = probe._normalise(X)
        Xn_calib, y_calib, self_samples = cls._cap_for_calibration(
            Xn, y, max_calib=3000, random_state=agent_kwargs.get("random_state"))

        self_class = cls._detect_self_class(Xn_calib, y_calib,
                                             random_state=agent_kwargs.get("random_state"))
        label_flip = (self_class == 1.0)
        if label_flip:
            y_calib = 1.0 - y_calib
            self_samples = Xn_calib[y_calib == 0]
            if len(self_samples) == 0:
                self_samples = Xn_calib

        if probe.radius_calibration == "cv":
            r, cand, scores = probe._select_radius_cv(Xn_calib, y_calib, n_detectors_search=35,
                                                        n_candidates=14, search_folds=5)
            y_label = "mean F1 (cross-validated)"
        else:
            d_self_all = probe._nearest_self_distances(Xn_calib, y_calib, self_samples)
            r, cand, scores = probe._select_radius_supervised(d_self_all, y_calib)
            y_label = "F1 (in-sample)"

        debug = {"x": cand, "y": scores, "chosen_x": r,
                 "x_label": "candidate self_radius", "y_label": y_label}
        return {"self_radius": r, "label_flip": label_flip, "_calibration_debug": debug}

    @staticmethod
    def _cap_for_calibration(Xn, y, max_calib=3000, random_state=None):
        rng = np.random.default_rng(random_state)
        if len(Xn) > max_calib:
            keep = rng.choice(len(Xn), size=max_calib, replace=False)
            Xn_calib, y_calib = Xn[keep], y[keep]
        else:
            Xn_calib, y_calib = Xn, y
        self_samples = Xn_calib[y_calib == 0]
        if len(self_samples) == 0:
            self_samples = Xn_calib
        if len(self_samples) > max_calib:
            keep2 = rng.choice(len(self_samples), size=max_calib, replace=False)
            self_samples = self_samples[keep2]
        return Xn_calib, y_calib, self_samples


    def _normalise(self, X):
        return (X - self._min) / (self._max - self._min + 1e-12)

    @staticmethod
    def _pairwise_dist(A, B):
        A2 = np.sum(A ** 2, axis=1)[:, None]
        B2 = np.sum(B ** 2, axis=1)[None, :]
        sq = np.maximum(A2 + B2 - 2 * (A @ B.T), 0.0)
        return np.sqrt(sq)

    @classmethod
    def _min_dist_to_set(cls, A, B, batch_size=2000):
        if len(B) == 0:
            return np.full(len(A), np.inf)
        if len(A) <= batch_size:
            return cls._pairwise_dist(A, B).min(axis=1)
        out = np.empty(len(A))
        for start in range(0, len(A), batch_size):
            end = min(start + batch_size, len(A))
            out[start:end] = cls._pairwise_dist(A[start:end], B).min(axis=1)
        return out

    @classmethod
    def _nearest_self_distances(cls, X_norm, y, self_pts):
        """For every training sample, distance to nearest OTHER self sample
        (leave-one-out for self-class points, so self points don't just match
        themselves at distance 0)."""
        D = cls._pairwise_dist(X_norm, self_pts)  # (n_samples, n_self)
        self_mask = y == 0
        if self_mask.any() and D.shape[1] > 1:
            self_rows = np.where(self_mask)[0]
            row_min_idx = np.argmin(D[self_rows], axis=1)
            D_masked = D.copy()
            D_masked[self_rows, row_min_idx] = np.inf
            d = D.min(axis=1)
            d[self_rows] = D_masked[self_rows].min(axis=1)
        else:
            d = D.min(axis=1) if D.shape[1] > 0 else np.zeros(len(y))
        return d

    @staticmethod
    def _select_radius_supervised(d_self_all, y):
        candidates = np.unique(np.quantile(d_self_all, np.linspace(0.02, 0.97, 30)))
        best_r, best_f1 = 0.25, -1.0
        scored_candidates, scores = [], []
        for r in candidates:
            if r <= 0:
                continue
            pred = (d_self_all >= 1.5 * r).astype(int)
            tp = np.sum((pred == 1) & (y == 1))
            fp = np.sum((pred == 1) & (y == 0))
            fn = np.sum((pred == 0) & (y == 1))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            scored_candidates.append(float(r))
            scores.append(float(f1))
            if f1 > best_f1:
                best_f1, best_r = f1, r
        return max(float(best_r), 1e-3), np.array(scored_candidates), np.array(scores)

    def _select_radius_cv(self, Xn, y, seed=0, n_detectors_search=20,
                           n_candidates=9, search_folds=3):
        folds = _stratified_kfold_indices(y, k=search_folds, seed=seed)
        n_features = Xn.shape[1]
        self_all = Xn[y == 0]
        if len(self_all) == 0:
            self_all = Xn
        d_self_all = self._nearest_self_distances(Xn, y, self_all)
        candidates = np.unique(np.quantile(d_self_all, np.linspace(0.02, 0.95, n_candidates)))

        rng = np.random.default_rng(seed)
        best_r = float(candidates[len(candidates) // 2]) if len(candidates) else 0.25
        best_f1 = -1.0
        scored_candidates, scores = [], []

        for r in candidates:
            if r <= 0:
                continue
            fold_f1s = []
            for i in range(search_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(search_folds) if j != i])
                self_train = Xn[train_idx][y[train_idx] == 0]
                if len(self_train) == 0:
                    continue

                if self.detector_strategy == "mutation":
                    fold_detectors = self._generate_detectors_mutation(
                        self_train, r, n_features, rng, n_override=n_detectors_search)
                else:
                    fold_detectors = self._generate_detectors_random(
                        self_train, r, n_features, rng, n_override=n_detectors_search)

                X_val = Xn[val_idx]
                d_val = self._min_dist_to_set(X_val, self_train)
                proba = np.clip((d_val - r) / r, 0.0, 1.0)
                if fold_detectors.shape[0] > 0:
                    d_det = self._min_dist_to_set(X_val, fold_detectors)
                    matched = d_det <= r
                    proba = np.where(matched, np.maximum(proba, 0.7), proba)
                pred = (proba >= 0.5).astype(int)

                y_val = y[val_idx]
                tp = np.sum((pred == 1) & (y_val == 1))
                fp = np.sum((pred == 1) & (y_val == 0))
                fn = np.sum((pred == 0) & (y_val == 1))
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                fold_f1s.append(f1)

            mean_f1 = float(np.mean(fold_f1s)) if fold_f1s else -1.0
            scored_candidates.append(float(r))
            scores.append(mean_f1)
            if mean_f1 > best_f1:
                best_f1, best_r = mean_f1, r
        return max(float(best_r), 1e-3), np.array(scored_candidates), np.array(scores)

    def _generate_detectors_random(self, self_samples, r, n_features, rng, n_override=None):
        n = self.n_detectors if n_override is None else n_override
        if n <= 0 or len(self_samples) == 0:
            return np.empty((0, n_features))
        budget = max(n * self.max_tries_per_detector, n)
        batch_size = max(64, n * 4)
        detectors = []
        tried = 0
        while len(detectors) < n and tried < budget:
            batch = rng.uniform(0.0, 1.0, size=(batch_size, n_features))
            d = self._min_dist_to_set(batch, self_samples)
            valid = batch[d > r]
            if len(valid) > 0:
                detectors.append(valid[: n - len(detectors)])
            tried += batch_size
        if not detectors:
            return np.empty((0, n_features))
        return np.vstack(detectors)[:n]

    def _generate_detectors_mutation(self, self_samples, r, n_features, rng, n_override=None):
        n = self.n_detectors if n_override is None else n_override
        if n <= 0 or len(self_samples) == 0:
            return np.empty((0, n_features))
        binary_mask = self._binary_mask
        cont_mask = ~binary_mask
        n_self = len(self_samples)
        budget = max(n * self.max_tries_per_detector, n)
        batch_size = max(64, n * 4)
        detectors = []
        tried = 0
        while len(detectors) < n and tried < budget:
            base_idx = rng.integers(0, n_self, size=batch_size)
            batch = self_samples[base_idx].copy()

            jitter_scale = rng.uniform(0.05, 0.6, size=(batch_size, 1))
            if cont_mask.any():
                noise = rng.normal(0.0, 1.0, size=(batch_size, n_features)) * jitter_scale
                batch[:, cont_mask] = batch[:, cont_mask] + noise[:, cont_mask]
            if binary_mask.any():
                flip_dims = (rng.random((batch_size, n_features)) < self.flip_prob) & binary_mask[None, :]
                batch[flip_dims] = 1.0 - batch[flip_dims]

            batch = np.clip(batch, 0.0, 1.0)
            d = self._min_dist_to_set(batch, self_samples)
            valid = batch[d > r]
            if len(valid) > 0:
                detectors.append(valid[: n - len(detectors)])
            tried += batch_size
        if not detectors:
            return np.empty((0, n_features))
        return np.vstack(detectors)[:n]

    @classmethod
    def _detect_self_class(cls, Xn, y, sample_size=1500, random_state=None):
        rng = np.random.default_rng(random_state)
        n = len(Xn)
        if n > sample_size:
            idx = rng.choice(n, size=sample_size, replace=False)
            Xn, y = Xn[idx], y[idx]

        def separability(self_label):
            self_mask = (y == self_label)
            if self_mask.sum() < 5 or (~self_mask).sum() < 5:
                return 0.5
            self_pts = Xn[self_mask]
            # 0 for candidate-self rows, matching _nearest_self_distances'
            # own internal convention (it always treats y==0 as self).
            y_for_call = (~self_mask).astype(np.float64)
            d = cls._nearest_self_distances(Xn, y_for_call, self_pts)
            d_self, d_other = d[self_mask], d[~self_mask]
            rng2 = np.random.default_rng(0)
            m = min(len(d_self), len(d_other), 1500)
            a = rng2.choice(d_self, size=m)
            b = rng2.choice(d_other, size=m)
            return (b > a).mean()  # P(other class farther than self-cluster)

        sep_natural = separability(0.0)
        sep_flipped = separability(1.0)
        return 0.0 if sep_natural >= sep_flipped else 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        X = self._as_2d(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_features = X.shape[1]

        # one-hot/binary columns take only {0, 1} in the RAW data - detect
        # this before normalising so it's robust to columns that are
        # already 0/1 (get_dummies output, or binary indicator features).
        self._binary_mask = np.array([
            set(np.unique(X[:, j])) <= {0.0, 1.0} for j in range(n_features)
        ])

        self._min = X.min(axis=0)
        self._max = X.max(axis=0)
        Xn = self._normalise(X)

        self_class = (1.0 if self.label_flip else 0.0) if self.label_flip is not None \
            else self._detect_self_class(Xn, y, random_state=self.random_state)
        self._label_flipped = (self_class == 1.0)
        if self._label_flipped:
            y = 1.0 - y  # everything below now treats the (relabeled) y==0 as self, as usual

        self_samples = Xn[y == 0]
        if len(self_samples) == 0:
            self_samples = Xn
        self.self_samples_ = self_samples

        if self.self_radius == "auto":
            Xn_calib, y_calib, self_samples_calib = self._cap_for_calibration(
                Xn, y, max_calib=3000, random_state=self.random_state)
            if self.radius_calibration == "cv":
                r, cand, scores = self._select_radius_cv(Xn_calib, y_calib)
                y_label = "mean F1 (cross-validated)"
            else:
                d_self_all = self._nearest_self_distances(Xn_calib, y_calib, self_samples_calib)
                r, cand, scores = self._select_radius_supervised(d_self_all, y_calib)
                y_label = "F1 (in-sample)"
            self._resolved_radius = r
            self.calibration_history = {"x": cand, "y": scores, "chosen_x": r,
                                         "x_label": "candidate self_radius", "y_label": y_label}
        else:
            self._resolved_radius = float(self.self_radius)
            self.calibration_history = None
        r = self._resolved_radius

        if self.detector_strategy == "mutation":
            self.detectors = self._generate_detectors_mutation(self_samples, r, n_features, rng)
        else:
            self.detectors = self._generate_detectors_random(self_samples, r, n_features, rng)
        self._fitted = True

    def predictProba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = self._as_2d(X)
        Xn = self._normalise(X)
        r = self._resolved_radius

        d_self = self._min_dist_to_set(Xn, self.self_samples_)
        proba = np.clip((d_self - r) / r, 0.0, 1.0)

        if self.detectors.shape[0] > 0:
            d_det = self._min_dist_to_set(Xn, self.detectors)
            matched = d_det <= r
            proba = np.where(matched, np.maximum(proba, 0.7), proba)

        if self._label_flipped:
            proba = 1.0 - proba
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predictProba(X) >= 0.5).astype(int)

    def get_name(self) -> str:
        return "AISAgent"


# BaggingEnsemble
class BaggingEnsemble:

    def __init__(self, base_agent_factory, n_estimators: int = 15,
                 bootstrap: bool = True, random_state: int = None,
                 agent_kwargs: dict = None):
        self.base_agent_factory = base_agent_factory
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.agent_kwargs = agent_kwargs or {}
        self.models = []
        self.oob_score = float("nan")
        self.oob_proba_ = None  # (n_train,) OOB-estimated P(y=1), NaN where no OOB coverage
        self.calibration_debug = None  # set by precalibrate(), if the base agent defines one - a
                                        # search curve (e.g. AIS's F1-vs-candidate-radius) for
                                        # visualization, since no individual member re-runs it
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples = X.shape[0]

        agent_kwargs = self.agent_kwargs
        if hasattr(self.base_agent_factory, "precalibrate"):
            overrides = self.base_agent_factory.precalibrate(X, y, agent_kwargs)
            if overrides:
                overrides = dict(overrides)
                self.calibration_debug = overrides.pop("_calibration_debug", None)
                agent_kwargs = {**agent_kwargs, **overrides}

        self.models = []
        oob_proba_sum = np.zeros(n_samples)
        oob_count = np.zeros(n_samples)

        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.integers(0, n_samples, n_samples)
            else:
                idx = np.arange(n_samples)
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[np.unique(idx)] = False

            model = self.base_agent_factory(**agent_kwargs)
            model.fit(X[idx], y[idx])
            self.models.append(model)

            if oob_mask.any():
                oob_proba = model.predictProba(X[oob_mask])
                oob_proba_sum[oob_mask] += oob_proba
                oob_count[oob_mask] += 1

        has_oob = oob_count > 0
        if has_oob.any():
            oob_mean_proba = np.full(n_samples, np.nan)
            oob_mean_proba[has_oob] = oob_proba_sum[has_oob] / oob_count[has_oob]
            self.oob_proba_ = oob_mean_proba
            oob_pred = (oob_mean_proba[has_oob] >= 0.5).astype(int)
            self.oob_score = float(np.mean(oob_pred == y[has_oob]))
        else:
            self.oob_proba_ = np.full(n_samples, np.nan)
            self.oob_score = float("nan")

        self._fitted = True

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("BaggingEnsemble must be fit() before predicting.")

    def predictProba(self, X: np.ndarray) -> np.ndarray:
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)
        preds = np.array([m.predictProba(X) for m in self.models])
        return preds.mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predictProba(X) >= 0.5).astype(int)

    def get_oob_score(self) -> float:
        return self.oob_score

    def get_member_loss_curves(self):
        """Loss curves (steps, loss_history, step_label) for each bagged member
        that tracks one, for learning-curve visualization. Members that don't
        train via iterative loss minimization (e.g. RandomForest) are skipped."""
        curves = []
        for m in self.models:
            if getattr(m, "loss_history", None):
                steps = getattr(m, "loss_steps", list(range(len(m.loss_history))))
                label = getattr(m, "loss_step_label", "step")
                curves.append((steps, m.loss_history, label))
        return curves


AGENT_REGISTRY = {
    "LogisticRegression": (LogisticRegressionAgent, dict(lr=0.15, n_iters=1200)),
    "SVM": (SVMAgent, dict(C=1.0, kernel="rbf", max_passes=5, max_iter=40, random_state=0)),
    "RandomForest": (RandomForestAgent, dict(n_estimators=8, max_depth=8, min_samples_leaf=2, random_state=0)),
    "MLP": (MLPAgent, dict(hidden_layer_sizes=(32, 16), lr=0.05, n_epochs=150, seed=0)),
    "AIS": (AISAgent, dict(n_detectors=150, self_radius="auto",
                           detector_strategy="mutation", radius_calibration="cv", random_state=0)),
}

