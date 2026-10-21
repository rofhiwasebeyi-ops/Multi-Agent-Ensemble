"""Figure builders for model structure: MLP network diagram and random-forest
tree diagrams. Return matplotlib Figure objects built with the object API
(no pyplot), so they are safe to create on any thread and embed in Tk."""
import numpy as np
from matplotlib.figure import Figure


def _short(name, n=22):
    name = str(name)
    return name if len(name) <= n else name[:n - 1] + "…"


def plot_mlp_network(agent, feature_names=None, max_inputs=12, max_hidden_nodes=16,
                     title=None, figsize=(9.5, 6.5), dpi=100):
    """Layer/neuron diagram of a fitted MLPAgent. Edge colour = weight sign
    (blue positive, red negative), edge width/opacity = |weight|. Inputs are
    the `max_inputs` features with the largest total outgoing |weight|;
    wide hidden layers are truncated to `max_hidden_nodes` (largest first)."""
    Ws = agent.Ws
    if Ws is None:
        raise ValueError("MLP is not fitted")
    n_in = Ws[0].shape[0]
    names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(n_in)]

    shown = []  # per layer: indices displayed
    in_strength = np.abs(Ws[0]).sum(axis=1)
    shown.append(np.sort(np.argsort(-in_strength)[:max_inputs]))
    for W in Ws[:-1]:  # hidden layers: rank by outgoing strength where possible
        n = W.shape[1]
        shown.append(np.arange(n))
    # rank hidden nodes by outgoing weight magnitude
    for li in range(1, len(Ws)):
        out_strength = np.abs(Ws[li]).sum(axis=1)
        shown[li] = np.sort(np.argsort(-out_strength)[:max_hidden_nodes])
    shown.append(np.array([0]))  # output

    fig = Figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    n_layers = len(shown)
    xs = np.linspace(0.08, 0.92, n_layers)
    pos = []
    for li, idx in enumerate(shown):
        k = len(idx)
        ys = np.linspace(0.92, 0.08, k) if k > 1 else np.array([0.5])
        pos.append(ys)

    wmax = max(np.abs(W).max() for W in Ws) or 1.0
    for li, W in enumerate(Ws):
        for a, i in enumerate(shown[li]):
            for b, j in enumerate(shown[li + 1]):
                w = W[i, j]
                s = abs(w) / wmax
                ax.plot([xs[li], xs[li + 1]], [pos[li][a], pos[li + 1][b]],
                        color="#2563eb" if w > 0 else "#dc2626",
                        alpha=0.12 + 0.75 * s, lw=0.3 + 2.2 * s, zorder=1)
    for li, idx in enumerate(shown):
        color = "#e5e7eb" if li == 0 else ("#fde68a" if li == n_layers - 1 else "#bfdbfe")
        ax.scatter([xs[li]] * len(idx), pos[li], s=260, c=color, edgecolors="#374151", zorder=3)
    for a, i in enumerate(shown[0]):
        ax.text(xs[0] - 0.02, pos[0][a], _short(names[i]), ha="right", va="center", fontsize=8)
    ax.text(xs[-1] + 0.02, 0.5, "P(at risk)", ha="left", va="center", fontsize=9, fontweight="bold")

    hidden = list(getattr(agent, "hidden_layer_sizes", ()))
    labels = [f"Input\n({n_in} features" + (f", top {len(shown[0])} shown)" if n_in > len(shown[0]) else ")")]
    for li, h in enumerate(hidden, start=1):
        labels.append(f"Hidden {li}\n({h} neurons" + (f", {len(shown[li])} shown)" if h > len(shown[li]) else ")"))
    labels.append("Output\n(sigmoid)")
    for li, lab in enumerate(labels):
        ax.text(xs[li], 0.99, lab, ha="center", va="bottom", fontsize=9, fontweight="bold")
    n_params = sum(W.size + b.size for W, b in zip(agent.Ws, agent.bs))
    fig.suptitle(title or f"MLP architecture {tuple(hidden)}  -  {n_params} trainable parameters",
                 fontsize=11)
    fig.text(0.5, 0.01, "blue = positive weight, red = negative weight; thicker/darker = larger |weight|",
             ha="center", fontsize=8, color="#555")
    ax.set_xlim(-0.12, 1.05)
    ax.set_ylim(0.0, 1.08)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.05)
    return fig


def _layout(node, depth, max_depth, counter, out):
    """In-order x layout. Returns x of node. Nodes deeper than max_depth are
    collapsed into a single '…' stub."""
    if node.value is not None or depth >= max_depth:
        x = counter[0]; counter[0] += 1
        out.append((node, depth, x, None, None, node.value is None))
        return x
    xl = _layout(node.left, depth + 1, max_depth, counter, out)
    xr = _layout(node.right, depth + 1, max_depth, counter, out)
    x = (xl + xr) / 2.0
    out.append((node, depth, x, xl, xr, False))
    return x


def tree_stats(tree):
    """(n_nodes, n_leaves, depth) of a fitted _CARTTree."""
    def rec(n):
        if n.value is not None:
            return 1, 1, 0
        a = rec(n.left); b = rec(n.right)
        return a[0] + b[0] + 1, a[1] + b[1], 1 + max(a[2], b[2])
    return rec(tree.root)


def plot_forest_tree(tree, feature_names=None, max_depth=3, title=None, dpi=100):
    """Draw one CART tree down to `max_depth` levels. Internal nodes show the
    split rule (feature <= threshold; left branch = True); leaves show the
    P(at risk) they predict and are coloured green (low) to red (high)."""
    if tree.root is None:
        raise ValueError("tree is not fitted")
    out = []
    _layout(tree.root, 0, max_depth, [0], out)
    n_leaf_slots = max(o[2] for o in out) + 1
    fig = Figure(figsize=(max(8.0, 1.7 * n_leaf_slots), 1.6 + 1.5 * (max_depth + 1)), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    coords = {id(o[0]): (o[2], -o[1]) for o in out}
    cmap_lo, cmap_hi = np.array([0.55, 0.85, 0.6]), np.array([0.95, 0.5, 0.5])
    for node, d, x, xl, xr, collapsed in out:
        if xl is not None:
            for child, lab in ((node.left, "yes"), (node.right, "no")):
                cx, cy = coords[id(child)]
                ax.plot([x, cx], [-d, cy], color="#6b7280", lw=1, zorder=1)
                ax.text((x + cx) / 2, (-d + cy) / 2 + 0.05, lab, fontsize=7, color="#6b7280", ha="center")
    for node, d, x, xl, xr, collapsed in out:
        if collapsed:
            txt, fc = "… subtree\ncollapsed", "#f3f4f6"
        elif node.value is not None:
            txt = f"P(at risk)\n{node.value:.2f}"
            fc = tuple(cmap_lo + (cmap_hi - cmap_lo) * float(node.value))
        else:
            fn = feature_names[node.feature] if feature_names is not None and node.feature < len(feature_names) \
                else f"x{node.feature}"
            txt, fc = f"{_short(fn, 26)}\n<= {node.threshold:.3g}", "#dbeafe"
        ax.text(x, -d, txt, ha="center", va="center", fontsize=8, zorder=3,
                bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec="#374151", lw=0.8))
    n_nodes, n_leaves, depth = tree_stats(tree)
    fig.suptitle(title or f"Decision tree  ({n_nodes} nodes, {n_leaves} leaves, depth {depth}; "
                          f"top {max_depth} levels shown)", fontsize=11)
    ax.set_xlim(-0.7, n_leaf_slots - 0.3)
    ax.set_ylim(-max_depth - 0.6, 0.5)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.02)
    return fig


def plot_forest_overview(forest, feature_names=None, title=None, figsize=(9, 5), dpi=100):
    """Forest-level summary: per-tree size/depth bars plus how often each
    feature is used as a split across all trees (top 12)."""
    from collections import Counter
    stats = [tree_stats(t) for t in forest.trees]
    cnt = Counter()

    def rec(n):
        if n.value is None:
            cnt[n.feature] += 1; rec(n.left); rec(n.right)
    for t in forest.trees:
        rec(t.root)
    fig = Figure(figsize=figsize, dpi=dpi)
    a1 = fig.add_subplot(121); a2 = fig.add_subplot(122)
    idx = np.arange(len(stats))
    a1.bar(idx, [s[1] for s in stats], color="#60a5fa")
    a1.set_xlabel("Tree #"); a1.set_ylabel("Leaves")
    a1.set_title("Leaves per tree (depth: " + ", ".join(str(s[2]) for s in stats[:6]) +
                 ("…" if len(stats) > 6 else "") + ")", fontsize=9)
    top = cnt.most_common(12)[::-1]
    nm = [_short(feature_names[f] if feature_names is not None and f < len(feature_names) else f"x{f}", 24)
          for f, _ in top]
    a2.barh(nm, [c for _, c in top], color="#34d399")
    a2.set_xlabel("Times used as a split"); a2.set_title("Most-used split features", fontsize=9)
    a2.tick_params(axis="y", labelsize=7)
    fig.suptitle(title or f"Random forest overview ({len(stats)} trees)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig
