"""Tkinter GUI wrapping the MAS ensemble pipeline: dataset loading, agent/
fusion configuration, running analyses, architecture search, learning
curves, SHAP explanations, and chart/report export.
"""
import numpy as np
import os
import shap
import threading
import time

from scratch_pipeline import (
    AGENT_REGISTRY, BaggingEnsemble, MASController,
    mlp_architecture_search, rf_architecture_search,
)
from shared import (
    accuracy, confusion_matrix, precision_recall_f1,
    load_and_preprocess, make_synthetic_dataset, train_test_split, group_train_test_split,
    fairness_report, ExplainabilityEngine, save_model, load_model,
)


class UserInterface:

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        self.tk, self.ttk = tk, ttk
        self.filedialog, self.messagebox = filedialog, messagebox

        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self._plt = plt
        self._Figure = Figure
        self._FigureCanvasTkAgg = FigureCanvasTkAgg
        self.chart_canvas = None
        self.chart_history = []   # list of (name, fig, filepath) for every chart saved this session
        self.session_dir = None   # output folder, created lazily on the first saved chart

        self.root = root
        self.root.title("Multi-Agent Ensemble - Prototype")
        self.root.geometry("1500x800")
        self._destroyed = False
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.dataset_path = None
        self.dataset_type = "UCI"
        self.X = self.y = self.header = None
        self.grade_idx = self.behaviour_idx = self.full_idx = None
        self.protected_attributes = None
        self.groups = None
        self.feature_groups = None
        self.last_results = None  # populated by run_analysis()
        self._explain_context = None  # populated by run_analysis(), used by explain_results()

        self._build_layout()

    # UI construction
    def _build_layout(self):
        tk, ttk = self.tk, self.ttk

        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Dataset:").pack(side="left")
        self.dataset_var = tk.StringVar(value="UCI Student Performance")
        self.dataset_menu = ttk.Combobox(
            top, textvariable=self.dataset_var,
            values=["UCI Student Performance", "OULAD"],
            state="readonly", width=25
        )
        self.dataset_menu.pack(side="left", padx=5)
        self.select_dataset_button = ttk.Button(top, text="Select Dataset", command=self.select_dataset)
        self.select_dataset_button.pack(side="left", padx=(5, 0))
        self.dataset_label = ttk.Label(top, text="No dataset loaded (synthetic demo data available)")
        self.dataset_label.pack(side="left", padx=10)

        params = ttk.LabelFrame(self.root, text="Configure Parameters", padding=10)
        params.pack(fill="x", padx=10, pady=5)

        ttk.Label(params, text="Agent:").grid(row=0, column=0, sticky="w")
        self.agent_var = tk.StringVar(value="RandomForest")
        agent_menu = ttk.Combobox(params, textvariable=self.agent_var,
                                   values=list(AGENT_REGISTRY.keys()), state="readonly", width=20)
        agent_menu.grid(row=0, column=1, padx=5)

        ttk.Label(params, text="Use BaggingEnsemble:").grid(row=0, column=2, sticky="w", padx=(20, 0))
        self.bagging_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(params, variable=self.bagging_var).grid(row=0, column=3)

        self.full_ensemble_var = tk.BooleanVar(value=False)

        ttk.Label(params, text="Fusion strategy:").grid(row=2, column=0, sticky="w", pady=(6, 0))
        self.fusion_type_var = tk.StringVar(value="Logistic stacker")
        self.fusion_menu = ttk.Combobox(
            params, textvariable=self.fusion_type_var,
            values=["Logistic stacker", "Gating network (context-aware, per-instance)"],
            state="disabled", width=40)
        self.fusion_menu.grid(row=2, column=1, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Checkbutton(params, text="Full ensemble (all 5 agents + fusion)",
                         variable=self.full_ensemble_var,
                         command=lambda: [
                             agent_menu.config(state="disabled" if self.full_ensemble_var.get() else "readonly"),
                             self.fusion_menu.config(state="readonly" if self.full_ensemble_var.get() else "disabled"),
                             auto_tune_check.config(state="normal" if self.full_ensemble_var.get() else "disabled"),
                             auto_tune_rf_check.config(state="normal" if self.full_ensemble_var.get() else "disabled"),
                             ais_grades_only_check.config(state="normal" if self.full_ensemble_var.get() else "disabled"),
                         ]).grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))

        self.auto_tune_mlp_var = tk.BooleanVar(value=False)
        auto_tune_check = ttk.Checkbutton(
            params, text="Auto-tune MLP architecture (5-fold CV search before training)",
            variable=self.auto_tune_mlp_var, state="disabled")
        auto_tune_check.grid(row=5, column=0, columnspan=4, sticky="w", pady=(6, 0))

        self.auto_tune_rf_var = tk.BooleanVar(value=False)
        auto_tune_rf_check = ttk.Checkbutton(
            params, text="Auto-tune RandomForest (depth/leaf-size/class-weight CV search before training)",
            variable=self.auto_tune_rf_var, state="disabled")
        auto_tune_rf_check.grid(row=7, column=0, columnspan=4, sticky="w", pady=(6, 0))

        self.ais_grades_only_var = tk.BooleanVar(value=True)
        ais_grades_only_check = ttk.Checkbutton(
            params, text="Use specialised feature groups for the heterogeneous agents",
            variable=self.ais_grades_only_var, state="disabled")
        ais_grades_only_check.grid(row=6, column=0, columnspan=4, sticky="w", pady=(6, 0))

        ttk.Label(params, text="n_estimators:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.n_estimators_var = tk.IntVar(value=10)
        ttk.Entry(params, textvariable=self.n_estimators_var, width=8).grid(row=3, column=1, sticky="w", pady=(6, 0))

        ttk.Label(params, text="Test fraction:").grid(row=3, column=2, sticky="w", padx=(20, 0), pady=(6, 0))
        self.test_frac_var = tk.DoubleVar(value=0.25)
        ttk.Entry(params, textvariable=self.test_frac_var, width=8).grid(row=3, column=3, sticky="w", pady=(6, 0))

        ttk.Label(params, text="Feature set:").grid(row=4, column=0, sticky="w", pady=(6, 0))
        self.feature_set_var = tk.StringVar(value="Full features")
        ttk.Combobox(params, textvariable=self.feature_set_var,
                     values=["Full features", "Academic/assessment features", "Engagement/behaviour features"],
                     state="readonly", width=28).grid(row=4, column=1, columnspan=2, sticky="w", pady=(6, 0))

        actions = ttk.Frame(self.root, padding=10)
        actions.pack(fill="x")
        self.run_button = ttk.Button(actions, text="Run Analysis", command=self.run_analysis)
        self.run_button.pack(side="left")
        self.explain_button = ttk.Button(actions, text="Explain", command=self.explain_results)
        self.explain_button.pack(side="left", padx=10)
        self.mlp_search_button = ttk.Button(actions, text="MLP Architecture Search", command=self.run_mlp_search)
        self.mlp_search_button.pack(side="left", padx=10)
        self.rf_search_button = ttk.Button(actions, text="RandomForest Architecture Search", command=self.run_rf_search)
        self.rf_search_button.pack(side="left", padx=10)
        self.learning_curves_button = ttk.Button(actions, text="Learning Curves", command=self.show_learning_curves)
        self.learning_curves_button.pack(side="left", padx=10)
        self.export_button = ttk.Button(actions, text="Export Report", command=self.export_report)
        self.export_button.pack(side="left")
        self.open_folder_button = ttk.Button(actions, text="Open Visuals Folder", command=self.reveal_session_dir)
        self.open_folder_button.pack(side="left", padx=10)
        self.save_model_button = ttk.Button(actions, text="Save Model", command=self.save_model_to_disk)
        self.save_model_button.pack(side="left")
        self.load_model_button = ttk.Button(actions, text="Load Model", command=self.load_model_from_disk)
        self.load_model_button.pack(side="left", padx=10)
        self._busy_buttons = [self.select_dataset_button, self.run_button, self.explain_button,
                               self.mlp_search_button, self.rf_search_button,
                               self.learning_curves_button, self.export_button,
                               self.open_folder_button, self.save_model_button, self.load_model_button]

        body = tk.PanedWindow(self.root, orient="horizontal", sashrelief="raised", sashwidth=6)
        body.pack(fill="both", expand=True, padx=10, pady=5)

        results_frame = ttk.LabelFrame(body, text="Results", padding=10)
        self.results_text = tk.Text(results_frame, height=20, width=48, wrap="word")
        self.results_text.pack(fill="both", expand=True)
        body.add(results_frame, stretch="always", minsize=350)

        self.chart_frame = ttk.LabelFrame(body, text="Latest chart (all charts are auto-saved as PNGs)", padding=10)
        body.add(self.chart_frame, stretch="always", minsize=720)

        chart_scroll_canvas = tk.Canvas(self.chart_frame, highlightthickness=0)
        v_scroll = ttk.Scrollbar(self.chart_frame, orient="vertical", command=chart_scroll_canvas.yview)
        h_scroll = ttk.Scrollbar(self.chart_frame, orient="horizontal", command=chart_scroll_canvas.xview)
        chart_scroll_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.chart_frame.rowconfigure(0, weight=1)
        self.chart_frame.columnconfigure(0, weight=1)
        chart_scroll_canvas.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        self.chart_inner = ttk.Frame(chart_scroll_canvas)
        chart_scroll_canvas.create_window((0, 0), window=self.chart_inner, anchor="nw")
        self.chart_inner.bind(
            "<Configure>",
            lambda e: chart_scroll_canvas.configure(scrollregion=chart_scroll_canvas.bbox("all")))
        self._chart_scroll_canvas = chart_scroll_canvas

        self._chart_placeholder = ttk.Label(
            self.chart_inner, text="Run an analysis, Explain, or search MLP\narchitectures to see charts here.\n"
                                    "Every chart is also saved automatically\nto a per-session output folder.",
            justify="center")
        self._chart_placeholder.pack(expand=True, padx=40, pady=40)

    # output-folder helpers
    def _ensure_session_dir(self) -> str:
        """Creates (once) a timestamped folder for this session's visuals,
        e.g. ./mas_visuals/run_20260802_101500/, and returns its path."""
        if self.session_dir is None:
            import os
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = os.path.join(os.getcwd(), "mas_visuals")
            self.session_dir = os.path.join(base_dir, f"run_{timestamp}")
            os.makedirs(self.session_dir, exist_ok=True)
        return self.session_dir

    def reveal_session_dir(self) -> None:
        if self.session_dir is None:
            self.messagebox.showinfo("No visuals yet", "No charts have been generated this session yet.")
            return
        self.messagebox.showinfo("Visuals folder", f"Charts for this session are saved to:\n{self.session_dir}")

    # chart helpers
    def _show_chart(self, fig, name: str = "chart") -> None:
        """Immediately saves the figure (full size, on its own) to this
        session's output folder, then swaps it into the chart preview panel
        at its own native pixel size (figsize * dpi)."""
        import os
        session_dir = self._ensure_session_dir()
        idx = len(self.chart_history) + 1
        fname = f"{idx:02d}_{name}.png"
        fpath = os.path.join(session_dir, fname)
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        self.chart_history.append((name, fig, fpath))

        if self.chart_canvas is not None:
            self.chart_canvas.get_tk_widget().destroy()
            self.chart_canvas = None
        if self._chart_placeholder is not None:
            self._chart_placeholder.destroy()
            self._chart_placeholder = None

        w_px = max(1, int(round(fig.get_figwidth() * fig.get_dpi())))
        h_px = max(1, int(round(fig.get_figheight() * fig.get_dpi())))
        self.chart_canvas = self._FigureCanvasTkAgg(fig, master=self.chart_inner)
        widget = self.chart_canvas.get_tk_widget()
        widget.configure(width=w_px, height=h_px)
        self.chart_canvas.draw()
        widget.pack(padx=10, pady=10)
        self.chart_inner.update_idletasks()
        self._chart_scroll_canvas.configure(scrollregion=self._chart_scroll_canvas.bbox("all"))
        return fpath

    def _plot_run_results(self, result: dict) -> None:
        """Two separate, full-size bar charts (each auto-saved on its own):
        per-agent standalone test accuracy, and fusion/gate trust weights."""
        fig1 = self._Figure(figsize=(6.4, 4.2), dpi=100)
        ax1 = fig1.add_subplot(111)
        names = list(result["per_agent_test_acc"].keys())
        accs = list(result["per_agent_test_acc"].values())
        ax1.bar(names, accs, color="#4C72B0")
        ax1.set_ylim(0, 1)
        ax1.set_title("Per-agent standalone test accuracy")
        ax1.tick_params(axis="x", rotation=30)
        for i, v in enumerate(accs):
            ax1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
        fig1.tight_layout()
        self._show_chart(fig1, name="per_agent_accuracy")

        fig2 = self._Figure(figsize=(6.4, 4.2), dpi=100)
        ax2 = fig2.add_subplot(111)
        fnames = list(result["fusion_weights"].keys())
        fweights = list(result["fusion_weights"].values())
        label = "Mean gate weight" if result.get("fusion_type") == "gating" else "Fusion weight"
        ax2.bar(fnames, fweights, color="#55A868")
        ax2.set_title(f"{label} per agent")
        ax2.tick_params(axis="x", rotation=30)
        ax2.axhline(0, color="black", linewidth=0.6)
        fig2.tight_layout()
        self._show_chart(fig2, name="fusion_weights")

    def _plot_fusion_impact(self, result: dict) -> None:
        per_agent_acc = result["per_agent_test_acc"]
        fused_acc = result["accuracy"]
        best_agent_name = max(per_agent_acc, key=per_agent_acc.get)
        best_agent_acc = per_agent_acc[best_agent_name]
        lift = fused_acc - best_agent_acc

        names = list(per_agent_acc.keys()) + ["Fused\n(learned)"]
        vals = list(per_agent_acc.values()) + [fused_acc]
        colors = ["#4C72B0" if n != best_agent_name else "#DD8452" for n in per_agent_acc.keys()]
        colors.append("#55A868")

        fig = self._Figure(figsize=(6.8, 4.6), dpi=100)
        ax = fig.add_subplot(111)
        bars = ax.bar(names, vals, color=colors)
        ax.set_ylim(0, 1)
        fusion_label = "Gating network" if result.get("fusion_type") == "gating" else "Logistic stacker"
        ax.set_title(f"Fusion layer impact ({fusion_label})\n"
                      f"best single agent = {best_agent_name} ({best_agent_acc:.3f}), "
                      f"fused = {fused_acc:.3f}", fontsize=10)
        ax.set_ylabel("Test accuracy")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                    ha="center", fontsize=8)

        ax.annotate(
            f"{'+' if lift >= 0 else ''}{lift:.3f} vs. best single agent",
            xy=(len(names) - 1, fused_acc), xytext=(len(names) - 2.6, min(fused_acc + 0.18, 0.97)),
            arrowprops=dict(arrowstyle="->", color="#55A868"),
            fontsize=9, color="#2E7D46", fontweight="bold")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        self._show_chart(fig, name="fusion_impact_vs_best_agent")

    def _plot_shap_global(self, explanation) -> None:
        """Global SHAP feature importance, rendered with SHAP's own
        matplotlib plotting (shap.plots.bar) rather than a hand-rolled bar
        chart, then captured into a Figure and auto-saved."""
        plt = self._plt
        plt.figure(figsize=(6.8, 4.6))
        shap.plots.bar(explanation, show=False)
        fig = plt.gcf()
        fig.suptitle("Global feature importance (SHAP, mean |value|)", fontsize=10)
        fig.tight_layout()
        self._show_chart(fig, name="shap_global_bar")

    def _plot_shap_local(self, explanation) -> None:
        """Local, per-student SHAP explanation for a single instance,
        rendered with SHAP's own matplotlib waterfall plot."""
        plt = self._plt
        plt.figure(figsize=(6.8, 4.6))
        shap.plots.waterfall(explanation, show=False)
        fig = plt.gcf()
        fig.suptitle("Local SHAP explanation (this student)", fontsize=10)
        fig.tight_layout()
        self._show_chart(fig, name="shap_local_waterfall")

    def _plot_gate_weights_instance(self, gate_pairs) -> None:
        """Per-instance agent trust weights from the gating fusion network,
        for the one student being explained."""
        fig = self._Figure(figsize=(5.6, 4.0), dpi=100)
        ax = fig.add_subplot(111)
        names3 = [n for n, _ in gate_pairs]
        vals3 = [v for _, v in gate_pairs]
        ax.bar(names3, vals3, color="#8172B2")
        ax.set_ylim(0, 1)
        ax.set_title("Gate trust for this student (sums to 1)")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        self._show_chart(fig, name="gate_weights_instance")

    def _plot_mlp_search(self, leaderboard) -> None:
        """Bar chart of mean CV accuracy (with std error bars) per architecture,
        sorted best-first, from an MLP architecture search."""
        fig = self._Figure(figsize=(6.8, 4.6), dpi=100)
        ax = fig.add_subplot(111)
        labels = [str(e["hidden_layer_sizes"]) for e in leaderboard]
        accs = [e["mean_acc"] for e in leaderboard]
        stds = [e["std_acc"] for e in leaderboard]
        colors = ["#55A868"] + ["#4C72B0"] * (len(labels) - 1) if labels else []
        ax.bar(labels, accs, yerr=stds, capsize=4, color=colors)
        ax.set_ylabel("Mean CV accuracy")
        ax.set_title("MLP architecture search (best = green)")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        self._show_chart(fig, name="mlp_architecture_search")

    def _plot_rf_search(self, leaderboard) -> None:
        """Bar chart of mean CV accuracy (with std error bars) per
        hyperparameter combination, sorted best-first, from a RandomForest
        architecture search."""
        fig = self._Figure(figsize=(6.8, 4.6), dpi=100)
        ax = fig.add_subplot(111)
        labels = [f"d{e['max_depth']}/mss{e['min_samples_split']}/msl{e['min_samples_leaf']}"
                  f"{'/bal' if e['class_weight'] == 'balanced' else ''}" for e in leaderboard]
        accs = [e["mean_acc"] for e in leaderboard]
        stds = [e["std_acc"] for e in leaderboard]
        colors = ["#55A868"] + ["#4C72B0"] * (len(labels) - 1) if labels else []
        ax.bar(labels, accs, yerr=stds, capsize=4, color=colors)
        ax.set_ylabel("Mean CV accuracy")
        ax.set_title("RandomForest hyperparameter search (best = green)")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        fig.tight_layout()
        self._show_chart(fig, name="rf_architecture_search")

    @staticmethod
    def _loss_curve_of(model):
        """Returns (steps, loss_history, step_label) if the model tracked one
        during training, else None. Covers LogisticRegression, SVM (Platt
        calibration), MLP, FusionLayer, and GatingFusionLayer."""
        if getattr(model, "loss_history", None):
            steps = getattr(model, "loss_steps", list(range(len(model.loss_history))))
            label = getattr(model, "loss_step_label", "step")
            return steps, model.loss_history, label
        return None

    @staticmethod
    def _calibration_curve_of(model):
        """Returns a calibration_history dict if the model tracked one, else
        None."""
        hist = getattr(model, "calibration_history", None)
        if hist and len(hist.get("x", [])) > 0:
            return hist
        if isinstance(model, BaggingEnsemble):
            debug = getattr(model, "calibration_debug", None)
            if debug and len(debug.get("x", [])) > 0:
                return debug
        return None

    def _plot_learning_curves(self, agents_dict: dict, fusion_model=None) -> None:
        """One SEPARATE, full-size figure per model that exposes a loss
        history OR a calibration-search history."""
        loss_entries = []          # (title, steps, losses, step_label, is_fusion)
        calibration_entries = []   # (title, history_dict)

        for name, model in agents_dict.items():
            if isinstance(model, BaggingEnsemble):
                member_curves = model.get_member_loss_curves()
                if member_curves:
                    min_len = min(len(c[1]) for c in member_curves)
                    steps = member_curves[0][0][:min_len]
                    label = member_curves[0][2]
                    losses = np.array([c[1][:min_len] for c in member_curves]).mean(axis=0)
                    loss_entries.append((f"{name} (bagged mean, n={len(member_curves)})",
                                          steps, losses, label, False))
                cal = self._calibration_curve_of(model)
                if cal:
                    calibration_entries.append((f"{name} (radius precalibrated once, "
                                                  f"shared by all bagged members)", cal))
            else:
                curve = self._loss_curve_of(model)
                if curve:
                    steps, losses, label = curve
                    loss_entries.append((name, steps, losses, label, False))
                cal = self._calibration_curve_of(model)
                if cal:
                    calibration_entries.append((name, cal))

        if fusion_model is not None:
            curve = self._loss_curve_of(fusion_model)
            if curve:
                steps, losses, label = curve
                loss_entries.append((f"Fusion: {fusion_model.get_name()}", steps, losses, label, True))
            cal = self._calibration_curve_of(fusion_model)
            if cal:
                calibration_entries.append((f"Fusion: {fusion_model.get_name()}", cal))

        if not loss_entries and not calibration_entries:
            self.messagebox.showinfo(
                "No learning curves",
                "None of the trained models in this run expose a loss or calibration "
                "history (e.g. RandomForest doesn't train via iterative loss minimization "
                "or hyperparameter search).")
            return

        for title, steps, losses, label, is_fusion in loss_entries:
            fig = self._Figure(figsize=(6.0, 3.6), dpi=100)
            ax = fig.add_subplot(111)
            color = "#55A868" if is_fusion else "#4C72B0"
            ax.plot(steps, losses, color=color, linewidth=1.8)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel("loss", fontsize=9)
            if is_fusion and len(losses) > 1:
                ax.annotate(f"start={losses[0]:.4f}\nend={losses[-1]:.4f}",
                            xy=(steps[-1], losses[-1]), xytext=(0.55, 0.75),
                            textcoords="axes fraction", fontsize=8,
                            arrowprops=dict(arrowstyle="->", color=color))
            fig.tight_layout()
            safe_title = "".join(c if c.isalnum() else "_" for c in title.lower())[:40]
            chart_name = f"learning_curve_fusion_{safe_title}" if is_fusion else f"learning_curve_{safe_title}"
            self._show_chart(fig, name=chart_name)

        for title, hist in calibration_entries:
            self._plot_calibration_curve(title, hist)

    def _plot_calibration_curve(self, title: str, hist: dict) -> None:
        """Plots a hyperparameter calibration search (score vs. candidate
        value)."""
        x = np.asarray(hist["x"], dtype=float)
        y_vals = np.asarray(hist["y"], dtype=float)
        order = np.argsort(x)
        x, y_vals = x[order], y_vals[order]

        fig = self._Figure(figsize=(6.0, 3.6), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x, y_vals, color="#C44E52", linewidth=1.8, marker="o", markersize=3)
        if hist.get("x_log"):
            ax.set_xscale("log")  # e.g. FusionLayer's L2 candidates span orders of magnitude
        chosen_x = hist.get("chosen_x")
        if chosen_x is not None:
            ax.axvline(chosen_x, color="#4C72B0", linestyle="--", linewidth=1.2)
            ax.annotate(f"chosen={chosen_x:.3g}", xy=(chosen_x, ax.get_ylim()[1]),
                        xytext=(4, -4), textcoords="offset points", fontsize=8, color="#4C72B0")
        ax.set_title(f"{title}\nHyperparameter calibration search (grid search, not gradient descent)",
                      fontsize=9)
        ax.set_xlabel(hist.get("x_label", "candidate"), fontsize=9)
        ax.set_ylabel(hist.get("y_label", "score"), fontsize=9)
        fig.tight_layout()
        safe_title = "".join(c if c.isalnum() else "_" for c in title.lower())[:40]
        self._show_chart(fig, name=f"calibration_curve_{safe_title}")

    def show_learning_curves(self) -> None:
        ctx = self._explain_context
        if ctx is None:
            self.messagebox.showwarning("Nothing to plot", "Run an analysis first.")
            return
        if ctx.get("controller") is not None:
            controller = ctx["controller"]
            self._plot_learning_curves(controller.agents, controller.fusion)
        elif ctx.get("single_model") is not None:
            self._plot_learning_curves({"Model": ctx["single_model"]})
        else:
            self.messagebox.showwarning("Nothing to plot", "Run an analysis first.")

    # dialog methods
    def select_dataset(self) -> None:
        """Select raw data according to the dataset chosen in the GUI.
        """
        dataset_choice = self.dataset_var.get()

        if dataset_choice == "OULAD":
            path = self.filedialog.askdirectory(title="Select extracted OULAD raw-data folder")
            if not path:
                return
            dataset_type = "oulad"
        else:
            path = self.filedialog.askopenfilename(
                title="Select UCI Student Performance CSV",
                filetypes=[("CSV files", "*.csv")]
            )
            if not path:
                return
            dataset_type = "uci"

        try:
            (self.X, self.y, self.header, feature_groups,
             self.protected_attributes, self.groups) = load_and_preprocess(
                path, dataset_type=dataset_type
            )
            self.dataset_type = dataset_choice
            self.feature_groups = feature_groups
            self.grade_idx = feature_groups.get("academic_idx", [])
            self.behaviour_idx = feature_groups.get("behaviour_idx", [])
            self.engagement_idx = feature_groups.get("engagement_idx", [])
            self.demographic_idx = feature_groups.get("demographic_idx", [])
            self.full_idx = feature_groups.get("full_idx", list(range(self.X.shape[1])))
            self._finish_dataset_load(path)

        except Exception as e:
            self.messagebox.showerror(
                f"Failed to load {dataset_choice} dataset", str(e)
            )

    def _finish_dataset_load(self, path: str) -> None:
        self.dataset_path = path
        extra = ""
        if self.groups is not None:
            extra += ", group-aware student split"
        if self.protected_attributes:
            extra += f", protected attrs: {list(self.protected_attributes.keys())}"
        display_name = os.path.basename(os.path.normpath(path))
        self.dataset_label.config(text=f"{self.dataset_type}: {display_name} "
                                        f"({self.X.shape[0]} rows, {self.X.shape[1]} features, "
                                        f"class balance={self.y.mean():.2f}{extra})")

    def _ask_threshold(self, explanation: str):
        """Small modal dialog asking for a numeric cutoff to binarize the target column."""
        dialog = self.tk.Toplevel(self.root)
        dialog.title("Binarize target column")
        dialog.grab_set()
        self.tk.Label(dialog, text=explanation, wraplength=420, justify="left").pack(padx=10, pady=10)
        self.tk.Label(dialog, text="At-risk threshold (samples BELOW this value -> at-risk=1):").pack(padx=10)
        entry_var = self.tk.StringVar()
        entry = self.ttk.Entry(dialog, textvariable=entry_var)
        entry.pack(padx=10, pady=5)
        entry.focus_set()
        result = {"value": None}

        def on_ok():
            result["value"] = entry_var.get()
            dialog.destroy()

        def on_cancel():
            dialog.destroy()

        btns = self.ttk.Frame(dialog)
        btns.pack(pady=(0, 10))
        self.ttk.Button(btns, text="OK", command=on_ok).pack(side="left", padx=5)
        self.ttk.Button(btns, text="Cancel", command=on_cancel).pack(side="left", padx=5)
        dialog.wait_window()
        return result["value"]

    def configure_parameters(self) -> dict:
        """Reads current widget state into a plain parameter dict."""
        fusion_label = self.fusion_type_var.get()
        fusion_type = "gating" if fusion_label.startswith("Gating") else "logistic"
        return {
            "agent_name": self.agent_var.get(),
            "use_bagging": self.bagging_var.get(),
            "use_full_ensemble": self.full_ensemble_var.get(),
            "fusion_type": fusion_type,
            "auto_tune_mlp": self.auto_tune_mlp_var.get(),
            "auto_tune_rf": self.auto_tune_rf_var.get(),
            "ais_grades_only": self.ais_grades_only_var.get(),
            "n_estimators": int(self.n_estimators_var.get()),
            "test_frac": float(self.test_frac_var.get()),
            "feature_set": self.feature_set_var.get(),
        }

    # background-job helper (keeps the window responsive)
    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for btn in self._busy_buttons:
            btn.configure(state=state)

    def _run_in_background(self, status_message, work_fn, on_success, on_error=None, clear=True):
        """Runs work_fn() on a background thread so the window stays
        responsive during long computation (model training, CV search, SHAP
        sampling)."""
        if clear:
            self.results_text.delete("1.0", "end")
        self.results_text.insert("end", status_message)
        self.results_text.update_idletasks()
        self._set_busy(True)

        def worker():
            try:
                result = work_fn()
            except Exception as exc:
                err = exc
                self.root.after(0, lambda: self._background_error(err, on_error))
            else:
                self.root.after(0, lambda: self._background_success(result, on_success))

        threading.Thread(target=worker, daemon=True).start()

    def _background_success(self, result, on_success):
        if self._destroyed:
            return  # window closed while the job was running - nothing to update
        self._set_busy(False)
        on_success(result)

    def _background_error(self, exc, on_error):
        if self._destroyed:
            return  # window closed while the job was running - nothing to update
        self._set_busy(False)
        if on_error is not None:
            on_error(exc)
        else:
            self.messagebox.showerror("Operation failed", str(exc))

    def _on_close(self) -> None:
        self._destroyed = True
        self.root.destroy()

    def run_analysis(self) -> None:
        params = self.configure_parameters()

        if self.X is None:
            self.dataset_type = "Synthetic demo"
            self.X, self.y = make_synthetic_dataset()
            self.header = [f"feature_{i}" for i in range(self.X.shape[1])] + ["target"]
            self.full_idx = list(range(self.X.shape[1]))
            self.grade_idx = self.behaviour_idx = None
            self.engagement_idx = []
            self.groups = None
            self.feature_groups = None

        feature_set = params["feature_set"]
        if feature_set == "Academic/assessment features" and self.grade_idx:
            cols = self.grade_idx
        elif feature_set == "Engagement/behaviour features" and self.behaviour_idx:
            cols = self.behaviour_idx
        else:
            cols = self.full_idx if self.full_idx is not None else list(range(self.X.shape[1]))
        X_subset = self.X[:, cols]
        subset_feature_names = [self.header[i] for i in cols] if self.header else \
            [f"f{i}" for i in cols]

        if self.groups is not None:
            X_train, X_test, y_train, y_test, train_idx, test_idx = group_train_test_split(
                X_subset, self.y, self.groups, test_frac=params["test_frac"], seed=0)
            test_protected = {k: np.asarray(v)[test_idx] for k, v in self.protected_attributes.items()}
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X_subset, self.y, test_frac=params["test_frac"], seed=0)
            test_protected = None

        agent_feature_overrides = None
        if params["use_full_ensemble"] and params["ais_grades_only"]:
            col_pos = {orig_idx: pos for pos, orig_idx in enumerate(cols)}
            def map_positions(original_indices):
                return [col_pos[i] for i in original_indices if i in col_pos]
            academic_positions = map_positions(self.grade_idx or [])
            behaviour_positions = map_positions(self.behaviour_idx or [])
            engagement_positions = map_positions(getattr(self, "engagement_idx", []) or [])
            if academic_positions or behaviour_positions or engagement_positions:
                agent_feature_overrides = {}
                if academic_positions: agent_feature_overrides["LogisticRegression"] = academic_positions
                if behaviour_positions: agent_feature_overrides["RandomForest"] = behaviour_positions
                if engagement_positions: agent_feature_overrides["SVM"] = engagement_positions
                # MLP remains holistic; AIS can use the lower-dimensional academic+behaviour space.
                ais_positions = sorted(set(academic_positions + behaviour_positions))
                if ais_positions and len(ais_positions) < len(cols):
                    agent_feature_overrides["AIS"] = ais_positions

        status = ("Running analysis - this can take a while on larger datasets like "
                   "OULAD. Rresults appear here when done.\n")
        if params["use_full_ensemble"] and (params["auto_tune_mlp"] or params["auto_tune_rf"]):
            tuning_msg = " and ".join(
                label for cond, label in
                [(params["auto_tune_mlp"], "MLP architecture"), (params["auto_tune_rf"], "RandomForest hyperparameters")]
                if cond)
            status = f"Auto-tuning {tuning_msg} (CV search) before training the full ensemble.\n" + status

        def work():
            return self._fit_and_evaluate(params, feature_set, X_subset, subset_feature_names,
                                           X_train, X_test, y_train, y_test, test_protected,
                                           agent_feature_overrides)

        def on_success(payload):
            self.last_results, self._explain_context = payload
            self.display_results()

        def on_error(exc):
            self.results_text.delete("1.0", "end")
            self.results_text.insert("end", f"Analysis failed: {exc}")
            self.messagebox.showerror("Analysis failed", str(exc))

        self._run_in_background(status, work, on_success, on_error)

    def _fit_and_evaluate(self, params, feature_set, X_subset, subset_feature_names,
                           X_train, X_test, y_train, y_test, test_protected,
                           agent_feature_overrides):
        """The actual model-fitting/scoring work behind Run Analysis."""
        import os
        t0 = time.time()

        if params["use_full_ensemble"]:
            controller = MASController(use_bagging=params["use_bagging"],
                                        n_estimators=params["n_estimators"],
                                        fusion_type=params["fusion_type"],
                                        auto_tune_mlp=params["auto_tune_mlp"],
                                        auto_tune_rf=params["auto_tune_rf"],
                                        agent_feature_overrides=agent_feature_overrides)
            result = controller.run_pipeline(X_train, y_train, X_test, y_test,
                                              feature_names=subset_feature_names)
            y_pred = result["y_pred"]
            acc = accuracy(y_test, y_pred)
            tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
            precision, recall, f1 = precision_recall_f1(tp, tn, fp, fn)
            dt = time.time() - t0

            last_results = {
                "mode": "fusion",
                "model_name": f"Heterogeneous Ensemble (5 agents + {controller.fusion.get_name()})",
                "params": params, "accuracy": acc, "oob_score": float("nan"),
                "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
                "train_size": len(y_train), "test_size": len(y_test), "time_sec": dt,
                "n_features": X_subset.shape[1], "feature_set": feature_set,
                "per_agent_test_acc": result["per_agent_test_acc"],
                "fusion_weights": result["fusion_weights"],
                "fusion_type": params["fusion_type"],
                "best_mlp_config": result.get("best_mlp_config"),
                "best_rf_config": result.get("best_rf_config"),
                "agent_feature_overrides": agent_feature_overrides,
                "test_protected": test_protected,
            }
            predict_fn = controller.predictProba
            single_model = None
        else:
            controller = None
            agent_cls, agent_kwargs = AGENT_REGISTRY[params["agent_name"]]

            if params["use_bagging"]:
                model = BaggingEnsemble(base_agent_factory=agent_cls, n_estimators=params["n_estimators"],
                                         bootstrap=True, random_state=7, agent_kwargs=agent_kwargs)
                model.fit(X_train, y_train)
                oob = model.get_oob_score()
                model_name = f"Bagged {agent_cls.__name__}"
            else:
                model = agent_cls(**agent_kwargs)
                model.fit(X_train, y_train)
                oob = float("nan")
                model_name = agent_cls.__name__

            y_pred = model.predict(X_test)
            acc = accuracy(y_test, y_pred)
            tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
            precision, recall, f1 = precision_recall_f1(tp, tn, fp, fn)
            dt = time.time() - t0

            last_results = {
                "mode": "single",
                "model_name": model_name, "params": params, "accuracy": acc,
                "oob_score": oob, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "precision": precision, "recall": recall, "f1": f1,
                "train_size": len(y_train), "test_size": len(y_test), "time_sec": dt,
                "n_features": X_subset.shape[1], "feature_set": feature_set,
                "test_protected": test_protected,
            }
            predict_fn = model.predictProba
            single_model = model

        # Fairness diagnostics on the untouched test set. For OULAD this is
        # evaluated on held-out students (group-aware split), not on training data.
        if test_protected:
            fr = fairness_report(y_test, y_pred, test_protected)
            fairness_path = "fairness_report.csv"
            if self.session_dir is not None:
                fairness_path = os.path.join(self.session_dir, fairness_path)
            fr.to_csv(fairness_path, index=False)
            last_results["fairness_report"] = fr
            last_results["fairness_path"] = fairness_path

        # stash context for the Explain / Learning Curves buttons
        explain_context = {
            "predict_fn": predict_fn,
            "X_train": X_train, "X_test": X_test, "y_test": y_test,
            "feature_names": subset_feature_names,
            "controller": controller,
            "single_model": single_model,
            "fusion_type": params["fusion_type"] if params["use_full_ensemble"] else None,
        }
        return last_results, explain_context

    def display_results(self) -> None:
        r = self.last_results
        self.results_text.delete("1.0", "end")
        if r is None:
            self.results_text.insert("end", "No results yet - click Run Analysis.")
            return

        lines = [
            f"Model: {r['model_name']}",
            f"Feature set: {r.get('feature_set', 'Full features')}  ({r.get('n_features', '?')} features)",
            f"Train size: {r['train_size']}   Test size: {r['test_size']}   "
            f"Time: {r['time_sec']:.2f}s",
            "",
            f"Accuracy:  {r['accuracy']:.4f}",
            f"OOB score: {r['oob_score']:.4f}" if r['oob_score'] == r['oob_score'] else "OOB score: n/a",
            f"Precision: {r['precision']:.4f}",
            f"Recall:    {r['recall']:.4f}",
            f"F1:        {r['f1']:.4f}",
            "",
            "Confusion matrix:",
            f"   TP={r['tp']}  FN={r['fn']}",
            f"   FP={r['fp']}  TN={r['tn']}",
        ]

        if r.get("mode") == "fusion":
            lines += ["", "Per-agent standalone test accuracy:"]
            for name, acc in r["per_agent_test_acc"].items():
                lines.append(f"   {name:<18} {acc:.4f}")
            if r.get("agent_feature_overrides") and "AIS" in r["agent_feature_overrides"]:
                lines.append("   (heterogeneous feature subsets were used for specialised agents)")

            best_agent_name = max(r["per_agent_test_acc"], key=r["per_agent_test_acc"].get)
            best_agent_acc = r["per_agent_test_acc"][best_agent_name]
            lift = r["accuracy"] - best_agent_acc
            lines += ["", f"Fusion impact: best single agent = {best_agent_name} "
                          f"({best_agent_acc:.4f}), fused = {r['accuracy']:.4f}  "
                          f"(lift = {'+' if lift >= 0 else ''}{lift:.4f})"]

            if r.get("best_mlp_config") is not None:
                bmc = r["best_mlp_config"]
                lines += ["", f"Auto-tuned MLP architecture: hidden_layer_sizes={bmc['hidden_layer_sizes']}, "
                               f"lr={bmc['lr']}, l2={bmc['l2']}",
                          f"  (5-fold CV mean acc={bmc['mean_acc']:.4f}, mean F1={bmc['mean_f1']:.4f})"]

            if r.get("best_rf_config") is not None:
                brc = r["best_rf_config"]
                lines += ["", f"Auto-tuned RandomForest: max_depth={brc['max_depth']}, "
                               f"min_samples_split={brc['min_samples_split']}, "
                               f"min_samples_leaf={brc['min_samples_leaf']}, "
                               f"class_weight={brc['class_weight']}, n_estimators={brc['n_estimators']}",
                          f"  (CV mean acc={brc['mean_acc']:.4f}, mean F1={brc['mean_f1']:.4f})"]

            if r.get("fusion_type") == "gating":
                lines += ["", "GatingFusionLayer - MEAN gate weight per agent over training set",
                           "(actual weights vary per student's own features; click Explain for a real example):"]
            else:
                lines += ["", "FusionLayer learned weights (trust per agent):"]
            for name, w in r["fusion_weights"].items():
                lines.append(f"   {name:<18} {w:+.4f}")

            self._plot_run_results(r)
            self._plot_fusion_impact(r)

        if r.get("fairness_report") is not None:
            lines += ["", "Fairness evaluation (test set):"]
            fr = r["fairness_report"]
            for attr in fr["attribute"].unique():
                g = fr[(fr["attribute"] == attr) & (fr["group"] != "GAP(max-min)")]
                gap = fr[(fr["attribute"] == attr) & (fr["group"] == "GAP(max-min)")]
                for _, row in g.iterrows():
                    lines.append(f"   {attr}={row['group']}: TPR={row['recall_tpr']:.3f}, FPR={row['fpr']:.3f}, PositiveRate={row['positive_rate']:.3f}")
                if not gap.empty:
                    row=gap.iloc[0]
                    lines.append(f"   {attr} gaps: TPR={row['recall_tpr']:.3f}, FPR={row['fpr']:.3f}, PositiveRate={row['positive_rate']:.3f}")
            lines.append(f"   Fairness CSV: {r.get('fairness_path','fairness_report.csv')}")

        if self.session_dir is not None:
            lines += ["", f"Charts saved to: {self.session_dir}"]

        self.results_text.insert("end", "\n".join(lines))

    def run_mlp_search(self) -> None:
        if self.X is None:
            self.dataset_type = "Synthetic demo"
            self.X, self.y = make_synthetic_dataset()
            self.header = [f"feature_{i}" for i in range(self.X.shape[1])] + ["target"]
            self.full_idx = list(range(self.X.shape[1]))
            self.grade_idx = self.behaviour_idx = None
            self.engagement_idx = []
            self.groups = None
            self.feature_groups = None

        cols = self.full_idx if self.full_idx is not None else list(range(self.X.shape[1]))
        X_subset = self.X[:, cols]
        y = self.y

        def work():
            return mlp_architecture_search(X_subset, y, k_folds=5, n_epochs=120, verbose=False)

        def on_success(payload):
            leaderboard, best = payload
            lines = ["MLP ARCHITECTURE SEARCH RESULTS (5-fold CV, sorted by mean accuracy)", "=" * 70,
                     f"{'Architecture':<18}{'lr':<8}{'l2':<10}{'Mean Acc':<12}{'Mean F1':<10}{'Params':<8}"]
            for entry in leaderboard:
                lines.append(
                    f"{str(entry['hidden_layer_sizes']):<18}{entry['lr']:<8}{entry['l2']:<10}"
                    f"{entry['mean_acc']:.4f}±{entry['std_acc']:.3f}  {entry['mean_f1']:<10.4f}"
                    f"{entry['n_params']:<8}")

            lines += ["", f"BEST: hidden_layer_sizes={best['hidden_layer_sizes']}, lr={best['lr']}, "
                           f"l2={best['l2']}  (mean acc={best['mean_acc']:.4f}, mean F1={best['mean_f1']:.4f})",
                      "", "To use this architecture, update AGENT_REGISTRY['MLP'] with:",
                      f"  {best['agent_kwargs']}"]

            self.results_text.insert("end", "\n".join(lines))
            self._plot_mlp_search(leaderboard)
            if self.session_dir is not None:
                self.results_text.insert("end", f"\n\nCharts saved to: {self.session_dir}")

        self._run_in_background(
            "Running MLP architecture search (5-fold CV) - this trains several networks, "
            "so it will take a moment. The window will stay responsive.\n\n",
            work, on_success)

    def run_rf_search(self) -> None:
        if self.X is None:
            self.dataset_type = "Synthetic demo"
            self.X, self.y = make_synthetic_dataset()
            self.header = [f"feature_{i}" for i in range(self.X.shape[1])] + ["target"]
            self.full_idx = list(range(self.X.shape[1]))
            self.grade_idx = self.behaviour_idx = None
            self.engagement_idx = []
            self.groups = None
            self.feature_groups = None

        cols = self.full_idx if self.full_idx is not None else list(range(self.X.shape[1]))
        X_subset = self.X[:, cols]
        y = self.y

        def work():
            return rf_architecture_search(X_subset, y, verbose=False)

        def on_success(payload):
            leaderboard, best = payload
            lines = ["RANDOMFOREST HYPERPARAMETER SEARCH RESULTS (CV, sorted by mean accuracy)", "=" * 70,
                     f"{'Depth':<8}{'MinSplit':<10}{'MinLeaf':<10}{'ClassWt':<12}{'Mean Acc':<12}{'Mean F1':<10}"]
            for entry in leaderboard:
                lines.append(
                    f"{entry['max_depth']:<8}{entry['min_samples_split']:<10}{entry['min_samples_leaf']:<10}"
                    f"{str(entry['class_weight']):<12}{entry['mean_acc']:.4f}±{entry['std_acc']:.3f}  "
                    f"{entry['mean_f1']:<10.4f}")

            lines += ["", f"BEST: max_depth={best['max_depth']}, min_samples_split={best['min_samples_split']}, "
                           f"min_samples_leaf={best['min_samples_leaf']}, class_weight={best['class_weight']}, "
                           f"n_estimators={best['n_estimators']}  "
                           f"(mean acc={best['mean_acc']:.4f}, mean F1={best['mean_f1']:.4f})",
                      "", "To use this configuration, update AGENT_REGISTRY['RandomForest'] with:",
                      f"  {best['agent_kwargs']}"]

            self.results_text.insert("end", "\n".join(lines))
            self._plot_rf_search(leaderboard)
            if self.session_dir is not None:
                self.results_text.insert("end", f"\n\nCharts saved to: {self.session_dir}")

        self._run_in_background(
            "Running RandomForest hyperparameter search (CV) - this takes a moment. "
            "The window will stay responsive.\n\n",
            work, on_success)

    def explain_results(self) -> None:
        if self._explain_context is None:
            self.messagebox.showwarning("Nothing to explain", "Run an analysis first.")
            return

        ctx = self._explain_context
        engine = ExplainabilityEngine()

        def work():
            global_pairs, global_explanation = engine.global_importance_and_explanation(
                ctx["predict_fn"], ctx["X_test"], feature_names=ctx["feature_names"],
                X_background=ctx["X_train"], background_size=50, max_samples=30, nsamples=100)

            sample_x = ctx["X_test"][0]
            instance_pairs, baseline_proba, instance_explanation = engine.explain_instance_and_explanation(
                ctx["predict_fn"], sample_x, ctx["X_train"],
                feature_names=ctx["feature_names"], background_size=50, nsamples=100)
            return global_pairs, global_explanation, sample_x, instance_pairs, baseline_proba, instance_explanation

        def on_success(payload):
            global_pairs, global_explanation, sample_x, instance_pairs, baseline_proba, instance_explanation = payload

            lines = ["", "=" * 60, "EXPLAINABILITY (SHAP)", "=" * 60, "",
                     "Global feature importance (mean |SHAP value|, top 10):",
                     "  (bigger = more influence on predictions across the test set)"]
            lines += engine.format_bar_chart(global_pairs[:10], value_fmt="{:.4f}")

            lines += ["", f"Local SHAP explanation for test sample #1 "
                           f"(predicted P(at-risk)={baseline_proba:.3f}), top 10 features:",
                      "  (how much this student's actual feature values pushed the prediction "
                      "up (+) or down (-) relative to the background distribution)"]
            lines += engine.format_bar_chart(instance_pairs[:10], value_fmt="{:+.4f}")

            # matplotlib SHAP plots (bar + waterfall), each auto-saved separately
            self._plot_shap_global(global_explanation)
            self._plot_shap_local(instance_explanation)

            # Extra: per-instance agent trust breakdown, only meaningful for the gating fusion
            if ctx.get("controller") is not None and ctx.get("fusion_type") == "gating":
                gate_weights = ctx["controller"].get_instance_gate_weights(sample_x.reshape(1, -1))[0]
                agent_names = list(ctx["controller"].agent_registry.keys())
                gate_pairs = list(zip(agent_names, gate_weights))
                gate_pairs.sort(key=lambda t: -t[1])
                lines += ["", "Per-instance agent trust for THIS student, from THEIR OWN features",
                           "(gating network, sums to 1):"]
                lines += engine.format_bar_chart(gate_pairs, value_fmt="{:.3f}")
                self._plot_gate_weights_instance(gate_pairs)

            if self.session_dir is not None:
                lines += ["", f"Charts saved to: {self.session_dir}"]

            self.results_text.insert("end", "\n" + "\n".join(lines) + "\n")
            self.results_text.see("end")

        self._run_in_background(
            "\n\nComputing SHAP values (this samples the model repeatedly, so it can "
            "take a little while).",
            work, on_success, clear=False)

    def export_report(self) -> None:
        if self.last_results is None:
            self.messagebox.showwarning("Nothing to export", "Run an analysis first.")
            return
        path = self.filedialog.asksaveasfilename(defaultextension=".txt",
                                                   filetypes=[("Text file", "*.txt")])
        if not path:
            return
        with open(path, "w") as f:
            f.write(self.results_text.get("1.0", "end"))
        self.messagebox.showinfo("Exported", f"Report saved to {path}")

    def save_model_to_disk(self) -> None:
        """Pickles last_results + explain_context together, so a reload can
        immediately re-display the analysis AND use Explain/Learning Curves
        without retraining."""
        if self.last_results is None or self._explain_context is None:
            self.messagebox.showwarning("Nothing to save", "Run an analysis first.")
            return
        path = self.filedialog.asksaveasfilename(
            title="Save trained model", defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if not path:
            return
        try:
            save_model({"last_results": self.last_results,
                        "explain_context": self._explain_context}, path)
        except Exception as exc:
            self.messagebox.showerror("Save failed", str(exc))
            return
        self.messagebox.showinfo("Saved", f"Model and results saved to:\n{path}")

    def load_model_from_disk(self) -> None:
        """Restores a session saved by save_model_to_disk."""
        path = self.filedialog.askopenfilename(
            title="Load a previously saved model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")])
        if not path:
            return
        try:
            payload = load_model(path)
            last_results = payload["last_results"]
            explain_context = payload["explain_context"]
        except Exception as exc:
            self.messagebox.showerror("Load failed", str(exc))
            return
        self.last_results = last_results
        self._explain_context = explain_context
        self.display_results()
        self.results_text.insert("end", f"\n\n(Loaded from: {path})")


