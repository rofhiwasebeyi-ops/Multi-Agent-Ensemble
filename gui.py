"""Tkinter GUI wrapping the MAS ensemble pipeline: dataset loading, agent/
fusion configuration, running analyses, architecture search, learning
curves, SHAP explanations, and chart/report export.

matplotlib and tkinter are imported lazily inside UserInterface.__init__
(not at module level) so that importing this module doesn't itself require
a display - main.py's headless fallback only needs this import to succeed
lazily, at GUI-construction time, not at module-import time.
"""
import os
import numpy as np
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
    agreement_report, interpret_agreement,
)
from shared import report as report_view
from shared import visuals


class UserInterface:

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
        self.tk, self.ttk = tk, ttk
        self.filedialog, self.messagebox = filedialog, messagebox

        # matplotlib's Tk backend needs tkinter present, so it's imported
        # here rather than at module level (keeps the headless fallback
        # path working on machines without tkinter installed at all).
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        self._plt = plt
        self._Figure = Figure
        self._FigureCanvasTkAgg = FigureCanvasTkAgg
        self.chart_canvas = None
        self.chart_history = []   # (name, png_bytes, filepath) for every chart of the CURRENT run
        self._chart_counter = 0   # session-wide, so PNG filenames never collide across runs
        self._chart_photo = None
        self._chart_index = -1
        self.session_dir = None   # output folder, created lazily on the first saved chart

        self.root = root
        self.root.title("Multi-Agent Ensemble - Prototype")
        # Wide enough that the chart pane can comfortably fit the largest
        # figure this app draws (6.8in x 4.6in @ dpi=100 = 680x460px) at its
        # native size without the PanedWindow having to squeeze it down.
        self.root.geometry("1500x800")

        # Background jobs (see _run_in_background) finish on a daemon
        # thread and hand their result back via root.after(...), which can
        # only run once the mainloop gets to it. If the window is closed
        # while a job is still running, that queued callback would fire
        # against widgets that no longer exist. daemon=True already means
        # the process exits cleanly either way (the thread is killed, not
        # left hanging) - this flag just stops the callback from touching
        # a destroyed window in the meantime, rather than raising.
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
        self._params_frame = params
        self._top_frame = top
        self.settings_btn = ttk.Button(top, text="Hide settings \u25b2", width=16, command=self.toggle_settings)
        self.settings_btn.pack(side="right")

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
                             auto_tune_all_check.config(state="normal" if self.full_ensemble_var.get() else "disabled"),
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

        self.auto_tune_all_var = tk.BooleanVar(value=False)
        auto_tune_all_check = ttk.Checkbutton(
            params, text="Auto-tune ALL agents (LR, SVM, RF, MLP, AIS - CV search; slower, includes the two above)",
            variable=self.auto_tune_all_var, state="disabled")
        auto_tune_all_check.grid(row=8, column=0, columnspan=4, sticky="w", pady=(6, 0))

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
        self.visualize_button = ttk.Button(actions, text="Visualize Models", command=self.visualize_models)
        self.visualize_button.pack(side="left", padx=10)
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
        # Disabled together while a background analysis/search/explain job is
        # running (see _run_in_background) so a second heavy job can't be
        # kicked off - and re-enabled together when it finishes.
        self._busy_buttons = [self.select_dataset_button, self.run_button, self.explain_button,
                               self.mlp_search_button, self.rf_search_button, self.visualize_button,
                               self.learning_curves_button, self.export_button,
                               self.open_folder_button, self.save_model_button, self.load_model_button]

        # Classic (non-ttk) PanedWindow rather than ttk.PanedWindow: only
        # the classic widget supports a per-pane minsize, which is what
        # stops the chart pane from ever being squeezed smaller than the
        # largest chart (see the comment on the chart_frame add() below).
        # ttk.PanedWindow only exposes a "weight" option, no size floor.
        body = tk.PanedWindow(self.root, orient="horizontal", sashrelief="raised", sashwidth=6)
        body.pack(fill="both", expand=True, padx=10, pady=5)

        results_frame = ttk.LabelFrame(body, text="Results", padding=6)
        self.results_nb = ttk.Notebook(results_frame)

        # Status strip under the tabs: shows what a background job is doing
        # (instead of yanking the user to another tab) with a busy indicator.
        status_row = ttk.Frame(results_frame)
        self.status_var = tk.StringVar(value="Ready.")
        self.status_label = ttk.Label(status_row, textvariable=self.status_var, foreground="#475569")
        self.status_label.pack(side="left", fill="x", expand=True)
        self.progress = ttk.Progressbar(status_row, mode="indeterminate", length=110)
        status_row.pack(side="bottom", fill="x", pady=(6, 0))
        self.results_nb.pack(fill="both", expand=True)

        # One scrollable page per results view; each is rebuilt on every
        # display (see the _render_* methods).
        import gui_widgets as gw
        self.gw = gw
        self.pages = {}
        for key, label in (("overview", "Overview"), ("models", "Models"), ("agreement", "Agreement"),
                           ("fairness", "Fairness"), ("explain", "Explain"), ("tuning", "Tuning")):
            page = gw.ScrollPage(self.results_nb)
            self.pages[key] = page
            self.results_nb.add(page, text=label)
            gw.empty_state(page, "Nothing here yet",
                           "Click Run Analysis to see results." if key != "tuning" else
                           "Run an MLP or RandomForest search to see the leaderboard here.")
        self._tab_index = {k: i for i, k in enumerate(self.pages)}

        # Log tab: raw status/progress messages. Kept as a Text widget because
        # background jobs append to it; Export Report also appends it.
        log_tab = ttk.Frame(self.results_nb)
        self.results_text = tk.Text(log_tab, height=20, width=56, wrap="word",
                                    font=("Consolas", 9), padx=8, pady=6, relief="flat",
                                    background="#f8fafc", foreground="#334155")
        sb = ttk.Scrollbar(log_tab, orient="vertical", command=self.results_text.yview)
        self.results_text.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        self.results_text.pack(side="left", fill="both", expand=True)
        t = self.results_text
        t.tag_configure("h1", font=("Segoe UI", 12, "bold"))
        t.tag_configure("h2", font=("Segoe UI", 10, "bold"), foreground="#1d4ed8")
        t.tag_configure("muted", foreground="#64748b")
        t.tag_configure("warn", foreground="#b45309", font=("Consolas", 9, "bold"))
        t.tag_configure("mono", font=("Consolas", 9))
        self.results_nb.add(log_tab, text="Log")
        self._tab_index["log"] = self.results_nb.index("end") - 1
        body.add(results_frame, stretch="always", minsize=520)

        self.chart_frame = ttk.LabelFrame(body, text="Latest chart (all charts are auto-saved as PNGs)", padding=10)
        # Never let the sash squeeze this pane below the largest chart's
        # native width - that squeeze is what previously forced charts to
        # be rescaled (and distorted - see _show_chart) to fit.
        body.add(self.chart_frame, stretch="always", minsize=720)

        # Charts are shown inside a scrollable canvas rather than packed
        # straight into chart_frame. Previously the FigureCanvasTkAgg widget
        # was packed with fill="both", expand=True, so Tkinter stretched or
        # squeezed the rendered chart to whatever size the pane happened to
        # be. matplotlib text is sized in fixed points, not inches, so that
        # rescaling didn't scale the fonts/markers along with the canvas -
        # shrinking the pane made text and bars overlap and look "zoomed
        # in", and enlarging it left a chart with tiny, sparse-looking text
        # floating in empty space ("zoomed out"). Showing the chart at its
        # own native pixel size (figsize * dpi) and scrolling if the pane is
        # smaller keeps every chart exactly as it was designed to look.
        chart_scroll_canvas = tk.Canvas(self.chart_frame, highlightthickness=0)
        v_scroll = ttk.Scrollbar(self.chart_frame, orient="vertical", command=chart_scroll_canvas.yview)
        h_scroll = ttk.Scrollbar(self.chart_frame, orient="horizontal", command=chart_scroll_canvas.xview)
        chart_scroll_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        # Chart navigator: step through / pick any chart generated this run.
        nav = ttk.Frame(self.chart_frame)
        nav.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.chart_prev_btn = ttk.Button(nav, text="◀ Prev", width=8, command=lambda: self._step_chart(-1))
        self.chart_prev_btn.pack(side="left")
        self.chart_choice_var = tk.StringVar()
        self.chart_choice = ttk.Combobox(nav, textvariable=self.chart_choice_var, state="readonly", width=44)
        self.chart_choice.pack(side="left", padx=6, fill="x", expand=True)
        self.chart_choice.bind("<<ComboboxSelected>>", lambda e: self._display_chart(self.chart_choice.current()))
        self.chart_next_btn = ttk.Button(nav, text="Next ▶", width=8, command=lambda: self._step_chart(1))
        self.chart_next_btn.pack(side="left")
        self.chart_count_label = ttk.Label(nav, text="0 charts")
        self.chart_count_label.pack(side="left", padx=8)

        self.chart_frame.rowconfigure(1, weight=1)
        self.chart_frame.columnconfigure(0, weight=1)
        chart_scroll_canvas.grid(row=1, column=0, sticky="nsew")
        v_scroll.grid(row=1, column=1, sticky="ns")
        h_scroll.grid(row=2, column=0, sticky="ew")

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

    def toggle_settings(self, show=None) -> None:
        """Collapse/expand the parameter panel so the results get more height."""
        shown = bool(self._params_frame.winfo_ismapped())
        show = (not shown) if show is None else show
        if show and not shown:
            self._params_frame.pack(fill="x", padx=10, pady=5, after=self._top_frame)
        elif not show and shown:
            self._params_frame.pack_forget()
        self.settings_btn.configure(text="Hide settings \u25b2" if show else "Show settings \u25bc")

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
        """Saves the figure as a 150-dpi PNG in the session folder, adds it
        to this run's chart gallery and displays it at native size (never
        rescaled; the pane scrolls instead). Use the Prev/Next buttons or the
        drop-down above the chart to revisit any chart from the run."""
        import os, io, base64
        session_dir = self._ensure_session_dir()
        self._chart_counter += 1
        fpath = os.path.join(session_dir, f"{self._chart_counter:02d}_{name}.png")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=fig.get_dpi(), bbox_inches="tight")
        self.chart_history.append((name, base64.b64encode(buf.getvalue()), fpath))
        self.chart_choice["values"] = [f"{i + 1}. {n.replace('_', ' ')}" for i, (n, _, _) in enumerate(self.chart_history)]
        self._display_chart(len(self.chart_history) - 1)
        return fpath

    def _display_chart(self, index: int) -> None:
        if not (0 <= index < len(self.chart_history)):
            return
        self._chart_index = index
        if self._chart_placeholder is not None:
            self._chart_placeholder.destroy()
            self._chart_placeholder = None
        for child in self.chart_inner.winfo_children():
            child.destroy()
        _, data, _ = self.chart_history[index]
        self._chart_photo = self.tk.PhotoImage(data=data)
        self.ttk.Label(self.chart_inner, image=self._chart_photo).pack(padx=10, pady=10)
        self.chart_choice.current(index)
        self.chart_count_label.configure(text=f"{index + 1} / {len(self.chart_history)}")
        self.chart_prev_btn.state(["!disabled"] if index > 0 else ["disabled"])
        self.chart_next_btn.state(["!disabled"] if index < len(self.chart_history) - 1 else ["disabled"])
        self.chart_inner.update_idletasks()
        self._chart_scroll_canvas.configure(scrollregion=self._chart_scroll_canvas.bbox("all"))
        self._chart_scroll_canvas.xview_moveto(0)
        self._chart_scroll_canvas.yview_moveto(0)

    def _step_chart(self, delta: int) -> None:
        self._display_chart(self._chart_index + delta)

    def _reset_charts(self) -> None:
        """Start a fresh gallery (called at the beginning of each analysis run)."""
        self.chart_history = []
        self._chart_index = -1
        self.chart_choice["values"] = []
        self.chart_choice_var.set("")
        self.chart_count_label.configure(text="0 charts")
        for child in self.chart_inner.winfo_children():
            child.destroy()
        self._chart_placeholder = self.ttk.Label(
            self.chart_inner, text="Charts from this run will appear here.\nUse Prev / Next or the list above to browse them.",
            justify="center")
        self._chart_placeholder.pack(expand=True, padx=40, pady=40)

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
        """The chart that answers 'did learning the fusion layer actually
        help?' - per-agent standalone accuracy vs. the fused ensemble's
        accuracy, with the best single agent highlighted and the fusion
        lift (fused acc - best single agent acc) annotated directly on the
        chart. This is direct visual evidence of what the fusion layer
        (FusionLayer / GatingFusionLayer) learned to contribute beyond
        simply picking the best individual agent."""
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

    def _plot_agreement(self, ag: dict) -> None:
        """Two heatmaps side by side: pairwise agreement and pairwise kappa."""
        names = ag["agent_names"]
        fig = self._Figure(figsize=(9.6, 4.6), dpi=100)
        for k, (key, title, vmin) in enumerate((("pairwise_agreement", "Pairwise agreement", 0.5),
                                                 ("pairwise_kappa", "Pairwise Cohen's kappa", 0.0))):
            ax = fig.add_subplot(1, 2, k + 1)
            M = np.array(ag[key])
            im = ax.imshow(M, vmin=vmin, vmax=1.0, cmap="Blues")
            ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
            ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
            ax.set_yticklabels(names, fontsize=8)
            for i in range(len(names)):
                for j in range(len(names)):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] > (vmin + 1) / 2 + 0.1 else "black")
            ax.set_title(title, fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle("How much do the agents agree?  (darker = more similar)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        self._show_chart(fig, name="agent_agreement")

    def _draw_structures(self, mlp=None, mlp_names=None, forest=None, forest_names=None,
                         n_trees=3, tree_depth=3, tag="") -> None:
        """Main-thread drawing of the MLP network and random-forest trees."""
        if mlp is not None:
            self._show_chart(visuals.plot_mlp_network(mlp, mlp_names), name=f"mlp_network{tag}")
        if forest is not None:
            self._show_chart(visuals.plot_forest_overview(forest, forest_names), name=f"rf_overview{tag}")
            for i, tree in enumerate(forest.trees[:n_trees]):
                self._show_chart(visuals.plot_forest_tree(
                    tree, forest_names, max_depth=tree_depth,
                    title=None), name=f"rf_tree_{i + 1}{tag}")

    def visualize_models(self) -> None:
        ctx = self._explain_context
        if ctx is None:
            self.messagebox.showwarning("Nothing to visualize", "Run an analysis first.")
            return
        mlp = forest = None
        mlp_names = forest_names = None
        ctrl = ctx.get("controller")
        names = ctx.get("feature_names")
        if ctrl is not None:
            def first_member(agent):
                return agent.models[0] if hasattr(agent, "models") else agent
            if "MLP" in ctrl.agents:
                mlp = first_member(ctrl.agents["MLP"])
                cols = ctrl.agent_feature_overrides.get("MLP")
                mlp_names = [names[c] for c in cols] if cols is not None and names else names
            if "RandomForest" in ctrl.agents:
                forest = first_member(ctrl.agents["RandomForest"])
                cols = ctrl.agent_feature_overrides.get("RandomForest")
                forest_names = [names[c] for c in cols] if cols is not None and names else names
        else:
            m = ctx.get("single_model")
            m = m.models[0] if hasattr(m, "models") else m
            if hasattr(m, "Ws"):
                mlp, mlp_names = m, names
            elif hasattr(m, "trees"):
                forest, forest_names = m, names
        if mlp is None and forest is None:
            self.messagebox.showinfo("Nothing to visualize",
                                     "The current model has no neural network or random forest.")
            return
        self.results_nb.select(self._tab_index['log'])
        self._draw_structures(mlp, mlp_names, forest, forest_names)
        self._w("\nModel structure charts added (MLP network, forest overview, first 3 trees). "
                "Blue/red edges = positive/negative weights.\n", "muted")

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
        for the one student being explained - not a SHAP plot, since it's the
        gate's own output rather than a feature attribution, so it stays as
        a plain matplotlib bar chart."""
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
        architecture search. Labels show depth/split/leaf/class_weight since
        n_estimators is fixed during the search (see rf_architecture_search)."""
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
        None. Covers any model that scans a grid of candidate hyperparameter
        values and scores each one (AISAgent's self_radius via
        _select_radius_supervised/_select_radius_cv, FusionLayer's l2 via
        _select_l2_cv) rather than a gradient-descent loss curve - this is
        that search's analogue of a learning curve (score vs. candidate
        hyperparameter rather than loss vs. training step)."""
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
        history OR a calibration-search history - each auto-saved on its own
        rather than squeezed into a shared grid, so every curve stays
        readable. Bagged agents are shown as the mean curve across their
        members (truncated to the shortest member's length); for AIS, the
        radius search runs ONCE on the full training set (see
        BaggingEnsemble.precalibrate handling) rather than per member, so its
        curve is plotted once per bagged group instead of averaged. The
        fusion layer's own curve is what shows the 'impact of fusion layer
        learning' over training time."""
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
        value) - AIS's/FusionLayer's analogue of a loss curve, since neither
        has a gradient-descent trajectory over this hyperparameter. The
        chosen value is marked with a vertical line."""
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

        UCI uses a CSV file; OULAD uses the extracted raw-data directory.
        The user therefore never has to provide a ZIP archive or a pre-built
        feature CSV through this interface.
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
            "auto_tune_all": self.auto_tune_all_var.get(),
            "ais_grades_only": self.ais_grades_only_var.get(),
            "n_estimators": int(self.n_estimators_var.get()),
            "test_frac": float(self.test_frac_var.get()),
            "feature_set": self.feature_set_var.get(),
        }

    # background-job helper (keeps the window responsive)
    def _set_busy(self, busy: bool) -> None:
        if not busy:
            self.progress.stop()
            self.progress.pack_forget()
        state = "disabled" if busy else "normal"
        for btn in self._busy_buttons:
            btn.configure(state=state)

    def _run_in_background(self, status_message, work_fn, on_success, on_error=None, clear=True):
        """Runs work_fn() on a background thread so the window stays
        responsive during long computation (model training, CV search, SHAP
        sampling). This is what was previously missing: every heavy button
        (Run Analysis, MLP/RandomForest search, Explain) used to call its
        heavy function directly from the Tk button callback, which runs on
        the same thread as the GUI's event loop - so for anything slower
        than a fraction of a second (which is normal at OULAD's scale, even
        after the RandomForest/CART speedups), the whole window stopped
        processing OS messages and the OS marked it "Not Responding" until
        the call returned. It wasn't actually frozen or crashed, just busy.

        work_fn MUST NOT touch any Tk widget or matplotlib figure - it runs
        off the main thread, and both toolkits are only safe to use from the
        thread that created them. on_success(result) and on_error(exc) DO
        run back on the main thread (via root.after), so they're the right
        place for any results_text/chart updates.
        """
        if clear:
            self.results_text.delete("1.0", "end")
        msg = status_message.strip()
        self.results_text.insert("end", msg + "\n")
        self.status_var.set(msg.split("\n")[-1][:140])
        self.progress.pack(side="right", padx=(8, 0))
        self.progress.start(12)
        self._set_busy(True)

        def worker():
            try:
                result = work_fn()
            except Exception as exc:
                # Python clears the "as exc" binding as soon as this except
                # block exits, but the lambda below only runs later (once
                # Tk's event loop gets to it via root.after) - by then `exc`
                # would already be gone, raising a NameError. Rebinding it to
                # a plain local first keeps it alive for the closure.
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
        """WM_DELETE_WINDOW handler. Setting _destroyed first means any
        background job's root.after callback that fires after this point
        (see _background_success/_background_error) sees the flag and
        no-ops instead of touching widgets that are about to be destroyed.
        The job's own thread is daemon=True, so it's simply killed with the
        process on exit rather than left running or blocking shutdown."""
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

        # Map self.grade_idx (positions in the ORIGINAL, unsubsetted feature
        # matrix) into positions within `cols` (the subset actually in use
        # this run) - only meaningful when the grade columns are actually
        # present in the chosen subset (e.g. not when "Behavioural only" was
        # picked, since G1/G2 aren't in that subset at all).
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
                   "OULAD. The window will stay responsive; results appear here when done.\n")
        if params["use_full_ensemble"] and (params["auto_tune_mlp"] or params["auto_tune_rf"] or params["auto_tune_all"]):
            tuning_msg = "ALL agents' hyperparameters" if params["auto_tune_all"] else " and ".join(
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
            self.results_text.insert("end", f"Analysis failed: {exc}\n", "warn")
            self.status_var.set("Analysis failed - see the Log tab.")
            self.messagebox.showerror("Analysis failed", str(exc))

        self._run_in_background(status, work, on_success, on_error)

    def _fit_and_evaluate(self, params, feature_set, X_subset, subset_feature_names,
                           X_train, X_test, y_train, y_test, test_protected,
                           agent_feature_overrides):
        """The actual model-fitting/scoring work behind Run Analysis. Runs on
        a background thread (see _run_in_background) - must not touch any Tk
        widget. Returns (last_results, explain_context), which the caller's
        on_success callback stores and displays back on the main thread."""
        import os
        t0 = time.time()

        if params["use_full_ensemble"]:
            controller = MASController(use_bagging=params["use_bagging"],
                                        n_estimators=params["n_estimators"],
                                        fusion_type=params["fusion_type"],
                                        auto_tune_mlp=params["auto_tune_mlp"],
                                        auto_tune_rf=params["auto_tune_rf"],
                                        auto_tune_all=params["auto_tune_all"],
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
                "best_configs": result.get("best_configs") or {},
                "agent_feature_overrides": agent_feature_overrides,
                "test_protected": test_protected,
            }
            last_results["agreement"] = agreement_report(
                result["agent_names"], result["agent_test_pred"], y_test, y_pred,
                agent_proba=result["agent_test_proba"])
            gstd = getattr(controller.fusion, "gate_weight_std_", None)
            if params["fusion_type"] == "gating" and gstd is not None:
                last_results["gate_weight_std"] = dict(zip(result["agent_names"], [float(v) for v in gstd]))
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

        # stash context for the Explain / Learning Curves buttons - works
        # identically for the fused controller or a single agent, since both
        # expose predictProba(X)
        explain_context = {
            "predict_fn": predict_fn,
            "X_train": X_train, "X_test": X_test, "y_test": y_test,
            "feature_names": subset_feature_names,
            "controller": controller,
            "single_model": single_model,
            "fusion_type": params["fusion_type"] if params["use_full_ensemble"] else None,
        }
        return last_results, explain_context

    # ---- structured results rendering ----
    def _page(self, key):
        """Empty the named results page and return it ready for new content."""
        page = self.pages[key]
        page.clear()
        return page

    def _w(self, text, *tags):
        self.results_text.insert("end", text, tags)

    def _safe(self, label, fn):
        """Run one display step; on failure, report it in the Log tab instead
        of silently aborting the rest of the results display."""
        try:
            fn()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._w(f"\n[Could not render {label}: {type(exc).__name__}: {exc}]\n", "warn")
            self.status_var.set(f"Could not render {label} - see the Log tab.")

    def _plot_fairness(self, fr) -> None:
        """Recall (TPR) and false-alarm (FPR) gap per protected attribute."""
        g = fr[fr["group"] == "GAP(max-min)"]
        if g.empty:
            return
        names = list(g["attribute"])
        y = np.arange(len(names))
        fig = self._Figure(figsize=(7.2, 0.9 + 0.55 * len(names) + 1.2), dpi=100)
        ax = fig.add_subplot(111)
        ax.barh(y - 0.2, g["recall_tpr"], height=0.38, color="#DD8452", label="Recall (TPR) gap")
        ax.barh(y + 0.2, g["fpr"], height=0.38, color="#4C72B0", label="False-alarm (FPR) gap")
        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9)
        ax.invert_yaxis(); ax.set_xlim(0, 1)
        ax.set_xlabel("Largest difference between groups (0 = equal treatment)")
        ax.set_title("Fairness gaps by protected attribute", fontsize=11)
        ax.legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        self._show_chart(fig, name="fairness_gaps")

    # ------------------------------------------------------------------
    # Results views. Each _render_* builds one tab from `last_results`.
    # ------------------------------------------------------------------
    def _render_overview(self, r):
        gw, rv = self.gw, report_view
        p = self._page("overview")
        sentence, cards = rv.headline(r)
        title = gw.label(p.body, r["model_name"], 15, True)
        p.wrap(title, 40)
        title.pack(anchor="w", fill="x", padx=16, pady=(16, 0))
        sub = (f"{r.get('feature_set', 'Full features')} ({r.get('n_features', '?')} features)  \u00b7  "
               f"{r['train_size']:,} train / {r['test_size']:,} test students  \u00b7  {r['time_sec']:.0f}s")
        gw.paragraph(p, sub, gw.MUTED, 9, pady=(0, 4))
        gw.banner(p, sentence, "info", "What this means")
        gw.metric_cards(p, cards)
        if r["oob_score"] == r["oob_score"]:
            gw.caption(p, f"Out-of-bag score: {r['oob_score']:.4f}")
        gw.heading(p, "Confusion matrix", "How the ensemble's decisions on unseen students turned out.")
        gw.confusion_grid(p, r["tp"], r["fn"], r["fp"], r["tn"])
        gw.heading(p, "Key findings")
        gw.bullets(p, rv.key_findings(r))
        tun = rv.tuning_lines(r)
        if tun or r.get("agent_feature_overrides"):
            gw.heading(p, "How it was configured")
            items = [("muted", t) for t in tun]
            if r.get("agent_feature_overrides"):
                items.append(("muted", "Specialised agents each used their own feature subset (heterogeneous views)."))
            gw.bullets(p, items)
        gw.paragraph(p, "Details are in the other tabs: Models, Agreement, Fairness, Explain.", gw.MUTED, 9, pady=(14, 16))

    def _render_models(self, r):
        gw, rv = self.gw, report_view
        p = self._page("models")
        if r.get("mode") != "fusion":
            gw.empty_state(p, "Single model", "This run used one model, so there are no per-agent results. "
                           "Tick Full ensemble to compare several agents.")
            return
        cols, rows, _ = rv.agents_view(r)
        gw.heading(p, "Individual models", "Each agent's accuracy on the same unseen students, and how much the "
                   "fusion layer trusts it.")
        note = rv.fusion_note(r)
        gw.banner(p, note, "good" if "helped" in note else "warn" if "worse" in note else "info")
        gw.table(p, cols, rows, kinds=["text", "bar", "delta", "bar", "bar"], widths=[3, 3, 2, 3, 3], bar_max=1.0)
        legend = ("\u2605 = best single agent.  Fusion weight = how much the final decision leans on that agent"
                  + (" (average over students; the gate varies it per student)." if r.get("fusion_type") == "gating"
                     else " (logistic-regression coefficient, scaled)."))
        gw.caption(p, legend)
        std = r.get("gate_weight_std")
        if std and r.get("fusion_weights"):
            gw.heading(p, "How much the gate varies", "Average weight \u00b1 spread across students. A tiny spread "
                       "means the gate is acting like a fixed weighting.")
            grows = [[k, (f"{r['fusion_weights'][k]:.3f}", r["fusion_weights"][k]), f"\u00b1 {v:.3f}"]
                     for k, v in std.items()]
            gw.table(p, ["Agent", "Mean weight", "Spread"], grows, kinds=["text", "bar", "text"], widths=[3, 4, 2])
        tun = rv.tuning_lines(r)
        if tun:
            gw.heading(p, "Tuned settings")
            gw.bullets(p, [("muted", t) for t in tun])

    def _render_agreement(self, r):
        gw, rv = self.gw, report_view
        p = self._page("agreement")
        ag = r.get("agreement")
        if not ag:
            gw.empty_state(p, "No agreement data", "Agreement needs a full-ensemble run (several agents).")
            return
        gw.heading(p, "In plain words", "Do the agents give different answers - and does that help?")
        gw.bullets(p, [("info", x) for x in interpret_agreement(ag)])
        for title, rows in rv.agreement_groups(ag):
            if rows:
                gw.heading(p, title)
                gw.table(p, ["Metric", "Value", "Meaning"], rows, kinds=["text", "text", "text"],
                         widths=[4, 1, 6], first_col_w=170, wraps={2: 200})
        names = ag["agent_names"]
        for title, key, lo in (("Pairwise agreement", "pairwise_agreement", 0.5),
                               ("Pairwise Cohen's kappa", "pairwise_kappa", 0.0)):
            gw.heading(p, title, "Darker = more alike." if key == "pairwise_agreement" else
                       "Chance-corrected; darker = more alike. Low values mean the agents genuinely differ.")
            rows = [[n] + [(f"{v:.2f}", v) for v in ag[key][i]] for i, n in enumerate(names)]
            gw.table(p, ["Agent"] + names, rows, kinds=["text"] + ["heat"] * len(names), heat_range=(lo, 1.0),
                     first_col_w=120, zebra=False)
        if "probability_correlation" in ag:
            gw.heading(p, "Probability correlation", "How alike the agents' risk scores are (Pearson r).")
            rows = [[n] + [(f"{v:.2f}", v) for v in ag["probability_correlation"][i]] for i, n in enumerate(names)]
            gw.table(p, ["Agent"] + names, rows, kinds=["text"] + ["heat"] * len(names), heat_range=(0.0, 1.0),
                     first_col_w=120, zebra=False)
        gw.paragraph(p, "", pady=(0, 10))

    def _render_fairness(self, r):
        gw, rv = self.gw, report_view
        p = self._page("fairness")
        fr = r.get("fairness_report")
        if fr is None or not len(fr):
            gw.empty_state(p, "Fairness not available",
                           "Fairness needs protected attributes. OULAD provides gender, age band, region, "
                           "education, IMD band and disability; synthetic and UCI data have none loaded here.")
            return
        summary, blocks = rv.fairness_view(fr)
        gw.heading(p, "Overview", "For each protected attribute: the gap between the best- and worst-treated group. "
                   "0 means every group is treated alike.")
        gw.banner(p, rv.fairness_note(fr), "info")
        gw.table(p, ["Attribute", "Recall gap", "False-alarm gap", "Severity"], summary,
                 kinds=["text", "bar", "bar", "tag"], widths=[3, 3, 3, 2], bar_max=1.0)
        gw.caption(p, "Recall gap = difference in how often truly at-risk students are caught.  "
                      "False-alarm gap = difference in how often students who are fine get flagged.  "
                      "Severity (rule of thumb): small < 0.10, moderate < 0.20, large \u2265 0.20.")
        gw.heading(p, "Group by group", "Worst attribute first.")
        for a, tg, fg, rows in blocks:
            label, tone = rv.severity(tg)
            head = tk_frame = self.tk.Frame(p.body, bg=gw.BG)
            head.pack(fill="x", padx=16, pady=(12, 0))
            gw.label(head, a, 10, True).pack(side="left")
            fgc, bgc = gw.TONES[tone]
            self.tk.Label(head, text=f"recall gap {tg:.2f} \u00b7 {label}", font=(gw.FONT, 9, "bold"), fg=fgc,
                          bg=bgc, padx=8, pady=1).pack(side="left", padx=10)
            gw.table(p, ["Group", "Recall", "False alarms", "Flagged at-risk"], rows,
                     kinds=["text", "bar", "bar", "bar"], widths=[3, 3, 3, 3], bar_max=1.0, first_col_w=200)
        if r.get("fairness_path"):
            gw.caption(p, f"CSV saved to: {r['fairness_path']}")

    def _render_explain(self, view=None):
        """Explain tab: a prompt after a run, filled in by explain_results()."""
        gw = self.gw
        p = self._page("explain")
        if view is None:
            gw.empty_state(p, "Not computed yet", "Click Explain (SHAP) to see which features drive the model, "
                           "and why one example student got their prediction. It samples the model repeatedly, "
                           "so it takes a little while.")
            return
        gw.heading(p, "In plain words")
        gw.bullets(p, [("info", x) for x in view["story"]])
        gw.heading(p, "What drives the model overall", "Average push each feature gives to the predicted at-risk "
                   "probability (mean |SHAP|), top 10.")
        gw.table(p, ["Feature", "Influence", "Share of total"],
                 [[n, (txt, frac), share] for n, txt, frac, share in view["global"]],
                 kinds=["text", "bar", "text"], widths=[4, 4, 2], first_col_w=220, bold_first=False)
        gw.heading(p, "Why this one student", "Test sample #1: what pushed their prediction up or down.")
        gw.table(p, ["Feature", "Value (scaled)", "Effect", "Contribution"],
                 [[n, val, (eff, "bad" if "raises" in eff else "good" if "lowers" in eff else "muted"), (c, -f)]
                  for n, val, eff, c, f in view["instance"]],
                 kinds=["text", "text", "tag", "delta"], widths=[4, 2, 3, 2], first_col_w=220, bold_first=False)
        gw.caption(p, "Red = pushes towards at-risk, green = pushes away.")
        if view.get("trust"):
            gw.heading(p, "Which agents the gate trusted for this student")
            gw.table(p, ["Agent", "Trust"], [[a, (t, w)] for a, t, w in view["trust"]],
                     kinds=["text", "bar"], widths=[3, 6])
        gw.paragraph(p, "SHAP charts are in the chart gallery on the right.", gw.MUTED, 9, pady=(10, 16))

    def _render_search(self, title, subtitle, cols, rows, best_text, how_to, bar_col):
        gw = self.gw
        p = self._page("tuning")
        gw.heading(p, title, subtitle)
        gw.banner(p, best_text, "good", "Winner")
        kinds = ["text"] * len(cols)
        kinds[bar_col] = "bar"
        gw.table(p, cols, rows, kinds=kinds, highlight={0: gw.GOOD_BG}, first_col_w=160)
        gw.caption(p, how_to)

    def display_results(self) -> None:
        r = self.last_results
        self._reset_charts()
        self._explain_lines = None
        if r is None:
            for key in ("overview", "models", "agreement", "fairness"):
                self.gw.empty_state(self._page(key), "No results yet", "Click Run Analysis.")
            return
        for label, fn in (("overview", self._render_overview), ("models", self._render_models),
                          ("agreement", self._render_agreement), ("fairness", self._render_fairness)):
            self._safe(label, lambda fn=fn: fn(r))
        self._safe("explain", self._render_explain)

        # ---- Charts (each guarded so one failure can't hide the sections above)
        fr = r.get("fairness_report")
        if r.get("mode") == "fusion":
            self._safe("run-results chart", lambda: self._plot_run_results(r))
            self._safe("fusion-impact chart", lambda: self._plot_fusion_impact(r))
            if r.get("agreement"):
                self._safe("agreement chart", lambda: self._plot_agreement(r["agreement"]))
        if fr is not None and len(fr):
            self._safe("fairness chart", lambda: self._plot_fairness(fr))

        self.results_nb.select(self._tab_index["overview"])
        self.toggle_settings(show=False)   # give the results room; button restores it
        self.status_var.set(f"Done - accuracy {r['accuracy']:.1%}, F1 {r['f1']:.3f}.")
        if self.session_dir is not None:
            self._w(f"\nCharts saved to: {self.session_dir}\n", "muted")

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

        feat_names = [self.header[c] for c in cols] if self.header is not None else None

        def work():
            leaderboard, best = mlp_architecture_search(X_subset, y, k_folds=5, n_epochs=120, verbose=False)
            # Train the winning architecture once on the data so its learned
            # weights can be drawn (the search itself only keeps CV scores).
            from scratch_pipeline.agents import MLPAgent
            rng = np.random.default_rng(0)
            take = rng.choice(len(y), min(len(y), 8000), replace=False)
            net = MLPAgent(**best["agent_kwargs"])
            net.fit(X_subset[take], y[take])
            return leaderboard, best, net

        def on_success(payload):
            leaderboard, best, net = payload
            self._safe("MLP search", lambda: self._render_search(
                "MLP architecture search", "5-fold cross-validation, best first.",
                ["Hidden layers", "Learning rate", "L2", "CV accuracy", "CV F1", "Parameters"],
                [[str(tuple(e["hidden_layer_sizes"])), str(e["lr"]), str(e["l2"]),
                  (f"{e['mean_acc']:.1%} \u00b1 {e['std_acc'] * 100:.1f}", e["mean_acc"]),
                  f"{e['mean_f1']:.3f}", f"{e['n_params']:,}"] for e in leaderboard],
                f"Best: hidden layers {tuple(best['hidden_layer_sizes'])}, lr {best['lr']}, L2 {best['l2']} "
                f"(CV accuracy {best['mean_acc']:.1%}, F1 {best['mean_f1']:.3f}).",
                f"To use it permanently, set AGENT_REGISTRY['MLP'] = {best['agent_kwargs']}", 3))
            self._plot_mlp_search(leaderboard)
            self._draw_structures(mlp=net, mlp_names=feat_names, tag="_after_search")
            self.results_nb.select(self._tab_index["tuning"])
            self.status_var.set("Search finished - leaderboard in the Tuning tab.")

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

        feat_names = [self.header[c] for c in cols] if self.header is not None else None

        def work():
            leaderboard, best = rf_architecture_search(X_subset, y, verbose=False)
            from scratch_pipeline.agents import RandomForestAgent
            rng = np.random.default_rng(0)
            take = rng.choice(len(y), min(len(y), 8000), replace=False)
            kw = dict(best["agent_kwargs"]); kw["n_estimators"] = min(kw.get("n_estimators", 20), 20)
            rf = RandomForestAgent(**kw)
            rf.fit(X_subset[take], y[take])
            return leaderboard, best, rf

        def on_success(payload):
            leaderboard, best, rf = payload
            self._safe("RF search", lambda: self._render_search(
                "RandomForest search", "Cross-validation, best first.",
                ["Max depth", "Min split", "Min leaf", "Class weight", "CV accuracy", "CV F1"],
                [[str(e["max_depth"]), str(e["min_samples_split"]), str(e["min_samples_leaf"]),
                  str(e["class_weight"]), (f"{e['mean_acc']:.1%} \u00b1 {e['std_acc'] * 100:.1f}", e["mean_acc"]),
                  f"{e['mean_f1']:.3f}"] for e in leaderboard],
                f"Best: depth {best['max_depth']}, min split {best['min_samples_split']}, min leaf "
                f"{best['min_samples_leaf']}, class weight {best['class_weight']}, {best['n_estimators']} trees "
                f"(CV accuracy {best['mean_acc']:.1%}, F1 {best['mean_f1']:.3f}).",
                f"To use it permanently, set AGENT_REGISTRY['RandomForest'] = {best['agent_kwargs']}", 4))
            self._plot_rf_search(leaderboard)
            self._draw_structures(forest=rf, forest_names=feat_names, tag="_after_search")
            self.results_nb.select(self._tab_index["tuning"])
            self.status_var.set("Search finished - leaderboard in the Tuning tab.")

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
            names = ctx.get("feature_names")

            gate_pairs, gate_note = None, ""
            if ctx.get("controller") is not None and ctx.get("fusion_type") == "gating":
                gate_weights = ctx["controller"].get_instance_gate_weights(sample_x.reshape(1, -1))[0]
                agent_names = list(ctx["controller"].agent_registry.keys())
                gate_pairs = sorted(zip(agent_names, gate_weights), key=lambda t: -t[1])
                gate_note = (f"The gate trusted {gate_pairs[0][0]} most for this student "
                             f"({gate_pairs[0][1]:.0%}).")

            view = report_view.explain_view(global_pairs, instance_pairs, names, sample_x,
                                            proba=baseline_proba, gate_pairs=gate_pairs)
            if gate_note:
                view["story"].append(gate_note)
            self._explain_lines = view["story"]
            self._safe("explain tab", lambda: self._render_explain(view))

            # ---- charts
            self._plot_shap_global(global_explanation)
            self._plot_shap_local(instance_explanation)
            if gate_pairs:
                self._plot_gate_weights_instance(gate_pairs)

            self.results_nb.select(self._tab_index["explain"])
            self.status_var.set("Explanation ready.")

        self._run_in_background(
            "Computing SHAP values (this samples the model repeatedly, so it can "
            "take a little while). The window will stay responsive.",
            work, on_success, clear=False)

    def export_report(self) -> None:
        if self.last_results is None:
            self.messagebox.showwarning("Nothing to export", "Run an analysis first.")
            return
        path = self.filedialog.asksaveasfilename(defaultextension=".txt",
                                                   filetypes=[("Text file", "*.txt")])
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(report_view.text_report(self.last_results, getattr(self, "_explain_lines", None)))
        self.messagebox.showinfo("Exported", f"Report saved to:\n{path}")

    def save_model_to_disk(self) -> None:
        """Pickles last_results + explain_context together, so a reload can
        immediately re-display the analysis AND use Explain/Learning Curves
        without retraining - not just the bare model."""
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
        """Restores a session saved by save_model_to_disk - the loaded
        model can be used immediately with Explain/Learning Curves, no
        retraining needed. Does NOT touch the currently loaded dataset
        (self.X/self.y); it only restores the trained model and the
        train/test split it was evaluated on."""
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
        self._w(f"\n(Loaded from: {path})\n", "muted")


