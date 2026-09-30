"""Entry point. Run with `python main.py`.

With no arguments: tries to launch the Tkinter GUI, falling back to a
headless console demo on synthetic data if no display is available.
"""
import argparse
import json
import os
import time

from scratch_pipeline import AGENT_REGISTRY, BaggingEnsemble, run_full_ensemble_pipeline, MASController
from shared import (
    accuracy, confusion_matrix, precision_recall_f1,
    make_synthetic_dataset, train_test_split, group_train_test_split,
    load_and_preprocess, fairness_report, ExplainabilityEngine, save_model,
)
from sklearn_pipeline import run_sklearn_single, run_sklearn_ensemble_pipeline, SKLEARN_AGENT_REGISTRY
from gui import UserInterface


def run_headless_demo():
    """Console fallback: exercises every agent + BaggingEnsemble without a display."""
    X, y = make_synthetic_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    print("STANDALONE AGENTS")
    for name, (cls, kwargs) in AGENT_REGISTRY.items():
        agent = cls(**kwargs)
        agent.fit(X_train, y_train)
        acc = accuracy(y_test, agent.predict(X_test))
        print(f"  {agent.get_name():<24} acc={acc:.3f}")

    print("\nBAGGED AGENTS")
    for name, (cls, kwargs) in AGENT_REGISTRY.items():
        bag = BaggingEnsemble(base_agent_factory=cls, n_estimators=8,
                               bootstrap=True, random_state=7, agent_kwargs=kwargs)
        bag.fit(X_train, y_train)
        acc = accuracy(y_test, bag.predict(X_test))
        print(f"  Bagged {cls.__name__:<20} acc={acc:.3f}  oob={bag.get_oob_score():.3f}")

    print("\nHETEROGENEOUS ENSEMBLE, with MLP architecture auto-tuned via 5-fold CV "
          "(auto_tune_mlp=True)")
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    controller = None

    for fusion_type in ("logistic", "gating"):
        print(f"\nfusion_type='{fusion_type}'")
        result = run_full_ensemble_pipeline(X_train, y_train, X_test, y_test,
                                             use_bagging=True, n_estimators=8,
                                             feature_names=feature_names,
                                             fusion_type=fusion_type,
                                             auto_tune_mlp=True,
                                             mlp_search_kwargs=dict(k_folds=5, n_epochs=100))
        controller = result["controller"]
        fusion_acc = accuracy(y_test, result["y_pred"])

        bmc = result["best_mlp_config"]
        print(f"  Auto-tuned MLP: hidden_layer_sizes={bmc['hidden_layer_sizes']}, "
              f"lr={bmc['lr']}, l2={bmc['l2']}  "
              f"(CV mean acc={bmc['mean_acc']:.4f}, CV mean F1={bmc['mean_f1']:.4f})")
        print(f"  Fusion ({controller.fusion.get_name()}) test accuracy: {fusion_acc:.3f}")
        print("  Per-agent standalone test accuracy:")
        for name, acc in result["per_agent_test_acc"].items():
            print(f"    {name:<18} {acc:.3f}")

        label = "Mean gate weight (training set)" if fusion_type == "gating" else "Learned static weight"
        print(f"  {label} per agent:")
        for name, w in result["fusion_weights"].items():
            print(f"    {name:<18} {w:+.4f}")

        if fusion_type == "gating":
            per_instance = controller.get_instance_gate_weights(X_test[:5])
            print("  Gate weights for the first 5 test students (rows sum to 1, "
                  "computed FROM each student's own features):")
            print("   ", "  ".join(f"{n:<7}" for n in result["agent_names"]))
            for row in per_instance:
                print("   ", "  ".join(f"{v:.3f}  " for v in row))

    print("\nEXPLAINABILITY (SHAP, via the last-trained gating controller)")
    engine = ExplainabilityEngine()
    global_pairs = controller.explain_global(X_test, background_size=30, max_samples=20,
                                              nsamples=60, top_k=5)
    print("  Global feature importance (mean |SHAP value|, top 5):")
    for line in engine.format_bar_chart(global_pairs, value_fmt="{:.4f}"):
        print(" " + line)

    instance_pairs, baseline = controller.explain_instance(
        X_test[0], X_train, background_size=30, nsamples=60, top_k=5)
    print(f"  Local SHAP explanation for test sample #1 (P(at-risk)={baseline:.3f}), top 5:")
    for line in engine.format_bar_chart(instance_pairs, value_fmt="{:+.4f}"):
        print(" " + line)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description="MAS ensemble - launches the GUI by default, or runs a "
                     "headless/scriptable analysis when --dataset is given.")
    p.add_argument("--dataset", choices=["oulad", "uci"], default=None,
                    help="Run headlessly (no GUI) against this dataset instead of launching the GUI.")
    p.add_argument("--path", default=None,
                    help="OULAD: path to the extracted raw-data folder. "
                         "UCI: path to student-mat.csv / student-por.csv.")
    p.add_argument("--headless", action="store_true",
                    help="Force the synthetic-data headless demo, skipping the GUI attempt entirely.")
    p.add_argument("--pipeline", choices=["scratch", "sklearn"], default="scratch",
                    help="'scratch': the from-scratch NumPy agents/fusion (agents.py/fusion.py). "
                         "'sklearn': the existing-library pipeline (sklearn_pipeline.py) - four "
                         "scikit-learn models fused with StackingClassifier. Both are trained/"
                         "evaluated on the exact same data/split/metrics for direct comparison. "
                         "Default: scratch.")
    p.add_argument("--agent-mode", choices=["full", "single"], default="full",
                    help="'full': all agents + fusion. "
                         "'single': one agent, optionally bagged (--pipeline=scratch only - "
                         "scikit-learn's own estimators already parallelize/bag internally where "
                         "relevant, e.g. RandomForestClassifier). Default: full.")
    p.add_argument("--agent", default="RandomForest",
                    help="Agent/model to use when --agent-mode=single. Must be a key in "
                         "AGENT_REGISTRY (--pipeline=scratch) or SKLEARN_AGENT_REGISTRY "
                         "(--pipeline=sklearn). Default: RandomForest.")
    p.add_argument("--fusion", choices=["logistic", "gating"], default="logistic",
                    help="Fusion strategy when --agent-mode=full. Default: logistic.")
    p.add_argument("--no-bagging", action="store_true",
                    help="Disable BaggingEnsemble (only meaningful with --agent-mode=single).")
    p.add_argument("--n-estimators", type=int, default=10)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0,
                    help="Seed for the train/test split, so a run can be exactly reproduced.")
    p.add_argument("--auto-tune-mlp", action="store_true")
    p.add_argument("--auto-tune-rf", action="store_true")
    p.add_argument("--heterogeneous-agents", action="store_true",
                    help="Give LogisticRegression/RandomForest/SVM/AIS restricted, specialized "
                         "feature views (academic, behaviour, engagement, academic+behaviour "
                         "respectively) instead of the full feature set; MLP keeps the full "
                         "view. This is the GUI's 'heterogeneous feature subsets' option - "
                         "only meaningful with --agent-mode=full and a dataset that has "
                         "feature groups (OULAD; UCI has no equivalent grouping).")
    p.add_argument("--explain", action="store_true",
                    help="Also compute SHAP global + local explanations after fitting.")
    p.add_argument("--output-dir", default=None,
                    help="Where to write the run manifest (config + results) and, if the "
                         "dataset has protected attributes, fairness_report.csv. "
                         "Default: ./runs/<timestamp>/")
    p.add_argument("--save-model", action="store_true",
                    help="Also pickle the fitted model (MASController or single agent/"
                         "BaggingEnsemble) to model.pkl in --output-dir, so it can be reloaded "
                         "later without retraining (see persistence.load_model).")
    return p


def run_headless_dataset(args):
    """Reproducible, scriptable equivalent of clicking 'Run Analysis' in the
    GUI against real data."""
    if not args.path:
        raise SystemExit(f"--path is required when --dataset={args.dataset} is given.")

    output_dir = args.output_dir or os.path.join("runs", time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading {args.dataset.upper()} from {args.path} ...")
    X, y, feature_names, feature_groups, protected, groups = load_and_preprocess(
        args.path, dataset_type=args.dataset)
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features.")

    if groups is not None:
        X_train, X_test, y_train, y_test, train_idx, test_idx = group_train_test_split(
            X, y, groups, test_frac=args.test_frac, seed=args.seed)
        test_protected = {k: v[test_idx] for k, v in protected.items()} if protected else {}
        split_kind = "group-aware (no student appears in both train and test)"
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_frac=args.test_frac, seed=args.seed)
        test_protected = None
        split_kind = "plain random"

    print(f"Train: {X_train.shape}, Test: {X_test.shape}  (split: {split_kind})\n")

    t0 = time.time()
    if args.pipeline == "sklearn":
        if args.agent_mode == "full":
            result = run_sklearn_ensemble_pipeline(X_train, y_train, X_test, y_test,
                                                     feature_names=feature_names,
                                                     n_estimators=args.n_estimators,
                                                     random_state=args.seed)
            y_pred = result["y_pred"]
            predict_fn = result["predict_proba"]
            model_name = result["model_name"]
            fitted_model = result["model"]
            extra = {"per_agent_test_acc": result["per_agent_test_acc"],
                     "fusion_weights": result["fusion_weights"]}
        else:
            if args.agent not in SKLEARN_AGENT_REGISTRY:
                raise SystemExit(f"--agent={args.agent!r} is not in SKLEARN_AGENT_REGISTRY. "
                                  f"Available: {list(SKLEARN_AGENT_REGISTRY.keys())}")
            result = run_sklearn_single(args.agent, X_train, y_train, X_test, y_test,
                                         n_estimators=args.n_estimators, random_state=args.seed)
            y_pred = result["y_pred"]
            predict_fn = result["predict_proba"]
            model_name = result["model_name"]
            fitted_model = result["model"]
            extra = {}
    elif args.agent_mode == "full":
        agent_feature_overrides = None
        if args.heterogeneous_agents:
            # CLI always trains on the FULL feature matrix (no --feature-set
            # subsetting like the GUI has), so feature_groups' indices are
            # already correct positions within X_train.
            academic_idx = feature_groups.get("academic_idx", [])
            behaviour_idx = feature_groups.get("behaviour_idx", [])
            engagement_idx = feature_groups.get("engagement_idx", [])
            if academic_idx or behaviour_idx or engagement_idx:
                agent_feature_overrides = {}
                if academic_idx: agent_feature_overrides["LogisticRegression"] = academic_idx
                if behaviour_idx: agent_feature_overrides["RandomForest"] = behaviour_idx
                if engagement_idx: agent_feature_overrides["SVM"] = engagement_idx
                # MLP stays holistic (full feature view); AIS gets the
                # combined academic+behaviour view 
                ais_idx = sorted(set(academic_idx + behaviour_idx))
                if ais_idx and len(ais_idx) < X.shape[1]:
                    agent_feature_overrides["AIS"] = ais_idx
            else:
                print("Warning: --heterogeneous-agents given but this dataset has no feature "
                      "groups (only OULAD provides them) - ignoring.")

        controller = MASController(use_bagging=not args.no_bagging,
                                    n_estimators=args.n_estimators,
                                    fusion_type=args.fusion,
                                    auto_tune_mlp=args.auto_tune_mlp,
                                    auto_tune_rf=args.auto_tune_rf,
                                    agent_feature_overrides=agent_feature_overrides)
        result = controller.run_pipeline(X_train, y_train, X_test, y_test,
                                          feature_names=feature_names)
        y_pred = result["y_pred"]
        predict_fn = controller.predictProba
        model_name = f"Heterogeneous Ensemble (5 agents + {controller.fusion.get_name()})"
        fitted_model = controller
        extra = {
            "per_agent_test_acc": result["per_agent_test_acc"],
            "fusion_weights": result["fusion_weights"],
            "best_mlp_config": result.get("best_mlp_config"),
            "best_rf_config": result.get("best_rf_config"),
        }
    else:
        if args.agent not in AGENT_REGISTRY:
            raise SystemExit(f"--agent={args.agent!r} is not in AGENT_REGISTRY. "
                              f"Available: {list(AGENT_REGISTRY.keys())}")
        agent_cls, agent_kwargs = AGENT_REGISTRY[args.agent]
        if not args.no_bagging:
            model = BaggingEnsemble(base_agent_factory=agent_cls, n_estimators=args.n_estimators,
                                     bootstrap=True, random_state=7, agent_kwargs=agent_kwargs)
            model.fit(X_train, y_train)
            extra = {"oob_score": model.get_oob_score()}
            model_name = f"Bagged {agent_cls.__name__}"
        else:
            model = agent_cls(**agent_kwargs)
            model.fit(X_train, y_train)
            extra = {}
            model_name = agent_cls.__name__
        y_pred = model.predict(X_test)
        predict_fn = model.predictProba
        fitted_model = model
    dt = time.time() - t0

    acc = accuracy(y_test, y_pred)
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
    precision, recall, f1 = precision_recall_f1(tp, tn, fp, fn)

    print(f"Model: {model_name}")
    print(f"Time: {dt:.2f}s")
    print(f"Accuracy: {acc:.4f}   Precision: {precision:.4f}   Recall: {recall:.4f}   F1: {f1:.4f}")
    print(f"Confusion matrix: TP={tp} TN={tn} FP={fp} FN={fn}")

    manifest = {
        "config": vars(args),
        "dataset": {"n_rows": int(X.shape[0]), "n_features": int(X.shape[1]),
                    "split_kind": split_kind},
        "model_name": model_name,
        "time_sec": dt,
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
        **extra,
    }

    if test_protected:
        fr = fairness_report(y_test, y_pred, test_protected)
        fr_path = os.path.join(output_dir, "fairness_report.csv")
        fr.to_csv(fr_path, index=False)
        manifest["fairness_report_path"] = fr_path
        print(f"\nFairness report written to {fr_path}")

    if args.explain:
        print("\nComputing SHAP explanations...")
        engine = ExplainabilityEngine()
        global_pairs, _ = engine.global_importance_and_explanation(
            predict_fn, X_test, feature_names=feature_names,
            X_background=X_train, background_size=50, max_samples=30, nsamples=100)
        print("Global feature importance (mean |SHAP value|, top 10):")
        for line in engine.format_bar_chart(global_pairs[:10], value_fmt="{:.4f}"):
            print(" " + line)
        manifest["shap_global_top10"] = [(name, float(v)) for name, v in global_pairs[:10]]

    if args.save_model:
        model_path = os.path.join(output_dir, "model.pkl")
        save_model(fitted_model, model_path)
        manifest["model_path"] = model_path
        proba_call = ".predict_proba(X)[:, 1]" if args.pipeline == "sklearn" else ".predictProba(X)"
        print(f"\nFitted model saved to {model_path} "
              f"(reload with persistence.load_model, then call {proba_call}).")

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nRun manifest written to {manifest_path}")


if __name__ == "__main__":
    cli_args = _build_arg_parser().parse_args()

    if cli_args.dataset is not None:
        run_headless_dataset(cli_args)
    elif cli_args.headless:
        run_headless_demo()
    else:
        try:
            import tkinter as tk
            root = tk.Tk()
            app = UserInterface(root)
            root.mainloop()
        except Exception as e:
            print(f"[No GUI available: {e}]\nFalling back to headless console demo.\n")
            run_headless_demo()
