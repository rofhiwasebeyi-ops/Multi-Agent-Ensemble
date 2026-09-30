# MAS Ensemble - Student At-Risk Prediction

Two pipelines predicting at-risk students on OULAD and the UCI Student
Performance dataset:

1. **From-scratch pipeline** (`scratch_pipeline/`) - five agents
   (Logistic Regression, SVM via SMO, Random Forest via CART, an MLP with
   manual backprop, and an Artificial Immune System negative-selection
   detector) implemented in NumPy only, fused via a learned stacking or
   gating layer, also implemented from scratch.
2. **Existing-library pipeline** (`sklearn_pipeline/`) - the same
   comparison, built with scikit-learn (`LogisticRegression`, `SVC`,
   `RandomForestClassifier`, `MLPClassifier`, fused via
   `StackingClassifier`), trained and evaluated on the exact same data,
   split, and metrics as the from-scratch pipeline, for a direct,
   apples-to-apples comparison. Note: scikit-learn has no standard
   equivalent of the AIS agent's negative-selection algorithm, so this
   slot reuses the from-scratch `AISAgent` itself (via a small
   `FromScratchAdapter`, with the exact same hyperparameters as the
   from-scratch pipeline's own `AGENT_REGISTRY["AIS"]` entry) rather than
   omitting that agent type or forcing in an unrelated substitute
   (`IsolationForest`, `OneClassSVM`) that solves a different problem.
   This is the one deliberate, disclosed exception to "everything in this
   pipeline is library code" - four of five agents are genuine library
   implementations.

Both are driven identically from `main.py --pipeline {scratch,sklearn}`
(see below) and report through the same `shared/metrics.py` functions
(accuracy, precision, recall, F1, confusion matrix) plus a shared
`fairness_report` across protected attributes, so a run's success is
measured the same way regardless of which pipeline produced it.

## Setup

```
pip install -r requirements.txt
```

tkinter (needed for the GUI) is part of the Python standard library, not
a pip package. If `python3 -c "import tkinter"` fails on Linux:
`sudo apt install python3-tk`.

## Project layout

```
mas_ensemble/
├── main.py                  entry point (GUI + headless CLI) - see "Running it" below
├── gui.py                   the Tkinter GUI, drives either pipeline identically
│
├── shared/                  infrastructure BOTH pipelines depend on - neither
│   │                        pipeline package depends on the other, only on this
│   ├── metrics.py             accuracy, confusion_matrix, precision_recall_f1, stratified k-fold
│   ├── data.py                 OULAD/UCI loading + feature building, train/test splitting
│   │                          (plain and group-aware), fairness_report, synthetic demo data
│   ├── explain.py               SHAP-based ExplainabilityEngine
│   └── persistence.py            save_model/load_model (pickle-based)
│
├── scratch_pipeline/        the FROM-SCRATCH pipeline (NumPy only)
│   ├── agents.py               the five agents (LogisticRegression, SVM, RandomForest,
│   │                           MLP, AIS) + BaggingEnsemble + AGENT_REGISTRY
│   ├── search.py                k-fold CV architecture/hyperparameter search
│   └── fusion.py                 FusionLayer / GatingFusionLayer + MASController
│                                (run_full_ensemble_pipeline is the entry point)
│
├── sklearn_pipeline/        the EXISTING-LIBRARY pipeline (scikit-learn)
│   └── pipeline.py             SKLEARN_AGENT_REGISTRY, run_sklearn_single,
│                               run_sklearn_ensemble_pipeline - depends on
│                               scratch_pipeline only for AISAgent (see below)
│
└── tests/                    correctness tests (see "Tests" below)
```

Import direction is one-way - `shared` has no dependency on either pipeline;
`scratch_pipeline` depends only on `shared`; `sklearn_pipeline` depends on
`shared` and on `scratch_pipeline` (for `AISAgent`/`AGENT_REGISTRY`, reused
for the one slot scikit-learn has no equivalent for - see `sklearn_pipeline/
pipeline.py`'s module docstring). No circular imports.

## Running it

**GUI** (default):
```
python main.py
```
Falls back to a headless console demo on synthetic data if no display is
available.

**Headless run against real data**, for reproducible results (e.g. for a
paper) without touching the GUI:
```
python main.py --dataset oulad --path /path/to/extracted/oulad
python main.py --dataset uci --path /path/to/student-mat.csv
```
Add `--pipeline sklearn` to run the existing-library pipeline instead of
the from-scratch one (default: `scratch`) - same data, same split (given
the same `--seed`), same metrics, for a direct comparison:
```
python main.py --dataset oulad --path /path/to/oulad --pipeline scratch --output-dir runs/scratch
python main.py --dataset oulad --path /path/to/oulad --pipeline sklearn --output-dir runs/sklearn
```
Run `python main.py --help` for the full set of options (agent choice,
fusion type, bagging, auto-tuning, `--heterogeneous-agents` to give each
scratch-pipeline agent a restricted, specialized feature view instead of
the full set (OULAD only), `--explain` for SHAP, `--seed` for exact
reproducibility). Every headless run writes a `manifest.json` (full
config + results) and, when the dataset has protected attributes, a
`fairness_report.csv`, to `--output-dir` (default: `./runs/<timestamp>/`).

OULAD's `--path` must be the **extracted** raw-data folder (containing
`studentInfo.csv`, `studentAssessment.csv`, `studentVle.csv`,
`studentRegistration.csv`, `assessments.csv`, `vle.csv`) - not a `.zip`.

**Save the fitted model** for later reuse (e.g. a panel demo without
retraining live) by adding `--save-model` to a headless run - it's
written as `model.pkl` in `--output-dir`:
```
python main.py --dataset oulad --path /path/to/oulad --save-model
```
Reload it anywhere with:
```python
from persistence import load_model
model = load_model("runs/<timestamp>/model.pkl")
model.predictProba(X_new)
```
The GUI has the same thing as "Save Model" / "Load Model" buttons, which
round-trip the whole analysis (not just the model) so Explain/Learning
Curves work immediately after loading, with no retraining.

**Reproducibility:** every agent (`SVMAgent`, `RandomForestAgent`,
`BaggingEnsemble`, `AISAgent`, `MLPAgent`) takes a `random_state`/`seed`
and uses a locally-scoped `np.random.Generator` - none of them mutate
global NumPy random state, so training one model never silently changes
another's result depending on call order. `AGENT_REGISTRY`'s entries also
each fix that value explicitly (rather than leaving it at the class
default of `None`), since that's what makes the *actual* pipeline - not
just a directly-parameterized agent in isolation - reproducible: without
it, each bagged member's own internal randomness (SMO's variable
selection, a tree's feature subsampling, MLP's weight init, AIS's
detector generation) was still drawn from OS entropy even though the
outer bagging bootstrap draw was seeded. Pass `--seed` on the CLI to fix
the train/test split as well, for an exactly reproducible end-to-end run.

**Force the synthetic headless demo** without attempting the GUI at all:
```
python main.py --headless
```

## Tests

```
python -m unittest discover -s tests
```

These aren't accuracy benchmarks - they check that each from-scratch
algorithm actually satisfies the mathematical property it claims to
(gradient near zero at logistic regression's convergence, KKT conditions
for the SVM's SMO solver, min-leaf-size enforcement in the CART splitter,
etc.), plus a couple of pipeline-level regression tests (e.g. the
pandas-3.0 string-dtype issue that once broke OULAD loading silently).
Good accuracy on one dataset is weak evidence a from-scratch
implementation is correct; failing one of these checks is strong evidence
it isn't.
