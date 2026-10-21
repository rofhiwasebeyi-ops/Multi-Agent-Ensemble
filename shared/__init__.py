"""Infrastructure shared by BOTH pipelines: scoring, dataset loading/
splitting, fairness diagnostics, SHAP explainability, and model
persistence. Neither pipeline package (scratch_pipeline, sklearn_pipeline)
depends on the other - both depend only on this, which is what makes their
results directly comparable (same data, same split, same metrics) rather
than two disconnected experiments.
"""
from .metrics import accuracy, confusion_matrix, precision_recall_f1, _stratified_kfold_indices
from .data import (
    load_and_preprocess, make_synthetic_dataset, train_test_split,
    group_train_test_split, fairness_report,
)
from .explain import ExplainabilityEngine
from .persistence import save_model, load_model
from .agreement import agreement_report, interpret as interpret_agreement, cohen_kappa, fleiss_kappa
