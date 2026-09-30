"""Infrastructure shared by BOTH pipelines: scoring, dataset loading/
splitting, fairness diagnostics, SHAP explainability, and model
persistence.
"""
from .metrics import accuracy, confusion_matrix, precision_recall_f1, _stratified_kfold_indices
from .data import (
    load_and_preprocess, make_synthetic_dataset, train_test_split,
    group_train_test_split, fairness_report,
)
from .explain import ExplainabilityEngine
from .persistence import save_model, load_model
