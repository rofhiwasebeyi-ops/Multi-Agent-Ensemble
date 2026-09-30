"""The from-scratch pipeline: five agents (Logistic Regression, SVM via
SMO, Random Forest via CART, an MLP with manual backprop, and an
Artificial Immune System negative-selection detector) implemented in
NumPy only, plus BaggingEnsemble and a learned fusion layer (static
stacking or context-aware gating), also implemented from scratch.

Depends only on shared/ (metrics, data, explain).
"""
from .agents import (
    BaseAgent, LogisticRegressionAgent, SVMAgent, RandomForestAgent,
    MLPAgent, AISAgent, BaggingEnsemble, AGENT_REGISTRY,
)
from .search import mlp_architecture_search, rf_architecture_search
from .fusion import FusionLayer, GatingFusionLayer, MASController, run_full_ensemble_pipeline
