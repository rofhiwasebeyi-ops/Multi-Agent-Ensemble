"""The existing-library pipeline: four genuine scikit-learn models
(LogisticRegression, SVC, RandomForestClassifier, MLPClassifier) fused via
StackingClassifier. The fifth slot (AIS) has no scikit-learn equivalent -
see pipeline.py's module docstring for why it reuses scratch_pipeline's
own AISAgent there instead of omitting it or substituting an unrelated
unsupervised detector.

Depends on shared/ (via scratch_pipeline, which itself depends on shared/)
and on scratch_pipeline (for AISAgent, BaggingEnsemble, AGENT_REGISTRY).
"""
from .pipeline import (
    SKLEARN_AGENT_REGISTRY, Subsampled, FromScratchAdapter,
    run_sklearn_single, run_sklearn_ensemble_pipeline, tune_sklearn_agents,
)
