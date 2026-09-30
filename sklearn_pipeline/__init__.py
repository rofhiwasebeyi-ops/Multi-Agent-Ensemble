"""The existing-library pipeline: four genuine scikit-learn models
(LogisticRegression, SVC, RandomForestClassifier, MLPClassifier) fused via
StackingClassifier. The fifth slot (AIS) has no scikit-learn equivalent -
see pipeline.py's module docstring for why it reuses scratch_pipeline's
own AISAgent there instead of omitting it or substituting an unrelated
unsupervised detector.
"""
from .pipeline import (
    SKLEARN_AGENT_REGISTRY, Subsampled, FromScratchAdapter,
    run_sklearn_single, run_sklearn_ensemble_pipeline,
)
