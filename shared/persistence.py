"""Save/load trained models to/from disk, so a session's result doesn't
disappear the moment the process exits - useful for demoing a pre-trained
model (e.g. in a panel presentation) without retraining live, and for
keeping an exact copy of whatever produced a specific reported number.

Uses pickle: every object here (agents, BaggingEnsemble, MASController,
ExplainabilityEngine, the numpy arrays and plain dicts/lists around them)
is a plain Python/NumPy object with no open file handles, sockets, or Tk
widgets, so it round-trips through pickle without special-casing. Do not
unpickle a model file from a source you don't trust - like any pickle,
loading one can execute arbitrary code.
"""
import pickle


def save_model(obj, path: str) -> None:
    """Pickles obj (a single agent, BaggingEnsemble, MASController, or any
    plain dict of these - e.g. a GUI's last_results/explain_context) to
    path. Raises on failure rather than silently producing a truncated or
    missing file."""
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str):
    """Unpickles and returns whatever save_model wrote to path.

    SECURITY NOTE: unpickling executes arbitrary code if the file is
    malicious. Only load files you created yourself or that came from a
    trusted source - never a model file downloaded from somewhere
    unverified.
    """
    with open(path, "rb") as f:
        return pickle.load(f)
