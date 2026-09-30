"""Save/load trained models to/from disk, so a session's result doesn't
disappear the moment the process exits 
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
    """
    with open(path, "rb") as f:
        return pickle.load(f)
