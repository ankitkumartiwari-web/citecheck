"""Inference for the sentence rhetorical-role classifier.

Loads ml/model.joblib lazily. If the model hasn't been trained yet, the helpers
degrade gracefully (return None) so the rest of the app keeps working.
"""
import re
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
_model = None
_load_failed = False


def _get_model():
    global _model, _load_failed
    if _model is None and not _load_failed:
        try:
            import joblib
            _model = joblib.load(MODEL_PATH)
        except Exception:
            _load_failed = True  # not trained yet — don't retry every call
    return _model


def _sentences(text: str):
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if len(p) > 25]  # skip fragments / headings


def classify_passage(text: str):
    """Predict the dominant rhetorical role of a passage, or None if unavailable.

    Classifies each sentence and returns the most common label.
    """
    model = _get_model()
    if model is None:
        return None
    sents = _sentences(text) or [text]
    try:
        preds = list(model.predict(sents))
    except Exception:
        return None
    if not preds:
        return None
    return max(set(preds), key=preds.count)
