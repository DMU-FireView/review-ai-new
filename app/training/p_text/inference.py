"""Compatibility imports; runtime inference lives in app.analyzers.p_text."""
from app.analyzers.p_text import (
    DEFAULT_MODEL_PATH, TextScorePredictor, predict_text_score,
    probability_result, unavailable_result,
)
