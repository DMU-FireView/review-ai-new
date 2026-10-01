"""API tests never load model weights or download anything."""
import pytest


@pytest.fixture(autouse=True)
def mock_ptext_model(monkeypatch):
    monkeypatch.setattr("app.services.analysis.predict_text_score", lambda content: {
        "text_score": 100, "suspicious_probability": 0., "predicted_label": "NORMAL",
    })
