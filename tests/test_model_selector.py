import os

from app.router.model_selector import select_model


def test_select_model_with_allowed_env(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "minimax-m3,kimi-k2p7-code,gemma-4-31b-it")
    chosen, rationale = select_model({"category": "text_summarisation", "confidence": 0.9}, None)
    assert chosen in ["minimax-m3", "kimi-k2p7-code", "gemma-4-31b-it"]
    assert rationale["reason"] == "high_confidence_choose_smallest"


def test_select_model_low_confidence(monkeypatch):
    monkeypatch.setenv("ALLOWED_MODELS", "minimax-m3,kimi-k2p7-code,gemma-4-31b-it")
    chosen, rationale = select_model({"category": "mathematical_reasoning", "confidence": 0.2}, None)
    assert rationale["reason"] == "low_confidence_choose_more_capable"
