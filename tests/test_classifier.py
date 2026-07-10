from app.clients.fireworks_client import extract_message_text
from app.router.classifier import classify, classify_task


def test_classify_summarize():
    res = classify("Summarise this text in one sentence: Hello world.")
    assert res["category"] == "text_summarisation"
    assert res["confidence"] >= 0.7


def test_classify_code():
    res = classify("Write a Python function that reverses a string")
    assert res["category"] == "code_generation"


def test_classify_task_augments():
    t = {"task_id": "t1", "prompt": "Calculate 2+2"}
    out = classify_task(t)
    assert "classification" in out


def test_extract_message_text_prefers_answer_field():
    response = {
        "choices": [
            {
                "message": {
                    "answer": "Neutral",
                    "content": "This should not be used"
                }
            }
        ]
    }

    assert extract_message_text(response) == "Neutral"


def test_classify_without_match_returns_valid_category():
    res = classify("This prompt has no obvious routing keywords.")
    assert res["method"] == "rules"
    assert res["category"] == "factual_knowledge"


def test_classify_uses_ai_router_for_ambiguous_category(monkeypatch):
    def fake_classify_rules(text, confidence_margin):
        return {
            "category": None,
            "confidence": 0.1,
            "method": "rules",
            "ambiguous": True,
        }

    def fake_call_router_model(*args, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"category":"factual_knowledge","confidence":0.2}'
                    }
                }
            ]
        }

    monkeypatch.setattr("app.router.classifier._classify_rules", fake_classify_rules)
    monkeypatch.setattr("app.router.classifier.call_router_model", fake_call_router_model)

    res = classify(
        "This prompt is ambiguous.",
        router_model="router-model",
        base_url="https://example.invalid",
        api_key="test-key",
    )

    assert res["method"] == "ai_router"
    assert res["category"] == "factual_knowledge"


def test_classify_preserves_rule_result_on_invalid_ai_category(monkeypatch):
    def fake_classify_rules(text, confidence_margin):
        return {
            "category": None,
            "confidence": 0.2,
            "method": "rules",
            "ambiguous": True,
        }

    def fake_call_router_model(*args, **kwargs):
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"category":"unknown","confidence":0.1}'
                    }
                }
            ]
        }

    monkeypatch.setattr("app.router.classifier._classify_rules", fake_classify_rules)
    monkeypatch.setattr("app.router.classifier.call_router_model", fake_call_router_model)

    res = classify(
        "This prompt is ambiguous.",
        router_model="router-model",
        base_url="https://example.invalid",
        api_key="test-key",
    )

    assert res["method"] == "rules"
    assert res["category"] == "factual_knowledge"
