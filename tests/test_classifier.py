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
