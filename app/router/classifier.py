"""Rule-based classifier for routing tasks to capability categories.

This simple classifier inspects the prompt text and returns a best-guess
category and a confidence score. It's intentionally lightweight so it can
be used without external dependencies during development and evaluation.

Categories (align with `context.md`):
- factual_knowledge
- mathematical_reasoning
- sentiment_classification
- text_summarisation
- named_entity_recognition
- code_debugging
- logical_deductive_reasoning
- code_generation

Functions:
- `classify(prompt: str) -> dict` — classify a prompt.
- `classify_task(task: dict) -> dict` — classify a task dict containing `prompt`.
"""
from __future__ import annotations

import re
from typing import Dict, Optional
import os

from app.clients.fireworks_client import call_chat_model, extract_message_text
from app.router.model_selector import MODEL_PROFILES

_PATTERNS = [
    (re.compile(r"\b(summariz|summary|summarise|summarize)\b", re.I), "text_summarisation", 0.9),
    (re.compile(r"\b(sentiment|opinion|tone|attitude)\b", re.I), "sentiment_classification", 0.9),
    (re.compile(r"\b(named entity|ner|entities|extract entities|extract|entity recognition)\b", re.I), "named_entity_recognition", 0.9),
    (re.compile(r"\b(debug|bug|fix|traceback|stack trace|error in)\b", re.I), "code_debugging", 0.9),
    (re.compile(r"\b(write a function|implement a function|generate code|create a function|code snippet|return the code)\b", re.I), "code_generation", 0.9),
    (re.compile(r"\b(calculate|compute|solve|sum|percentage|percent|equation|integral|derivative|math|arithmetic)\b", re.I), "mathematical_reasoning", 0.9),
    (re.compile(r"\b(explain|what is|define|describe|how does)\b", re.I), "factual_knowledge", 0.75),
    (re.compile(r"\b(logic|logical|deduce|deductive|puzzle|constraint|satisfy the conditions)\b", re.I), "logical_deductive_reasoning", 0.9),
]


def _normalize(text: str) -> str:
    return (text or "").strip()


def classify(prompt: str) -> Dict[str, object]:
    """Classify a prompt into one of the predefined categories.

    Returns a dict with `category` (str) and `confidence` (0.0-1.0).
    If nothing matches, returns `factual_knowledge` with low confidence.
    """
    text = _normalize(prompt)
    if not text:
        return {"category": "factual_knowledge", "confidence": 0.0}

    scores = []
    # apply patterns
    for pattern, category, base_conf in _PATTERNS:
        if pattern.search(text):
            scores.append((category, base_conf))

    if scores:
        # choose highest base_conf; if multiple same category matches, boost confidence
        agg = {}
        for cat, conf in scores:
            agg[cat] = max(agg.get(cat, 0.0), conf)

        # pick best
        best_cat, best_conf = max(agg.items(), key=lambda x: x[1])
        # small boost if prompt is short and explicit
        explicitness = 1.0 if len(text.split()) < 8 else 0.0
        confidence = min(1.0, best_conf + 0.05 * explicitness)
        return {"category": best_cat, "confidence": round(confidence, 2)}

    # fallback heuristics: look for question words or code markers
    # determine classifier model: prefer explicit CLASSIFIER_MODEL, otherwise
    # choose the lightest model from ALLOWED_MODELS (if provided)
    model_name = os.environ.get("CLASSIFIER_MODEL")
    if not model_name:
        allowed = os.environ.get("ALLOWED_MODELS")
        if allowed:
            allowed_list = [m.strip() for m in allowed.split(",") if m.strip()]
            # prefer models with known profiles, pick smallest size_rank
            candidates = [m for m in allowed_list if m in MODEL_PROFILES]
            if candidates:
                model_name = min(candidates, key=lambda m: MODEL_PROFILES[m].get("size_rank", 999))
            else:
                # fall back to first allowed model if profiles unknown
                model_name = allowed_list[0] if allowed_list else None

    if model_name:
        # craft a small instruction asking for JSON output
        instruct = (
            "You are a classifier that maps user prompts to one of the categories: "
            "factual_knowledge, mathematical_reasoning, sentiment_classification, text_summarisation, "
            "named_entity_recognition, code_debugging, logical_deductive_reasoning, code_generation. "
            "Respond ONLY with a JSON object with keys 'category' and 'confidence' (0.0-1.0).\n\n"
            f"Prompt: {text}\n"
        )
        resp = call_chat_model(instruct, model_name)
        if resp:
            out_text = extract_message_text(resp)
            if out_text:
                # try to parse a simple JSON substring
                import json

                try:
                    parsed = json.loads(out_text)
                    if isinstance(parsed, dict) and "category" in parsed:
                        return {"category": parsed.get("category"), "confidence": float(parsed.get("confidence", 0.0))}
                except Exception:
                    # ignore parsing errors and fall back to rules
                    pass

    if "```" in text or "def " in text or "class " in text:
        return {"category": "code_generation", "confidence": 0.6}

    if text.endswith("?"):
        return {"category": "factual_knowledge", "confidence": 0.6}

    # default
    return {"category": "factual_knowledge", "confidence": 0.4}


def classify_task(task: Dict[str, object]) -> Dict[str, object]:
    """Classify a task dict. Expects a `prompt` field.

    Returns the original task augmented with `classification` key.
    """
    prompt = task.get("prompt") if isinstance(task, dict) else None
    cls = classify(prompt or "")
    out = dict(task)
    out["classification"] = cls
    return out


if __name__ == "__main__":
    # quick manual smoke test
    examples = [
        "Summarise the following text in one sentence: The quick brown...",
        "Fix the bug in this function that raises a TypeError",
        "Calculate the percentage increase from 50 to 75",
        "Write a Python function that reverses a string",
        "Is climate change real?",
    ]
    for e in examples:
        print(e, "=>", classify(e))
