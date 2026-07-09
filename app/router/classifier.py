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

import json
import re
from pathlib import Path
from typing import Dict, Optional
import os

from app.clients.response_parser import call_chat_model, extract_message_text

MODEL_CAPABILITIES_PATH = Path(__file__).with_name("model_capabilities.json")

_PATTERNS = [
    (re.compile(r"\b(summariz|summary|summarise|summarize)\b", re.I), "text_summarisation", 0.9),
    (re.compile(r"\b(sentiment|opinion|tone|attitude)\b", re.I), "sentiment_classification", 0.9),
    (re.compile(r"\b(named entity|ner|entities|extract entities|extract|entity recognition)\b", re.I), "named_entity_recognition", 0.9),
    (re.compile(r"\b(debug|bug|fix|traceback|stack trace|error in)\b", re.I), "code_debugging", 0.9),
    (re.compile(r"\b(write a function|implement a function|generate code|create a function|code snippet|return the code)\b", re.I), "code_generation", 0.9),
    (re.compile(r"\b(calculate|compute|solve|sum|percentage|percent|equation|integral|derivative|math|arithmetic|increase|decrease)\b", re.I), "mathematical_reasoning", 0.9),
    (re.compile(r"\b(logic|logical|deduce|deductive|puzzle|constraint|satisfy|who has|each have|different)\b", re.I), "logical_deductive_reasoning", 0.9),
    (re.compile(r"\b(explain|what is|define|describe|how does|what are)\b", re.I), "factual_knowledge", 0.75),
]


def _normalize(text: str) -> str:
    return (text or "").strip()


def _allowed_models() -> list[str]:
    allowed = os.environ.get("ALLOWED_MODELS")
    if not allowed:
        return []
    return [m.strip() for m in allowed.split(",") if m.strip()]


def _router_model(allowed: list[str]) -> str | None:
    explicit = os.environ.get("ROUTER_MODEL")
    if explicit:
        explicit = explicit.strip()
        if explicit and explicit in allowed:
            return explicit
    return allowed[0] if allowed else None


def _extract_usage(response: object) -> dict[str, int]:
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _router_capability_context() -> str:
    return (
        "Capability context: factual_knowledge = explanations and definitions; "
        "mathematical_reasoning = arithmetic, percentages, and word problems; "
        "sentiment_classification = positive/negative/neutral labeling; "
        "text_summarisation = concise condensation of passages; "
        "named_entity_recognition = people, organizations, locations, and dates; "
        "code_debugging = finding bugs and correcting code; "
        "logical_deductive_reasoning = constraint puzzles and logic checks; "
        "code_generation = writing functions or code from a spec."
    )


def _load_model_capabilities() -> dict[str, str]:
    if not MODEL_CAPABILITIES_PATH.exists():
        return {}

    try:
        raw = MODEL_CAPABILITIES_PATH.read_text(encoding="utf-8")
        parsed = json.loads(raw)
    except Exception:
        return {}

    if not isinstance(parsed, dict):
        return {}

    return {
        str(key).strip(): str(value).strip()
        for key, value in parsed.items()
        if str(key).strip() and str(value).strip()
    }


def _router_model_capabilities(allowed: list[str]) -> str:
    capabilities = _load_model_capabilities()
    if not capabilities:
        return "Model capabilities are not provided; use allowed model IDs only."

    entries = [f"{model} = {capabilities[model]}" for model in allowed if model in capabilities]
    if not entries:
        return "Model capabilities are not provided for the currently allowed models."
    return "Model capabilities: " + "; ".join(entries) + "."


def classify(prompt: str) -> Dict[str, object]:
    """Classify a prompt into one of the predefined categories.

    Returns a dict with `category` (str) and `confidence` (0.0-1.0).
    If nothing matches, returns `factual_knowledge` with low confidence.
    """
    text = _normalize(prompt)
    if not text:
        return {"category": "factual_knowledge", "confidence": 0.0, "method": "rules"}

    allowed = _allowed_models()
    model_name = _router_model(allowed)

    if model_name:
        allowed_models_text = ", ".join(allowed)
        instruct = (
            "You are a router for user prompts. Choose one category only from: "
            "factual_knowledge, mathematical_reasoning, sentiment_classification, text_summarisation, "
            "named_entity_recognition, code_debugging, logical_deductive_reasoning, code_generation. "
            f"{_router_capability_context()} "
            f"{_router_model_capabilities(allowed)} "
            f"Available router models: {allowed_models_text}. "
            "Respond ONLY with a JSON object with keys 'category' and 'confidence' (0.0-1.0).\n\n"
            f"Prompt: {text}\n"
        )
        resp = call_chat_model(instruct, model_name)
        if resp:
            out_text = extract_message_text(resp)
            router_usage = _extract_usage(resp)
            if out_text:
                import json

                try:
                    parsed = json.loads(out_text)
                    if isinstance(parsed, dict) and "category" in parsed:
                        confidence = float(parsed.get("confidence", 0.0))
                        if confidence >= 0.4:
                            return {
                                "category": parsed.get("category"),
                                "confidence": confidence,
                                "method": "ai_router",
                                "router_model": model_name,
                                "router_usage": router_usage,
                            }
                except Exception:
                    # ignore parsing errors and fall back to rules
                    pass

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
        return {"category": best_cat, "confidence": round(confidence, 2), "method": "rules"}

    if "```" in text or "def " in text or "class " in text:
        return {"category": "code_generation", "confidence": 0.6, "method": "rules"}

    if text.endswith("?"):
        return {"category": "factual_knowledge", "confidence": 0.6, "method": "rules"}

    # default
    return {"category": "factual_knowledge", "confidence": 0.4, "method": "rules"}


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
