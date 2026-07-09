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
from pathlib import Path
from typing import Dict, Optional
import os

from app.clients.response_parser import (
    call_router_model,
    extract_message_text,
)
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

def classify(prompt: str) -> Dict[str, object]:
    """Classify a prompt into one of the predefined categories.

    Returns a dict with `category` (str) and `confidence` (0.0-1.0).
    If nothing matches, returns `factual_knowledge` with low confidence.
    """
    text = _normalize(prompt)
    if not text:
        return {"category": "factual_knowledge", "confidence": 0.0, "method": "rules"}
    rule_result = _classify_rules(text)

    if rule_result["confidence"] >= 0.85:
        return rule_result
    
    allowed = _allowed_models()
    model_name = _router_model(allowed)

    if model_name:
        instruct = (
            "Classify the task into one category.\n"
            "Categories:\n"
            "- factual_knowledge\n"
            "- mathematical_reasoning\n"
            "- sentiment_classification\n"
            "- text_summarisation\n"
            "- named_entity_recognition\n"
            "- code_debugging\n"
            "- logical_deductive_reasoning\n"
            "- code_generation\n\n"
            'Return ONLY JSON: {"category":"...","confidence":0.0}\n\n'
            f"Task:\n{text}"
        )
        
        resp = call_router_model(
            instruct,
            model_name,
        )
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
                
                return rule_result

def classify_task(task: Dict[str, object]) -> Dict[str, object]:
    """Classify a task dict. Expects a `prompt` field.

    Returns the original task augmented with `classification` key.
    """
    prompt = task.get("prompt") if isinstance(task, dict) else None
    cls = classify(prompt or "")
    out = dict(task)
    out["classification"] = cls
    return out

def _classify_rules(text: str) -> Dict[str, object]:
    """
    Rule-based classifier.

    Returns:
    {
        "category": str,
        "confidence": float,
        "method": "rules",
    }
    """

    scores: list[tuple[str, float]] = []

    # Apply regex patterns
    for pattern, category, base_conf in _PATTERNS:
        if pattern.search(text):
            scores.append((category, base_conf))

    if scores:
        # Keep the highest confidence for each category
        aggregated: dict[str, float] = {}

        for category, confidence in scores:
            aggregated[category] = max(
                aggregated.get(category, 0.0),
                confidence,
            )

        best_category, best_confidence = max(
            aggregated.items(),
            key=lambda item: item[1],
        )

        # Small confidence boost for short, explicit prompts
        if len(text.split()) < 8:
            best_confidence = min(best_confidence + 0.05, 1.0)

        return {
            "category": best_category,
            "confidence": round(best_confidence, 2),
            "method": "rules",
        }

    # Detect obvious code
    if any(token in text for token in ("```", "def ", "class ", "function ", "import ")):
        return {
            "category": "code_generation",
            "confidence": 0.65,
            "method": "rules",
        }

    # Questions usually request factual knowledge
    if text.endswith("?"):
        return {
            "category": "factual_knowledge",
            "confidence": 0.60,
            "method": "rules",
        }

    # Default fallback
    return {
        "category": "factual_knowledge",
        "confidence": 0.40,
        "method": "rules",
    }