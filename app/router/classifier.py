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
import logging
import re
from typing import Dict

from app.clients.response_parser import (
    call_router_model,
    extract_message_text,
)

logger = logging.getLogger("eco_router")

ALLOWED_CATEGORIES = {
    "factual_knowledge",
    "mathematical_reasoning",
    "sentiment_classification",
    "text_summarisation",
    "named_entity_recognition",
    "code_debugging",
    "logical_deductive_reasoning",
    "code_generation",
}

_PATTERNS = [
    (re.compile(r"\b(summariz|summary|summarise|summarize)\b", re.I), "text_summarisation", 0.85),
    (re.compile(r"\b(sentiment|opinion|tone|attitude)\b", re.I), "sentiment_classification", 0.85),
    (re.compile(r"\b(named entity|ner|named entities|extract entities|entity recognition)\b", re.I), "named_entity_recognition", 0.9),
    (re.compile(
            r"\b("
            r"debug|bug|fix code|fix this code|"
            r"traceback|stack trace|"
            r"exception|syntax error|runtime error|"
            r"compile error|error in code"
            r")\b",
            re.I,
        ),
        "code_debugging",
        0.85,
    ),
    (re.compile(
            r"\b("
            r"write a function|implement|generate code|"
            r"create a function|code snippet|"
            r"write code|"
            r"python function|javascript function"
            r")\b",
            re.I,
        ),
        "code_generation",
        0.85,
    ),
    (re.compile(
            r"\b("
            r"calculate|compute|percentage|percent|"
            r"equation|integral|derivative|arithmetic|"
            r"average|multiplication|division|subtraction|addition"
            r")\b",
            re.I,
        ),
        "mathematical_reasoning",
        0.80,
    ),
    (re.compile(
            r"\b("
            r"logic|logical|deduce|deductive|"
            r"constraint puzzle|logic puzzle|"
            r"who owns|who has|"
            r"each person|each friend|"
            r"given that"
            r")\b",
            re.I,
        ),
        "logical_deductive_reasoning",
        0.80,
    ),
    (re.compile(
            r"\b("
            r"what is|what are|who is|"
            r"where is|when did|"
            r"define|describe|explain"
            r")\b",
            re.I,
        ),
        "factual_knowledge",
        0.70,
    ),
]


def _normalize(text: str) -> str:
    return (text or "").strip()

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


def _finalize_category(result: Dict[str, object]) -> Dict[str, object]:
    if result.get("category"):
        return result

    finalized = dict(result)
    finalized["category"] = "factual_knowledge"
    finalized["confidence"] = float(finalized.get("confidence", 0.0) or 0.0)
    finalized.setdefault("method", "rules")
    finalized.setdefault("router_model", None)
    finalized.setdefault("router_usage", {})
    return finalized

def classify(
        prompt: str,
        router_model: str = "",
        base_url: str = "",
        api_key: str = "",
        rule_high_confidence: float = 0.85,
        ai_min_confidence: float = 0.60,
        confidence_margin: float = 0.15,
    ) -> Dict[str, object]:
    """Classify a prompt into one of the predefined categories.

    Returns a dict with `category` (str) and `confidence` (0.0-1.0).
    If nothing matches, returns `factual_knowledge` with low confidence.
    """
    text = _normalize(prompt)
    if not text:
        return {"category": "factual_knowledge", "confidence": 0.0, "method": "rules", "router_model": None, "router_usage": {}}
    rule_result = _classify_rules(
            text,
            confidence_margin,
        )
    
    logger.debug(f"Rule result: category={rule_result.get('category')}, confidence={rule_result.get('confidence')}, ambiguous={rule_result.get('ambiguous')}")

    if (
        rule_result.get("category")
        and rule_result["confidence"] >= rule_high_confidence
        and not rule_result.get("ambiguous", False)
    ):
        logger.debug(f"High confidence detected ({rule_result['confidence']} >= {rule_high_confidence}), returning rules result")
        # High confidence: return rules result with router fields
        result = dict(rule_result)
        result.setdefault("router_model", None)
        result.setdefault("router_usage", {})
        return result
    
    # If low confidence OR ambiguous, try AI router
    confidence = rule_result.get("confidence", 0.0)
    is_ambiguous = rule_result.get("ambiguous", False)
    is_low_confidence = confidence < ai_min_confidence
    
    logger.debug(f"Confidence check: {confidence} < {ai_min_confidence}? {is_low_confidence}, ambiguous={is_ambiguous}, router_model={bool(router_model)}")
    
    should_call_router = bool(router_model) and (
        is_ambiguous
        or confidence < rule_high_confidence
    )
    
    logger.debug(f"should_call_router={should_call_router}")

    if should_call_router:
        
        instruct = (
            "Classify this task into exactly one category:\n"
            "factual_knowledge, mathematical_reasoning, sentiment_classification, "
            "text_summarisation, named_entity_recognition, code_debugging, "
            "logical_deductive_reasoning, code_generation.\n\n"
            'Return only JSON: {"category":"category_name","confidence":0.0}\n\n'
            f"Task:\n{text[:1000]}"
        )
        logger.info(
            "Calling AI router model=%s",
            router_model,
        )

        logger.debug(
            "Router prompt:\n%s",
            instruct,
        )

        resp = call_router_model(
            instruct,
            router_model,
            base_url,
            api_key,
        )
        
        logger.info(
            "AI router raw response received: %s",
            bool(resp),
        )
        
        if resp:
            out_text = extract_message_text(resp)
            
            logger.info(
                "AI router output: %s",
                out_text,
            )
            
            router_usage = _extract_usage(resp)

            try:
                parsed = json.loads(out_text)
                if isinstance(parsed, dict) and "category" in parsed:
                    category = parsed.get("category")
                    confidence = float(parsed.get("confidence", 0.0))

                    if (
                        category in ALLOWED_CATEGORIES
                        and (
                            confidence >= ai_min_confidence
                        )
                    ):
                        return {
                            "category": category,
                            "confidence": confidence,
                            "method": "ai_router",
                            "router_model": router_model,
                            "router_usage": router_usage,
                        }
                        
            except Exception:
                # ignore parsing errors and fall back to rules
                pass

            return _finalize_category(rule_result)

    return _finalize_category(rule_result)
    

def classify_task(
        task: Dict[str, object],
        router_model: str = "",
        base_url: str = "",
        api_key: str = "",
        rule_high_confidence: float = 0.85,
        ai_min_confidence: float = 0.60,
        confidence_margin: float = 0.15,
    ) -> Dict[str, object]:
    """Classify a task dict. Expects a `prompt` field.

    Returns the original task augmented with `classification` key.
    """
    prompt = task.get("prompt") if isinstance(task, dict) else None
    cls = classify(
        prompt or "",
        router_model,
        base_url,
        api_key,
        rule_high_confidence,
        ai_min_confidence,
        confidence_margin,
    )
    out = dict(task)
    out["classification"] = cls
    return out

def _classify_rules(
        text: str,
        confidence_margin: float,
    ) -> Dict[str, object]:
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
        sorted_scores = sorted(
            aggregated.items(),
            key=lambda item: item[1],
            reverse=True,
        )
        best_category, best_confidence = sorted_scores[0]
        
            
        ambiguous = False

        if len(sorted_scores) > 1:
            margin = (
                sorted_scores[0][1]
                -
                sorted_scores[1][1]
            )

            if margin < confidence_margin:
                ambiguous = True
                return {
                    "category": None,
                    "confidence": round(margin, 2),
                    "method": "rules",
                    "ambiguous": True,
                }
                
        return {
            "category": best_category,
            "confidence": round(best_confidence, 2),
            "method": "rules",
            "ambiguous": ambiguous,
        }

    # Detect obvious code
    if any(token in text for token in ("```", "def ", "class ", "function ", "import ")):
        return {
            "category": "code_generation",
            "confidence": 0.65,
            "method": "rules",
        }

    if text.endswith("?"):
        return {
            "category": None,
            "confidence": 0.30,
            "method": "rules",
            "ambiguous": True,
        }

    # Default fallback
    return {
        "category": None,
        "confidence": 0.40,
        "method": "rules",
        "ambiguous": True,
    }
    
    