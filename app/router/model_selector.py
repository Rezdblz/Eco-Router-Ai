"""Simple model selector that picks an allowed model based on classification.

This module exposes `select_model(classification, allowed_models)` which
returns a tuple `(model_id, rationale)`.

The implementation is intentionally minimal and deterministic for testing.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple


def _normalize_allowed(allowed: Optional[Iterable[str]]) -> List[str]:
    if not allowed:
        env = os.environ.get("ALLOWED_MODELS")
        if not env:
            return []
        allowed = [m.strip() for m in env.split(",") if m.strip()]
    return [m for m in allowed]


def _tier_index(category: Optional[str], count: int, confidence: float) -> int:
    if count <= 1:
        return 0

    if category in {"code_generation", "code_debugging", "mathematical_reasoning", "logical_deductive_reasoning"}:
        if confidence < 0.4:
            return min(count - 1, 2)
        if confidence < 0.75:
            return min(count - 1, 1)
        return 0

    if confidence < 0.4:
        return count - 1
    if confidence < 0.75:
        return min(count - 1, 1)
    return 0


def select_model(classification: Dict[str, object], allowed_models: Optional[Iterable[str]] = None) -> Tuple[Optional[str], Dict[str, object]]:
    """Select a model given a `classification` dict and allowed models.

    `classification` expected shape: {"category": str, "confidence": float}

    Returns (model_id or None, rationale dict).
    """
    category = classification.get("category") if isinstance(classification, dict) else None
    confidence = float(classification.get("confidence", 0.0)) if isinstance(classification, dict) else 0.0

    allowed = _normalize_allowed(allowed_models)
    rationale: Dict[str, object] = {"category": category, "confidence": confidence, "considered": allowed}

    if not allowed:
        rationale["reason"] = "no_allowed_models"
        return None, rationale

    index = _tier_index(category, len(allowed), confidence)
    chosen = allowed[index]

    if confidence < 0.4:
        rationale["reason"] = "low_confidence_choose_more_capable"
    elif confidence < 0.75:
        rationale["reason"] = "medium_confidence"
    else:
        rationale["reason"] = "high_confidence_choose_smallest"

    rationale["chosen"] = chosen
    return chosen, rationale


if __name__ == "__main__":
    # quick manual test
    os.environ["ALLOWED_MODELS"] = "model-a,model-b,model-c"
    examples = [
        ({"category": "text_summarisation", "confidence": 0.9}),
        ({"category": "code_generation", "confidence": 0.85}),
        ({"category": "mathematical_reasoning", "confidence": 0.3}),
    ]
    for ex in examples:
        print(ex, "=>", select_model(ex))
