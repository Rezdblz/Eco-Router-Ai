"""Simple model selector that picks an allowed model based on classification.

This module exposes `select_model(classification, allowed_models)` which
returns a tuple `(model_id, rationale)`.

The implementation is intentionally minimal and deterministic for testing.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

# Minimal model profiles used for ranking. In real systems this would be
# populated from telemetry or a config; here we include a few example names.
MODEL_PROFILES: Dict[str, Dict[str, object]] = {
    # small / cheap models
    "minimax-m3": {"size_rank": 1, "capabilities": ["factual_knowledge", "text_summarisation", "sentiment_classification", "named_entity_recognition"]},
    # medium models
    "kimi-k2p7-code": {"size_rank": 2, "capabilities": ["code_generation", "code_debugging", "factual_knowledge"]},
    # larger / reasoning models
    "gemma-4-31b-it": {"size_rank": 4, "capabilities": ["mathematical_reasoning", "logical_deductive_reasoning", "code_generation", "code_debugging", "factual_knowledge"]},
}


def _normalize_allowed(allowed: Optional[Iterable[str]]) -> List[str]:
    if not allowed:
        env = os.environ.get("ALLOWED_MODELS")
        if not env:
            return []
        allowed = [m.strip() for m in env.split(",") if m.strip()]
    return [m for m in allowed]


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

    # Candidate filtering: models that claim the category capability
    candidates: List[str] = []
    for m in allowed:
        prof = MODEL_PROFILES.get(m)
        if prof and category in prof.get("capabilities", []):
            candidates.append(m)

    # If no candidates matched capability, fall back to allowed order
    if not candidates:
        candidates = list(allowed)
        rationale["fallback"] = "no_capability_match"

    # Scoring: prefer lower size_rank for higher confidence, otherwise pick first
    def score(m: str) -> int:
        prof = MODEL_PROFILES.get(m)
        if not prof:
            return 999
        return prof.get("size_rank", 999)

    candidates.sort(key=lambda x: score(x))

    # Threshold logic: if low confidence, pick a larger model if available
    if confidence < 0.4:
        # pick the largest allowed candidate (max size_rank)
        chosen = max(candidates, key=lambda x: MODEL_PROFILES.get(x, {}).get("size_rank", 999))
        rationale["reason"] = "low_confidence_choose_more_capable"
    elif confidence < 0.75:
        # medium confidence: pick medium-sized candidate
        chosen = candidates[min(1, len(candidates) - 1)]
        rationale["reason"] = "medium_confidence"
    else:
        # high confidence: pick smallest candidate
        chosen = candidates[0]
        rationale["reason"] = "high_confidence_choose_smallest"

    rationale["chosen"] = chosen
    return chosen, rationale


if __name__ == "__main__":
    # quick manual test
    import json

    os.environ["ALLOWED_MODELS"] = "minimax-m3,kimi-k2p7-code,gemma-4-31b-it"
    examples = [
        ({"category": "text_summarisation", "confidence": 0.9}),
        ({"category": "code_generation", "confidence": 0.85}),
        ({"category": "mathematical_reasoning", "confidence": 0.3}),
    ]
    for ex in examples:
        print(ex, "=>", select_model(ex))
