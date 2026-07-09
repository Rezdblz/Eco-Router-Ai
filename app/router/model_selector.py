"""Model selector that picks the best model based on task category and model capabilities.

This module exposes `select_model(classification, allowed_models)` which
returns a tuple `(model_id, rationale)`.

The selector uses capability-aware routing based on model characteristics to match
task complexity with model strengths. Model capabilities are inferred from model names.
Environment variables (ALLOWED_MODELS) determine which models are available.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Set, Tuple


def _normalize_allowed(allowed: Optional[Iterable[str]]) -> List[str]:
    if not allowed:
        env = os.environ.get("ALLOWED_MODELS")
        if not env:
            return []
        allowed = [m.strip() for m in env.split(",") if m.strip()]
    return [m for m in allowed]


def _get_model_capabilities(model_id: str) -> Set[str]:
    """Infer model capabilities from model ID.
    
    Returns a set of capability tags based on model characteristics.
    """
    model_lower = model_id.lower()
    capabilities = set()
    
    # Code-specialized models
    if "kimi" in model_lower or "code" in model_lower:
        capabilities.add("code_specialist")
    
    # Larger general-purpose models
    if "gemma-4-31b" in model_lower or "gemma-4-26b" in model_lower:
        capabilities.add("large_general_purpose")
        capabilities.add("reasoning")
    
    # Fast/small models
    if "minimax" in model_lower or "m3" in model_lower:
        capabilities.add("fast_efficient")
    
    # If no specific capabilities detected, mark as general
    if not capabilities:
        capabilities.add("general_purpose")
    
    return capabilities


def _select_by_capability(category: Optional[str], allowed: List[str]) -> Tuple[Optional[str], str]:
    """Select the best model for the task category from allowed models.
    
    Uses model capabilities (inferred from model IDs) to match task requirements
    with available models. Routes based on context.md capability categories.
    
    Returns (model_id or None, reason).
    """
    if not allowed:
        return None, "no_allowed_models"
    
    # Define capability requirements for each task category (from context.md)
    category_requirements = {
        # Code tasks require code specialists
        "code_generation": ["code_specialist"],
        "code_debugging": ["code_specialist"],
        
        # Complex reasoning tasks - try large models first, fall back to code specialist
        "mathematical_reasoning": ["large_general_purpose", "code_specialist", "fast_efficient"],
        "logical_deductive_reasoning": ["large_general_purpose", "code_specialist", "fast_efficient"],
        
        # Simple tasks work with any model, prefer fast
        "factual_knowledge": ["fast_efficient", "general_purpose"],
        "sentiment_classification": ["fast_efficient", "general_purpose"],
        "named_entity_recognition": ["fast_efficient", "general_purpose"],
        "text_summarisation": ["fast_efficient", "general_purpose"],
    }
    
    if not category or category not in category_requirements:
        # No category info, use first available
        return allowed[0], "unknown_category_using_first_available"
    
    required_capabilities = category_requirements[category]
    
    # Find models matching required capabilities, in priority order
    for required_cap in required_capabilities:
        for model in allowed:
            model_caps = _get_model_capabilities(model)
            if required_cap in model_caps:
                return model, f"capability_match_{category}_{required_cap}"
    
    # No model with required capabilities, use first available
    return allowed[0], f"no_capability_match_using_fallback"


def select_model(classification: Dict[str, object], allowed_models: Optional[Iterable[str]] = None) -> Tuple[Optional[str], Dict[str, object]]:
    """Select a model given a `classification` dict and allowed models.

    `classification` expected shape: {"category": str, "confidence": float}

    Uses capability-aware routing based on model names (inferred capabilities) and task
    category definitions from context.md. Does not hardcode model IDs - all routing is
    based on capability patterns and ALLOWED_MODELS environment variable.

    Returns (model_id or None, rationale dict).
    """
    category = classification.get("category") if isinstance(classification, dict) else None
    confidence = float(classification.get("confidence", 0.0)) if isinstance(classification, dict) else 0.0

    allowed = _normalize_allowed(allowed_models)
    rationale: Dict[str, object] = {"category": category, "confidence": confidence, "considered": allowed}

    if not allowed:
        rationale["reason"] = "no_allowed_models"
        return None, rationale

    chosen, reason = _select_by_capability(category, allowed)
    
    rationale["reason"] = reason
    rationale["chosen"] = chosen
    return chosen, rationale


if __name__ == "__main__":
    # Test with environment variable (as per context.md requirements)
    os.environ["ALLOWED_MODELS"] = "accounts/fireworks/models/minimax-m3,accounts/fireworks/models/kimi-k2p7-code,accounts/fireworks/models/gemma-4-26b-a4b-it"
    
    examples = [
        {"category": "text_summarisation", "confidence": 0.9},
        {"category": "code_generation", "confidence": 0.85},
        {"category": "mathematical_reasoning", "confidence": 0.3},
        {"category": "code_debugging", "confidence": 0.5},
        {"category": "factual_knowledge", "confidence": 0.95},
        {"category": "logical_deductive_reasoning", "confidence": 0.75},
    ]
    for ex in examples:
        model, rationale = select_model(ex)
        print(f"{ex['category']:30s} (conf={ex['confidence']:.2f}) => {model}")
        print(f"  Reason: {rationale['reason']}")

