import math


CATEGORY_BASES = {
    "sentiment_classification": 80,
    "named_entity_recognition": 160,
    "text_summarisation": 300,
    "mathematical_reasoning": 250,
    "factual_knowledge": 280,
    "logical_deductive_reasoning": 300,
    "code_generation": 450,
    "code_debugging": 500,
}

CATEGORY_MULTIPLIERS = {
    "sentiment_classification": 1.0,
    "named_entity_recognition": 1.2,
    "text_summarisation": 1.3,
    "mathematical_reasoning": 1.2,
    "factual_knowledge": 1.3,
    "logical_deductive_reasoning": 1.3,
    "code_generation": 1.4,
    "code_debugging": 1.4,
}


def _prompt_complexity_bonus(words: int) -> int:
    if words > 500:
        return 224
    if words > 300:
        return 160
    if words > 150:
        return 96
    if words > 75:
        return 48
    if words > 25:
        return 24
    return 0


def _structure_bonus(prompt: str) -> int:
    bonus = 0

    if "```" in prompt:
        bonus += 128

    if "|" in prompt:
        bonus += 48

    if prompt.count("\n") > 20:
        bonus += 64

    reasoning_keywords = (
        "explain",
        "why",
        "prove",
        "derive",
        "reason",
        "compare",
        "analyze",
        "analyse",
        "justify",
    )

    if any(k in prompt.lower() for k in reasoning_keywords):
        bonus += 48

    return bonus


def _model_multiplier(model_id: str) -> float:
    model = model_id.lower()

    if "minimax" in model or "m3" in model:
        return 0.9

    if "gemma" in model:
        return 1.1

    if "kimi" in model:
        return 1.25

    return 1.0


def _category_multiplier(category: str | None) -> float:
    return CATEGORY_MULTIPLIERS.get(category, 1.0)


def _category_floor(category: str | None) -> int:
    return CATEGORY_BASES.get(category, 96)


def allocate_tokens(
    model_id: str,
    category: str | None,
    prompt: str,
) -> int:
    """
    Dynamically allocate completion tokens based on:
    - task category
    - prompt length
    - model capability

    The goal is to minimize token usage while avoiding truncation.
    """

    words = len(prompt.split())

    base = _category_floor(category)
    base += _prompt_complexity_bonus(words)
    base += _structure_bonus(prompt)

    # Longer factual and summarization prompts need proportionally more room.
    if category in {"factual_knowledge", "text_summarisation"}:
        base += int(math.ceil(words * 0.8))
    elif category == "named_entity_recognition":
        base += int(math.ceil(words * 0.45))
    elif category in {"mathematical_reasoning", "logical_deductive_reasoning"}:
        base += int(math.ceil(words * 0.35))
    elif category in {"code_generation", "code_debugging"}:
        base += int(math.ceil(words * 0.5))

    base = int(math.ceil(base * _category_multiplier(category)))
    base = int(math.ceil(base * _model_multiplier(model_id)))

    return max(24, min(base, 1024))