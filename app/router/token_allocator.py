import math
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

    # Base allocation per task category
    base = {
        "sentiment_classification": 24,
        "named_entity_recognition": 48,
        "text_summarisation": 72,
        "mathematical_reasoning": 96,
        "factual_knowledge": 128,
        "logical_deductive_reasoning": 192,
        "code_generation": 320,
        "code_debugging": 384,
    }.get(category, 128)

    # Prompt complexity
    words = len(prompt.split())

    if words > 500:
        base += 192
    elif words > 300:
        base += 128
    elif words > 150:
        base += 64
    elif words > 75:
        base += 32

    # Detect code
    if "```" in prompt:
        base += 128

    # Detect tables
    if "|" in prompt:
        base += 48

    # Detect long numbered lists
    if prompt.count("\n") > 20:
        base += 64

    # Detect reasoning keywords
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
        base += 64

    model = model_id.lower()

    # Small fast models
    if "minimax" in model or "m3" in model:
        base *= 0.9

    # Medium models
    elif "gemma" in model:
        base *= 1.1

    # Large reasoning/code models
    elif "kimi" in model:
        base *= 1.25

    return max(24, min(int(math.ceil(base)), 1024))