from typing import Optional, Dict, Any
import httpx
import logging
import os

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://llm:11434")


def call_router_model(
    prompt: str,
    model: str = "eco-router",
    timeout: int = 10,
) -> Optional[Dict[str, Any]]:

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": [
                        "factual_knowledge",
                        "mathematical_reasoning",
                        "sentiment_classification",
                        "text_summarisation",
                        "named_entity_recognition",
                        "code_debugging",
                        "logical_deductive_reasoning",
                        "code_generation",
                    ],
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": [
                "category",
                "confidence",
            ],
            "additionalProperties": False,
        },
        "options": {
            "temperature": 0,
            "num_predict": 32,
            "top_p": 0.1,
        },
    }

    try:
        response = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()
        return response.json()

    except Exception as exc:
        logger.exception("Ollama request failed: %s", exc)
        return None