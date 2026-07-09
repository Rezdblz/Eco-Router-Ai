import os
from typing import Any, Dict, Optional
import httpx
import json
from app.clients.fireworks_client import _build_chat_completions_url

def extract_message_text(response: Dict[str, Any]) -> Optional[str]:
    """
    Extract the assistant's final answer from a Fireworks/OpenAI chat response.
    """

    if not response:
        return None

    if isinstance(response, str):
        return response.strip() or None

    choices = response.get("choices")
    if not choices:
        return None

    message = choices[0].get("message", {})

    # Preferred: assistant content
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    # Some reasoning models only expose reasoning_content
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()

    # Streaming responses
    delta = choices[0].get("delta", {})
    delta_content = delta.get("content")
    if isinstance(delta_content, str) and delta_content.strip():
        return delta_content.strip()

    return None


def call_chat_model(prompt: str, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 10, max_tokens: int = 256) -> Optional[Dict[str, Any]]:
    """Call a chat-style endpoint at the Fireworks proxy and return parsed JSON.

    Returns the provider response as a dict on success, or None on failure.
    The function attempts to call `${base_url}/v1/chat/completions` which is
    compatible with many OpenAI-compatible proxies. If `base_url` or `api_key`
    are not provided, the function will read them from the environment.
    """
    base = base_url or os.environ.get("FIREWORKS_BASE_URL")
    key = api_key or os.environ.get("FIREWORKS_API_KEY")
    if not base or not key:
        return None

    url = _build_chat_completions_url(base)
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=data, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception:
        return None