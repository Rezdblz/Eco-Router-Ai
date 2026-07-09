from typing import Any, Dict, Optional
import httpx
import logging
from app.clients.fireworks_client import _build_chat_completions_url

logger = logging.getLogger(__name__)

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

def call_inference_model(
    prompt: str,
    model: str,
    base_url: str ,
    api_key: str ,
    timeout: int = 10,
    max_tokens: int = 256,
) -> Optional[Dict[str, Any]]:

    return _post_chat_request(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )
    
def call_router_model(
    prompt: str,
    model: str,
    base_url: str = None,
    api_key: str = None,
    timeout: int = 10,
) -> Optional[Dict[str, Any]]:

    return _post_chat_request(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an AI routing classifier."
                    "Return ONLY valid JSON."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
        max_tokens=24,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )

def _post_chat_request(
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    base_url: str,
    api_key: str,
    timeout: int = 10,
) -> Optional[Dict[str, Any]]:

    url = _build_chat_completions_url(base_url)

    headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()
        
    except Exception as exc:
        logger.exception("Fireworks request failed: %s", exc)
        return None