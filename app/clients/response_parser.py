from typing import Any, Dict, Optional
import httpx
import logging
from app.clients.fireworks_client import _build_chat_completions_url

logger = logging.getLogger(__name__)

def extract_message_text(response: Dict[str, Any]) -> Optional[str]:
    """
    Extract assistant text from either:
    - Fireworks / OpenAI
    - Ollama
    """

    if not response:
        return None

    # Already a plain string
    if isinstance(response, str):
        return response.strip() or None

    # -------------------------
    # Ollama format
    # -------------------------
    message = response.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

    # -------------------------
    # Fireworks / OpenAI format
    # -------------------------
    choices = response.get("choices")
    if choices:
        message = choices[0].get("message", {})

        answer = message.get("answer")
        if isinstance(answer, str) and answer.strip():
            return answer.strip()

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

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
        json_mode=True,
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a routing classification API. "
                    "Classify the user task into exactly one category. "
                    "Return ONLY a valid JSON object. "
                    "No explanations. "
                    "No markdown. "
                    "No extra fields. "
                    'Schema: {"category":"category_name","confidence":0.0}'
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.0,
        max_tokens=32,
        base_url=base_url,
        api_key=api_key,
        timeout=timeout,
    )

def _post_chat_request(
    *,
    json_mode=False,
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
    
    if json_mode:
        payload["response_format"] = {
            "type": "json_object"
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