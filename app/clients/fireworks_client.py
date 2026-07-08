"""Minimal Fireworks AI client (OpenAI-compatible) used for light classification.

The client is intentionally small and fault-tolerant: it tries to POST a
chat-style request to the provided `base_url` using the `FIREWORKS_API_KEY`.
If the call fails for any reason the caller should handle the exception and
fall back to a non-network classifier.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit, urlunsplit

import httpx


def _load_dotenv_if_present() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    candidates = [Path(__file__).resolve().parents[3] / ".env", Path.cwd() / ".env"]
    for candidate in candidates:
        if candidate.exists():
            load_dotenv(candidate)
            break


_load_dotenv_if_present()


def _first_string_value(payload: object, preferred_keys: tuple[str, ...]) -> Optional[str]:
    if isinstance(payload, str):
        stripped = payload.strip()
        return stripped or None

    if isinstance(payload, list):
        parts: list[str] = []
        for item in payload:
            extracted = _first_string_value(item, preferred_keys)
            if extracted:
                parts.append(extracted)
        return "".join(parts) if parts else None

    if isinstance(payload, dict):
        for key in preferred_keys:
            value = payload.get(key)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped:
                    return stripped
            if isinstance(value, (dict, list)):
                nested = _first_string_value(value, preferred_keys)
                if nested:
                    return nested

    return None


def _build_chat_completions_url(base_url: str) -> str:
    """Build the Fireworks chat-completions URL without duplicating `/v1`."""
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path

    if path.endswith("/v1"):
        chat_path = path + "/chat/completions"
    else:
        chat_path = path + "/v1/chat/completions"

    return urlunsplit((parsed.scheme, parsed.netloc, chat_path, parsed.query, parsed.fragment))


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


def extract_message_text(response: Dict[str, Any]) -> Optional[str]:
    """Extract assistant text from common chat-completion shapes.

    Prefers answer-style fields, then common text fields, and finally
    stringifies the payload if no plain text is present.
    """
    if not response:
        return None

    if isinstance(response, str):
        stripped = response.strip()
        return stripped or None

    preferred_fields = ("answer", "text", "content", "message", "output", "result", "label", "sentiment")

    direct = _first_string_value(response, preferred_fields)
    if direct:
        return direct

    choices = response.get("choices") or []
    if choices:
        first = choices[0]
        msg = first.get("message") or {}
        msg_text = _first_string_value(msg, preferred_fields)
        if msg_text:
            return msg_text

        delta = first.get("delta") or {}
        delta_text = _first_string_value(delta, preferred_fields)
        if delta_text:
            return delta_text

        choice_text = _first_string_value(first, preferred_fields)
        if choice_text:
            return choice_text

    output_text = _first_string_value(response.get("output"), preferred_fields)
    if output_text:
        return output_text

    result_text = _first_string_value(response.get("result"), preferred_fields)
    if result_text:
        return result_text

    try:
        return json.dumps(response, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(response)