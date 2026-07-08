"""Minimal Fireworks AI client (OpenAI-compatible) used for light classification.

The client is intentionally small and fault-tolerant: it tries to POST a
chat-style request to the provided `base_url` using the `FIREWORKS_API_KEY`.
If the call fails for any reason the caller should handle the exception and
fall back to a non-network classifier.
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from typing import Optional, Dict, Any

import httpx


# Attempt to load a local .env for development if python-dotenv is available.
def _load_dotenv_if_present():
    try:
        from dotenv import load_dotenv
    except Exception:
        return

    # common locations: project root and current working directory
    candidates = [Path(__file__).resolve().parents[3] / ".env", Path.cwd() / ".env"]
    for p in candidates:
        if p.exists():
            load_dotenv(p)
            break


_load_dotenv_if_present()


def _build_chat_completions_url(base_url: str) -> str:
    """Build the Fireworks chat-completions URL without duplicating `/v1`."""
    parsed = urlsplit(base_url.rstrip("/"))
    path = parsed.path

    if path.endswith("/v1"):
        chat_path = path + "/chat/completions"
    else:
        chat_path = path + "/v1/chat/completions"

    return urlunsplit((parsed.scheme, parsed.netloc, chat_path, parsed.query, parsed.fragment))


def call_chat_model(prompt: str, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout: int = 10) -> Optional[Dict[str, Any]]:
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
        "max_tokens": 256,
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, json=data, headers=headers)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        return None


def extract_message_text(response: Dict[str, Any]) -> Optional[str]:
    """Extract assistant text from common chat-completion shapes.

    Supports OpenAI-like responses (choices[].message.content) and some
    provider variants. Returns the first assistant text found or None.
    """
    if not response:
        return None

    if isinstance(response, str):
        return response.strip() or None

    if isinstance(response.get("content"), str):
        return response.get("content")

    if isinstance(response.get("text"), str):
        return response.get("text")

    message = response.get("message")
    if isinstance(message, str):
        return message
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "".join(parts)

    # OpenAI-like
    choices = response.get("choices") or []
    if choices:
        first = choices[0]
        msg = first.get("message") or {}
        content = msg.get("content")
        if isinstance(content, str) and content:
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    text = part.get("text") or part.get("content")
                    if isinstance(text, str):
                        parts.append(text)
            if parts:
                return "".join(parts)

        delta = first.get("delta") or {}
        delta_content = delta.get("content")
        if isinstance(delta_content, str) and delta_content:
            return delta_content

        choice_content = first.get("content")
        if isinstance(choice_content, str) and choice_content:
            return choice_content
        # older completion-style
        text = first.get("text")
        if isinstance(text, str) and text:
            return text

    # fallback: top-level `output` or `result` fields
    output = response.get("output")
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts: list[str] = []
        for part in output:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("content")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "".join(parts)

    result = response.get("result")
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("content", "text", "message"):
            value = result.get(key)
            if isinstance(value, str):
                return value

    try:
        return json.dumps(response, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        return str(response)

    return None
