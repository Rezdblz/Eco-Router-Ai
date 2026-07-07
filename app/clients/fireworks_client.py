"""Minimal Fireworks AI client (OpenAI-compatible) used for light classification.

The client is intentionally small and fault-tolerant: it tries to POST a
chat-style request to the provided `base_url` using the `FIREWORKS_API_KEY`.
If the call fails for any reason the caller should handle the exception and
fall back to a non-network classifier.
"""
from __future__ import annotations

import os
from pathlib import Path
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

    url = base.rstrip("/") + "/v1/chat/completions"
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
    except Exception:
        return None


def extract_message_text(response: Dict[str, Any]) -> Optional[str]:
    """Extract assistant text from common chat-completion shapes.

    Supports OpenAI-like responses (choices[].message.content) and some
    provider variants. Returns the first assistant text found or None.
    """
    if not response:
        return None

    # OpenAI-like
    choices = response.get("choices") or []
    if choices:
        first = choices[0]
        msg = first.get("message") or {}
        content = msg.get("content")
        if content:
            return content
        # older completion-style
        text = first.get("text")
        if text:
            return text

    # fallback: top-level `output` or `result` fields
    if isinstance(response.get("output"), str):
        return response.get("output")
    if isinstance(response.get("result"), str):
        return response.get("result")

    return None
