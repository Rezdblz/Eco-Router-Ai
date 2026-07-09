"""Minimal Fireworks AI client (OpenAI-compatible) used for light classification.

The client is intentionally small and fault-tolerant: it tries to POST a
chat-style request to the provided `base_url` using the `FIREWORKS_API_KEY`.
If the call fails for any reason the caller should handle the exception and
fall back to a non-network classifier.
"""
from __future__ import annotations



from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit




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


