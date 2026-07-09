"""Model inference service."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import logging

from app.clients.response_parser import (
    call_inference_model,
    extract_message_text,
)
from app.core.config import Settings
from app.router.prompt_templates import build_answer_prompt
from app.router.token_allocator import allocate_tokens

logger = logging.getLogger("eco_router")


@dataclass(slots=True)
class InferenceResult:
    answer: str
    usage: dict[str, int]
    elapsed: float
    model: str
    allocated_tokens: int


def execute_task(
    task: Any,
    category: str | None,
    chosen_model: str,
    allowed_models: list[str],
    settings: Settings,
) -> InferenceResult:
    """
    Execute inference with automatic fallback and retry on truncation.
    """

    models = [chosen_model]

    fallback_models = [
        model
        for model in allowed_models
        if model != chosen_model
    ]
    
    models.extend(fallback_models)
    
    prompt = build_answer_prompt(task.prompt, category)

    for model in models:

        allocated = allocate_tokens(
            model_id=model,
            category=category,
            prompt=task.prompt,
        )

        response, elapsed = _call_once(
            prompt,
            model,
            allocated,
            settings,
        )

        if response is None:
            _log_failure(task.task_id, model, chosen_model)
            continue

        response, retry_elapsed, allocated = _retry_if_truncated(
            response,
            prompt,
            model,
            allocated,
            settings,
            task.task_id,
        )

        elapsed += retry_elapsed

        if model != chosen_model:
            logger.info(
                "Task %s fell back from %s to %s",
                task.task_id,
                chosen_model,
                model,
            )

        return InferenceResult(
            answer=extract_message_text(response) or "",
            usage=_extract_usage(response),
            elapsed=elapsed,
            model=model,
            allocated_tokens=allocated,
        )

    return InferenceResult(
        answer="",
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        elapsed=0.0,
        model=chosen_model,
        allocated_tokens=0,
    )


def _call_once(
    prompt: str,
    model: str,
    max_tokens: int,
    settings: Settings,
) -> tuple[dict[str, Any] | None, float]:

    started = perf_counter()

    response = call_inference_model(
        prompt,
        model,
        base_url=settings.fireworks_base_url,
        api_key=settings.fireworks_api_key,
        timeout=settings.request_timeout,
        max_tokens=max_tokens,
    )

    return response, perf_counter() - started


def _retry_if_truncated(
    response: dict[str, Any],
    prompt: str,
    model: str,
    allocated: int,
    settings: Settings,
    task_id: str,
) -> tuple[dict[str, Any], float, int]:

    finish_reason = (
        response.get("choices", [{}])[0]
        .get("finish_reason", "")
    )

    if finish_reason != "length":
        return response, 0.0, allocated

    new_limit = min(allocated * 2, 2048)

    logger.info(
        "Task %s truncated. Retrying with %d tokens.",
        task_id,
        new_limit,
    )

    retry_response, retry_elapsed = _call_once(
        prompt,
        model,
        new_limit,
        settings,
    )

    if retry_response is None:
        return response, retry_elapsed, allocated

    return retry_response, retry_elapsed, new_limit


def _extract_usage(
    response: dict[str, Any],
) -> dict[str, int]:

    usage = response.get("usage") or {}

    prompt = int(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or 0
    )

    completion = int(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or 0
    )

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": int(
            usage.get("total_tokens")
            or prompt + completion
        ),
    }


def _log_failure(
    task_id: str,
    model: str,
    primary_model: str,
) -> None:

    if model == primary_model:
        logger.warning(
            "Model %s returned no response for task %s",
            model,
            task_id,
        )
    else:
        logger.warning(
            "Fallback model %s also failed for task %s",
            model,
            task_id,
        )