"""Task processing service."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from app.core.config import Settings
from app.models.result import Result
from app.router.classifier import classify_task
from app.router.model_selector import select_model
from app.services.analytics import build_analytics_row
from app.services.inference import execute_task

logger = logging.getLogger("eco_router")


@dataclass(slots=True)
class ClassificationResult:
    classification: dict[str, Any]
    category: str | None
    router_model: str | None
    router_usage: dict[str, int]


def extract_classification(
    task: Any,
    settings: Settings,
):
    """
    Run the router/classifier and normalize its output.
    """

    classified = classify_task(
        task.model_dump(),
        settings.router_model,
        settings.fireworks_base_url,
        settings.fireworks_api_key,
    )

    classification = classified.get("classification", {})
    if not isinstance(classification, dict):
        classification = {}

    return ClassificationResult(
        classification=classification,
        category=classification.get("category"),
        router_model=classification.get("router_model"),
        router_usage=classification.get(
            "router_usage",
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        ),
    )


def process_task(
    task: Any,
    settings: Settings,
) -> tuple[Result, dict[str, Any]]:
    """
    Process a single task.
    """

    routing = extract_classification(
        task,
        settings,
    )

    chosen_model, rationale = select_model(
        routing.classification,
        settings.allowed_models,
    )

    if not chosen_model:
        logger.warning(
            "No model selected for task %s",
            task.task_id,
        )

        inference = None

    else:
        inference = execute_task(
            task=task,
            category=routing.category,
            chosen_model=chosen_model,
            allowed_models=settings.allowed_models,
            settings=settings,
        )

    result = Result(
        task_id=task.task_id,
        answer=inference.answer if inference else "",
    )

    analytics = build_analytics_row(
        task=task,
        classification=routing.classification,
        category=routing.category,
        router_model=routing.router_model,
        router_usage=routing.router_usage,
        chosen_model=chosen_model,
        actual_model=inference.model if inference else None,
        rationale=rationale,
        usage=inference.usage if inference else {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
        elapsed=inference.elapsed if inference else 0.0,
        allocated_tokens=inference.allocated_tokens if inference else 0,
    )

    return result, analytics


def process_batch(
    tasks: list[Any],
    settings: Settings,
) -> list[tuple[Result, dict[str, Any]]]:
    """
    Process all tasks concurrently.
    """

    if not tasks:
        return []

    max_workers = min(8, len(tasks))

    if max_workers == 1:
        return [
            process_task(task, settings)
            for task in tasks
        ]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(
            executor.map(
                lambda task: process_task(task, settings),
                tasks,
            )
        )