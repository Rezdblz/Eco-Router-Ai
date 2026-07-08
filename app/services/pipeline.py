"""Execution pipeline for loading tasks, routing, calling models, and writing outputs."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

import logging

from app.clients.fireworks_client import call_chat_model, extract_message_text
from app.core.config import Settings
from app.io.reader import load_tasks
from app.io.writer import write_analytics, write_analytics_history, write_results
from app.models.result import Result
from app.router.classifier import classify_task
from app.router.model_selector import select_model
from app.router.prompt_templates import build_answer_prompt, get_category_prompt_config


logger = logging.getLogger("eco_router")


@dataclass(slots=True)
class RunArtifacts:
	results: list[Result]
	analytics_rows: list[dict[str, Any]]
	run_id: str
	run_started_at: str
	pipeline_started: float


def _make_run_artifacts() -> RunArtifacts:
	return RunArtifacts(
		results=[],
		analytics_rows=[],
		run_id=uuid4().hex,
		run_started_at=datetime.now(timezone.utc).isoformat(),
		pipeline_started=perf_counter(),
	)


def _build_summary(artifacts: RunArtifacts) -> dict[str, Any]:
	return {
		"run_id": artifacts.run_id,
		"generated_at": artifacts.run_started_at,
		"tasks": len(artifacts.results),
		"pipeline_elapsed_seconds": round(perf_counter() - artifacts.pipeline_started, 4),
		"prompt_tokens": sum(row["prompt_tokens"] for row in artifacts.analytics_rows),
		"completion_tokens": sum(row["completion_tokens"] for row in artifacts.analytics_rows),
		"total_tokens": sum(row["total_tokens"] for row in artifacts.analytics_rows),
		"average_model_elapsed_seconds": round(
			sum(row["model_elapsed_seconds"] for row in artifacts.analytics_rows) / len(artifacts.analytics_rows),
			4,
		) if artifacts.analytics_rows else 0.0,
	}


def _extract_classification(task: Any) -> dict[str, Any]:
	classified = classify_task(task.model_dump())
	classification = classified.get("classification", {})
	if not isinstance(classification, dict):
		return {"classification": {}, "category": None}
	return {
		"classification": classification,
		"category": classification.get("category"),
	}


def _call_model_for_task(task: Any, category: str | None, chosen: str, prompt_config: dict[str, Any], settings: Settings) -> tuple[str, dict[str, int], float]:
	answer_text = ""
	usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
	call_started = perf_counter()
	resp = call_chat_model(
		build_answer_prompt(task.prompt, category),
		chosen,
		base_url=settings.fireworks_base_url,
		api_key=settings.fireworks_api_key,
		timeout=settings.request_timeout,
		max_tokens=int(prompt_config["max_tokens"]),
	)
	model_elapsed_seconds = perf_counter() - call_started
	if not resp:
		logger.warning("Model call returned no response for task %s using model %s", task.task_id, chosen)
		return answer_text, usage, model_elapsed_seconds

	text = extract_message_text(resp)
	answer_text = text or ""
	resp_usage = resp.get("usage") if isinstance(resp, dict) else None
	if isinstance(resp_usage, dict):
		prompt_tokens = int(resp_usage.get("prompt_tokens") or resp_usage.get("input_tokens") or 0)
		completion_tokens = int(resp_usage.get("completion_tokens") or resp_usage.get("output_tokens") or 0)
		total_tokens = int(resp_usage.get("total_tokens") or (prompt_tokens + completion_tokens))
		usage = {
			"prompt_tokens": prompt_tokens,
			"completion_tokens": completion_tokens,
			"total_tokens": total_tokens,
		}
	return answer_text, usage, model_elapsed_seconds


def _build_analytics_row(task: Any, classification: dict[str, Any], category: str | None, chosen: str | None, rationale: dict[str, Any], prompt_config: dict[str, Any], usage: dict[str, int], model_elapsed_seconds: float) -> dict[str, Any]:
	return {
		"task_id": task.task_id,
		"category": category,
		"confidence": classification.get("confidence"),
		"method": classification.get("method"),
		"chosen_model": chosen,
		"reason": rationale.get("reason"),
		"prompt_template": prompt_config["instruction"],
		"max_tokens": int(prompt_config["max_tokens"]),
		"model_elapsed_seconds": round(model_elapsed_seconds, 4),
		**usage,
	}


def _process_task(task: Any, settings: Settings) -> tuple[Result, dict[str, Any]]:
	classification_data = _extract_classification(task)
	classification = classification_data["classification"]
	category = classification_data["category"]
	chosen, rationale = select_model(classification, settings.allowed_models)
	prompt_config = get_category_prompt_config(category)

	answer_text = ""
	usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
	model_elapsed_seconds = 0.0

	if chosen:
		answer_text, usage, model_elapsed_seconds = _call_model_for_task(
			task,
			category,
			chosen,
			prompt_config,
			settings,
		)

	result = Result(task_id=task.task_id, answer=answer_text)
	analytics_row = _build_analytics_row(
		task,
		classification,
		category,
		chosen,
		rationale,
		prompt_config,
		usage,
		model_elapsed_seconds,
	)
	if not chosen:
		logger.warning("No model chosen for task %s (classification=%s)", task.task_id, classification)
	return result, analytics_row


def run_pipeline(settings: Settings) -> int:
	try:
		tasks = load_tasks(settings.input_path)
	except Exception as exc:
		logger.exception("Failed to load tasks: %s", exc)
		return 3

	artifacts = _make_run_artifacts()

	for task in tasks:
		result, analytics_row = _process_task(task, settings)
		artifacts.results.append(result)
		artifacts.analytics_rows.append(analytics_row)

	try:
		write_results(artifacts.results, settings.output_path)
		payload = {
			"summary": _build_summary(artifacts),
			"tasks": artifacts.analytics_rows,
		}
		write_analytics(payload, settings.analytics_output_path)
		write_analytics_history(payload, settings.analytics_history_dir)
	except Exception as exc:
		logger.exception("Failed to write output artifacts: %s", exc)
		return 4

	logger.info("Completed processing %d tasks", len(artifacts.results))
	return 0