"""Integrated runner using project's settings, IO and router modules.

Behavior:
- Load settings from environment via `app.core.config.load_settings()`
- Read tasks via `app.io.reader.load_tasks()`
- For each task: classify, select model, call Fireworks via client
- Write results via `app.io.writer.write_results()`

This runner is suitable for local testing and for the evaluation harness.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from uuid import uuid4
from time import perf_counter
from typing import List

from app.core.config import load_settings
from app.io.reader import load_tasks
from app.io.writer import write_analytics, write_results
from app.router.classifier import classify_task
from app.router.model_selector import select_model
from app.clients.fireworks_client import call_chat_model, extract_message_text
from app.models.result import Result


logger = logging.getLogger("eco_router")


def run() -> int:
	try:
		settings = load_settings()
	except Exception as e:
		logger.exception("Failed to load settings: %s", e)
		return 2

	logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

	try:
		tasks = load_tasks(settings.input_path)
	except Exception as e:
		logger.exception("Failed to load tasks: %s", e)
		return 3

	results: List[Result] = []
	analytics_rows: list[dict] = []
	pipeline_started = perf_counter()
	run_id = uuid4().hex
	run_started_at = datetime.now(timezone.utc).isoformat()

	for task in tasks:
		classified = classify_task(task.model_dump())
		chosen, rationale = select_model(classified.get("classification", {}), settings.allowed_models)

		answer_text = ""
		usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
		model_elapsed_seconds = 0.0
		if chosen:
			call_started = perf_counter()
			resp = call_chat_model(task.prompt, chosen, base_url=settings.fireworks_base_url, api_key=settings.fireworks_api_key, timeout=settings.request_timeout)
			model_elapsed_seconds = perf_counter() - call_started
			if resp:
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
			else:
				logger.warning("Model call returned no response for task %s using model %s", task.task_id, chosen)
		else:
			logger.warning("No model chosen for task %s (classification=%s)", task.task_id, classified.get("classification"))

		results.append(Result(task_id=task.task_id, answer=answer_text))
		analytics_rows.append(
			{
				"task_id": task.task_id,
				"category": classified.get("classification", {}).get("category"),
				"confidence": classified.get("classification", {}).get("confidence"),
				"method": classified.get("classification", {}).get("method"),
				"chosen_model": chosen,
				"reason": rationale.get("reason"),
				"model_elapsed_seconds": round(model_elapsed_seconds, 4),
				**usage,
			}
		)

	try:
		write_results(results, settings.output_path)
		write_analytics(
			{
				"summary": {
					"run_id": run_id,
					"generated_at": run_started_at,
					"tasks": len(results),
					"pipeline_elapsed_seconds": round(perf_counter() - pipeline_started, 4),
					"prompt_tokens": sum(row["prompt_tokens"] for row in analytics_rows),
					"completion_tokens": sum(row["completion_tokens"] for row in analytics_rows),
					"total_tokens": sum(row["total_tokens"] for row in analytics_rows),
					"average_model_elapsed_seconds": round(
						sum(row["model_elapsed_seconds"] for row in analytics_rows) / len(analytics_rows),
						4,
					) if analytics_rows else 0.0,
				},
				"tasks": analytics_rows,
			},
			settings.analytics_output_path,
		)
	except Exception as e:
		logger.exception("Failed to write output artifacts: %s", e)
		return 4

	logger.info("Completed processing %d tasks", len(results))
	return 0


if __name__ == "__main__":
	sys.exit(run())
