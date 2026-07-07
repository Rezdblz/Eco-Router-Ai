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
from typing import List

from app.core.config import load_settings
from app.io.reader import load_tasks
from app.io.writer import write_results
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

	for task in tasks:
		classified = classify_task(task.model_dump())
		chosen, rationale = select_model(classified.get("classification", {}), settings.allowed_models)

		answer_text = ""
		if chosen:
			resp = call_chat_model(task.prompt, chosen, base_url=settings.fireworks_base_url, api_key=settings.fireworks_api_key, timeout=settings.request_timeout)
			if resp:
				text = extract_message_text(resp)
				answer_text = text or ""
			else:
				logger.warning("Model call returned no response for task %s using model %s", task.task_id, chosen)
		else:
			logger.warning("No model chosen for task %s (classification=%s)", task.task_id, classified.get("classification"))

		results.append(Result(task_id=task.task_id, answer=answer_text))

	try:
		write_results(results, settings.output_path)
	except Exception as e:
		logger.exception("Failed to write results: %s", e)
		return 4

	logger.info("Completed processing %d tasks", len(results))
	return 0


if __name__ == "__main__":
	sys.exit(run())
