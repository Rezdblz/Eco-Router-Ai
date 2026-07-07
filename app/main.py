"""Minimal runner for the Eco-Router-Ai evaluation harness.

This script implements the minimum required behavior for local testing:
- reads `/input/tasks.json` (list of tasks with `task_id` and `prompt`)
- classifies each task
- selects a model from `ALLOWED_MODELS`
- writes `/output/results.json` with classification and chosen model

The script intentionally does not call Fireworks for inference; it only
demonstrates the classification and selection capability to keep tests
fast and deterministic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Dict

from app.router.classifier import classify_task
from app.router.model_selector import select_model
from app.clients.fireworks_client import call_chat_model, extract_message_text


INPUT_PATH = Path("/input/tasks.json")
OUTPUT_PATH = Path("/output/results.json")


def load_tasks(path: Path) -> List[Dict]:
	if not path.exists():
		return []
	with path.open("r", encoding="utf-8") as f:
		return json.load(f)


def write_results(path: Path, results: List[Dict]) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as f:
		json.dump(results, f, ensure_ascii=False, indent=2)


def run() -> int:
	tasks = load_tasks(INPUT_PATH)
	if not tasks:
		print("No tasks found at /input/tasks.json")
		return 1

	allowed_env = os.environ.get("ALLOWED_MODELS")
	allowed_list = [m.strip() for m in allowed_env.split(",")] if allowed_env else []

	results = []
	for t in tasks:
		classified = classify_task(t)
		chosen, rationale = select_model(classified.get("classification", {}), allowed_list)
		answer = None
		if chosen:
			prompt = t.get("prompt") or ""
			# call fireworks proxy; call_chat_model reads env if needed
			resp = call_chat_model(prompt, chosen)
			if resp:
				text = extract_message_text(resp)
				answer = text

		results.append({
			"task_id": t.get("task_id"),
			"classification": classified.get("classification"),
			"chosen_model": chosen,
			"rationale": rationale,
			"answer": answer,
		})

	write_results(OUTPUT_PATH, results)
	print(f"Wrote {len(results)} results to {OUTPUT_PATH}")
	return 0


if __name__ == "__main__":
	raise SystemExit(run())
