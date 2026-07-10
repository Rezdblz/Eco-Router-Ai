from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any
from uuid import uuid4

from app.models.result import Result

@dataclass(slots=True)
class RunArtifacts:
	results: list[Result]
	analytics_rows: list[dict[str, Any]]
	run_id: str
	run_started_at: str
	pipeline_started: float

def make_run_artifacts() -> RunArtifacts:
	return RunArtifacts(
		results=[],
		analytics_rows=[],
		run_id=uuid4().hex,
		run_started_at=datetime.now(timezone.utc).isoformat(),
		pipeline_started=perf_counter(),
	)

def build_summary(artifacts: RunArtifacts) -> dict[str, Any]:
    rows = artifacts.analytics_rows

    return {
        "run_id": artifacts.run_id,
        "generated_at": artifacts.run_started_at,
        "tasks": len(artifacts.results),
        "pipeline_elapsed_seconds": round(
            perf_counter() - artifacts.pipeline_started,
            4,
        ),

        "router_prompt_tokens": sum(r["router_prompt_tokens"] for r in rows),
        "router_completion_tokens": sum(r["router_completion_tokens"] for r in rows),
        "router_total_tokens": sum(r["router_total_tokens"] for r in rows),
        "overall_total_tokens": (
            sum(r["router_total_tokens"] for r in rows)
            + sum(r["total_tokens"] for r in rows)
        ),
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "total_tokens": sum(r["total_tokens"] for r in rows),

        "average_model_elapsed_seconds": (
            round(
                sum(r["model_elapsed_seconds"] for r in rows) / len(rows),
                4,
            )
            if rows
            else 0.0
        ),
    }

def build_analytics_row(
    task,
    classification,
    category,
    router_model,
    router_usage,
    chosen_model,
    actual_model,
    rationale,
    usage,
    elapsed,
    allocated_tokens,
) -> dict[str, Any]:

    prompt = usage.get("prompt_tokens", 0)
    completion = usage.get("completion_tokens", 0)
    total = usage.get("total_tokens", prompt + completion)

    return {
        "task_id": task.task_id,
        "category": category,
        "confidence": classification.get("confidence"),
        "method": classification.get("method"),

        "router_model": router_model,
        "router_prompt_tokens": router_usage.get("prompt_tokens", 0),
        "router_completion_tokens": router_usage.get("completion_tokens", 0),
        "router_total_tokens": router_usage.get("total_tokens", 0),

        "chosen_model": actual_model or chosen_model,
        "reason": rationale.get("reason"),

        "allocated_tokens": allocated_tokens,

        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,

        "completion_efficiency": (
            round(completion / allocated_tokens, 3)
            if allocated_tokens
            else 0.0
        ),

        "model_elapsed_seconds": round(elapsed, 4),
    }
 
