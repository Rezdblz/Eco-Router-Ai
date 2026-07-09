"""Pipeline entrypoint."""
from __future__ import annotations

import logging
from pathlib import Path

from app.core.config import Settings
from app.io.reader import load_tasks
from app.io.writer import (
    write_results,
    write_analytics,
    write_analytics_history,
)

from app.services.processor import process_task
from app.services.analytics import (
    make_run_artifacts,
    build_summary,
)

logger = logging.getLogger("eco_router")


def run_pipeline(settings: Settings) -> int:
    _clear_previous_results(settings.output_path)

    # Load tasks
    try:
        tasks = load_tasks(settings.input_path)
    except Exception as exc:
        logger.exception("Failed to load tasks: %s", exc)
        return 3

    artifacts = make_run_artifacts()

    # Execute pipeline
    try:
        for task in tasks:
            result, analytics = process_task(
                task,
                settings,
            )

            artifacts.results.append(result)
            artifacts.analytics_rows.append(analytics)
            
    except Exception as exc:
        logger.exception("Pipeline execution failed: %s", exc)
        return 5

    # Write outputs
    try:
        write_results(
            artifacts.results,
            settings.output_path,
        )

        payload = {
            "summary": build_summary(artifacts),
            "tasks": artifacts.analytics_rows,
        }

        write_analytics(
            payload,
            settings.analytics_output_path,
        )

        write_analytics_history(
            payload,
            settings.analytics_history_dir,
        )

    except Exception as exc:
        logger.exception("Failed writing outputs: %s", exc)
        return 4

    logger.info(
        "Completed processing %d tasks",
        len(artifacts.results),
    )

    return 0


def _clear_previous_results(output_path: str) -> None:
    path = Path(output_path)
    try:
        path.unlink(missing_ok=True)
    except Exception:
        logger.warning("Could not clear previous results file: %s", path)