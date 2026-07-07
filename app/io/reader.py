import json
import logging
from pathlib import Path

from pydantic import ValidationError

from app.models.task import Task

logger = logging.getLogger(__name__)


def load_tasks(input_path: str) -> list[Task]:
    path = Path(input_path)

    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = path.read_bytes()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Input file is not valid JSON: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at root, got {type(data).__name__}")

    tasks: list[Task] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning("Item %d is not an object — skipped entirely", i)
            continue

        task_id = item.get("task_id")
        if not isinstance(task_id, str) or not task_id.strip():
            logger.warning("Item %d has no valid task_id — skipped entirely", i)
            continue

        try:
            tasks.append(Task.model_validate(item))
        except ValidationError as e:
            logger.warning(
                "Item %d (task_id=%r) failed validation — will produce error answer: %s",
                i, task_id, e,
            )
            # Keep a placeholder so writer can emit an error entry for this task_id
            tasks.append(Task(task_id=task_id, prompt=""))

    logger.info("Loaded %d tasks from %s", len(tasks), input_path)
    return tasks
