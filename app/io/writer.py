import logging
import os
from pathlib import Path

import orjson

from app.models.result import Result

logger = logging.getLogger(__name__)


def write_results(results: list[Result], output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [{"task_id": r.task_id, "answer": r.answer} for r in results]
    serialized = orjson.dumps(payload, option=orjson.OPT_INDENT_2)

    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_bytes(serialized)
    os.replace(tmp_path, path)  # atomic on POSIX; safe on Windows when dest exists

    logger.info("Wrote %d results to %s", len(results), output_path)


def write_analytics(analytics: dict, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = orjson.dumps(analytics, option=orjson.OPT_INDENT_2)

    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_bytes(serialized)
    os.replace(tmp_path, path)  # atomic on POSIX; safe on Windows when dest exists

    logger.info("Wrote analytics to %s", output_path)
