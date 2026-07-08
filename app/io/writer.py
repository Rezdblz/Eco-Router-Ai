import logging
import os
import tempfile
from pathlib import Path

import orjson

from app.models.result import Result

logger = logging.getLogger(__name__)


def _write_json_overwrite(payload: object, output_path: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    serialized = orjson.dumps(payload, option=orjson.OPT_INDENT_2)

    with tempfile.NamedTemporaryFile(
        mode="wb",
        delete=False,
        dir=path.parent,
        prefix=f"{path.stem}.",
        suffix=".tmp",
    ) as handle:
        handle.write(serialized)
        temp_path = Path(handle.name)

    os.replace(temp_path, path)


def write_results(results: list[Result], output_path: str) -> None:
    payload = [{"task_id": r.task_id, "answer": r.answer} for r in results]
    _write_json_overwrite(payload, output_path)

    logger.info("Wrote %d results to %s", len(results), output_path)


def write_analytics(analytics: dict, output_path: str) -> None:
    _write_json_overwrite(analytics, output_path)

    logger.info("Wrote analytics to %s", output_path)
