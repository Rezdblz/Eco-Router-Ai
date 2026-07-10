"""CLI entrypoint for the EcoRoute AI pipeline."""
from __future__ import annotations

import logging
import sys


def setup_logging():
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        force=True,
    )


setup_logging()

from app.core.config import load_settings
from app.services.pipeline import run_pipeline


logger = logging.getLogger("eco_router")


def run() -> int:
    logger.info("Starting EcoRoute AI")

    try:
        settings = load_settings()
    except Exception as exc:
        logger.exception("Failed to load settings: %s", exc)
        return 2

    logging.getLogger().setLevel(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    )

    logger.info("Settings loaded")
    return run_pipeline(settings)


if __name__ == "__main__":
    sys.exit(run())