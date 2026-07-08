"""CLI entrypoint for the EcoRoute AI pipeline."""
from __future__ import annotations

import logging
import sys

from app.core.config import load_settings
from app.services.pipeline import run_pipeline


logger = logging.getLogger("eco_router")


def run() -> int:
	try:
		settings = load_settings()
	except Exception as exc:
		logger.exception("Failed to load settings: %s", exc)
		return 2

	logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
	return run_pipeline(settings)


if __name__ == "__main__":
	sys.exit(run())
