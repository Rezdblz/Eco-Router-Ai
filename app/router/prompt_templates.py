"""Category-specific prompt templates and token caps.

This module keeps the main runner small while making it easy to tune
instructions and token budgets per task category.
"""
from __future__ import annotations

from typing import Any


CATEGORY_PROMPTS: dict[str, dict[str, Any]] = {
	"factual_knowledge": {"instruction": "Answer accurately. No markdown unless requested. Do not explain what the user is asking—just answer directly."},
	"mathematical_reasoning": {"instruction": "Solve accurately and show steps only if requested. Answer directly without meta-commentary."},
	"sentiment_classification": {"instruction": "Return the sentiment label first; add a brief note only if asked."},
	"text_summarisation": {"instruction": "Summarize exactly as requested."},
	"named_entity_recognition": {"instruction": "Extract only the requested entities."},
	"code_debugging": {"instruction": "Fix or explain only the requested code. Answer directly without explaining what you will do."},
	"logical_deductive_reasoning": {"instruction": "Solve the reasoning task accurately. Answer directly without explaining your approach."},
	"code_generation": {"instruction": "Generate the requested code and nothing extra unless needed."},
}


DEFAULT_PROMPT_CONFIG: dict[str, Any] = {
	"instruction": "Answer accurately in English.",
}


COMMON_PROMPT_RULES = "IMPORTANT: Do not include meta-commentary, internal reasoning, or references to the user. Answer the task directly without explaining what you will do. No preambles. Use the requested format only."


def get_category_prompt_config(category: str | None) -> dict[str, Any]:
	if not category:
		return DEFAULT_PROMPT_CONFIG
	return CATEGORY_PROMPTS.get(category, DEFAULT_PROMPT_CONFIG)


def build_answer_prompt(task_prompt: str, category: str | None) -> str:
	prompt_config = get_category_prompt_config(category)
	return (
		f"{prompt_config['instruction']} {COMMON_PROMPT_RULES}\n"
		f"Task:\n{task_prompt.strip()}"
	)