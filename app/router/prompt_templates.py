"""Category-specific prompt templates and token caps.

This module keeps the main runner small while making it easy to tune
instructions and token budgets per task category.
"""
from __future__ import annotations

from typing import Any


CATEGORY_PROMPTS: dict[str, dict[str, Any]] = {
	"factual_knowledge": {
		"instruction": "Answer briefly and clearly in plain English.",
		"max_tokens": 120,
	},
	"mathematical_reasoning": {
		"instruction": "Show only the necessary steps and the final answer.",
		"max_tokens": 180,
	},
	"sentiment_classification": {
		"instruction": "Return only one label: Positive, Negative, or Neutral.",
		"max_tokens": 32,
	},
	"text_summarisation": {
		"instruction": "Summarise the passage in one sentence.",
		"max_tokens": 120,
	},
	"named_entity_recognition": {
		"instruction": "Extract people, organizations, and locations as a compact list.",
		"max_tokens": 96,
	},
	"code_debugging": {
		"instruction": "Identify the bug and provide a corrected version with a brief explanation.",
		"max_tokens": 220,
	},
	"logical_deductive_reasoning": {
		"instruction": "Solve the puzzle with concise reasoning and the final answer.",
		"max_tokens": 180,
	},
	"code_generation": {
		"instruction": "Write the function or code solution directly with minimal explanation.",
		"max_tokens": 240,
	},
}


DEFAULT_PROMPT_CONFIG: dict[str, Any] = {
	"instruction": "Answer the task directly in English with the smallest useful output.",
	"max_tokens": 160,
}


def get_category_prompt_config(category: str | None) -> dict[str, Any]:
	if not category:
		return DEFAULT_PROMPT_CONFIG
	return CATEGORY_PROMPTS.get(category, DEFAULT_PROMPT_CONFIG)


def build_answer_prompt(task_prompt: str, category: str | None) -> str:
	prompt_config = get_category_prompt_config(category)
	return (
		f"{prompt_config['instruction']} "
		"Do not add any preamble, headings, or meta commentary unless the task explicitly asks for it.\n\n"
		f"Task:\n{task_prompt.strip()}"
	)