"""Category-specific prompt templates and token caps.

This module keeps the main runner small while making it easy to tune
instructions and token budgets per task category.
"""
from __future__ import annotations

from typing import Any


CATEGORY_PROMPTS: dict[str, dict[str, Any]] = {
	"factual_knowledge": {
		"instruction": "Answer briefly and clearly in plain English.",
	},
	"mathematical_reasoning": {
		"instruction": "Show only the necessary steps and the final answer.",
	},
	"sentiment_classification": {
		"instruction": "Determine the sentiment and answer according to the task. Be concise and avoid unnecessary explanation.",
	},
	"text_summarisation": {
		"instruction": "Summarise the passage in one sentence.",
	},
	"named_entity_recognition": {
		"instruction": "Extract people, organizations, and locations as a compact list.",
	},
	"code_debugging": {
		"instruction": "Identify the bug and provide a corrected version with a brief explanation.",
	},
	"logical_deductive_reasoning": {
		"instruction": "Solve the puzzle with concise reasoning and the final answer.",
	},
	"code_generation": {
		"instruction": "Write the function or code solution directly with minimal explanation.",
	},
}


DEFAULT_PROMPT_CONFIG: dict[str, Any] = {
	"instruction": "Answer the task directly in English with the smallest useful and accurate output.",
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