"""Category-specific prompt templates and token caps.

This module keeps the main runner small while making it easy to tune
instructions and token budgets per task category.
"""
from __future__ import annotations

from typing import Any


CATEGORY_PROMPTS: dict[str, dict[str, Any]] = {
	"factual_knowledge": {
		"instruction": (
			"Answer in 1-2 plain-English sentences."
			" Do not add markdown unless the task asks for it."
    	),
	},
	"mathematical_reasoning": {
		"instruction": (
			"Show only the essential steps and the final answer."
			" Keep numbers, units, and calculations exact."
		),
	},
	"sentiment_classification": {
		"instruction": (
			"Return the sentiment label first."
			" Add a brief justification only if the task asks for one."
		),
	},
	"text_summarisation": {
		"instruction": (
			"Summarise the passage in one sentence."
			" Do not use bullets, headings, or commentary."
		),
	},
	"named_entity_recognition": {
		"instruction": (
			"Extract only entities present in the text."
			" Group them by type in a compact list."
		),
	},
	"code_debugging": {
		"instruction": (
			"Identify the bug, then provide the corrected code and a short explanation."
			" Do not rewrite unrelated parts of the solution."
		),
	},
	"logical_deductive_reasoning": {
		"instruction": (
			"Solve the puzzle with concise reasoning and the final answer."
			" Do not speculate beyond the given constraints."
		),
	},
	"code_generation": {
		"instruction": (
			"Write the code directly with minimal explanation."
			" Include imports and function signatures if needed."
		),
	},
}


DEFAULT_PROMPT_CONFIG: dict[str, Any] = {
	"instruction": "Answer directly in English with the smallest useful accurate output.",
}


COMMON_PROMPT_RULES = (
	"Follow the requested output format exactly."
	" If the task asks for a label, number, list, or code, provide only that format."
	" Use the fewest words needed."
	" Do not add preambles, meta commentary, or self-reference."
	" If the task is ambiguous or missing required information, say so briefly."
)


def get_category_prompt_config(category: str | None) -> dict[str, Any]:
	if not category:
		return DEFAULT_PROMPT_CONFIG
	return CATEGORY_PROMPTS.get(category, DEFAULT_PROMPT_CONFIG)


def build_answer_prompt(task_prompt: str, category: str | None) -> str:
	prompt_config = get_category_prompt_config(category)
	return (
		f"{prompt_config['instruction']} {COMMON_PROMPT_RULES}\n\n"
		f"Task:\n{task_prompt.strip()}"
	)