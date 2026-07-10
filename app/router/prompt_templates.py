"""Category-specific prompt templates and token caps.

This module keeps the main runner small while making it easy to tune
instructions and token budgets per task category.
"""
from __future__ import annotations

from typing import Any


CATEGORY_PROMPTS: dict[str, dict[str, Any]] = {
	"factual_knowledge": {
		"instruction": (
			"Answer the user's question accurately and follow any requested output format exactly."
			"Do not add markdown unless the task asks for it."
    	),
	},
	"mathematical_reasoning": {
		"instruction": (
			"Solve the problem accurately and follow any instructions about showing reasoning or steps."
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
			"Complete the user's summarization request exactly as specified."
		),
	},
	"named_entity_recognition": {
		"instruction": (
			"Extract only the entities present in the text and follow the output format requested by the task."
		),
	},
	"code_debugging": {
		"instruction": (
			"Complete the requested debugging task exactly as specified. Modify or explain the code only as requested."
			"Do not rewrite unrelated parts of the solution."
		),
	},
	"logical_deductive_reasoning": {
		"instruction": (
			"Solve the reasoning task accurately and follow any instructions about showing reasoning."
		),
	},
	"code_generation": {
		"instruction": (
			"Generate the requested code and follow the requested format exactly. Include additional code only if the task requires it."
		),
	},
}


DEFAULT_PROMPT_CONFIG: dict[str, Any] = {
	"instruction": "Answer accurately in English and follow the requested output format exactly.",
}


COMMON_PROMPT_RULES = (
	"Follow the requested output format exactly."
	"If the task asks for a label, number, list, or code, provide only that format."
	"Do not add preambles, meta commentary, or self-reference."
	"If the task is ambiguous or missing required information, say so briefly."
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