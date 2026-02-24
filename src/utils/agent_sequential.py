"""Sequential ABSA agent powered by the synchronous vLLM ``LLM`` engine.

This module implements a straightforward sequential agent for Aspect-Based
Sentiment Analysis (ABSA).  Reviews are processed one at a time; within each
review the agent runs an extraction → evaluation → (optional) retry loop until
the evaluator accepts the result or the retry budget is exhausted.

Compared to :mod:`agent_batch`, this approach has lower GPU throughput but
simpler control flow and is easier to debug on individual examples.

Typical usage::

    system = SequentialABSASystem(model_name="path/to/model")
    result = system.process_review(input_text="Hotel review...")
"""

import json
import logging
import os
from typing import List, Dict, Any
import torch
from vllm import LLM, SamplingParams

from .base_agent import BaseABSASystem

logger = logging.getLogger(__name__)

# Turn off multiprocessing to make the scheduling deterministic, or
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

class SequentialABSASystem(BaseABSASystem):
	"""Sequential ABSA agent that processes one review at a time.

	Uses the synchronous vLLM ``LLM`` engine.  Each review goes through an
	extraction → evaluation → retry loop, with the full (extraction, critique)
	history prepended to every subsequent extractor prompt so the model can see
	all prior mistakes before making a new attempt.

	The agent uses a **two-phase generation** strategy for every LLM call:

	1. *Thinking phase* – the model reasons freely at moderate temperature and
	   stops at the ``</think>`` sentinel.
	2. *Answering phase* – the model produces a structured answer at lower
	   temperature, conditioned on the full thought block.

	Args:
		model_name: HuggingFace model ID or local path to the LLM weights.
		prompts_dir: Directory containing the Markdown prompt templates.
		max_model_len: Maximum sequence length (in tokens) for the engine.
			Currently commented out so the engine auto-detects the context
			window from the model config.
	"""

	def __init__(self, model_name: str, prompts_dir: str = "prompts", max_model_len: int = 4096, seed: int = 42) -> None:
		"""Initialise the sequential agent and load the vLLM synchronous engine.

		Calls the parent :class:`BaseABSASystem` constructor to load prompt
		templates before instantiating the ``LLM`` engine and the three sets of
		:class:`~vllm.SamplingParams` used across the two generation phases.

		Args:
			model_name: HuggingFace model ID or local path to the LLM weights.
			prompts_dir: Directory containing the Markdown prompt templates.
				Defaults to ``"prompts"``.
			max_model_len: Maximum sequence length used if the commented-out
				``max_model_len`` argument is re-enabled in the engine config.
				Defaults to ``4096``.
			seed: Global random seed passed to the vLLM engine for
				reproducible sampling.  Defaults to ``42``.
		"""
		super().__init__(model_name, prompts_dir)

		logger.info(f"Loading vLLM model: {model_name}")
		self.llm = LLM(
			model=model_name,
			trust_remote_code=True,
			# max_model_len=max_model_len,
			dtype="auto",
			gpu_memory_utilization=0.7,
			seed=seed,
			# Prefix caching is required for the two-step (think → answer)
            # generation to reuse the KV-cache between the two phases.
            enable_prefix_caching=True,
		)

		# Step 1 (shared): free-form reasoning at moderate temperature.
		self.thinking_params = SamplingParams(
			temperature=0.6,
			top_p=0.95,
			max_tokens=4096,
			stop=["</think>"],
			include_stop_str_in_output=True,
		)
		# Step 2a (extractor): compact, deterministic JSON list output.
		self.extractor_answer_params = SamplingParams(
			temperature=0.05,
			top_p=0.1,
			max_tokens=512,
		)
		# Step 2b (evaluator): richer output with reasoning + verdict.
		self.evaluator_answer_params = SamplingParams(
			temperature=0.65,
			top_p=0.95,
			max_tokens=2048,
		)

	def _generate_two_step(self, prompt: str, answer_params: SamplingParams) -> str:
		"""Run two-phase (think → answer) generation for a single prompt.

		Phase 1 lets the model reason freely at moderate temperature until it
		emits ``</think>``.  Phase 2 conditions on the full thought block and
		produces the final structured answer.

		If the thinking phase exhausts ``max_tokens`` before emitting
		``</think>``, the tag is appended manually so phase 2 can still start
		from a well-formed prefix.

		Args:
			prompt: The fully formatted prompt (system + user turns) used as
				the shared prefix for both phases.
			answer_params: :class:`~vllm.SamplingParams` applied only to the
				answer phase.  Use :attr:`extractor_answer_params` or
				:attr:`evaluator_answer_params` as appropriate.

		Returns:
			Concatenation of the raw thought text and the answer text,
			i.e. ``<thought_block></think>\n<answer_block>``.
		"""
		# STEP 1: THINKING PHASE
		generated_thought = self.llm.generate(
			[prompt], self.thinking_params, use_tqdm=False
		)[0].outputs[0].text

		answer_prompt = prompt + generated_thought
		if "</think>" not in generated_thought:
			logger.warning("Thinking phase hit max_tokens before </think>. Force-closing.")
			answer_prompt += "\n</think>\n"

		# STEP 2: ANSWERING PHASE
		generated_answer = self.llm.generate(
			[answer_prompt], answer_params, use_tqdm=False
		)[0].outputs[0].text
		logger.debug(f"Answer output: {generated_answer}")

		return generated_thought + generated_answer

	def run_extractor(self, input_text: str, critique_history: List[Dict] = None) -> List[Dict]:
		"""Extract ABSA tuples from a review using the extractor LLM.

		Builds the extractor prompt, optionally prepending a structured block
		that lists *all* previous extraction attempts together with their
		evaluator critiques.  This lets the model see the full correction
		history rather than only the most recent feedback.

		Args:
			input_text: The raw review text to analyse.
			critique_history: Ordered list of prior attempt dicts (earliest
				first), each containing:

				- ``extraction`` (``List[Dict]``): the JSON from that attempt.
				- ``critique`` (``str``): the evaluator's feedback.

				Pass ``None`` or an empty list on the first attempt.

		Returns:
			A list of dicts, each representing one ABSA tuple
			(e.g. ``{"aspect": ..., "opinion": ..., "sentiment": ...}``).
			If the model output cannot be parsed as a JSON list, returns a
			single-element list containing an error dict with the raw output.
		"""
		critique_text = self._format_critique_history(critique_history or [])

		user_prompt_filled = self.prompts["extractor_user"].format(
			input_text=input_text,
			critique_instruction=critique_text,
		)

		full_prompt = self._format_prompt(self.prompts["extractor_system"], user_prompt_filled)

		if critique_text:
			logger.debug(f"Extractor prompt with critique history:\n{full_prompt}")

		raw_output = self._generate_two_step(full_prompt, self.extractor_answer_params)
		parsed_reasoning = self._parse_reasoning_output(raw_output)
		result = self._parse_json(parsed_reasoning["content"])

		if not isinstance(result, list):
			return [{"error": "Invalid JSON format", "raw_output": parsed_reasoning["content"]}]
		return result

	def run_evaluator(self, input_text: str, extraction: List[Dict]) -> Dict:
		"""Evaluate whether an extraction is correct using the evaluator LLM.

		Sends the original review together with the proposed ABSA tuples to
		the evaluator and parses its structured verdict.

		Args:
			input_text: The original review text that was analysed.
			extraction: The list of ABSA tuples produced by
				:meth:`run_extractor` for this review.

		Returns:
			A dict with at minimum:

			- ``is_correct`` (``bool``): whether the extraction is accepted.
			- ``reasoning`` (``str``): the evaluator's justification.
			- ``critique`` (``str``): actionable feedback for the next
			  extraction attempt (empty string if ``is_correct`` is ``True``).

			If the model output cannot be parsed, returns a fallback dict
			with ``is_correct=False`` and a ``"raw_output"`` key.
		"""
		user_prompt_filled = self.prompts["evaluator_user"].format(
			input_text=input_text,
			extracted_json=json.dumps(extraction, indent=2),
		)

		full_prompt = self._format_prompt(self.prompts["evaluator_system"], user_prompt_filled)

		raw_output = self._generate_two_step(full_prompt, self.evaluator_answer_params)
		parsed_reasoning = self._parse_reasoning_output(raw_output)
		result = self._parse_json(parsed_reasoning["content"])

		if not isinstance(result, dict) or "is_correct" not in result:
			return {"is_correct": False, "reasoning": "Parser failed", "critique": "", "raw_output": parsed_reasoning["content"]}
		return result

	def process_review(self, input_text: str, max_retries: int = 3) -> Dict:
		"""Run the full extract → evaluate → retry agent loop for one review.

		On each attempt the extractor produces ABSA tuples which are then judged
		by the evaluator.  If the evaluator rejects the result, *both* the
		extraction output and the critique are appended to a running
		``critique_history`` list that is passed to every subsequent extraction
		attempt, giving the model the complete correction trail.

		The loop exits early on the first accepted result or after
		``max_retries`` attempts, whichever comes first.

		Args:
			input_text: The raw review text to process.
			max_retries: Maximum number of extraction attempts before giving
				up.  Defaults to ``3``.

		Returns:
			A dict with the following keys:

			- ``final_output`` (``List[Dict]``): ABSA tuples from the last
			  extraction attempt.
			- ``status`` (``str``): ``"success"`` if the evaluator accepted
			  the result within the allowed attempts, otherwise ``"failed"``.
			- ``attempts`` (``int``): number of extraction attempts made.
			- ``history`` (``List[Dict]``): per-attempt record of each
			  extraction and its evaluation, useful for debugging.
		"""
		# Accumulates {"extraction": ..., "critique": ...} for every rejected
		# attempt so the extractor always has the full correction history.
		critique_history: List[Dict] = []
		history: List[Dict] = []

		logger.info(f"Processing: length={len(input_text)} chars")

		for attempt in range(max_retries):
			extraction = self.run_extractor(input_text, critique_history)
			evaluation = self.run_evaluator(input_text, extraction)

			history.append({
				"attempt": attempt + 1,
				"extraction": extraction,
				"evaluation": evaluation,
			})

			if evaluation.get("is_correct") is True:
				logger.info(f"Attempt {attempt + 1} successful")
				return {
					"final_output": extraction,
					"status": "success",
					"attempts": attempt + 1,
					"history": history,
				}

			current_critique = evaluation.get("critique", "Incorrect extraction.")
			critique_history.append({"extraction": extraction, "critique": current_critique})
			logger.warning(f"Attempt {attempt + 1} rejected. Critique: {current_critique}")

		logger.error("Max retries reached")
		return {
			"final_output": extraction,
			"status": "failed",
			"attempts": max_retries,
			"history": history,
		}

if __name__ == "__main__":
	pass