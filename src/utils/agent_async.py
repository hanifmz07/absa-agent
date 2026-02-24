"""Async ABSA agent powered by a vLLM AsyncLLMEngine.

This module implements an asynchronous, two-phase (think → answer) agent for
Aspect-Based Sentiment Analysis (ABSA).  Multiple reviews can be processed
concurrently through ``asyncio.gather``; each review goes through an
extraction → evaluation → (optional) retry loop.

Typical usage::

    system = AsyncABSASystem(model_name="path/to/model")
    result = asyncio.run(system.process_review(item_id="0", input_text="..."))
"""

import json
import os
import uuid
import asyncio
import logging
from typing import List, Dict, Any
import torch

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams

from .base_agent import BaseABSASystem

logger = logging.getLogger(__name__)

# Turn off multiprocessing to make the scheduling deterministic, or
# os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" 
# Apparently, online inference with AsyncLLMEngine doesn't work well with multiprocessing disabled, so we'll leave it on for now and just be aware that the scheduling may be non-deterministic.
# https://docs.vllm.ai/en/latest/usage/reproducibility/

class AsyncABSASystem(BaseABSASystem):
    """Async ABSA agent that processes reviews concurrently via vLLM.

    Each call to :meth:`process_review` runs an extraction → evaluation loop
    for a single review.  Because every step is ``await``-ed, many reviews
    can be dispatched simultaneously with ``asyncio.gather`` without blocking
    the event loop.

    The agent uses a **two-phase generation** strategy:

    1. *Thinking phase* – the model reasons freely at higher temperature and
       stops at the ``</think>`` sentinel.
    2. *Answering phase* – the model produces a structured answer at lower
       temperature, reusing the KV-cache from step 1 via prefix caching.

    Args:
        model_name: HuggingFace model ID or local path to the LLM weights.
        prompts_dir: Directory containing the Markdown prompt templates.
        max_model_len: Maximum sequence length (in tokens) for the engine.
    """

    def __init__(self, model_name: str, prompts_dir: str = "prompts", max_model_len: int = 4096, seed: int = 42):
        super().__init__(model_name, prompts_dir)

        logger.info(f"Initializing Async vLLM Engine: {model_name}")

        # Build engine arguments; max_model_len is kept commented out so the
        # engine auto-detects the context window from the model config.
        engine_args = AsyncEngineArgs(
            model=model_name,
            trust_remote_code=True,
            # max_model_len=max_model_len,
            dtype="auto",
            gpu_memory_utilization=0.7,
            tensor_parallel_size=1,
            seed=seed,
            # Prefix caching is required for the two-step (think → answer)
            # generation to reuse the KV-cache between the two phases.
            enable_prefix_caching=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        # --- Sampling parameters for two-step generation ---
        # Step 1 (shared): free-form reasoning at moderate temperature.
        # Generation stops at </think> so the thought block is clearly bounded.
        self.thinking_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=4096,
            stop=["</think>"],
            include_stop_str_in_output=True,  # Keep the closing tag in the prompt prefix
        )

        # Step 2a (extractor): low temperature / narrow nucleus for a compact,
        # deterministic JSON list output.
        self.extractor_answer_params = SamplingParams(
            temperature=0.05,
            top_p=0.1,
            max_tokens=512,
        )

        # Step 2b (evaluator): higher temperature for richer natural-language
        # reasoning alongside the structured verdict.
        self.evaluator_answer_params = SamplingParams(
            temperature=0.65,
            top_p=0.95,
            max_tokens=2048,
        )

    async def _generate(self, prompt: str, sampling_params: SamplingParams) -> str:
        """Submit a single generation request to the vLLM async engine.

        Iterates over the async generator returned by the engine and returns
        only the final (complete) output text.

        Args:
            prompt: The fully formatted text prompt to send to the model.
            sampling_params: vLLM ``SamplingParams`` controlling decoding.

        Returns:
            The generated text string from the first output sequence.
        """
        request_id = str(uuid.uuid4())
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        # Drain the async generator; intermediate outputs are streaming chunks
        # — only the last one contains the complete text.
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        return final_output.outputs[0].text
    
    async def _generate_two_step(self, prompt: str, answer_params: SamplingParams) -> str:
        """Run two-phase (think → answer) generation for a single prompt.

        Phase 1 lets the model reason freely at moderate temperature until it
        emits ``</think>``.  Phase 2 conditions on the full thought block and
        produces the final structured answer.  Because both phases share the
        same prompt prefix, vLLM's prefix caching avoids re-computing the
        initial KV-cache entries.

        If the thinking phase exhausts ``max_tokens`` before emitting
        ``</think>``, the tag is appended manually so phase 2 can still
        start from a well-formed prefix.

        Args:
            prompt: The fully formatted prompt (system + user turns) to use
                as the shared prefix for both phases.
            answer_params: ``SamplingParams`` applied only to the answer phase.
                Use :attr:`extractor_answer_params` or
                :attr:`evaluator_answer_params` as appropriate.

        Returns:
            Concatenation of the raw thought text and the answer text,
            i.e. ``<thought_block></think>\n<answer_block>``.
        """
        # ------------------------------------------------------------------
        # PHASE 1: THINKING
        # ------------------------------------------------------------------
        thought_request_id = str(uuid.uuid4())
        thought_generator = self.engine.generate(prompt, self.thinking_params, thought_request_id)

        # Collect streaming chunks; only the last output is complete.
        thought_output = None
        async for output in thought_generator:
            thought_output = output
        generated_thought = thought_output.outputs[0].text

        # Append the thought to the original prompt so phase 2 sees the full
        # context.  Force-close the tag if the model was cut off by max_tokens.
        answer_prompt = prompt + generated_thought
        if "</think>" not in generated_thought:
            logger.warning("Thinking phase hit max_tokens before </think>. Force-closing.")
            answer_prompt += "\n</think>\n"

        # ------------------------------------------------------------------
        # PHASE 2: ANSWERING
        # ------------------------------------------------------------------
        answer_request_id = str(uuid.uuid4())
        answer_generator = self.engine.generate(answer_prompt, answer_params, answer_request_id)

        final_answer_output = None
        async for output in answer_generator:
            final_answer_output = output
        generated_answer = final_answer_output.outputs[0].text

        # Return the combined text so callers can parse both reasoning and answer.
        return generated_thought + generated_answer

    def _format_critique_history(self, critique_history: List[Dict]) -> str:
        """Render accumulated (extraction, critique) pairs into a prompt block.

        Each entry in *critique_history* must contain:

        - ``extraction`` (``List[Dict]``): the JSON output from that attempt.
        - ``critique`` (``str``): the evaluator's feedback for that attempt.

        The returned string is designed to be injected directly into the
        extractor user prompt so the model can see every mistake it made and
        the corresponding feedback before producing its next attempt.

        Args:
            critique_history: Ordered list of prior attempt dicts, earliest
                first.  An empty list returns an empty string.

        Returns:
            A formatted multi-line string, or an empty string when the list
            is empty.
        """
        if not critique_history:
            return ""

        lines = [
            "IMPORTANT FEEDBACK: Your previous extractions were rejected.",
            "Review ALL attempts and critiques below carefully before producing "
            "your new extraction.\n",
        ]

        for i, entry in enumerate(critique_history, start=1):
            extraction_str = json.dumps(entry["extraction"], indent=2, ensure_ascii=False)
            lines.append(f"--- Attempt {i} ---")
            lines.append("Your extraction:")
            lines.append(extraction_str)
            lines.append(f"Critique: \"{entry['critique']}\"\n")

        lines.append(
            "Make sure your new extraction addresses every critique listed above."
        )
        return "\n".join(lines)

    async def run_extractor(
        self,
        input_text: str,
        critique_history: List[Dict] = None,
    ) -> List[Dict]:
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
        # Build the structured feedback block from the full history.
        # On the first attempt this is an empty string.
        critique_text = self._format_critique_history(critique_history or [])

        user_prompt_filled = self.prompts["extractor_user"].format(
            input_text=input_text,
            critique_instruction=critique_text
        )


        full_prompt = self._format_prompt(self.prompts["extractor_system"], user_prompt_filled)

        # # Log the full prompt at debug level for troubleshooting; just log the prompt if the prompt contains critiques, since the full prompt can get very long.
        # if critique_text:
        #     logger.debug(f"Extractor prompt with critique history:\n{full_prompt}")

        raw_output = await self._generate_two_step(full_prompt, self.extractor_answer_params)
        parsed_reasoning = self._parse_reasoning_output(raw_output)
        result = self._parse_json(parsed_reasoning["content"])

        if not isinstance(result, list):
            # Propagate the raw model output so callers can log or retry.
            return [{"error": "Invalid JSON format", "raw_output": parsed_reasoning["content"]}]
        return result

    async def run_evaluator(self, input_text: str, extraction: List[Dict]) -> Dict:
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
            extracted_json=json.dumps(extraction, indent=2)
        )

        full_prompt = self._format_prompt(self.prompts["evaluator_system"], user_prompt_filled)

        raw_output = await self._generate_two_step(full_prompt, self.evaluator_answer_params)
        parsed_reasoning = self._parse_reasoning_output(raw_output)
        result = self._parse_json(parsed_reasoning["content"])

        if not isinstance(result, dict) or "is_correct" not in result:
            # Return a safe fallback so the caller's loop can continue.
            return {"is_correct": False, "reasoning": "Parser failed", "critique": "", "raw_output": parsed_reasoning["content"]}
        return result

    async def process_review(self, item_id: str, input_text: str, max_retries: int = 3) -> Dict:
        """Run the full extract → evaluate → retry agent loop for one review.

        On each attempt the extractor produces ABSA tuples which are then
        judged by the evaluator.  If the evaluator rejects the result, *both*
        the extraction output and the critique are appended to a running
        ``critique_history`` list that is passed to every subsequent extraction
        attempt.  This means the model always sees the complete correction
        history, not just the most recent feedback.

        The loop exits early on the first accepted result or after
        ``max_retries`` attempts, whichever comes first.

        Args:
            item_id: An opaque identifier for the review used in log messages.
            input_text: The raw review text to process.
            max_retries: Maximum number of extraction attempts before giving
                up.  Defaults to ``3``.

        Returns:
            A dict with the following keys:

            - ``final_output`` (``List[Dict]``): ABSA tuples from the last
              extraction attempt.
            - ``status`` (``str``): ``"success"`` if the evaluator accepted
              the result within the allowed attempts, otherwise ``"failed"``.
            - ``attempts`` (``int``): Number of extraction attempts made.
            - ``history`` (``List[Dict]``): Per-attempt record of each
              extraction and its evaluation, useful for debugging.
        """
        # Accumulates {"extraction": ..., "critique": ...} for every rejected
        # attempt so the extractor always has the full correction history.
        critique_history: List[Dict] = []
        history: List[Dict] = []

        logger.info(f"[ID: {item_id}] Processing: length={len(input_text)} chars")

        for attempt in range(max_retries):
            # On the first attempt critique_history is empty; on subsequent
            # attempts it contains every prior (extraction, critique) pair.
            extraction = await self.run_extractor(input_text, critique_history)
            evaluation = await self.run_evaluator(input_text, extraction)

            # Record every attempt for post-hoc analysis.
            history.append({
                "attempt": attempt + 1,
                "extraction": extraction,
                "evaluation": evaluation
            })

            if evaluation.get("is_correct") is True:
                logger.info(f"[ID: {item_id}] Attempt {attempt+1} successful")
                return {
                    "final_output": extraction,
                    "status": "success",
                    "attempts": attempt + 1,
                    "history": history
                }

            # Append this attempt's output and critique to the shared history
            # so all future extraction prompts include the full feedback trail.
            current_critique = evaluation.get("critique", "Incorrect extraction.")
            critique_history.append({
                "extraction": extraction,
                "critique": current_critique,
            })
            logger.warning(f"[ID: {item_id}] Attempt {attempt+1} rejected. Critique: {current_critique}")

        logger.error(f"[ID: {item_id}] Max retries reached")
        return {
            "final_output": extraction,  # Best attempt so far
            "status": "failed",
            "attempts": max_retries,
            "history": history
        }

if __name__ == "__main__":
    pass