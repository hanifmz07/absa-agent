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
from typing import List, Dict, Any, Optional, Tuple
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
        track_tokens: When ``True``, each generation call records the number
            of input, thinking, and output tokens.  The per-review totals are
            included under a ``"token_usage"`` key in the dict returned by
            :meth:`process_review`, and per-attempt breakdowns are included
            in the ``"history"`` list.  Set to ``False`` (the default) to
            skip all counting overhead.
    """

    def __init__(self, model_name: str, prompts_dir: str = "prompts", max_model_len: int = 24576, seed: int = 42, track_tokens: bool = False, gpu_memory_utilization: float = 0.55):
        super().__init__(model_name, prompts_dir)

        logger.info(f"Initializing Async vLLM Engine: {model_name}")

        #: Enable / disable per-generation token counting.
        self.track_tokens: bool = track_tokens

        # Build engine arguments.
        engine_args = AsyncEngineArgs(
            model=model_name,
            trust_remote_code=True,
            max_model_len=max_model_len,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
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
            max_tokens=8192,
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

    async def _generate(
        self, prompt: str, sampling_params: SamplingParams
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        """Submit a single generation request to the vLLM async engine.

        Iterates over the async generator returned by the engine and returns
        only the final (complete) output text, optionally accompanied by a
        token-usage breakdown.

        Args:
            prompt: The fully formatted text prompt to send to the model.
            sampling_params: vLLM ``SamplingParams`` controlling decoding.

        Returns:
            A 2-tuple ``(text, usage)`` where *text* is the generated string
            and *usage* is either a dict
            ``{"input_tokens": int, "thinking_tokens": int, "output_tokens": int}``
            when :attr:`track_tokens` is ``True``, or ``None`` otherwise.
        """
        request_id = str(uuid.uuid4())
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        # Drain the async generator; intermediate outputs are streaming chunks
        # — only the last one contains the complete text.
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        text = final_output.outputs[0].text

        usage: Optional[Dict[str, int]] = None
        if self.track_tokens:
            usage = {
                "input_tokens": len(final_output.prompt_token_ids),
                "thinking_tokens": 0,
                "output_tokens": len(final_output.outputs[0].token_ids),
            }

        return text, usage
    
    async def _generate_two_step(
        self, prompt: str, answer_params: SamplingParams
    ) -> Tuple[str, Optional[Dict[str, int]]]:
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
            A 2-tuple ``(combined_text, usage)`` where *combined_text* is the
            concatenation of the thought block and the answer
            (``<thought_block></think>\n<answer_block>``), and *usage* is
            either a dict
            ``{"input_tokens": int, "thinking_tokens": int, "output_tokens": int}``
            when :attr:`track_tokens` is ``True``, or ``None`` otherwise.

            *input_tokens* counts the tokens in the original prompt (phase 1
            input).  *thinking_tokens* counts the tokens generated by phase 1.
            *output_tokens* counts the tokens generated by phase 2.  The
            phase-2 prompt (original prompt + thought) reuses the cached KV
            entries so it is not double-counted.
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

        usage: Optional[Dict[str, int]] = None
        if self.track_tokens:
            usage = {
                # Prompt tokens for the initial phase-1 request.
                "input_tokens": len(thought_output.prompt_token_ids),
                # Tokens the model generated while thinking.
                "thinking_tokens": len(thought_output.outputs[0].token_ids),
                # Tokens generated for the final answer (phase 2).
                "output_tokens": len(final_answer_output.outputs[0].token_ids),
            }

        # Return the combined text so callers can parse both reasoning and answer.
        return generated_thought + generated_answer, usage

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
    ) -> Tuple[List[Dict], Optional[Dict[str, int]]]:
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
            A 2-tuple ``(extraction, usage)``.

            *extraction* is a list of dicts, each representing one ABSA tuple
            (e.g. ``{"aspect": ..., "opinion": ..., "sentiment": ...}``).
            If the model output cannot be parsed as a JSON list, returns a
            single-element list containing an error dict with the raw output.

            *usage* is a token-count dict
            ``{"input_tokens": int, "thinking_tokens": int, "output_tokens": int}``
            when :attr:`track_tokens` is ``True``, otherwise ``None``.
        """
        # Build the structured feedback block from the full history.
        # On the first attempt this is an empty string.
        critique_text = self._format_critique_history(critique_history or [])

        user_prompt_filled = self.prompts["extractor_user"].format(
            input_text=input_text,
            critique_instruction=critique_text
        )


        extractor_system = self._render_prompt_language(self.prompts["extractor_system"])
        full_prompt = self._format_prompt(extractor_system, user_prompt_filled)

        # # Log the full prompt at debug level for troubleshooting; just log the prompt if the prompt contains critiques, since the full prompt can get very long.
        # if critique_text:
        #     logger.debug(f"Extractor prompt with critique history:\n{full_prompt}")

        raw_output, usage = await self._generate_two_step(full_prompt, self.extractor_answer_params)
        parsed_reasoning = self._parse_reasoning_output(raw_output)
        result = self._parse_json(parsed_reasoning["content"])

        if not isinstance(result, list):
            # Propagate the raw model output so callers can log or retry.
            return [{"error": "Invalid JSON format", "raw_output": parsed_reasoning["content"]}], usage
        return result, usage

    async def run_evaluator(
        self, input_text: str, extraction: List[Dict]
    ) -> Tuple[Dict, Optional[Dict[str, int]]]:
        """Evaluate whether an extraction is correct using the evaluator LLM.

        Sends the original review together with the proposed ABSA tuples to
        the evaluator and parses its structured verdict.

        Args:
            input_text: The original review text that was analysed.
            extraction: The list of ABSA tuples produced by
                :meth:`run_extractor` for this review.

        Returns:
            A 2-tuple ``(evaluation, usage)``.

            *evaluation* is a dict with at minimum:

            - ``is_correct`` (``bool``): whether the extraction is accepted.
            - ``reasoning`` (``str``): the evaluator's justification.
            - ``critique`` (``str``): actionable feedback for the next
              extraction attempt (empty string if ``is_correct`` is ``True``).

            If the model output cannot be parsed, returns a fallback dict
            with ``is_correct=False`` and a ``"raw_output"`` key.

            *usage* is a token-count dict
            ``{"input_tokens": int, "thinking_tokens": int, "output_tokens": int}``
            when :attr:`track_tokens` is ``True``, otherwise ``None``.
        """
        user_prompt_filled = self.prompts["evaluator_user"].format(
            input_text=input_text,
            extracted_json=json.dumps(extraction, indent=2)
        )

        evaluator_system = self._render_prompt_language(self.prompts["evaluator_system"])
        full_prompt = self._format_prompt(evaluator_system, user_prompt_filled)

        raw_output, usage = await self._generate_two_step(full_prompt, self.evaluator_answer_params)
        parsed_reasoning = self._parse_reasoning_output(raw_output)
        result = self._parse_json(parsed_reasoning["content"])

        if not isinstance(result, dict) or "is_correct" not in result:
            # Return a safe fallback so the caller's loop can continue.
            return {"is_correct": False, "reasoning": "Parser failed", "critique": "", "raw_output": parsed_reasoning["content"]}, usage
        return result, usage

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
            - ``token_usage`` (``Dict[str, int] | None``): Aggregate token
              counts across all attempts —
              ``{"input_tokens": int, "thinking_tokens": int, "output_tokens": int}``
              — when :attr:`track_tokens` is ``True``, otherwise ``None``.
        """
        # Accumulates {"extraction": ..., "critique": ...} for every rejected
        # attempt so the extractor always has the full correction history.
        critique_history: List[Dict] = []
        history: List[Dict] = []

        # Running totals across all extractor + evaluator calls for this review.
        total_usage: Optional[Dict[str, int]] = (
            {"input_tokens": 0, "thinking_tokens": 0, "output_tokens": 0}
            if self.track_tokens else None
        )

        def _accumulate(usage: Optional[Dict[str, int]]) -> None:
            """Add per-call token counts to the review-level running totals."""
            if total_usage is not None and usage is not None:
                total_usage["input_tokens"] += usage["input_tokens"]
                total_usage["thinking_tokens"] += usage["thinking_tokens"]
                total_usage["output_tokens"] += usage["output_tokens"]

        logger.info(f"[ID: {item_id}] Processing: length={len(input_text)} chars")

        for attempt in range(max_retries):
            # On the first attempt critique_history is empty; on subsequent
            # attempts it contains every prior (extraction, critique) pair.
            extraction, ext_usage = await self.run_extractor(input_text, critique_history)
            evaluation, eval_usage = await self.run_evaluator(input_text, extraction)

            _accumulate(ext_usage)
            _accumulate(eval_usage)

            # Build per-attempt token breakdown for the history entry.
            attempt_usage: Optional[Dict[str, int]] = None
            if self.track_tokens:
                attempt_usage = {
                    k: (ext_usage or {}).get(k, 0) + (eval_usage or {}).get(k, 0)
                    for k in ("input_tokens", "thinking_tokens", "output_tokens")
                }

            # Record every attempt for post-hoc analysis.
            history_entry: Dict[str, Any] = {
                "attempt": attempt + 1,
                "extraction": extraction,
                "evaluation": evaluation,
            }
            if self.track_tokens:
                history_entry["token_usage"] = attempt_usage
            history.append(history_entry)

            if evaluation.get("is_correct") is True:
                logger.info(f"[ID: {item_id}] Attempt {attempt+1} successful")
                result = {
                    "final_output": extraction,
                    "status": "success",
                    "attempts": attempt + 1,
                    "history": history,
                }
                if self.track_tokens:
                    result["token_usage"] = total_usage
                    logger.debug(
                        f"[ID: {item_id}] Token usage: {total_usage}"
                    )
                return result

            # Append this attempt's output and critique to the shared history
            # so all future extraction prompts include the full feedback trail.
            current_critique = evaluation.get("critique", "Incorrect extraction.")
            critique_history.append({
                "extraction": extraction,
                "critique": current_critique,
            })
            logger.warning(f"[ID: {item_id}] Attempt {attempt+1} rejected. Critique: {current_critique}")

        logger.error(f"[ID: {item_id}] Max retries reached")
        result = {
            "final_output": extraction,  # Best attempt so far
            "status": "failed",
            "attempts": max_retries,
            "history": history,
        }
        if self.track_tokens:
            result["token_usage"] = total_usage
            logger.debug(f"[ID: {item_id}] Token usage: {total_usage}")
        return result

if __name__ == "__main__":
    pass