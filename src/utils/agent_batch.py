"""Batch ABSA agent powered by the synchronous vLLM ``LLM`` engine.

This module implements an optimised offline-batch agent for Aspect-Based
Sentiment Analysis (ABSA).  An entire dataset is processed round by round;
all items in each retry round are sent to the model in a single batched call,
maximising GPU throughput compared to the sequential alternative.

Typical usage::

    system = BatchABSASystem(model_name="path/to/model")
    results = system.process_dataset(dataset, max_retries=3)
"""

import json
import logging
from typing import List, Dict, Any
import torch
import os

from vllm import LLM, SamplingParams

from .base_agent import BaseABSASystem

logger = logging.getLogger(__name__)

# Turn off multiprocessing to make the scheduling deterministic, or
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"


class BatchABSASystem(BaseABSASystem):
    """Batch ABSA agent that processes datasets in optimised offline batches.

    Uses the synchronous vLLM ``LLM`` engine so every item in each retry round
    is dispatched to the model together in a single batched call, maximising
    GPU throughput compared to item-by-item sequential processing.

    The agent uses a **two-phase generation** strategy applied over the whole
    batch at once:

    1. *Thinking phase* – all prompts are fed to the model simultaneously;
       each sequence reasons freely at moderate temperature and stops at the
       ``</think>`` sentinel.
    2. *Answering phase* – the completed thought blocks are prepended to their
       respective prompts and the whole batch is forwarded again to produce
       the final structured answers.

    Args:
        model_name: HuggingFace model ID or local path to the LLM weights.
        prompts_dir: Directory containing the Markdown prompt templates.
    """

    def __init__(self, model_name: str, prompts_dir: str = "prompts", seed: int = 42) -> None:
        """Initialise the batch agent and load the vLLM synchronous engine.

        Calls the parent :class:`BaseABSASystem` constructor to load prompt
        templates before instantiating the ``LLM`` engine and the three sets
        of :class:`~vllm.SamplingParams` used across the two generation phases.

        Args:
            model_name: HuggingFace model ID or local path to the LLM weights.
            prompts_dir: Directory containing the Markdown prompt templates.
                Defaults to ``"prompts"``.
            seed: Global random seed passed to the vLLM engine for
                reproducible sampling.  Defaults to ``42``.
        """
        super().__init__(model_name, prompts_dir)

        logger.info(f"Initializing Synchronous vLLM Engine: {model_name}")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=0.7,
            tensor_parallel_size=1,
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

    def _batch_generate_two_step(
        self, prompts: List[str], answer_params: SamplingParams
    ) -> List[str]:
        """Run batched two-phase (think → answer) generation over a list of prompts.

        Both phases forward the entire prompt list as a single batch, making
        full use of the offline ``LLM`` engine's continuous batching.

        Phase 1 lets each model instance reason freely at moderate temperature
        until it emits ``</think>``.  Phase 2 conditions on the full thought
        block and produces the final structured answer.  Any thought that
        exhausts ``max_tokens`` before emitting ``</think>`` has the closing
        tag appended manually so phase 2 still starts from a well-formed prefix.

        Args:
            prompts: Fully formatted prompt strings (system + user turns), one
                per dataset item in the current batch.
            answer_params: :class:`~vllm.SamplingParams` applied only to the
                answer phase.  Use :attr:`extractor_answer_params` or
                :attr:`evaluator_answer_params` as appropriate.

        Returns:
            A list of strings, one per input prompt, each being the
            concatenation of the raw thought text and the answer text, i.e.
            ``<thought_block></think>\n<answer_block>``.
        """
        # STEP 1: THINKING PHASE (batch)
        thinking_outputs = self.llm.generate(prompts, self.thinking_params, use_tqdm=True)
        thoughts = [out.outputs[0].text for out in thinking_outputs]

        # Build answer prompts; force-close truncated thoughts.
        answer_prompts = []
        for prompt, thought in zip(prompts, thoughts):
            answer_prompt = prompt + thought
            if "</think>" not in thought:
                logger.warning("Thinking phase hit max_tokens before </think>. Force-closing.")
                answer_prompt += "\n</think>\n"
            answer_prompts.append(answer_prompt)

        # STEP 2: ANSWERING PHASE (batch)
        answer_outputs = self.llm.generate(answer_prompts, answer_params, use_tqdm=True)
        answers = [out.outputs[0].text for out in answer_outputs]

        return [thought + answer for thought, answer in zip(thoughts, answers)]

    def process_review(self, input_text: str, max_retries: int = 3) -> Dict:
        """Process a single review using the batch pipeline.

        A thin convenience wrapper that packages the single item into a
        one-element list, delegates to :meth:`process_dataset`, and unwraps
        the result.  Useful for ad-hoc inference without constructing a
        dataset object.

        Args:
            input_text: The raw review text to analyse.
            max_retries: Maximum number of extraction attempts before giving
                up.  Defaults to ``3``.

        Returns:
            A single result dict identical in structure to the dicts returned
            by :meth:`process_dataset`.  See that method for key descriptions.
        """
        results = self.process_dataset([{"id": "single", "text": input_text}], max_retries=max_retries)
        return results[0]

    def process_dataset(self, dataset: List[Dict], max_retries: int = 3) -> List[Dict]:
        """Process a list of reviews in optimised offline batches.

        Iterates over up to ``max_retries`` rounds.  In each round *all*
        currently active items (those not yet accepted by the evaluator) are
        processed together:

        1. **Extraction batch** – one extractor call for all active items,
           with prior (extraction, critique) pairs injected into each prompt.
        2. **Evaluation batch** – one evaluator call for all fresh extractions.
        3. **Routing** – accepted items are moved to ``completed_items``;
           rejected items have their critique appended to ``critique_history``
           for the next round.

        Items still active after all rounds are marked as ``"failed"`` and
        appended to the output with the best extraction seen so far.

        Args:
            dataset: List of dicts, each with at minimum:

                - ``id``: a unique identifier for the review.
                - ``text``: the raw review text to analyse.

            max_retries: Maximum number of extraction attempts per item.
                Defaults to ``3``.

        Returns:
            A list of result dicts, one per input item, each containing:

            - ``id``: the original item identifier.
            - ``text``: the original review text.
            - ``final_output`` (``List[Dict]``): ABSA tuples from the last
              extraction attempt.
            - ``status`` (``str``): ``"success"`` if the evaluator accepted
              the extraction within the allowed attempts, else ``"failed"``.
            - ``attempts`` (``int``): number of extraction rounds attempted.
            - ``history`` (``List[Dict]``): per-attempt record of each
              extraction and its evaluation, useful for debugging.
            - ``critique_history`` (``List[Dict]``): accumulated (extraction,
              critique) pairs injected into extractor prompts on retries.
        """

        # Keyed by item id for O(1) lookup when routing results after each round.
        # ``critique_history`` accumulates all rejected (extraction, critique)
        # pairs so every re-attempt sees the complete correction trail.
        active_items = {
            item['id']: {
                "id": item['id'],
                "text": item['text'],
                "critique_history": [],   # List[{"extraction": ..., "critique": ...}]
                "attempts": 0,
                "history": [],
                "final_output": None,
                "status": "pending",
            }
            for item in dataset
        }

        completed_items = []

        # Each iteration of the while-loop represents one full retry round
        # over all remaining (not-yet-accepted) items.
        while active_items:
            current_attempt = list(active_items.values())[0]["attempts"] + 1
            if current_attempt > max_retries:
                break

            logger.info(f"Starting batch loop {current_attempt}/{max_retries} | Processing {len(active_items)} items")
            active_ids = list(active_items.keys())

            # -----------------------------------------------------------------
            # PHASE 1: BATCH EXTRACTION (two-step)
            # -----------------------------------------------------------------
            ext_prompts = []
            for uid in active_ids:
                critique_text = self._format_critique_history(
                    active_items[uid]["critique_history"]
                )
                user_prompt = self.prompts["extractor_user"].format(
                    input_text=active_items[uid]["text"],
                    critique_instruction=critique_text,
                )
                ext_prompts.append(self._format_prompt(self.prompts["extractor_system"], user_prompt))

            logger.info("Running extractor inference (two-step)...")
            raw_ext_texts = self._batch_generate_two_step(ext_prompts, self.extractor_answer_params)

            parsed_extractions = []
            for text in raw_ext_texts:
                parsed = self._parse_reasoning_output(text)
                json_data = self._parse_json(parsed["content"])
                if not isinstance(json_data, list):
                    json_data = [{"error": "Invalid JSON format", "raw_output": parsed["content"]}]
                parsed_extractions.append(json_data)

            # -----------------------------------------------------------------
            # PHASE 2: BATCH EVALUATION (two-step)
            # -----------------------------------------------------------------
            eval_prompts = []
            for uid, ext_json in zip(active_ids, parsed_extractions):
                user_prompt = self.prompts["evaluator_user"].format(
                    input_text=active_items[uid]["text"],
                    extracted_json=json.dumps(ext_json, indent=2),
                )
                eval_prompts.append(self._format_prompt(self.prompts["evaluator_system"], user_prompt))

            logger.info("Running evaluator inference (two-step)...")
            raw_eval_texts = self._batch_generate_two_step(eval_prompts, self.evaluator_answer_params)

            parsed_evals = []
            for text in raw_eval_texts:
                parsed = self._parse_reasoning_output(text)
                json_data = self._parse_json(parsed["content"])
                if not isinstance(json_data, dict) or "is_correct" not in json_data:
                    json_data = {"is_correct": False, "critique": "Evaluator parser failed.", "raw_output": parsed["content"]}
                parsed_evals.append(json_data)

            # -----------------------------------------------------------------
            # PHASE 3: FILTER AND ROUTE
            # -----------------------------------------------------------------
            for uid, ext_json, eval_json in zip(active_ids, parsed_extractions, parsed_evals):
                item = active_items[uid]
                item["attempts"] += 1
                item["history"].append({
                    "attempt": item["attempts"],
                    "extraction": ext_json,
                    "evaluation": eval_json,
                })

                if eval_json.get("is_correct") is True:
                    item["status"] = "success"
                    item["final_output"] = ext_json
                    completed_items.append(item)
                    del active_items[uid]
                else:
                    current_critique = eval_json.get("critique", "Incorrect extraction format or semantics.")
                    if item["attempts"] >= max_retries:
                        item["status"] = "failed"
                        item["final_output"] = ext_json
                        completed_items.append(item)
                        del active_items[uid]
                    else:
                        # Append to history so the next attempt sees the full trail.
                        item["critique_history"].append({
                            "extraction": ext_json,
                            "critique": current_critique,
                        })

        # Any items still active here were left by the while-break (i.e. the
        # next attempt number would exceed max_retries before entering the loop
        # body).  Mark them failed and use the last recorded extraction.
        for uid, item in active_items.items():
            item["status"] = "failed"
            item["final_output"] = item["history"][-1]["extraction"] if item["history"] else []
            completed_items.append(item)

        logger.info(f"Batch processing complete | Total processed: {len(completed_items)}")
        return completed_items


# Example Usage Wrapper
if __name__ == "__main__":
    system = BatchABSASystem("Qwen/Qwen3-0.6B", prompts_dir="prompts")

    # Mock dataset
    test_cases = [
        {"id": 1, "text": "Kamarnya sangat luas dan bersih."},
        {"id": 2, "text": "Harganya mahal tapi fasilitas jelek."},
        {"id": 3, "text": "Wifi lambat, AC panas, staf judes!"}
    ]

    results = system.process_dataset(test_cases, max_retries=3)

    # Save results
    with open('results/batch_absa_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)