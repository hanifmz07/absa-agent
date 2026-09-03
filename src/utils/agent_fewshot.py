"""Single-pass few-shot ABSA agent powered by a vLLM AsyncLLMEngine.

Unlike `agent_async.py`'s `AsyncABSASystem`, this agent has no evaluator or
retry loop — each review gets exactly one extractor call, optionally
preceded by a `{fewshot_examples}` block built by the caller (see
`src/utils/fewshot_pool.py` and `src/main/run_agent_fewshot.py`).
"""

import uuid
import logging
from typing import List, Dict, Any, Optional, Tuple

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams

from .base_agent import BaseABSASystem

logger = logging.getLogger(__name__)


class FewShotABSASystem(BaseABSASystem):
    """Single-pass ABSA agent: one extractor call per review, no evaluator.

    Reuses the same two-phase (think -> answer) generation strategy as
    `AsyncABSASystem` for comparability with `exp1`/`exp2_staged` results.

    Args:
        model_name: HuggingFace model ID or local path to the LLM weights.
        prompts_dir: Directory containing the Markdown prompt templates.
            Must contain `extractor_system.md`/`extractor_user.md` (and,
            per the repo's prompt-set convention, `evaluator_system.md`/
            `evaluator_user.md`, even though they're unused here).
        track_tokens: When `True`, records input/thinking/output token
            counts for the single extractor call under `"token_usage"`.
    """

    def __init__(
        self,
        model_name: str,
        prompts_dir: str = "prompts",
        seed: int = 42,
        track_tokens: bool = False,
        gpu_memory_utilization: float = 0.55,
        max_model_len: int = 16384,
    ):
        super().__init__(model_name, prompts_dir)

        logger.info(f"Initializing FewShot vLLM Engine: {model_name}")

        self.track_tokens: bool = track_tokens

        engine_args = AsyncEngineArgs(
            model=model_name,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=1,
            seed=seed,
            enable_prefix_caching=True,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)

        self.thinking_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            max_tokens=8192,
            stop=["</think>"],
            include_stop_str_in_output=True,
        )

        self.extractor_answer_params = SamplingParams(
            temperature=0.05,
            top_p=0.1,
            max_tokens=512,
        )

    async def _generate_two_step(
        self, prompt: str, answer_params: SamplingParams
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        """Run two-phase (think -> answer) generation for a single prompt.

        Identical strategy to `AsyncABSASystem._generate_two_step`: phase 1
        reasons freely until `</think>`, phase 2 conditions on the thought
        block to produce the final structured answer, reusing the shared
        prompt prefix via vLLM's prefix caching.
        """
        thought_request_id = str(uuid.uuid4())
        thought_generator = self.engine.generate(prompt, self.thinking_params, thought_request_id)

        thought_output = None
        async for output in thought_generator:
            thought_output = output
        generated_thought = thought_output.outputs[0].text

        answer_prompt = prompt + generated_thought
        if "</think>" not in generated_thought:
            logger.warning("Thinking phase hit max_tokens before </think>. Force-closing.")
            answer_prompt += "\n</think>\n"

        answer_request_id = str(uuid.uuid4())
        answer_generator = self.engine.generate(answer_prompt, answer_params, answer_request_id)

        final_answer_output = None
        async for output in answer_generator:
            final_answer_output = output
        generated_answer = final_answer_output.outputs[0].text

        usage: Optional[Dict[str, int]] = None
        if self.track_tokens:
            usage = {
                "input_tokens": len(thought_output.prompt_token_ids),
                "thinking_tokens": len(thought_output.outputs[0].token_ids),
                "output_tokens": len(final_answer_output.outputs[0].token_ids),
            }

        return generated_thought + generated_answer, usage

    async def run_extractor(
        self,
        input_text: str,
        fewshot_block: str = "",
    ) -> Tuple[List[Dict], Optional[Dict[str, int]]]:
        """Extract ABSA tuples from a review, with optional few-shot examples.

        `critique_instruction=""` is passed alongside `fewshot_examples` so
        this single call works unmodified against both the new fewshot
        prompt templates (which reference `{fewshot_examples}`) and
        `prompts/exp1/extractor_user.md` (which references
        `{critique_instruction}` instead) — `str.format()` silently ignores
        kwargs that don't match a placeholder in the template. This is what
        lets the static-curated condition reuse `prompts/exp1/` unchanged.

        Args:
            input_text: The raw review text to analyse.
            fewshot_block: Pre-formatted few-shot example block (see
                `fewshot_pool.format_fewshot_block`), or `""` for zero-shot
                / static-curated (whose examples are baked into the system
                prompt instead).

        Returns:
            Same `(extraction, usage)` shape as `AsyncABSASystem.run_extractor`.
        """
        user_prompt_filled = self.prompts["extractor_user"].format(
            input_text=input_text,
            critique_instruction="",
            fewshot_examples=fewshot_block,
        )

        extractor_system = self._render_prompt_language(self.prompts["extractor_system"])
        full_prompt = self._format_prompt(extractor_system, user_prompt_filled)

        raw_output, usage = await self._generate_two_step(full_prompt, self.extractor_answer_params)
        parsed_reasoning = self._parse_reasoning_output(raw_output)
        result = self._parse_json(parsed_reasoning["content"])

        if not isinstance(result, list):
            return [{"error": "Invalid JSON format", "raw_output": parsed_reasoning["content"]}], usage
        return result, usage

    async def process_review(self, item_id: str, input_text: str, fewshot_block: str = "") -> Dict:
        """Run a single extractor call for one review — no evaluator, no retry.

        Args:
            item_id: An opaque identifier for the review used in log messages.
            input_text: The raw review text to process.
            fewshot_block: Pre-formatted few-shot example block, or `""`.

        Returns:
            A dict with the following keys:

            - `final_output` (`List[Dict]`): ABSA tuples from the extractor.
            - `status` (`str`): `"success"` if the extractor produced
              parseable JSON, `"failed"` otherwise. Unlike `exp1`, this is
              not an evaluator's correctness verdict — there is no
              evaluator in this pipeline — it only reflects whether the
              output parsed.
            - `attempts` (`int`): always `1`.
            - `history` (`List[Dict]`): single-entry list with the
              extraction (and `token_usage` if tracked), kept for schema
              symmetry with `AsyncABSASystem.process_review`.
            - `token_usage` (`Dict[str, int] | None`): present only when
              `track_tokens` is `True`.
        """
        logger.info(f"[ID: {item_id}] Processing: length={len(input_text)} chars")

        extraction, usage = await self.run_extractor(input_text, fewshot_block)

        parse_ok = isinstance(extraction, list) and not (
            len(extraction) == 1 and isinstance(extraction[0], dict) and "error" in extraction[0]
        )

        history_entry: Dict[str, Any] = {"attempt": 1, "extraction": extraction}
        if self.track_tokens:
            history_entry["token_usage"] = usage

        result = {
            "final_output": extraction,
            "status": "success" if parse_ok else "failed",
            "attempts": 1,
            "history": [history_entry],
        }
        if self.track_tokens:
            result["token_usage"] = usage
        return result


if __name__ == "__main__":
    pass
