"""OpenRouter-backed async ABSA agent.

This module mirrors the logic of :mod:`agent_async` but replaces the local
vLLM ``AsyncLLMEngine`` with calls to the `OpenRouter API
<https://openrouter.ai>`_, which exposes an OpenAI-compatible REST endpoint.

Multiple reviews can still be processed concurrently via ``asyncio.gather``
since every I/O step is ``await``-ed through the ``openai`` async client.

Unlike the vLLM variant there is no two-step (think → answer) generation
with prefix caching.  A single ``/chat/completions`` call is made per agent
step.  Models that natively emit ``<think>…</think>`` tokens (e.g. Qwen3 via
the ``qwen/qwen3-*`` family on OpenRouter) will have those tokens stripped
before JSON parsing; the raw reasoning is preserved in the history.

Typical usage::

    import asyncio, os
    from src.utils.agent_openrouter import OpenRouterABSASystem

    os.environ["OPENROUTER_API_KEY"] = "sk-or-..."
    system = OpenRouterABSASystem(
        model_name="qwen/qwen3-8b",
        prompts_dir="prompts/exp1/",
    )
    result = asyncio.run(system.process_review(item_id="0", input_text="..."))
"""

import json
import os
import re
import asyncio
import logging
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Prompt template file names — must match the filenames in *prompts_dir*.
_REQUIRED_PROMPT_FILES: Dict[str, str] = {
    "extractor_system": "extractor_system.md",
    "extractor_user": "extractor_user.md",
    "evaluator_system": "evaluator_system.md",
    "evaluator_user": "evaluator_user.md",
}


class OpenRouterABSASystem:
    """Async ABSA agent that calls the OpenRouter REST API.

    Unlike :class:`~agent_async.AsyncABSASystem`, this class does **not**
    require a local GPU or vLLM installation.  Inference is delegated to
    OpenRouter via the OpenAI-compatible ``/chat/completions`` endpoint.

    The extraction → evaluation → retry loop is identical to the vLLM
    variant.  Thinking tokens (``<think>…</think>``) emitted by compatible
    models (e.g. Qwen3) are stripped before JSON parsing, but the raw
    reasoning is preserved in the per-attempt history.

    Args:
        model_name: OpenRouter model identifier, e.g. ``"qwen/qwen3-8b"``
            or ``"anthropic/claude-3.5-sonnet"``.
        prompts_dir: Directory containing the four Markdown prompt files
            (``extractor_system.md``, ``extractor_user.md``,
            ``evaluator_system.md``, ``evaluator_user.md``).
        api_key: OpenRouter API key.  Falls back to the
            ``OPENROUTER_API_KEY`` environment variable when *None*.
        site_url: Optional ``HTTP-Referer`` header forwarded to OpenRouter
            for attribution / rate-limiting purposes.
        site_name: Optional ``X-Title`` header sent with every request.
        extractor_temperature: Sampling temperature for extractor completions.
            Defaults to ``0.05`` (near-deterministic for structured output).
        evaluator_temperature: Sampling temperature for evaluator completions.
            Defaults to ``0.65`` (richer reasoning).
        max_tokens_extractor: Token budget for extractor completions.
        max_tokens_evaluator: Token budget for evaluator completions.
    """

    def __init__(
        self,
        model_name: str,
        prompts_dir: str = "prompts",
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
        extractor_temperature: float = 0.05,
        evaluator_temperature: float = 0.65,
        max_tokens_extractor: int = 512,
        max_tokens_evaluator: int = 2048,
    ) -> None:
        self.model_name = model_name
        self.extractor_temperature = extractor_temperature
        self.evaluator_temperature = evaluator_temperature
        self.max_tokens_extractor = max_tokens_extractor
        self.max_tokens_evaluator = max_tokens_evaluator

        # Resolve API key (constructor arg takes precedence over env var).
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No OpenRouter API key found.  Either pass api_key= to the "
                "constructor or set the OPENROUTER_API_KEY environment variable."
            )

        # Build optional attribution headers expected by OpenRouter.
        extra_headers: Dict[str, str] = {}
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if site_name:
            extra_headers["X-Title"] = site_name

        self.client = AsyncOpenAI(
            api_key=resolved_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers=extra_headers or None,
        )

        logger.info(f"Loading prompts from directory: {prompts_dir}/")
        self.prompts = self._load_prompts(prompts_dir)
        logger.info(f"OpenRouterABSASystem initialised (model={model_name})")

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _load_prompts(self, directory: str) -> Dict[str, str]:
        """Load all required Markdown prompt files from *directory*."""
        prompts: Dict[str, str] = {}
        for key, filename in _REQUIRED_PROMPT_FILES.items():
            path = os.path.join(directory, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required prompt file: {path}")
            with open(path, "r", encoding="utf-8") as fh:
                prompts[key] = fh.read().strip()
        return prompts

    @staticmethod
    def _build_messages(
        system_prompt: str, user_prompt: str
    ) -> List[Dict[str, str]]:
        """Return a standard OpenAI chat ``messages`` list."""
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    # ------------------------------------------------------------------
    # Output parsing helpers  (mirrors BaseABSASystem)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(text: str) -> Any:
        """Strip markdown code fences and parse JSON; returns *None* on failure."""
        try:
            clean = re.sub(r"```json|```", "", text).strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _parse_reasoning_output(output_text: str) -> Dict[str, Any]:
        """Separate any ``<think>…</think>`` block from the final answer.

        Returns::

            {"reasoning": str | None, "content": str}
        """
        match = re.search(r"<think>(.*?)</think>", output_text, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            content = output_text.split("</think>")[-1].strip()
        else:
            reasoning = None
            content = output_text.strip()
        return {"reasoning": reasoning, "content": content}

    def _format_critique_history(self, critique_history: List[Dict]) -> str:
        """Render accumulated (extraction, critique) pairs into a prompt block.

        Mirrors :meth:`BaseABSASystem._format_critique_history` exactly.
        Returns an empty string when *critique_history* is empty.
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

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    async def _generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Send a chat-completion request to OpenRouter and return the reply.

        Args:
            messages: Fully formatted ``[{"role": ..., "content": ...}]`` list.
            temperature: Sampling temperature for this call.
            max_tokens: Maximum number of completion tokens to generate.

        Returns:
            The model's reply as a plain string (may include ``<think>`` tags).
        """
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    # ------------------------------------------------------------------
    # Agent steps
    # ------------------------------------------------------------------

    async def run_extractor(
        self,
        input_text: str,
        critique_history: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Extract ABSA tuples from *input_text* using the extractor LLM.

        On the first attempt *critique_history* is ``None`` / empty.  On
        subsequent attempts it carries every prior (extraction, critique) pair
        so the model sees the full correction trail.

        Args:
            input_text: The raw review text to analyse.
            critique_history: Ordered list of prior attempt dicts, each with
                ``extraction`` (``List[Dict]``) and ``critique`` (``str``) keys.

        Returns:
            A list of ABSA tuple dicts, or a single-element error list when
            the model output cannot be parsed as a JSON array.
        """
        critique_text = self._format_critique_history(critique_history or [])
        user_prompt = self.prompts["extractor_user"].format(
            input_text=input_text,
            critique_instruction=critique_text,
        )
        messages = self._build_messages(self.prompts["extractor_system"], user_prompt)

        raw_output = await self._generate(
            messages,
            temperature=self.extractor_temperature,
            max_tokens=self.max_tokens_extractor,
        )
        parsed = self._parse_reasoning_output(raw_output)
        result = self._parse_json(parsed["content"])

        if not isinstance(result, list):
            logger.warning("Extractor returned non-list JSON or parse failure.")
            return [{"error": "Invalid JSON format", "raw_output": parsed["content"]}]
        return result

    async def run_evaluator(
        self, input_text: str, extraction: List[Dict]
    ) -> Dict:
        """Evaluate whether *extraction* is correct using the evaluator LLM.

        Args:
            input_text: The original review text that was analysed.
            extraction: ABSA tuples produced by :meth:`run_extractor`.

        Returns:
            A dict with at minimum ``is_correct`` (bool), ``reasoning`` (str),
            and ``critique`` (str) keys.  Falls back to a safe
            ``is_correct=False`` dict on parse errors.
        """
        user_prompt = self.prompts["evaluator_user"].format(
            input_text=input_text,
            extracted_json=json.dumps(extraction, indent=2, ensure_ascii=False),
        )
        messages = self._build_messages(self.prompts["evaluator_system"], user_prompt)

        raw_output = await self._generate(
            messages,
            temperature=self.evaluator_temperature,
            max_tokens=self.max_tokens_evaluator,
        )
        parsed = self._parse_reasoning_output(raw_output)
        result = self._parse_json(parsed["content"])

        if not isinstance(result, dict) or "is_correct" not in result:
            logger.warning("Evaluator returned unexpected structure or parse failure.")
            return {
                "is_correct": False,
                "reasoning": "Parser failed",
                "critique": "",
                "raw_output": parsed["content"],
            }
        return result

    async def process_review(
        self,
        item_id: str,
        input_text: str,
        max_retries: int = 3,
    ) -> Dict:
        """Run the full extract → evaluate → retry agent loop for one review.

        Mirrors :meth:`~agent_async.AsyncABSASystem.process_review` exactly:
        a rejected extraction appends both the output and the evaluator
        critique to a running ``critique_history`` list that is forwarded to
        every subsequent extraction attempt.

        Args:
            item_id: An opaque identifier for the review used in log messages.
            input_text: The raw review text to process.
            max_retries: Maximum number of extraction attempts before giving up.

        Returns:
            A dict with the following keys:

            - ``final_output`` (``List[Dict]``): ABSA tuples from the last attempt.
            - ``status`` (``str``): ``"success"`` if accepted within the retries,
              otherwise ``"failed"``.
            - ``attempts`` (``int``): Number of extraction attempts made.
            - ``history`` (``List[Dict]``): Per-attempt record for debugging.
        """
        critique_history: List[Dict] = []
        history: List[Dict] = []

        logger.info(f"[ID: {item_id}] Processing: length={len(input_text)} chars")

        for attempt in range(max_retries):
            extraction = await self.run_extractor(input_text, critique_history)
            evaluation = await self.run_evaluator(input_text, extraction)

            history.append({
                "attempt": attempt + 1,
                "extraction": extraction,
                "evaluation": evaluation,
            })

            if evaluation.get("is_correct") is True:
                logger.info(f"[ID: {item_id}] Attempt {attempt + 1} successful")
                return {
                    "final_output": extraction,
                    "status": "success",
                    "attempts": attempt + 1,
                    "history": history,
                }

            current_critique = evaluation.get("critique", "Incorrect extraction.")
            critique_history.append({
                "extraction": extraction,
                "critique": current_critique,
            })
            logger.warning(
                f"[ID: {item_id}] Attempt {attempt + 1} rejected. "
                f"Critique: {current_critique}"
            )

        logger.error(f"[ID: {item_id}] Max retries reached")
        return {
            "final_output": extraction,  # Best attempt so far
            "status": "failed",
            "attempts": max_retries,
            "history": history,
        }
