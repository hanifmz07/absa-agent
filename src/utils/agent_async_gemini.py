"""Async ABSA agent powered by the Gemini API.

This module mirrors the extract -> evaluate -> retry loop used by the vLLM
async agent, but sends generation requests to Gemini instead.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from google import genai
from google.genai import types
from .const import LANGCODE2LANGNAME

logger = logging.getLogger(__name__)


class AsyncGeminiABSASystem:
    """Async ABSA agent that uses Gemini for extraction and evaluation.

    The control flow is the same as the existing async implementation:
    extraction -> evaluation -> optional retries with full critique history.

    Args:
        model_name: Gemini model id, e.g. "gemini-3-flash-preview".
        prompts_dir: Directory containing extractor/evaluator prompt templates.
        track_tokens: Whether to include token usage stats in outputs.
        api_key: Optional API key. If omitted, GEMINI_API_KEY is used.
    """

    REQUIRED_PROMPT_FILES = {
        "extractor_system": "extractor_system.md",
        "extractor_user": "extractor_user.md",
        "evaluator_system": "evaluator_system.md",
        "evaluator_user": "evaluator_user.md",
    }

    def __init__(
        self,
        model_name: str = "gemini-3-flash-preview",
        prompts_dir: str = "prompts",
        track_tokens: bool = False,
        api_key: Optional[str] = None,
        max_api_retries: int = 5,
        retry_base_sleep_seconds: float = 2.0,
    ) -> None:
        self.model_name = model_name
        self.prompts = self._load_prompts_from_dir(prompts_dir)
        self.language = "Indonesian"
        self.track_tokens = track_tokens
        self.max_api_retries = max(1, max_api_retries)
        self.retry_base_sleep_seconds = max(0.1, retry_base_sleep_seconds)

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "Missing Gemini API key. Set GEMINI_API_KEY or pass api_key."
            )

        self.client = genai.Client(api_key=key)

        self.thinking_config = types.GenerateContentConfig(
            temperature=0.6,
            top_p=0.95,
            max_output_tokens=8192,
        )
        self.extractor_answer_config = types.GenerateContentConfig(
            temperature=0.05,
            top_p=0.1,
            max_output_tokens=512,
        )
        self.evaluator_answer_config = types.GenerateContentConfig(
            temperature=0.65,
            top_p=0.95,
            max_output_tokens=2048,
        )

    def _is_retryable_api_error(self, exc: Exception) -> bool:
        """Return True when the Gemini API error should be retried.

        Handles common throttling/quota wording and HTTP status patterns such
        as 429 / RESOURCE_EXHAUSTED / TOO_MANY_REQUESTS.
        """
        text = str(exc).lower()
        retry_markers = (
            "429",
            "resource_exhausted",
            "too_many_requests",
            "too many requests",
            "rate limit",
            "quota",
            "temporarily unavailable",
            "service unavailable",
        )
        if any(marker in text for marker in retry_markers):
            return True

        code = getattr(exc, "status_code", None)
        if code in {429, 500, 502, 503, 504}:
            return True

        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        return response_status in {429, 500, 502, 503, 504}

    def _call_generate_with_retry(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
    ) -> Any:
        """Call Gemini generate_content with retry for retryable API errors."""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_api_retries + 1):
            try:
                return self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
            except Exception as exc:  # noqa: BLE001 - external SDK exception types vary
                last_error = exc
                if not self._is_retryable_api_error(exc) or attempt == self.max_api_retries:
                    raise

                sleep_seconds = self.retry_base_sleep_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Gemini API retryable error (attempt %s/%s): %s. Sleeping %.1fs before retry.",
                    attempt,
                    self.max_api_retries,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Gemini API call failed without an exception object.")

    def _load_prompts_from_dir(self, directory: str) -> Dict[str, str]:
        prompts: Dict[str, str] = {}
        for key, filename in self.REQUIRED_PROMPT_FILES.items():
            path = os.path.join(directory, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required prompt file: {path}")
            with open(path, "r", encoding="utf-8") as f:
                prompts[key] = f.read().strip()
        return prompts

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        return (
            "SYSTEM INSTRUCTION:\n"
            f"{system_prompt}\n\n"
            "USER INPUT:\n"
            f"{user_prompt}"
        )

    def set_language(self, language: str) -> None:
        self.language = language

    def set_language_from_code(self, language_code: str) -> None:
        self.language = LANGCODE2LANGNAME.get(language_code, language_code)

    def _render_prompt_language(self, prompt: str) -> str:
        return prompt.replace("{language}", self.language)

    def _parse_json(self, text: str) -> Any:
        try:
            clean_text = text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except json.JSONDecodeError:
            return None

    def _parse_reasoning_output(self, output_text: str) -> Dict[str, Any]:
        start = output_text.find("<think>")
        end = output_text.find("</think>")

        if start != -1 and end != -1 and end > start:
            reasoning = output_text[start + len("<think>") : end].strip()
            content = output_text[end + len("</think>") :].strip()
            return {"reasoning": reasoning, "content": content}

        # Gemini sometimes emits alternate tags in free-text mode.
        alt_start = output_text.find("<thinking>")
        alt_end = output_text.find("</thinking>")
        if alt_start != -1 and alt_end != -1 and alt_end > alt_start:
            reasoning = output_text[alt_start + len("<thinking>") : alt_end].strip()
            content = output_text[alt_end + len("</thinking>") :].strip()
            return {"reasoning": reasoning, "content": content}

        return {"reasoning": None, "content": output_text.strip()}

    def _format_critique_history(self, critique_history: List[Dict]) -> str:
        if not critique_history:
            return ""

        lines = [
            "IMPORTANT FEEDBACK: Your previous extractions were rejected.",
            "Review ALL attempts and critiques below carefully before producing your new extraction.\n",
        ]

        for i, entry in enumerate(critique_history, start=1):
            extraction_str = json.dumps(entry["extraction"], indent=2, ensure_ascii=False)
            lines.append(f"--- Attempt {i} ---")
            lines.append("Your extraction:")
            lines.append(extraction_str)
            lines.append(f"Critique: \"{entry['critique']}\"\n")

        lines.append("Make sure your new extraction addresses every critique listed above.")
        return "\n".join(lines)

    def _extract_text_parts(self, response: Any) -> Tuple[List[str], List[str], List[str]]:
        """Return (thought_parts, answer_parts, all_parts) from a Gemini response.

        Gemini thinking models can return thought parts separately from answer
        parts. This parser keeps both channels so downstream code can preserve
        reasoning while still parsing the final JSON answer robustly.
        """
        thought_parts: List[str] = []
        answer_parts: List[str] = []
        all_parts: List[str] = []

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if not content:
                continue
            for part in getattr(content, "parts", []) or []:
                text = getattr(part, "text", None)
                if not text:
                    continue
                all_parts.append(text)
                if bool(getattr(part, "thought", False)):
                    thought_parts.append(text)
                else:
                    answer_parts.append(text)

        # Fallback for SDK responses that expose only `.text`.
        if not all_parts:
            fallback_text = getattr(response, "text", None)
            if fallback_text:
                all_parts = [fallback_text]
                answer_parts = [fallback_text]

        return thought_parts, answer_parts, all_parts

    def _normalize_reasoning_and_answer(self, response: Any) -> Tuple[Optional[str], str, str]:
        """Normalize Gemini output into (reasoning, answer, combined_text).

        `combined_text` is intentionally formatted as `<think>...</think>\n...`
        whenever reasoning exists, so existing parsing and logging logic remains
        compatible with the vLLM-based implementation.
        """
        thought_parts, answer_parts, all_parts = self._extract_text_parts(response)

        reasoning = "\n".join(part.strip() for part in thought_parts if part.strip()).strip()
        answer = "\n".join(part.strip() for part in answer_parts if part.strip()).strip()
        combined = "\n".join(part.strip() for part in all_parts if part.strip()).strip()

        # If Gemini did not provide explicit thought parts, try XML-tag formats.
        if not reasoning and combined:
            parsed = self._parse_reasoning_output(combined)
            reasoning = parsed.get("reasoning") or ""
            answer = parsed.get("content") or answer

        if not answer and combined:
            answer = combined

        if reasoning:
            normalized = f"<think>{reasoning}</think>\n{answer}".strip()
            return reasoning, answer, normalized

        return None, answer, answer

    def _extract_usage(
        self,
        response: Any,
        thinking_tokens: Optional[int] = None,
    ) -> Dict[str, int]:
        usage_meta = getattr(response, "usage_metadata", None)
        prompt_tokens = int(getattr(usage_meta, "prompt_token_count", 0) or 0)
        output_tokens = int(getattr(usage_meta, "candidates_token_count", 0) or 0)

        return {
            "input_tokens": prompt_tokens,
            "thinking_tokens": int(thinking_tokens or 0),
            "output_tokens": output_tokens,
        }

    def _merge_configs(
        self,
        base_config: types.GenerateContentConfig,
        system_instruction: str,
    ) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            temperature=base_config.temperature,
            top_p=base_config.top_p,
            max_output_tokens=base_config.max_output_tokens,
            stop_sequences=base_config.stop_sequences,
            system_instruction=system_instruction,
        )

    def _generate_sync(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        response = self._call_generate_with_retry(prompt, config)
        _, _, text = self._normalize_reasoning_and_answer(response)

        usage: Optional[Dict[str, int]] = None
        if self.track_tokens:
            usage = self._extract_usage(response)

        return text, usage

    async def _generate(
        self,
        prompt: str,
        config: types.GenerateContentConfig,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        return await asyncio.to_thread(self._generate_sync, prompt, config)

    def _generate_two_step_sync(
        self,
        prompt: str,
        system_instruction: str,
        answer_config: types.GenerateContentConfig,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        # Phase 1: thinking.
        thinking_instruction = (
            f"{system_instruction}\n\n"
            "First, think step-by-step inside <think>...</think>. "
            "End your thinking with the exact closing tag </think>."
        )
        think_config = self._merge_configs(self.thinking_config, thinking_instruction)
        thought_response = self._call_generate_with_retry(prompt, think_config)
        thought_text, phase1_answer, combined_phase1 = self._normalize_reasoning_and_answer(
            thought_response
        )

        # Gemini may place reasoning in either thought parts or plain text.
        generated_thought = thought_text or phase1_answer or combined_phase1
        if not generated_thought:
            generated_thought = "No explicit reasoning generated."

        # Phase 2: answer.
        answer_instruction = (
            f"{system_instruction}\n\n"
            "You already completed the reasoning phase. Now provide only the final answer."
        )
        merged_answer_config = self._merge_configs(answer_config, answer_instruction)
        answer_prompt = (
            f"{prompt}\n\n"
            "Reasoning from prior phase:\n"
            f"<think>{generated_thought}</think>\n"
        )
        answer_response = self._call_generate_with_retry(
            answer_prompt,
            merged_answer_config,
        )
        _, generated_answer, normalized_answer = self._normalize_reasoning_and_answer(
            answer_response
        )

        usage: Optional[Dict[str, int]] = None
        if self.track_tokens:
            thought_usage = self._extract_usage(thought_response)
            answer_usage = self._extract_usage(answer_response)
            usage = {
                "input_tokens": thought_usage["input_tokens"],
                "thinking_tokens": thought_usage["output_tokens"],
                "output_tokens": answer_usage["output_tokens"],
            }

        combined_text = f"<think>{generated_thought}</think>\n{generated_answer}".strip()
        if not generated_answer and normalized_answer:
            combined_text = f"<think>{generated_thought}</think>\n{normalized_answer}".strip()

        return combined_text, usage

    async def _generate_two_step(
        self,
        prompt: str,
        system_instruction: str,
        answer_config: types.GenerateContentConfig,
    ) -> Tuple[str, Optional[Dict[str, int]]]:
        return await asyncio.to_thread(
            self._generate_two_step_sync,
            prompt,
            system_instruction,
            answer_config,
        )

    async def run_extractor(
        self,
        input_text: str,
        critique_history: Optional[List[Dict]] = None,
    ) -> Tuple[List[Dict], Optional[Dict[str, int]]]:
        critique_text = self._format_critique_history(critique_history or [])

        user_prompt_filled = self.prompts["extractor_user"].format(
            input_text=input_text,
            critique_instruction=critique_text,
        )
        extractor_system = self._render_prompt_language(self.prompts["extractor_system"])
        full_prompt = self._build_prompt(extractor_system, user_prompt_filled)

        raw_output, usage = await self._generate_two_step(
            full_prompt,
            extractor_system,
            self.extractor_answer_config,
        )
        parsed_reasoning = self._parse_reasoning_output(raw_output)
        result = self._parse_json(parsed_reasoning["content"])

        if not isinstance(result, list):
            return [
                {
                    "error": "Invalid JSON format",
                    "raw_output": parsed_reasoning["content"],
                }
            ], usage
        return result, usage

    async def run_evaluator(
        self,
        input_text: str,
        extraction: List[Dict],
    ) -> Tuple[Dict, Optional[Dict[str, int]]]:
        user_prompt_filled = self.prompts["evaluator_user"].format(
            input_text=input_text,
            extracted_json=json.dumps(extraction, indent=2),
        )
        evaluator_system = self._render_prompt_language(self.prompts["evaluator_system"])
        full_prompt = self._build_prompt(evaluator_system, user_prompt_filled)

        raw_output, usage = await self._generate_two_step(
            full_prompt,
            evaluator_system,
            self.evaluator_answer_config,
        )
        parsed_reasoning = self._parse_reasoning_output(raw_output)
        result = self._parse_json(parsed_reasoning["content"])

        if not isinstance(result, dict) or "is_correct" not in result:
            return {
                "is_correct": False,
                "reasoning": "Parser failed",
                "critique": "",
                "raw_output": parsed_reasoning["content"],
            }, usage
        return result, usage

    async def process_review(
        self,
        item_id: str,
        input_text: str,
        max_retries: int = 3,
    ) -> Dict:
        critique_history: List[Dict] = []
        history: List[Dict] = []

        total_usage: Optional[Dict[str, int]] = (
            {"input_tokens": 0, "thinking_tokens": 0, "output_tokens": 0}
            if self.track_tokens
            else None
        )

        def _accumulate(usage: Optional[Dict[str, int]]) -> None:
            if total_usage is not None and usage is not None:
                total_usage["input_tokens"] += usage["input_tokens"]
                total_usage["thinking_tokens"] += usage["thinking_tokens"]
                total_usage["output_tokens"] += usage["output_tokens"]

        logger.info("[ID: %s] Processing: length=%s chars", item_id, len(input_text))

        extraction: List[Dict] = []
        for attempt in range(max_retries):
            extraction, ext_usage = await self.run_extractor(input_text, critique_history)
            evaluation, eval_usage = await self.run_evaluator(input_text, extraction)

            _accumulate(ext_usage)
            _accumulate(eval_usage)

            attempt_usage: Optional[Dict[str, int]] = None
            if self.track_tokens:
                attempt_usage = {
                    k: (ext_usage or {}).get(k, 0) + (eval_usage or {}).get(k, 0)
                    for k in ("input_tokens", "thinking_tokens", "output_tokens")
                }

            history_entry: Dict[str, Any] = {
                "attempt": attempt + 1,
                "extraction": extraction,
                "evaluation": evaluation,
            }
            if self.track_tokens:
                history_entry["token_usage"] = attempt_usage
            history.append(history_entry)

            if evaluation.get("is_correct") is True:
                result = {
                    "final_output": extraction,
                    "status": "success",
                    "attempts": attempt + 1,
                    "history": history,
                }
                if self.track_tokens:
                    result["token_usage"] = total_usage
                return result

            current_critique = evaluation.get("critique", "Incorrect extraction.")
            critique_history.append(
                {
                    "extraction": extraction,
                    "critique": current_critique,
                }
            )
            logger.warning(
                "[ID: %s] Attempt %s rejected. Critique: %s",
                item_id,
                attempt + 1,
                current_critique,
            )

        result = {
            "final_output": extraction,
            "status": "failed",
            "attempts": max_retries,
            "history": history,
        }
        if self.track_tokens:
            result["token_usage"] = total_usage
        return result


if __name__ == "__main__":
    pass
