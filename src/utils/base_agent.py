import json
import re
import os
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from transformers import AutoTokenizer
from .const import LANGCODE2LANGNAME

logger = logging.getLogger(__name__)


class BaseABSASystem(ABC):
    """
    Abstract base class for all ABSA agent variants.

    Subclasses must implement `process_review`, and are responsible for
    initializing their own LLM engine and sampling parameters in `__init__`
    after calling `super().__init__`.

    Shared responsibilities handled here:
    - Loading and validating prompt files
    - Tokenizer initialization
    - Prompt formatting with chat template
    - JSON parsing helpers
    - Reasoning output parsing (Qwen3 <think> tags)
    """

    REQUIRED_PROMPT_FILES = {
        "extractor_system": "extractor_system.md",
        "extractor_user": "extractor_user.md",
        "evaluator_system": "evaluator_system.md",
        "evaluator_user": "evaluator_user.md",
        # "critique_template": "critique_template.md", # Not needed as a separate file since it's just a wrapper for the critique history.
    }

    def __init__(self, model_name: str, prompts_dir: str = "prompts"):
        logger.info(f"Loading prompts from directory: {prompts_dir}/")
        self.prompts = self._load_prompts_from_dir(prompts_dir)
        self.language = "Indonesian"

        logger.info(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def set_language(self, language: str) -> None:
        """Set the natural-language name used to render prompt placeholders."""
        self.language = language

    def set_language_from_code(self, language_code: str) -> None:
        """Resolve a dataset language code (e.g., 'indo') to display name."""
        self.language = LANGCODE2LANGNAME.get(language_code, language_code)

    def _render_prompt_language(self, prompt: str) -> str:
        """Replace the {language} placeholder without touching JSON braces."""
        return prompt.replace("{language}", self.language)

    def _load_prompts_from_dir(self, directory: str) -> Dict[str, str]:
        """Load all required `.md` prompt files from *directory* into a dict."""
        prompts = {}
        for key, filename in self.REQUIRED_PROMPT_FILES.items():
            path = os.path.join(directory, filename)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing required prompt file: {path}")
            with open(path, "r", encoding="utf-8") as f:
                prompts[key] = f.read().strip()
        return prompts

    def _format_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Apply the model's chat template to a system/user message pair."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

    def _parse_json(self, text: str) -> Any:
        """Strip markdown code fences and parse JSON; returns *None* on failure."""
        try:
            clean_text = re.sub(r"```json|```", "", text).strip()
            return json.loads(clean_text)
        except json.JSONDecodeError:
            return None

    def _parse_reasoning_output(self, output_text: str) -> Dict[str, Any]:
        """
        Separate the Qwen3 `<think>…</think>` reasoning block from the final answer.

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

    @abstractmethod
    def process_review(self, *args, **kwargs):
        """Run the full extractor-evaluator agent loop for a single review."""
        raise NotImplementedError

    def _format_critique_history(self, critique_history: List[Dict]) -> str:
        """Render accumulated (extraction, critique) pairs into a prompt block.

        Each entry in *critique_history* must contain:

        - ``extraction`` (``List[Dict]``): the JSON output from that attempt.
        - ``critique`` (``str``): the evaluator's feedback for that attempt.

        Returns a formatted multi-line string ready to be injected into the
        extractor user prompt, or an empty string when the list is empty.
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
