# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A prompt-based Aspect-Based Sentiment Analysis (ABSA) experimentation harness. It runs an LLM extractor→evaluator→retry loop over hotel-review datasets in multiple languages (Indonesian and regional languages) to extract `(aspect, opinion, sentiment)` triplets, then scores the results with several evaluation methods. There is no application code being shipped — this is a research/experimentation codebase; changes are almost always to prompts, runner scripts, or eval logic, not a deployed service.

## Environment

Uses `uv` with a local `.venv`, Python >=3.12.

```bash
uv sync
source .venv/bin/activate
```

API keys go in `.env` (see `.env.example` / README): `GEMINI_API_KEY` for Gemini runners, `OPENROUTER_API_KEY` for OpenRouter runners.

There is no test suite, linter, or CI config in this repo — don't invent commands for these.

## Running experiments

Everything is invoked either via `scripts/*.sh` wrappers (preferred, they loop over seeds/retry counts) or directly as `python -m src.main.<module>`. Common flags across runners: `--test_case_path`, `--max_retries`, `--seed`, `--prompt_set` (see "Prompt sets" below), `--track_tokens`.

Inference (produces `inference_results.json`):
```bash
./scripts/inference.sh indo                    # vLLM async, one language
./scripts/inference_loop.sh indo sun min        # vLLM async, multiple languages
./scripts/inference_gemini.sh                   # Gemini async
./scripts/inference_gemini_sequential.sh        # Gemini sequential
```
or directly, e.g. `python -m src.main.run_agent_async --model_path Qwen/Qwen3-8B --test_case_path dataset/hotel_reviews/indo/mvp_aos/test.json --max_retries 10 --prompt_set exp1 --seed 42`.

Evaluation (three independent, comparable scoring methods — all read only `final_extraction`/`target_text` from `inference_results.json`, so they're agnostic to how the extraction was produced):
```bash
bash scripts/eval_all_exact_match.sh results       # strict field-for-field match
bash scripts/eval_all_instruct_absa.sh results     # substring-tolerant string match
bash scripts/eval_all_semantic.sh results Qwen/Qwen3-Embedding-0.6B   # embedding cosine-similarity fallback for near-misses
```
Single-config variants (`scripts/eval_exact_match.sh`, `eval_instruct_absa.sh`, `eval_semantic.sh`) take positional args: `results_dir model dataset_type lang dataset_folder prompt_set max_retries_N seed_N run_id`. There's also a legacy/agent report path via `src/main/eval_agent.py` (`--runner_type` selects which runner's output shape to read: `async`, `sequential`, `batch`, `async_0.6B`, `sequential_gemini`).

Full command reference and per-evaluator explanation is in [README.md](README.md) — read it before adding a new runner or evaluator rather than re-deriving flag names.

## Architecture

**Prompt sets are pure directory conventions, not code.** `prompts/<name>/` (e.g. `prompts/exp1/`, `prompts/exp2_staged/`) must contain exactly `extractor_system.md`, `extractor_user.md`, `evaluator_system.md`, `evaluator_user.md`. Adding a new experiment variant almost always means adding a new `prompts/<name>/` directory and passing `--prompt_set <name>` — it does **not** require touching Python, as long as the pipeline shape (extractor→evaluator→retry, same final JSON schema) stays the same. Placeholders: `{language}` is substituted via plain `.replace` (not `.format`, to avoid colliding with JSON braces in the prompt text) in system prompts; `{input_text}` / `{critique_instruction}` (extractor) and `{input_text}` / `{extracted_json}` (evaluator) are substituted via `.format()` in user prompts. `{language}` is resolved automatically from the dataset path (`dataset/hotel_reviews/<langcode>/...` → display name via `LANGCODE2LANGNAME` in `src/utils/const.py`), so changing the dataset path changes prompt wording without any other change.

**`src/utils/base_agent.py` (`BaseABSASystem`) is the shared core** all runner classes build on: loads/validates the 4 required prompt files, applies the model's chat template, and parses model output — `_parse_reasoning_output` strips a Qwen3 `<think>...</think>` block before `_parse_json` (which strips ` ```json ` fences and `json.loads`s the rest). The extractor must emit a JSON list `[{"aspect", "opinion", "sentiment"}]`; the evaluator a JSON dict `{"reasoning", "is_correct", "critique"}`. `critique_template.md` is **not** loaded from disk anywhere — its content is hardcoded as English text in `_format_critique_history` (base_agent.py); don't expect editing that file to change runtime behavior.

**Runner backends** (`src/utils/agent_*.py`, one per backend) each implement `process_review()`, looping extractor→evaluator up to `--max_retries` until `is_correct: true` or attempts exhausted:
- `agent_async.py`, `agent_sequential.py`, `agent_batch.py` — subclass `BaseABSASystem`, use local vLLM.
- `agent_openrouter.py`, `agent_async_gemini.py`, `agent_sequential_gemini.py` — standalone classes (not subclasses) that duplicate the same `REQUIRED_PROMPT_FILES`/prompt-loading pattern independently, for Gemini/OpenRouter APIs. If you change loading/parsing behavior in `base_agent.py`, check whether the equivalent duplicated logic in these three needs the same fix.

Each backend has a matching CLI entrypoint in `src/main/run_agent_*.py` (thin argparse wrapper) and is invoked as `python -m src.main.run_agent_<backend>`.

**Result paths encode the full experiment config** and are the only record of it — there is no experiment config file (YAML/JSON); everything is CLI flags/env vars:
```
results/<runner_type>/<model_name>/<dataset_type>/<lang>/<dataset_folder>/<prompt_set>/max_retries_<N>/seed_<N>/<timestamp>/inference_results.json
```
Downstream eval scripts (`src/main/eval_exact_match.py`, `eval_instruct_absa.py`, `eval_semantic.py`) parse this metadata **positionally** from the path (see `extract_metadata()` in each) — so preserve this exact directory depth/ordering when adding new runners or output locations, or the eval scripts will silently mis-tag results.

**Dataset layout**: `dataset/<dataset_type>/<lang>/<variant>/{train,dev,test}.json`, e.g. `dataset/hotel_reviews/indo/mvp_aos/test.json`. `mvp_aos` is the current canonical triplet-annotated variant; other variants (`mvp`, `mvp_aso`, `gas`, `legoabsa_multitask`, etc.) are older/alternate task formats — check which variant a script expects before pointing it at a different one.

## In-progress work

`FEWSHOT_EXPERIMENT_PLAN.md` (repo root) is a draft plan for a not-yet-built few-shot experiment (zero-shot vs. static vs. dynamically-retrieved few-shot examples, single-pass, no evaluator/retry) — check it for open design questions before building anything in that direction.
