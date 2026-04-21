# ABSA Agent

ABSA Agent is a prompt-based Aspect-Based Sentiment Analysis (ABSA) experimentation repository.

It supports multiple runner backends:
- Local vLLM (async, sequential, batch)
- Gemini (async and sequential)
- OpenRouter (async)

It also includes:
- Reproducible inference/evaluation scripts
- Retry-based extraction-evaluation agent loops
- Multi-language dataset handling through dataset paths
- Experiment outputs organized under timestamped result folders

## Repository Layout

- `dataset/`: ABSA datasets (language-specific subfolders, e.g. `indo`, `sun`, `min`, `eng`)
- `prompts/`: prompt sets used by the agent (e.g. `prompts/exp1`)
- `scripts/`: shell wrappers for common experiment runs
- `src/main/`: runnable Python entry points
- `src/utils/`: core agent logic, parsing, metrics, utilities
- `results/`: generated experiment outputs

## Environment Setup

### 1. Create and sync environment

This project uses `uv` and a local `.venv`.

```bash
uv sync
```

### 2. Activate environment

```bash
source .venv/bin/activate
```

### 3. API keys (when using Gemini/OpenRouter)

- Gemini runners read `GEMINI_API_KEY` (dotenv is supported by runner scripts).
- OpenRouter runners read `OPENROUTER_API_KEY`.

Example `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

## Prompt Language Behavior

System prompts use a `{language}` placeholder.

At runtime, runners infer language from `--test_case_path`, for example:
- `dataset/hotel_reviews/indo/mvp_aos/test.json` -> `Indonesian`
- `dataset/hotel_reviews/sun/mvp_aos/test.json` -> `Sundanese`

So changing the dataset language path automatically updates the language wording used in prompts.

## Running Experiments: Inference

You can run inference either via scripts or directly via Python modules.

### A. Script workflow (recommended)

#### 1) vLLM async inference for one language

```bash
./scripts/inference.sh indo
```

If no argument is provided, it defaults to `indo`.

#### 2) vLLM async inference for multiple languages (sequential loop)

```bash
./scripts/inference_loop.sh indo sun min
```

#### 3) Gemini async inference

```bash
./scripts/inference_gemini.sh
```

Override defaults with env vars:

```bash
MODEL_NAME=gemini-3-flash-preview \
TEST_CASE_PATH=dataset/hotel_reviews/sun/mvp_aos/test.json \
MAX_RETRIES=5 \
SEED=42 \
./scripts/inference_gemini.sh
```

#### 4) Gemini sequential inference

```bash
./scripts/inference_gemini_sequential.sh
```

### B. Direct module workflow

#### vLLM async

```bash
python -m src.main.run_agent_async \
	--model_path Qwen/Qwen3-8B \
	--test_case_path dataset/hotel_reviews/indo/mvp_aos/test.json \
	--max_retries 10 \
	--prompt_set exp1 \
	--seed 42
```

#### Gemini async

```bash
python -m src.main.run_agent_async_gemini \
	--model_name gemini-3-flash-preview \
	--test_case_path dataset/hotel_reviews/indo/mvp_aos/test.json \
	--max_retries 3 \
	--prompt_set exp1 \
	--seed 42 \
	--track_tokens
```

#### Gemini sequential

```bash
python -m src.main.run_agent_sequential_gemini \
	--model_name gemini-3-flash-preview \
	--test_case_path dataset/hotel_reviews/indo/mvp_aos/test.json \
	--max_retries 10 \
	--prompt_set exp1 \
	--seed 42 \
	--track_tokens
```

#### OpenRouter async

```bash
python -m src.main.run_agent_openrouter \
	--model qwen/qwen3-8b \
	--test_case_path dataset/hotel_reviews/indo/mvp_aos/test.json \
	--max_retries 3 \
	--prompt_set exp1
```

## Running Experiments: Evaluation

Evaluation consumes generated `inference_results.json` and writes `agent_evaluation_results.json` to the same timestamp folder.

This repository now has two evaluation tracks:

- Agent evaluator (`eval_agent.py`) for the legacy/agent report format.
- Method-based evaluators (`exact_match`, `instruct_absa`, `semantic`) for cross-method comparison outputs.

### A. Script workflow (recommended)

#### 1) Evaluate one language

```bash
./scripts/eval.sh indo
```

#### 2) Evaluate multiple languages (sequential loop)

```bash
./scripts/eval_loop.sh indo sun min
```

### B. Direct module workflow

```bash
python -m src.main.eval_agent \
	--model_path google/gemini-3-flash-preview \
	--test_case_path dataset/hotel_reviews/indo/mvp_aos/test.json \
	--max_retries 10 \
	--seed 42 \
	--prompt_set exp1 \
	--runner_type sequential_gemini
```

`--runner_type` choices currently include:
- `async`
- `sequential`
- `batch`
- `async_0.6B`
- `sequential_gemini`

Use `--lowercase` if you want lowercased comparison before scoring.

### C. Method-Based Evaluation (Exact Match, InstructABSA, Semantic)

These evaluators read agent inference results from paths like:

```text
{output_dir}/async/{model_name}/{dataset_type}/{lang}/{dataset_folder}/{prompt_dir}/{num_of_max_retries}/{seed}/{timestamp_or_id}/inference_results.json
```

Example:

```text
results/async/Qwen3-8B/hotel_reviews/indo/mvp_aos/exp1/max_retries_10/seed_42/2026-03-02_01-21-36/inference_results.json
```

#### 1) Exact Match

How it works:

- Compares predicted and target triplets as structured tuples (`aspect`, `opinion`, `sentiment`) after normalization.
- A triplet counts as true positive only when all three fields match exactly.
- False positives are predicted triplets not found in target; false negatives are target triplets not found in prediction.
- Dataset-level precision/recall/F1 are computed from aggregated TP/FP/FN across all instances.

Single config:

```bash
bash scripts/eval_exact_match.sh \
	results \
	Qwen3-8B \
	hotel_reviews \
	indo \
	mvp_aos \
	exp1 \
	max_retries_10 \
	seed_42 \
	2026-03-02_01-21-36
```

All discovered async results:

```bash
bash scripts/eval_all_exact_match.sh results
```

Outputs (written beside each `inference_results.json`):

- `exact_match.json` (dataset-level precision/recall/f1)
- `exact_match_detail.json` (instance-level precision/recall/f1 + false positives/false negatives)

#### 2) InstructABSA

How it works:

- Converts each triplet into AOSTE-style text (`aspect:opinion:sentiment`) and compares at string level.
- Uses substring-friendly matching (`pred in gt` or `gt in pred`) to tolerate small wording differences.
- This is less strict than exact match, so partially equivalent phrasing can still count as correct.
- Precision/recall/F1 are computed from matched/unmatched triplet strings.

Single config:

```bash
bash scripts/eval_instruct_absa.sh \
	results \
	Qwen3-8B \
	hotel_reviews \
	indo \
	mvp_aos \
	exp1 \
	max_retries_10 \
	seed_42 \
	2026-03-02_01-21-36
```

All discovered async results:

```bash
bash scripts/eval_all_instruct_absa.sh results
```

Outputs:

- `instruct_absa.json`
- `instruct_absa_detail.json`

#### 3) Semantic

How it works:

- Converts agent triplets to tagged ABSA strings (`[A] ... [O] ... [S] ...`) to align with `example_eval` format.
- First performs exact string matching.
- For remaining mismatches, computes embedding cosine similarity between unmatched prediction/target strings.
- If similarity is above threshold (default `0.9`), it is treated as a semantic match; remaining items become FP/FN.

Single config:

```bash
bash scripts/eval_semantic.sh \
	results \
	Qwen/Qwen3-Embedding-0.6B \
	Qwen3-8B \
	hotel_reviews \
	indo \
	mvp_aos \
	exp1 \
	max_retries_10 \
	seed_42 \
	2026-03-02_01-21-36
```

All discovered async results:

```bash
bash scripts/eval_all_semantic.sh results Qwen/Qwen3-Embedding-0.6B
```

Outputs:

- `semantic_metrics.json`
- `semantic_metrics_detail.json`

Notes:

- Semantic evaluation requires `sentence-transformers`.
- Semantic evaluator converts agent triplets into `[A] ... [O] ... [S] ...` format before scoring, to match `example_eval` style.

## Result Paths

Outputs are grouped by runner/model/dataset/language/prompt set/retry/seed/timestamp.

Example inference result path:

```text
results/async/Qwen3-8B/hotel_reviews/indo/mvp_aos/exp1/max_retries_10/seed_42/<timestamp>/inference_results.json
```

Evaluation output is written alongside it:

```text
.../agent_evaluation_results.json
```

## Reproducibility Notes

- Seeds are supported by runners (`--seed`).
- Some async engine scheduling effects can still introduce non-determinism.
- Keep prompt set, dataset path, model, retry count, and seed fixed for fair comparisons.

## Useful Entry Points

- `src/main/run_agent_async.py`
- `src/main/run_agent_sequential.py`
- `src/main/run_agent_batch.py`
- `src/main/run_agent_async_gemini.py`
- `src/main/run_agent_sequential_gemini.py`
- `src/main/run_agent_openrouter.py`
- `src/main/eval_agent.py`
