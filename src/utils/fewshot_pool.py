"""Few-shot example pool loading, sampling, and retrieval.

Supports the static-random and dynamic conditions of the few-shot
experiment: `load_pool` reads a language's `train.json` as the example
pool, `sample_static_random` draws a fixed subset once, and
`build_embedding_index`/`retrieve_topk` retrieve the most similar examples
per input via `Qwen/Qwen3-Embedding-0.6B` cosine similarity (the same
embedding model already used by `src/main/eval_semantic.py`).
"""

import json
import os
import random
import logging
from collections import OrderedDict
from typing import Dict, List, Tuple

from sentence_transformers import SentenceTransformer, util

from .parsing import parse_absa_string
from .eval_utils import format_sts

logger = logging.getLogger(__name__)

RETRIEVAL_INSTRUCTION = (
    "Given a customer review, retrieve reviews that are semantically similar "
    "and would make good few-shot demonstration examples for extracting "
    "Aspect-Opinion-Sentiment triplets from the given review."
)


def load_pool(dataset_type: str, lang: str, dataset_folder: str) -> List[Dict]:
    """Load `dataset/<dataset_type>/<lang>/<dataset_folder>/train.json` as a
    few-shot example pool.

    Each item's `input` has the literal `'[A] [O] [S]'` marker stripped
    (matching how test cases are handled elsewhere), and gains a
    `parsed_target` key holding `parse_absa_string(item['target'])` — a
    list of `{"aspect", "opinion", "sentiment"}` dicts, already in the
    exact shape the extractor's own JSON output uses.
    """
    path = os.path.join("dataset", dataset_type, lang, dataset_folder, "train.json")
    logger.info(f"Loading few-shot example pool from: {path}")
    with open(path, "r", encoding="utf-8") as f:
        pool = json.load(f)

    for item in pool:
        item["input"] = item["input"].replace("[A] [O] [S]", "").strip()
        item["parsed_target"] = parse_absa_string(item["target"])

    return pool


def sample_static_random(pool: List[Dict], n: int, seed: int) -> List[Dict]:
    """Sample *n* examples from *pool* once, using a fixed *seed*."""
    return random.Random(seed).sample(pool, n)


# Percentile thresholds for `sample_diverse_curated`'s 5 categories, computed
# per-language from that language's own pool (not fixed word counts) since
# opinion/aspect term length distributions vary substantially by language.
SHORT_INPUT_PERCENTILE = 0.25
COMPLEX_TUPLE_PERCENTILE = 0.75
LONG_TERM_PERCENTILE = 0.9


def _percentile(values: List[int], q: float) -> float:
    """Nearest-rank percentile of *values* (0 <= q <= 1). No numpy dependency."""
    ordered = sorted(values)
    index = min(int(q * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]


def _term_lengths(pool: List[Dict], element: str) -> List[int]:
    """Word counts of every non-null `element` ("aspect"/"opinion") term in *pool*."""
    lengths = []
    for item in pool:
        for triplet in item["parsed_target"]:
            value = triplet.get(element)
            if value and value != "null":
                lengths.append(len(value.split()))
    return lengths


def _has_repeated_aspect(item: Dict) -> bool:
    """True if some non-null aspect in *item* appears in 2+ triplets (any polarity)."""
    counts: Dict[str, int] = {}
    for triplet in item["parsed_target"]:
        aspect = triplet.get("aspect")
        if aspect and aspect != "null":
            counts[aspect] = counts.get(aspect, 0) + 1
    return any(count >= 2 for count in counts.values())


def _max_term_length(item: Dict, element: str) -> int:
    """Longest (word count) non-null `element` term in *item*, or 0 if none."""
    lengths = [
        len(triplet[element].split())
        for triplet in item["parsed_target"]
        if triplet.get(element) and triplet[element] != "null"
    ]
    return max(lengths, default=0)


def sample_diverse_curated(pool: List[Dict], seed: int) -> List[Dict]:
    """Pick 5 examples from *pool*, one per structural-variation category.

    Implements the advisor's requested selection strategy: rather than random
    or similarity-based selection, pick a fixed, deterministic (per
    language/seed) set of demonstrations that each showcase a different
    phenomenon:

    1. Simple — exactly 1 triplet and a short input (<= 25th percentile
       input word length).
    2. Complex — many triplets (>= 75th percentile tuple count) and a long
       input (>= median input word length).
    3. One aspect, multiple opinions — the same non-null aspect appears in
       2+ triplets in one example (any polarity).
    4. Long aspect term — contains a non-null aspect term at or above the
       90th percentile aspect-term word length.
    5. Long opinion term — contains an opinion term at or above the 90th
       percentile opinion-term word length.

    Thresholds are computed from *pool* itself (percentile-relative, not
    fixed word counts), so this adapts to each language's own length
    distribution when called with a per-language pool (see `load_pool`).

    Always returns exactly 5 distinct examples, one per category, in the
    order listed above. Raises `RuntimeError` if any category has no
    remaining eligible candidate — this should not happen given the pool
    sizes these datasets have, and failing loudly beats silently swapping in
    a non-matching example that would corrupt the category comparison.
    """
    input_lens = [len(item["input"].split()) for item in pool]
    tuple_counts = [len(item["parsed_target"]) for item in pool]
    aspect_lens = _term_lengths(pool, "aspect")
    opinion_lens = _term_lengths(pool, "opinion")

    short_input_thresh = _percentile(input_lens, SHORT_INPUT_PERCENTILE)
    complex_tuple_thresh = _percentile(tuple_counts, COMPLEX_TUPLE_PERCENTILE)
    median_input_len = _percentile(input_lens, 0.5)
    long_aspect_thresh = _percentile(aspect_lens, LONG_TERM_PERCENTILE)
    long_opinion_thresh = _percentile(opinion_lens, LONG_TERM_PERCENTILE)

    categories = [
        (
            "simple (1 tuple, short)",
            lambda item: len(item["parsed_target"]) == 1
            and len(item["input"].split()) <= short_input_thresh,
        ),
        (
            "complex (>1 tuple, long)",
            lambda item: len(item["parsed_target"]) >= complex_tuple_thresh
            and len(item["input"].split()) >= median_input_len,
        ),
        ("one aspect, multiple opinions", _has_repeated_aspect),
        (
            "long aspect term",
            lambda item: _max_term_length(item, "aspect") >= long_aspect_thresh,
        ),
        (
            "long opinion term",
            lambda item: _max_term_length(item, "opinion") >= long_opinion_thresh,
        ),
    ]

    rng = random.Random(seed)
    used_indices: set = set()
    selected: List[Dict] = []

    for name, predicate in categories:
        candidates = [
            i for i, item in enumerate(pool) if i not in used_indices and predicate(item)
        ]
        if not candidates:
            raise RuntimeError(
                f"No eligible pool example found for category {name!r} "
                f"(pool size={len(pool)}, {len(used_indices)} already selected). "
                "This is unexpected given typical dataset sizes -- check the pool."
            )
        pick = rng.choice(candidates)
        used_indices.add(pick)
        selected.append(pool[pick])

    return selected


def build_embedding_index(
    pool: List[Dict],
    embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
) -> Tuple[SentenceTransformer, "torch.Tensor"]:
    """Load the embedding model and batch-encode the whole pool once.

    Loaded on CPU deliberately: retrieval is cheap relative to LLM
    generation, and keeping it off the GPU avoids competing with vLLM for
    VRAM on single-GPU setups where the extractor model's weights alone
    can already consume most of the card (e.g. Qwen3-8B's ~15 GiB on a
    22.5 GiB L4).
    """
    logger.info(f"Loading embedding model for dynamic retrieval (CPU): {embedding_model_name}")
    model = SentenceTransformer(
        embedding_model_name,
        trust_remote_code=True,
        cache_folder=os.getenv("HF_CACHE_DIR"),
        device="cpu",
    )
    texts = [format_sts(item["input"], RETRIEVAL_INSTRUCTION) for item in pool]
    pool_embeddings = model.encode(texts, convert_to_tensor=True, show_progress_bar=True)
    return model, pool_embeddings


def retrieve_topk(
    query_text: str,
    model: SentenceTransformer,
    pool: List[Dict],
    pool_embeddings: "torch.Tensor",
    k: int = 3,
) -> List[Dict]:
    """Retrieve the *k* most similar pool examples to *query_text*."""
    query_embedding = model.encode(format_sts(query_text, RETRIEVAL_INSTRUCTION), convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, pool_embeddings)[0]
    top_indices = scores.topk(k=min(k, len(pool))).indices.tolist()
    return [pool[i] for i in top_indices]


def _join_quoted(items: List[str]) -> str:
    """Join *items* as quoted, comma-separated text ending in "... and X"."""
    quoted = [f'"{item}"' for item in items]
    if len(quoted) == 1:
        return quoted[0]
    return ", ".join(quoted[:-1]) + " and " + quoted[-1]


def format_staged_reasoning(triplets: List[Dict]) -> str:
    """Synthesize an `exp2_staged`-style 3-stage reasoning trace from *triplets*.

    `train.json` only has final `{"aspect", "opinion", "sentiment"}` triplets,
    not a human-written reasoning trace, so this deterministically reconstructs
    one (Stage 1: sentiment markers, Stage 2: related aspects, Stage 3:
    polarity classification) purely from the triplet fields themselves —
    no extra LLM call. Mirrors `prompts/exp2_staged/extractor_system.md`'s
    example format so the staged few-shot examples match what the staged
    system prompt asks the model to produce.
    """
    if not triplets:
        return (
            "Stage 1 - Sentiment markers: no sentiment markers found in the text.\n"
            "Stage 2 - Related aspects: none.\n"
            "Stage 3 - Polarity classification: none."
        )

    # Group by opinion text (preserving first-seen order) so a marker shared
    # by multiple aspects is listed once in Stage 1 and fanned out in Stage 2.
    groups: "OrderedDict[str, List[Tuple[str, str]]]" = OrderedDict()
    for triplet in triplets:
        groups.setdefault(triplet["opinion"], []).append((triplet["aspect"], triplet["sentiment"]))

    stage1 = "Stage 1 - Sentiment markers: " + ", ".join(
        f'"{opinion}" ({pairs[0][1]} marker)' for opinion, pairs in groups.items()
    ) + "."

    stage2_sentences = []
    for opinion, pairs in groups.items():
        aspects = [aspect for aspect, _ in pairs]
        if aspects == ["null"]:
            stage2_sentences.append(
                f'"{opinion}" relates to no explicit feature named in the text -> aspect is null.'
            )
        else:
            stage2_sentences.append(f'"{opinion}" relates to {_join_quoted(aspects)}.')
    stage2 = "Stage 2 - Related aspects: " + " ".join(stage2_sentences)

    stage3 = "Stage 3 - Polarity classification: " + "; ".join(
        f'{triplet["aspect"]} + {triplet["opinion"]} -> {triplet["sentiment"]}' for triplet in triplets
    ) + "."

    return f"{stage1}\n{stage2}\n{stage3}"


def format_fewshot_block(examples: List[Dict], staged: bool = False) -> str:
    """Render example Input/Output pairs for injection into `{fewshot_examples}`.

    Returns an empty string for an empty list, so it substitutes cleanly
    for the zero-shot condition.

    Args:
        examples: Pool items (each with `input`/`parsed_target`, see
            `load_pool`) to render as demonstrations.
        staged: When `True`, insert a synthesized `format_staged_reasoning`
            trace between Input and Output in each example, matching the
            `fewshot_*_staged` prompt sets' reasoning-process instructions.
    """
    if not examples:
        return ""

    blocks = ["## Examples"]
    for example in examples:
        output_json = json.dumps(example["parsed_target"], ensure_ascii=False)
        if staged:
            reasoning = format_staged_reasoning(example["parsed_target"])
            blocks.append(
                f'### Example\nInput: "{example["input"]}"\nReasoning:\n{reasoning}\nOutput: {output_json}'
            )
        else:
            blocks.append(f'### Example\nInput: "{example["input"]}"\nOutput: {output_json}')
    return "\n\n".join(blocks) + "\n"
