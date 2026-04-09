#!/usr/bin/env bash
# Phase 1, step 00b — generate the IT-model entity prompts directory.
#
# Upstream's repo ships pre-processed entity_prompts/{model_alias}/ directories
# only for *base* models (gemma-2-2b, gemma-2-9b, meta-llama_Llama-3.1-8B). For
# the IT variants only `<alias>_wikidata_splits.json` was committed — a list
# of question strings with no greedy_completion or string_matching_sampled_labels.
#
# But `utils/activation_cache.py → load_wikidata_queries(model_alias)` reads
# the per-prompt-type JSON files in `dataset/processed/entity_prompts/{model_alias}/`
# and uses the per-query labels. So before we can cache activations for
# gemma-2-2b-it, we have to actually run upstream's
# `dataset/process_data/wikidata/create_wikidata_entity_queries.py` against
# the IT model and let it produce those files.
#
# What this script does:
#   - Runs the IT model via vllm over ~34k entities × ~5 prompts × 11 generations
#     (1 greedy + 10 sampled). Saves per-prompt-type JSONs under:
#       vendor/sae_entities/dataset/processed/entity_prompts/gemma-2-2b-it/
#   - Each generation is short (max 32 new tokens) but the total token volume
#     is large. Expect: ~5-8h on T4, ~1-3h on A6000.
#
# Smoke / quick mode:
#   MAX_QUERIES=200 ./scripts/00b_generate_entity_prompts.sh
#   (caps each entity_type at 200 entities — finishes in ~10-20 min, useful for
#    end-to-end pipeline plumbing checks. NOT enough data for the verification
#    gates — must re-run unbounded for the real Phase 1 deliverable.)
#
# Note on the buggy filter step: upstream also ships
# dataset/process_data/wikidata/filter_known_unknown_wikidata.py, which would
# normally aggregate per-query results into entity_knowledge.json. That script
# is broken for Gemma (gemma_filter_player_is_known is `pass` and the dispatch
# table hardcodes llama3_filter_*) — and crucially, it is NOT on our critical
# path. `feature_analysis_utils.py → mech_interp_utils.get_known_unknown_splits`
# derives known/unknown labels directly from per-query
# `string_matching_sampled_labels`, which `create_wikidata_entity_queries.py`
# already populates. We do not run filter_known_unknown_wikidata.py in Phase 1.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"${REPO_ROOT}/scripts/patch_upstream.sh"

MAX_QUERIES_ARG=()
if [[ -n "${MAX_QUERIES:-}" ]]; then
    MAX_QUERIES_ARG=(--max_num_queries "${MAX_QUERIES}")
    echo "[00b] MAX_QUERIES=${MAX_QUERIES} — running in capped mode (smoke / pipeline check only)"
fi

cd "${REPO_ROOT}/vendor/sae_entities"
exec uv run --project "${REPO_ROOT}" python -m dataset.process_data.wikidata.create_wikidata_entity_queries \
    --model_path google/gemma-2-2b-it \
    --model_engine vllm \
    --max_new_tokens 32 \
    --num_sampled_generations 10 \
    "${MAX_QUERIES_ARG[@]}"
