#!/usr/bin/env bash
# Phase 1, step 01 — cache entity-token activations on the Wikidata sweep.
#
# Iterates over all four entity types (player, movie, city, song) using the
# pre-processed prompts at vendor/sae_entities/dataset/processed/entity_prompts/.
# Saves memory-mapped float32 activations to:
#   vendor/sae_entities/dataset/cached_activations/entity/gemma-2-2b-it_wikidata_<entity>/...
#
# Batch size 16 is sized for cloud T4/P100 (16 GB VRAM). Bump to 32 if peak
# is comfortably under 12 GB on the first iteration.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"${REPO_ROOT}/scripts/patch_upstream.sh"

cd "${REPO_ROOT}/vendor/sae_entities"
exec uv run --project "${REPO_ROOT}" python -m utils.activation_cache \
    --model_alias gemma-2-2b-it \
    --tokens_to_cache entity \
    --batch_size 16 \
    --entity_type_and_entity_name_format \
    --dataset wikidata
