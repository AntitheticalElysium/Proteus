#!/usr/bin/env bash
# Phase 1, step 02 — cache random-token activations from the Pile baseline.
#
# Used by the feature analysis to filter out latents that fire on
# generic-language tokens (i.e., not entity-specific signal).
# Saves to:
#   vendor/sae_entities/dataset/cached_activations/random/gemma-2-2b-it_pile/...

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"${REPO_ROOT}/scripts/patch_upstream.sh"

cd "${REPO_ROOT}/vendor/sae_entities"
exec uv run --project "${REPO_ROOT}" python -m utils.activation_cache \
    --model_alias gemma-2-2b-it \
    --tokens_to_cache random \
    --batch_size 16 \
    --dataset pile
