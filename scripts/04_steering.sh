#!/usr/bin/env bash
# Phase 1, step 04 — steering experiments on the IT model.
#
# Runs upstream's mech_interp/steering_it.py: takes the top known/unknown
# latents identified in step 03 and applies them as additive steering vectors
# at coefficients 20-400. Compares refusal rates between original and
# steered generations on each entity type. This is the verification gate
# for the causal direction of the latents.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"${REPO_ROOT}/scripts/patch_upstream.sh"

cd "${REPO_ROOT}/vendor/sae_entities"
exec uv run --project "${REPO_ROOT}" python -m mech_interp.steering_it
