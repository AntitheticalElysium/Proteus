#!/usr/bin/env bash
# Phase 1, step 03 — compute SAE latent separation scores.
#
# Runs upstream's mech_interp/feature_analysis.py against the cached entity
# and pile activations from steps 01 and 02. Produces per-layer latent
# rankings, scatter plots of separation scores, and identifies "general
# latents" that fire across all four entity types.
#
# Note: feature_analysis.py is a notebook-style script (uses # %% cell
# markers). Running it as a module executes top-to-bottom, which is what we
# want for batch reproduction.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"${REPO_ROOT}/scripts/patch_upstream.sh"

cd "${REPO_ROOT}/vendor/sae_entities"
exec uv run --project "${REPO_ROOT}" python -m mech_interp.feature_analysis
