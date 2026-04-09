"""Project paths and constants.

Populated incrementally; Phase 1 only needs path constants for the
upstream submodule and our results directory.
"""

from __future__ import annotations

from pathlib import Path

# Repo root: src/proteus/config.py -> ../../..
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Upstream vendored as a git submodule.
VENDOR_DIR: Path = REPO_ROOT / "vendor"
SAE_ENTITIES_DIR: Path = VENDOR_DIR / "sae_entities"

# Upstream cache directory used by `utils.activation_cache`. The path is
# relative to upstream's working directory at run time:
#   {upstream}/dataset/cached_activations/{tokens_to_cache}/{model_alias}_{dataset_name}/...
UPSTREAM_CACHE_DIR: Path = SAE_ENTITIES_DIR / "dataset" / "cached_activations"

# Our outputs (gitignored).
RESULTS_DIR: Path = REPO_ROOT / "results"

# Model alias passed to upstream's `--model_alias` flag. Maps to
# `google/gemma-2-2b-it` via `vendor/sae_entities/utils/utils.py`.
MODEL_ALIAS: str = "gemma-2-2b-it"
