#!/usr/bin/env bash
# Patch upstream sae_entities scripts to use gemma-2-2b-it.
#
# Upstream's mech_interp/* scripts hard-code a model_alias literal at the top
# of each file (a notebook-style convention). For Phase 1 we want every
# critical-path script to point at gemma-2-2b-it. This script applies the
# narrow substitutions in-place and is idempotent — re-running is a no-op.
#
# Files patched (Phase 1 critical path only):
#   - mech_interp/feature_analysis.py  (line 22)
#   - mech_interp/steering_it.py       (line 57)
#
# Out-of-Phase-1 scripts (uncertain_features, patching, refusal_analysis,
# attn_analysis, visualize_latents.ipynb) are intentionally NOT patched —
# they are not in Phase 1's verification gates. Patch them in Phase 2 if
# needed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UPSTREAM="${REPO_ROOT}/vendor/sae_entities"
TARGET_ALIAS="gemma-2-2b-it"

if [[ ! -d "${UPSTREAM}" ]]; then
    echo "FAIL: ${UPSTREAM} not found. Run: git submodule update --init --recursive" >&2
    exit 1
fi

patch_line() {
    # patch_line <file> <old_python_literal> <new_python_literal>
    local file="$1"
    local old="$2"
    local new="$3"
    local full_old="model_alias = '${old}'"
    local full_new="model_alias = '${new}'"

    if grep -qF "${full_new}" "${file}"; then
        echo "  [skip] ${file##*/}: already patched"
        return 0
    fi
    if ! grep -qF "${full_old}" "${file}"; then
        echo "FAIL: ${file}: neither old nor new literal found — upstream layout changed?" >&2
        echo "      expected line: ${full_old}" >&2
        return 1
    fi
    # Use a sed delimiter that won't appear in the literals.
    sed -i "s|${full_old}|${full_new}|" "${file}"
    echo "  [done] ${file##*/}: '${old}' → '${new}'"
}

echo "Patching upstream sae_entities scripts → ${TARGET_ALIAS}"
patch_line "${UPSTREAM}/mech_interp/feature_analysis.py" "gemma-2-2b" "${TARGET_ALIAS}"
patch_line "${UPSTREAM}/mech_interp/steering_it.py" "meta-llama/Llama-3.1-8B" "${TARGET_ALIAS}"
echo "Done."
