#!/usr/bin/env bash
# Patch upstream sae_entities scripts to use gemma-2-2b-it.
#
# Upstream's mech_interp/* scripts hard-code a model_alias literal at the top
# of each file (a notebook-style convention). For Phase 1 we want every
# critical-path script to point at gemma-2-2b-it. This script applies the
# narrow substitutions in-place and is idempotent — re-running is a no-op.
#
# Files patched (Phase 1 critical path only):
#   - mech_interp/feature_analysis.py     (line 22)  — model_alias literal
#   - mech_interp/steering_it.py          (line 57)  — model_alias literal
#   - utils/hf_models/gemma_model.py      (line 99)  — bf16 → float16 on Pascal/Turing
#   - utils/generation_utils.py           (line 295) — vllm.LLM dtype on Pascal/Turing
#
# Also creates the missing dataset/process_data/__init__.py marker so
# `python -m dataset.process_data.wikidata.create_wikidata_entity_queries`
# resolves. Upstream forgot to ship it; without it the relative import
# `from .check_correctness_wikidata import ...` cannot be resolved.
#
# bf16/fp16 patches: Gemma 2 was trained in bfloat16, but bf16 requires
# compute capability >= 8.0 (Ampere/Ada/Hopper). On Pascal (P100, cc 6.0)
# and Turing (T4, cc 7.5), the model loaders crash with
#   ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0.
# We replace the hardcoded `torch.bfloat16` defaults with a runtime check
# that falls back to fp16 on older cards.
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

ensure_init() {
    # ensure_init <path>  — create an empty __init__.py if missing.
    local init="$1/__init__.py"
    if [[ -f "${init}" ]]; then
        echo "  [skip] ${init#${UPSTREAM}/}: already exists"
    else
        : > "${init}"
        echo "  [done] ${init#${UPSTREAM}/}: created"
    fi
}

patch_text() {
    # patch_text <file> <old_literal> <new_literal>
    # Generic substring replacement, idempotent (skip if new already present).
    # Membership and replacement are both done in Python so multi-line literals
    # work correctly (grep -F splits multi-line patterns into per-line
    # alternatives, which would false-positive on overlapping single-line text).
    local file="$1"
    local old="$2"
    local new="$3"

    local status
    status="$(python3 - "$file" "$old" "$new" <<'PY'
import sys, pathlib
path, old, new = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
src = path.read_text()
if new in src:
    print("skip")
elif old not in src:
    print("missing")
else:
    path.write_text(src.replace(old, new, 1))
    print("done")
PY
)"
    case "${status}" in
        skip) echo "  [skip] ${file##*/}: already patched" ;;
        done) echo "  [done] ${file##*/}: patched" ;;
        missing)
            echo "FAIL: ${file}: old literal not found — upstream layout changed?" >&2
            echo "      expected: ${old}" >&2
            return 1
            ;;
        *)
            echo "FAIL: ${file}: unexpected status '${status}'" >&2
            return 1
            ;;
    esac
}

echo "Patching upstream sae_entities scripts → ${TARGET_ALIAS}"
patch_line "${UPSTREAM}/mech_interp/feature_analysis.py" "gemma-2-2b" "${TARGET_ALIAS}"
patch_line "${UPSTREAM}/mech_interp/steering_it.py" "meta-llama/Llama-3.1-8B" "${TARGET_ALIAS}"

echo "Ensuring missing package markers"
ensure_init "${UPSTREAM}/dataset/process_data"

# --- bf16 → fp16 fallback for Pascal/Turing GPUs ---
# HF loader path (used by activation_cache.py via construct_model_base):
echo "Patching dtype defaults (bf16 → fp16 fallback on cc<8.0 GPUs)"
patch_text "${UPSTREAM}/utils/hf_models/gemma_model.py" \
    "def _load_model(self, model_path, dtype=torch.bfloat16):" \
    "def _load_model(self, model_path, dtype=(torch.float16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8 else torch.bfloat16)):"

# vllm path (used by create_wikidata_entity_queries.py via get_batch_completion_fn):
# We need `torch` available in generation_utils.py — it's not currently imported.
# Inject the import next to the existing `import vllm`, then update the LLM call.
patch_text "${UPSTREAM}/utils/generation_utils.py" \
    "import vllm" \
    "import torch
import vllm"

patch_text "${UPSTREAM}/utils/generation_utils.py" \
    "llm = vllm.LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization)" \
    "llm = vllm.LLM(model=model_path, gpu_memory_utilization=gpu_memory_utilization, dtype=('float16' if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8 else 'auto'))"

echo "Done."
