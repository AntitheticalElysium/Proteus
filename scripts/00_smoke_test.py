"""Phase 1 smoke test — verify the toolchain end-to-end on a single prompt.

Loads `google/gemma-2-2b-it` via upstream's `construct_model_base`, runs one
forward pass with a residual-stream hook on layer 9, and one short greedy
generation with the Gemma chat template. Prints shapes, generated text, and
peak VRAM. Exits non-zero on any deviation from expected.

This is the gate that proves the upstream pipeline imports and runs before
the long activation-caching jobs are kicked off. Intended to run on a cloud
GPU (Colab T4 16GB / Kaggle P100 16GB / RunPod A6000) — local 6GB VRAM
cannot fit Gemma 2 2B in bf16 with any meaningful headroom.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# ── sys.path setup ────────────────────────────────────────────────────────
# Upstream uses absolute imports rooted at the repo dir (e.g. `from
# utils.hf_models...`). Put `vendor/sae_entities` on the path so those
# imports resolve regardless of where this script is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[1]
SAE_ENTITIES_DIR = REPO_ROOT / "vendor" / "sae_entities"
if not SAE_ENTITIES_DIR.exists():
    raise SystemExit(
        f"Submodule missing: {SAE_ENTITIES_DIR}\n"
        "Run: git submodule update --init --recursive"
    )
sys.path.insert(0, str(SAE_ENTITIES_DIR))

# Upstream's `dataset/cached_activations/...` paths are relative to CWD.
# Run this script from the upstream dir to keep paths consistent with the
# rest of the pipeline.
os.chdir(SAE_ENTITIES_DIR)

from utils.hf_models.model_factory import construct_model_base  # noqa: E402
from utils.utils import model_alias_to_model_name  # noqa: E402

MODEL_ALIAS = "gemma-2-2b-it"
PROMPT = "What is the capital of France?"
TARGET_LAYER = 9  # middle of the 26-layer Gemma 2 2B residual stack
EXPECTED_D_MODEL = 2304
VRAM_GATE_GB = 14.0  # T4=15GB, P100=16GB; well under both


def fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not torch.cuda.is_available():
        fail("CUDA is not available. Smoke test requires a GPU runtime.")

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    torch.cuda.reset_peak_memory_stats()

    model_path = model_alias_to_model_name[MODEL_ALIAS]
    print(f"\nLoading {model_path}...")
    model_base = construct_model_base(model_path)
    model_base.tokenizer.pad_token = model_base.tokenizer.eos_token

    n_layers = len(model_base.model_block_modules)
    print(f"Model loaded. n_layers={n_layers}")
    if n_layers != 26:
        fail(f"expected 26 layers for Gemma 2 2B, got {n_layers}")

    # ── Forward pass with residual hook on TARGET_LAYER ───────────────────
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        # Gemma decoder layers return a tuple; first element is hidden state.
        resid = output[0] if isinstance(output, tuple) else output
        captured["resid"] = resid.detach()

    handle = model_base.model_block_modules[TARGET_LAYER].register_forward_hook(hook)
    try:
        tokenized = model_base.tokenize_instructions_fn(instructions=[PROMPT])
        input_ids = tokenized.input_ids.to(model_base.model.device)
        attention_mask = tokenized.attention_mask.to(model_base.model.device)
        with torch.no_grad():
            _ = model_base.model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        handle.remove()

    if "resid" not in captured:
        fail(f"forward hook on layer {TARGET_LAYER} did not fire")

    resid = captured["resid"]
    print(f"\nResidual stream @ L{TARGET_LAYER}: shape={tuple(resid.shape)} dtype={resid.dtype}")
    if resid.ndim != 3:
        fail(f"expected 3D residual [batch, seq, d_model], got shape {tuple(resid.shape)}")
    if resid.shape[-1] != EXPECTED_D_MODEL:
        fail(f"expected d_model={EXPECTED_D_MODEL}, got {resid.shape[-1]}")

    # ── Greedy generation through the chat template ───────────────────────
    print("\nGenerating completion (max 30 tokens, greedy)...")
    completions = model_base.generate_completions(
        instructions=[PROMPT],
        batch_size=1,
        max_new_tokens=30,
    )
    if not completions or not completions[0].strip():
        fail("generation returned empty output")
    print(f"Prompt: {PROMPT}")
    print(f"Output: {completions[0]}")

    # ── VRAM gate ─────────────────────────────────────────────────────────
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"\nPeak VRAM: {peak_gb:.2f} GB (gate: < {VRAM_GATE_GB} GB)")
    if peak_gb > VRAM_GATE_GB:
        fail(f"peak VRAM {peak_gb:.2f} GB exceeds gate {VRAM_GATE_GB} GB")

    print("\nOK — smoke test passed.")


if __name__ == "__main__":
    main()
