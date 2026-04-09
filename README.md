# Proteus

A Mechanistic Taxonomy of Hallucination.

Proteus investigates whether different hallucination types share internal
circuitry in Gemma 2 2B, using pre-trained GemmaScope SAEs as the lens.

## Phase 1 — Ferrando replication on `gemma-2-2b-it`

Phase 1 reproduces Ferrando et al., *"Do I Know This Entity?"* (ICLR 2025)
end-to-end on the instruction-tuned Gemma 2 2B model. Goal: confirm that
SAE latents on the residual stream cleanly separate known from unknown
entities, and that those latents causally drive refusal behavior.

This phase is a vendoring + orchestration job — no original code. The
upstream repo `javiferran/sae_entities` is included as a submodule under
`vendor/sae_entities` and provides the datasets, the activation-caching
CLI, the feature analysis script, the steering script, and the refusal
detector. Proteus owns the dependency manifest, the cloud bootstrap, and
the thin shell shims that wire everything together.

### Architecture

```
Proteus/
├── pyproject.toml             # uv-managed env (Linux x86_64, CUDA 12.1)
├── uv.lock                    # locked, 249 packages
├── vendor/sae_entities/       # submodule — Ferrando's pipeline + data
├── src/proteus/               # empty package skeleton (populated in Phase 2)
├── scripts/
│   ├── 00_smoke_test.py       # model load + shape + VRAM gate
│   ├── patch_upstream.sh      # idempotent IT-model patches
│   ├── 01_cache_entity_acts.sh
│   ├── 02_cache_pile_acts.sh
│   ├── 03_feature_analysis.sh
│   └── 04_steering.sh
└── notebooks/
    └── phase1_cloud_bootstrap.ipynb   # Colab / Kaggle / RunPod entry point
```

### Hardware

Phase 1 needs a GPU with **≥16 GB VRAM**. The cache jobs sweep four
Wikidata entity types and the Pile baseline at fp16 — too heavy for a
laptop GPU. Tested targets:

- **Colab T4** (16 GB) — free tier, ~5 hour runtime cap
- **Kaggle P100** (16 GB) — free, 12 hour cap
- **RunPod A6000** (48 GB) — paid, no cap, fastest

### Prerequisites (one-time)

1. Hugging Face account, accept the [Gemma 2 license](https://huggingface.co/google/gemma-2-2b-it).
2. Create a read-scope HF token at <https://huggingface.co/settings/tokens>.
3. Add the token to your runtime's secret store as `HF_TOKEN`:
   - **Colab**: `Secrets` (key icon in sidebar) → name `HF_TOKEN`
   - **Kaggle**: `Add-ons → Secrets` → label `HF_TOKEN`
   - **Local / RunPod**: `export HF_TOKEN=...`
4. Fork this repo (or use upstream directly) and update `REPO_URL` in
   the bootstrap notebook's first code cell.

### Reproduction

Open `notebooks/phase1_cloud_bootstrap.ipynb` in your runtime of choice
and run all cells in order. The notebook handles:

1. Cloning the repo (with submodules) and `cd`-ing into it
2. Installing `uv` and syncing the locked environment
3. HF authentication via the runtime secret store
4. Smoke test — loads the model, runs one forward + one generation,
   verifies layer count, residual width, and peak VRAM
5. Caching entity activations across player / movie / city / song
6. Caching the Pile baseline activations
7. Computing per-layer SAE latent separation scores
8. Running steering experiments at coefficients 20–400
9. Tar-ing up the lightweight outputs (plots, CSVs, latent rankings) for download

The cache jobs dominate the wall clock — expect **2–4 hours** for entity
activations on a T4 and **1–2 hours** for Pile.

### Verification gates

Phase 1 succeeds when **all** of these hold:

1. **Smoke test passes** — residual shape `[1, seq, 2304]`, peak VRAM
   under the runtime's budget, generation looks coherent.
2. **Entity activations cached** under
   `vendor/sae_entities/dataset/cached_activations/entity/gemma-2-2b-it_wikidata_*/`.
3. **Pile activations cached** under
   `vendor/sae_entities/dataset/cached_activations/random/gemma-2-2b-it_pile/`.
4. **Separation score peaks in the middle layers** (~L8–L14). Exact
   values do not need to match the paper, but the layerwise shape must.
5. **Steering is causal in both directions**: at least one
   (latent, coefficient) combination induces refusal on a previously-answered
   known entity, and at least one combination suppresses refusal on an
   unknown entity.

If gates 4 or 5 fail with the toolchain otherwise working, **stop** and
reassess before planning Phase 2.

## License

See `LICENSE`. Vendored upstream code under `vendor/sae_entities/` is
governed by its own license — see that directory.
