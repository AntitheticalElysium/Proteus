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
│   ├── 00_smoke_test.py            # model load + shape + VRAM gate
│   ├── patch_upstream.sh           # idempotent IT-model patches
│   ├── 00b_generate_entity_prompts.sh  # regenerate IT prompts (upstream gap)
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
5. **Generating IT-model entity prompts** — vllm pass over the raw
   Wikidata entities, producing the per-prompt-type JSON files that
   `activation_cache.py` consumes (upstream only ships these for base
   models)
6. Caching entity activations across player / movie / city / song
7. Caching the Pile baseline activations
8. Computing per-layer SAE latent separation scores
9. Running steering experiments at coefficients 20–400
10. Tar-ing up the lightweight outputs (plots, CSVs, latent rankings) for download

The prompt-generation pass (~5–8h on T4) and the entity cache
(~2–4h on T4) dominate the wall clock. Total Phase 1 budget on T4:
**~10–14 hours**; on an A6000 closer to **~4–6 hours**.

For a quick end-to-end pipeline check (does **not** satisfy the
verification gates), set `MAX_QUERIES=200` in the prompt-generation
cell. This caps each entity type at 200 entities and finishes the
generation pass in 10–20 minutes.

### Verification gates

Phase 1 succeeds when **all** of these hold:

1. **Smoke test passes** — residual shape `[1, seq, 2304]`, peak VRAM
   under the runtime's budget, generation looks coherent.
2. **Entity prompts generated** under
   `vendor/sae_entities/dataset/processed/entity_prompts/gemma-2-2b-it/`,
   one JSON per `<entity_type>_<attribute>` (e.g. `player_date_birth.json`)
   plus the `<entity_type>_is_known.json` files. Each query carries
   `greedy_completion` and a non-empty `string_matching_sampled_labels`.
3. **Entity activations cached** under
   `vendor/sae_entities/dataset/cached_activations/entity/gemma-2-2b-it_wikidata_*/`.
4. **Pile activations cached** under
   `vendor/sae_entities/dataset/cached_activations/random/gemma-2-2b-it_pile/`.
5. **Separation score peaks in the middle layers** (~L8–L14). Exact
   values do not need to match the paper, but the layerwise shape must.
6. **Steering is causal in both directions**: at least one
   (latent, coefficient) combination induces refusal on a previously-answered
   known entity, and at least one combination suppresses refusal on an
   unknown entity.

If gates 5 or 6 fail with the toolchain otherwise working, **stop** and
reassess before planning Phase 2.

### Known upstream gaps

- Upstream's `dataset/processed/entity_prompts/` ships pre-processed
  prompt files only for *base* models (`gemma-2-2b/`, `gemma-2-9b/`,
  `meta-llama_Llama-3.1-8B/`). For IT variants only the splits file is
  committed. `00b_generate_entity_prompts.sh` regenerates them.
- Upstream's `dataset/process_data/` is missing an `__init__.py`;
  `patch_upstream.sh` creates it so the generation script can be run
  with `python -m`.
- Upstream's `filter_known_unknown_wikidata.py` is broken for Gemma
  (one filter is `pass`, dispatch is hardcoded to Llama-3). It is
  **not** on Phase 1's critical path — `feature_analysis_utils.py`
  derives known/unknown labels directly from per-query
  `string_matching_sampled_labels`, so we skip the filter step.
- `mech_interp/feature_analysis.py` and `steering_it.py` hardcode
  their target model alias at the top of the file;
  `patch_upstream.sh` rewrites those literals to `gemma-2-2b-it`.

## License

See `LICENSE`. Vendored upstream code under `vendor/sae_entities/` is
governed by its own license — see that directory.
