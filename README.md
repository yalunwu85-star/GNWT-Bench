# GNWT

GNWT is a benchmark runner for graded-noise robustness experiments on text and image questions.

This repository contains the API-based evaluation pipeline used for closed-source (API-only) models: every sample is sent to an OpenAI-compatible `/chat/completions` endpoint once per perturbation condition, the model must answer through a fixed two-probe JSON schema, answers are scored by rule-based graders or an LLM judge, and per-sample outcomes are aggregated into *ignition thresholds* — the first perturbation level at which the model fails the task.

Not included in this release: the local open-weight inference pipeline, the full benchmark dataset, and any private provider catalog or credentials. The model-identity tables in `model_identity.py` ship empty for the same reason; see [Model identity](#model-identity).

## Repository layout

| Path | Role |
|---|---|
| `scripts/run_experiment.py` | Experiment driver: config resolution, request building, retries, threading, resume, manifests |
| `scripts/judge_answer.py` | Rule-based graders, LLM-as-judge, judge fallback chains, standalone judge CLI |
| `scripts/summarize_results.py` | Threshold and probe-1 aggregation over result directories |
| `scripts/api_request_config.py` | Request-profile engine shared by eval and judge calls |
| `scripts/model_identity.py` | Canonical model-id and identity resolution rules |
| `scripts/my_api.json` | Provider / model / experiment / judge catalog (placeholder example) |
| `scripts/api_request_params.json` | Sampling parameters per role (temperature, token limits, response format) |
| `prompts/` | Frozen task templates (one per question type) and the judge template |
| `examples/mini/` | Synthetic one-sample fixture for wiring checks |

## Install

```bash
python3 -m pip install -r requirements.txt
cp scripts/.env.example scripts/.env
```

Edit `.env`: set `GNWT_API_BASE_URL` and `GNWT_API_KEY` to your OpenAI-compatible endpoint. Replace `example_provider` / `example-chat` in `scripts/my_api.json` with a provider and model id that endpoint accepts.

## Configuration

| File | Role |
|---|---|
| `scripts/.env` | Local endpoint URL and keys. Never commit. |
| `scripts/my_api.json` | Providers, models, experiment profiles, and judge profiles. This copy is a placeholder example. |
| `scripts/api_request_params.json` | Per-role request profiles: temperature, token limit field, response format, retries, and model-substring overrides. |

`my_api.json` has four sections:

- `providers` — named OpenAI-compatible endpoints. `baseUrlEnv` / `apiKeyEnv` name environment variables (from `.env`); `models` lists registered ids with optional `request_model`, `canonical_model`, `display_name`, `identity_source`, `identity_status`.
- `experiments` — named profiles binding modality, model list, levels, methods, workers, timeout, and a judge profile.
- `judges` — judge profiles binding a provider, model, key env (`GNWT_JUDGE_API_KEY` falls back to `GNWT_API_KEY`), and an optional ordered `fallback_profiles` chain.
- `defaults` — profile names and fallback worker/timeout values.

`.env` values are loaded at startup and never override variables already present in the environment.

## Data format

The runner expects a data root containing `data/` and `perturbed/` trees (the bundled `examples/mini/` shows the minimal shape):

```text
<root>/
  data/
    text/text_1/text_1.json          # sample metadata
    text/text_1/text_1_0.txt         # clean input
    image/image_1/image_1.json
    image/image_1/image_1_0.png
  perturbed/
    text/text_1/delete/text_1_delete_20.txt
    image/image_1/contrast/image_1_contrast_20.png
```

Sample metadata (`text_1.json`) carries the task and the paths:

```json
{
  "id": "text_1",
  "modality": "text",
  "question_type": "judgment_detection",
  "question": "Does the input say the lamp is on?",
  "gt": {"answer": "yes", "scoring_method": "yes_no_exact", "aliases": ["yes"]},
  "paths": {
    "original_text": "data/text/text_1/text_1_0.txt",
    "perturbations": {"delete": "perturbed/text/text_1/delete"}
  }
}
```

Image samples use `original_sample` and `perturbed_samples` instead of `paths`.

- Question types: `objective_choice_discrimination`, `objective_fill_recognition`, `judgment_detection`, `localization_evidence`, `subjective_generation_integration` — each maps to a frozen template in `prompts/`.
- Perturbation methods: text `delete` / `typos` / `antonym`, image `impulse_noise` / `shot_noise` / `contrast`. Levels are 0–90 in steps of 10; a typical run uses `02468`, i.e. the clean input plus 4 perturbed levels per method (13 conditions per sample).

Each model call must return a single JSON object:

```json
{"probe1_answer": "yes", "probe1_confidence": 0.0, "answer": "...", "confidence": 0.0}
```

Probe 1 asks whether the task-relevant information is still accessible in the perturbed input; the main answer is the task response itself.

## Running experiments

Quick positional mode (provider, model, modality, levels):

```bash
python3 scripts/run_experiment.py example_provider example-chat text 02468 --root examples/mini --ids 1 --no-resume
```

Named single-model mode:

```bash
python3 scripts/run_experiment.py \
  --provider example_provider --model example-chat \
  --modality text --levels 02468 --no-resume
```

Profile mode (uses a profile from `my_api.json`):

```bash
python3 scripts/run_experiment.py --profile text_default_02468 --no-resume
python3 scripts/run_experiment.py --profile image_default_02468 --no-resume
```

Useful options: `--root` (data root, default: repository root), `--ids` / `--id-start` / `--id-end` / `--limit` (sample selection), `--models provider/model ...` (multi-model runs), `--workers`, `--timeout`, `--max-tokens`, `--max-image-base64-bytes` (in-memory JPEG recompression for oversized images), `--results-dir`.

Runs write to `<root>/results/<modality>/<model>/` and refuse to overwrite a non-empty directory; continue it with `--resume` (optionally a path) instead. Resume validates modality, canonical model, and inference variant against the stored manifest, then only runs missing rows.

Retry flags (require `--resume`):

- `--retry-failures` — rerun stored runtime/transport/JSON failures and judge infrastructure failures
- `--retry-schema-invalid` — rerun parsed rows whose `probe1_answer` or `confidence` violates the output schema
- `--retry-missing-confidence` — rerun parsed rows with a missing or invalid task confidence

Rows that only need a fresh judge verdict reuse the stored model answer and rerun the judge alone.

## Scoring and judging

- `objective_choice_discrimination` — exact match on choice label or option text
- `localization_evidence` — exact supporting-fact id list (text only; image localization is not supported)
- other objective types — normalized exact/alias match
- `subjective_generation_integration` (or any `scoring_method` containing "judge") — LLM-as-judge using `scripts/prompts/judge.txt`: the judge returns `{"verdict": "1|0|notsure", "confidence": 0.0}`, verdicts below 0.75 confidence become `notsure`, and `fallback_profiles` chains are tried until a decisive 0/1 verdict
- outputs without valid JSON, or empty answers, are scored incorrect without calling the judge

## Outputs

Each run directory contains:

- `<sample_id>.json` — per-sample result: raw outputs, probe-1 answers, `results_by_perturbation`, and `ignition_threshold_by_perturbation` (first failed level per method)
- `raw_rows.jsonl` — append-only row log used for resume and auditing
- `run_manifest.json` — run status, resolved route/identity, provider history, progress, and resume events
- `run_summary.json` — aggregate summary produced at the end of a run

Aggregate across runs:

```bash
python3 scripts/summarize_results.py --root . --results-dir results --out result_summary.json
```

Judge a single answer standalone:

```bash
python3 scripts/judge_answer.py --meta examples/mini/data/text/text_1/text_1.json --answer "yes"
```

## Model identity

Before the first request, every model route is resolved to a stable identity (canonical id, display name, published source). Resolution order:

1. explicit fields on the `my_api.json` model entry (`canonical_model`, `display_name`, `identity_source`, `identity_status`)
2. the `OFFICIAL_IDENTITIES` table in `model_identity.py`
3. a Hugging Face repo URL derived from a known org namespace (`org/model`)
4. a family catalog URL from `FAMILY_SOURCES`
5. fallback `provider_catalog` / `provider_catalog_only`

All tables ship empty; register your own models either in the tables or per model entry. Quick and named single-model runs reject `provider_catalog_only` identities so unmapped formal runs fail fast — set the identity fields explicitly (as the bundled `example-chat` entry does) or extend the tables.

## Smoke test

`examples/mini/` is a synthetic one-id fixture for CLI checks; it is not an evaluation set. With a reachable endpoint configured in `.env`:

```bash
python3 scripts/run_experiment.py example_provider example-chat text 02468 --root examples/mini --ids 1 --no-resume
```
