# GNWT

GNWT is a benchmark runner for graded-noise robustness experiments on text and image questions. Each sample is sent to an OpenAI-compatible chat completions endpoint at every perturbation level, the model answers through a fixed two-probe JSON schema, answers are scored by rule-based graders or an LLM judge, and per-sample results are aggregated into ignition thresholds: the first perturbation level at which the model fails the task.

## Repository layout

| Path | Role |
|---|---|
| `scripts/run_experiment.py` | Experiment driver |
| `scripts/judge_answer.py` | Graders and LLM judge |
| `scripts/summarize_results.py` | Result aggregation |
| `scripts/api_request_config.py` | Request profile engine |
| `scripts/model_identity.py` | Model identity rules |
| `scripts/my_api.json` | Provider, model, and judge catalog (example) |
| `scripts/api_request_params.json` | Sampling parameters per role |
| `prompts/` | Task and judge templates |
| `examples/mini/` | Synthetic single-sample fixture |

## Install and configure

```bash
python3 -m pip install -r requirements.txt
cp scripts/.env.example scripts/.env
```

Set `GNWT_API_BASE_URL` and `GNWT_API_KEY` in `scripts/.env` to an OpenAI-compatible endpoint. The `.env` file is local only and never committed; `GNWT_JUDGE_API_KEY` optionally sets a separate key for the judge. `scripts/api_request_params.json` holds the sampling parameters for eval and judge calls (temperature, token limit, response format, retries). Registering providers and models in `scripts/my_api.json` is covered in [Model identity](#model-identity).

## Data format

The runner expects a data root with `data/` and `perturbed/` trees; `examples/mini/` shows the minimal shape:

```text
<root>/
  data/
    text/text_1/text_1.json
    text/text_1/text_1_0.txt
    image/image_1/image_1.json
    image/image_1/image_1_0.png
  perturbed/
    text/text_1/delete/text_1_delete_20.txt
    image/image_1/contrast/image_1_contrast_20.png
```

Sample metadata carries the task and the paths:

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

Image samples use `original_sample` and `perturbed_samples` instead of `paths`. Question types are `objective_choice_discrimination`, `objective_fill_recognition`, `judgment_detection`, `localization_evidence`, and `subjective_generation_integration`; each maps to one template in `prompts/`. Perturbation methods are `delete`, `typos`, `antonym` for text and `impulse_noise`, `shot_noise`, `contrast` for image, at levels 0-90 in steps of 10; a typical run uses `02468`. Each model call must return a single JSON object:

```json
{"probe1_answer": "yes", "probe1_confidence": 0.0, "answer": "...", "confidence": 0.0}
```

Probe 1 asks whether the task-relevant information is still accessible in the perturbed input; `answer` is the task response itself.

## Running

```bash
python3 scripts/run_experiment.py example_provider example-chat text 02468 --root examples/mini --ids 1 --no-resume
```

The positional arguments are provider, model, modality, and levels, as registered in `scripts/my_api.json`. The same run can be expressed with `--provider`, `--model`, `--modality`, `--levels`, or with `--profile` to use an experiment profile; `--root` selects the data root. Results are written to `<root>/results/<modality>/<model>/`. An existing run directory is never overwritten; continue it with `--resume`.

## Outputs and scoring

A run directory contains:

- `<sample_id>.json` — per-sample raw outputs, probe-1 answers, and ignition thresholds per method
- `raw_rows.jsonl` — append-only row log
- `run_manifest.json` — run status, resolved route, and progress
- `run_summary.json` — aggregate summary

Objective question types are scored by exact or alias match in `scripts/judge_answer.py`; subjective answers are scored by an LLM judge using `prompts/judge.txt`. To aggregate across runs:

```bash
python3 scripts/summarize_results.py --root . --results-dir results
```

## Model identity

Providers and models are registered in `scripts/my_api.json`:

- `providers` — a named OpenAI-compatible endpoint (`baseUrlEnv`, `apiKeyEnv`) plus its model list
- `experiments` — a named profile binding modality, models, levels, methods, and judge
- `judges` — the judge model, its key env, and an optional ordered `fallback_profiles` chain
- `defaults` — default profile names and fallback worker/timeout values

Each model entry can set `request_model`, `canonical_model`, `display_name`, `identity_source`, and `identity_status`; these fields pin the identity used in results and resume state. The lookup tables in `scripts/model_identity.py` are empty by default, so add your models there or set the fields on each entry. Quick and named single-model runs reject `provider_catalog_only` identities, and unmapped models fail before any request is sent.
