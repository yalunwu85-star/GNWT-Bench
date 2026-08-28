#!/usr/bin/env python3
"""Run GNWT experiments from a packaged gnwtdata root."""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import mimetypes
import os
import re
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from pathlib import Path
from typing import Any

import api_request_config
import judge_answer as judge_answer_module
import model_identity
import summarize_results

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional at import time.
    tqdm = None


TEXT_METHODS = ("delete", "typos", "antonym")
IMAGE_METHODS = ("impulse_noise", "shot_noise", "contrast")
QUESTION_TYPE_PROMPTS = {
    "objective_choice_discrimination": "objective_choice_discrimination.txt",
    "objective_fill_recognition": "objective_fill_recognition.txt",
    "judgment_detection": "judgment_detection.txt",
    "localization_evidence": "localization_evidence.txt",
    "subjective_generation_integration": "subjective_generation_integration.txt",
}
EVIDENCE_LABEL_RE = re.compile(r"^\s*\[([^\]#]+)#(\d+)\]\s*(.*)$")


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8", errors="replace") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False))
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    obj = load_json(path)
    if not isinstance(obj, dict):
        raise SystemExit(f"Config must be a JSON object: {path}")
    return obj


def provider_config(config: dict[str, Any], name: str) -> dict[str, Any]:
    providers = config.get("providers") or {}
    if not isinstance(providers, dict) or name not in providers:
        raise SystemExit(f"Provider not found in config: {name}")
    provider = providers[name]
    if not isinstance(provider, dict):
        raise SystemExit(f"Provider config must be an object: {name}")
    return provider


def resolve_provider(config: dict[str, Any], name: str) -> tuple[str, str]:
    provider = provider_config(config, name)
    base_url_env = provider.get("baseUrlEnv") or provider.get("base_url_env")
    api_key_env = provider.get("apiKeyEnv") or provider.get("api_key_env")
    base_url = (os.environ.get(str(base_url_env), "") if base_url_env else "") or str(provider.get("baseUrl") or provider.get("base_url") or "")
    api_key = (os.environ.get(str(api_key_env), "") if api_key_env else "") or str(provider.get("apiKey") or provider.get("api_key") or "")
    return base_url, api_key


def provider_model_ids(provider: dict[str, Any]) -> set[str]:
    models = provider.get("models") or []
    ids: set[str] = set()
    if isinstance(models, list):
        for model in models:
            if isinstance(model, dict):
                model_id = str(model.get("id") or "").strip()
                if model_id:
                    ids.add(model_id)
            elif isinstance(model, str):
                ids.add(model.strip())
    return ids


def provider_model_entry(provider: dict[str, Any], model_id: str) -> dict[str, Any]:
    models = provider.get("models") or []
    if isinstance(models, list):
        for model in models:
            if isinstance(model, dict) and str(model.get("id") or "").strip() == model_id:
                return model
            if isinstance(model, str) and model.strip() == model_id:
                return {"id": model_id}
        for model in models:
            if not isinstance(model, dict):
                continue
            request_model = str(
                model.get("request_model")
                or model.get("api_model")
                or model.get("name")
                or ""
            ).strip()
            if request_model == model_id:
                return model
    return {}


def matching_model_providers(config: dict[str, Any], model: str) -> list[str]:
    providers = config.get("providers") or {}
    if not isinstance(providers, dict):
        return []
    matches: list[str] = []
    for provider_name, provider in providers.items():
        if isinstance(provider, dict) and provider_model_entry(provider, model):
            matches.append(str(provider_name))
    return matches


def split_provider_model_spec(config: dict[str, Any], model_spec: str) -> tuple[str, str]:
    providers = config.get("providers") or {}
    if isinstance(providers, dict) and "/" in model_spec:
        prefix, suffix = model_spec.split("/", 1)
        if prefix in providers and suffix:
            return prefix, suffix
    return "", model_spec


def resolve_registered_provider_model(config: dict[str, Any], provider_name: str, model_spec: str, label: str) -> tuple[str, str]:
    provider_hint, model = split_provider_model_spec(config, model_spec)
    if provider_hint and provider_name and provider_hint != provider_name:
        raise SystemExit(f"{label} model {model_spec} selects provider {provider_hint}, but provider override is {provider_name}.")
    resolved_provider = provider_hint or provider_name
    if not resolved_provider:
        raise SystemExit(f"{label} override requires a provider. Pass the provider option or use provider/model.")
    if not model:
        raise SystemExit(f"{label} override requires a model.")
    provider = provider_config(config, resolved_provider)
    model_entry = provider_model_entry(provider, model)
    if not model_entry:
        raise SystemExit(f"{label} model {model} is not registered under provider {resolved_provider}.")
    return resolved_provider, str(model_entry.get("id") or model)


def resolve_model_runtime(config: dict[str, Any], model_spec: str, provider_name: str, base_url: str, api_key: str) -> dict[str, Any]:
    provider_hint, model = split_provider_model_spec(config, model_spec)
    if provider_hint and provider_name and provider_hint != provider_name:
        raise SystemExit(f"Model {model_spec} selects provider {provider_hint}, but --provider is {provider_name}.")

    resolved_provider = provider_hint or provider_name
    provider_base_url = ""
    provider_api_key = ""
    if resolved_provider:
        provider = provider_config(config, resolved_provider)
        model_entry = provider_model_entry(provider, model)
        if not model_entry:
            raise SystemExit(f"Model {model} is not registered under provider {resolved_provider}.")
        provider_base_url, provider_api_key = resolve_provider(config, resolved_provider)
    else:
        matches = matching_model_providers(config, model)
        if len(matches) == 1:
            resolved_provider = matches[0]
            provider = provider_config(config, resolved_provider)
            model_entry = provider_model_entry(provider, model)
            provider_base_url, provider_api_key = resolve_provider(config, resolved_provider)
        elif len(matches) > 1:
            raise SystemExit(
                f"Model {model} appears in multiple providers: {', '.join(matches)}. "
                "Pass --provider or use provider/model."
            )
        else:
            raise SystemExit(f"Model {model} is not registered in my_api.json.")

    request_model = str(
        model_entry.get("request_model")
        or model_entry.get("api_model")
        or model_entry.get("name")
        or model
    )
    resolved_model_id = str(model_entry.get("id") or model)
    identity = model_identity.identity_for_request(request_model, model_entry)
    request_overrides = model_entry.get("request_overrides")
    if not isinstance(request_overrides, dict):
        request_overrides = provider.get("request_overrides") if isinstance(provider.get("request_overrides"), dict) else {}
    return {
        "model_spec": f"{resolved_provider}/{resolved_model_id}",
        "model_selector": model_spec,
        "model": resolved_model_id,
        "request_model": request_model,
        **identity,
        "provider": resolved_provider,
        "base_url": base_url or provider_base_url,
        "api_key": api_key or provider_api_key,
        "request_overrides": request_overrides,
    }


def resolve_judge_runtime(config: dict[str, Any], profile: dict[str, Any], timeout: int, default_max_tokens: int = 256) -> dict[str, Any]:
    provider_name = str(profile.get("provider") or "")
    model = str(profile.get("model") or "")
    if provider_name:
        provider = provider_config(config, provider_name)
        model_entry = provider_model_entry(provider, model) if model else {}
        if model and not model_entry:
            raise SystemExit(f"Judge model {model} is not registered under provider {provider_name}.")
        base_url, api_key = resolve_provider(config, provider_name)
    else:
        model_entry = {}
        base_url, api_key = "", ""
    profile_api_key_env = profile.get("apiKeyEnv") or profile.get("api_key_env")
    if profile_api_key_env:
        api_key = os.environ.get(str(profile_api_key_env), "") or api_key
    request_overrides = profile.get("request_overrides")
    if not isinstance(request_overrides, dict):
        request_overrides = model_entry.get("request_overrides")
    if not isinstance(request_overrides, dict):
        request_overrides = provider.get("request_overrides") if provider_name else {}
    if not isinstance(request_overrides, dict):
        request_overrides = {}
    return {
        "provider": provider_name,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "request_model": str(
            model_entry.get("request_model")
            or model_entry.get("api_model")
            or model_entry.get("name")
            or model
        ),
        "max_tokens": int(profile.get("max_tokens") or default_max_tokens),
        "timeout": int(profile.get("timeout") or timeout),
        "request_overrides": request_overrides,
    }


def resolve_judge_chain(
    config: dict[str, Any],
    profile_name: str,
    provider_override: str,
    base_url_override: str,
    api_key_override: str,
    model_override: str,
    timeout: int,
    default_max_tokens: int = 256,
) -> list[dict[str, Any]]:
    if provider_override or base_url_override or api_key_override or model_override:
        provider_name, model = resolve_registered_provider_model(config, provider_override, model_override, "Judge")
        base_url, api_key = resolve_provider(config, provider_name)
        provider = provider_config(config, provider_name)
        model_entry = provider_model_entry(provider, model)
        return [
            {
                "provider": provider_name,
                "base_url": base_url_override or base_url,
                "api_key": api_key_override or api_key,
                "model": model,
                "request_model": str(
                    model_entry.get("request_model")
                    or model_entry.get("api_model")
                    or model_entry.get("name")
                    or model
                ),
                "max_tokens": default_max_tokens,
                "timeout": timeout,
            }
        ]
    if not profile_name:
        return []
    judge_profiles = config.get("judges") or {}
    if not isinstance(judge_profiles, dict) or profile_name not in judge_profiles:
        raise SystemExit(f"Judge profile not found in config: {profile_name}")
    profile = judge_profiles[profile_name]
    if not isinstance(profile, dict):
        raise SystemExit(f"Judge profile must be an object: {profile_name}")
    fallback_profiles = profile.get("fallback_profiles")
    if isinstance(fallback_profiles, list):
        chain: list[dict[str, Any]] = []
        for fallback_name in fallback_profiles:
            fallback_profile = judge_profiles.get(str(fallback_name))
            if not isinstance(fallback_profile, dict):
                raise SystemExit(f"Fallback judge profile not found in config: {fallback_name}")
            chain.append(resolve_judge_runtime(config, fallback_profile, timeout, default_max_tokens))
        return chain
    return [resolve_judge_runtime(config, profile, timeout, default_max_tokens)]


def default_config_path() -> Path:
    return Path(__file__).with_name("my_api.json")


def default_api_params_path() -> Path:
    return api_request_config.default_api_params_path()


def default_env_path() -> Path:
    return Path(__file__).with_name(".env")


def default_prompt_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts"


def default_root_path() -> Path:
    return Path(__file__).resolve().parents[1]


def load_prompt(prompt_dir: Path, filename: str) -> str:
    path = prompt_dir / filename
    if not path.exists():
        raise SystemExit(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def resolve_prompt_filename(question_type: str, prompt_dir: Path) -> str:
    filename = QUESTION_TYPE_PROMPTS.get(question_type)
    if not filename:
        raise SystemExit(f"Unsupported question_type for prompt selection: {question_type}")
    path = prompt_dir / filename
    if not path.exists():
        raise SystemExit(f"Missing prompt file: {path}")
    return filename


def render_prompt(template: str, **values: Any) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", "" if value is None else str(value))
    return rendered


def sample_sort_key(path: Path, modality: str) -> tuple[int, str]:
    match = re.fullmatch(rf"{re.escape(modality)}_(\d+)", path.stem)
    if match:
        return int(match.group(1)), path.stem
    return 10**18, path.stem


def sample_json_paths(root: Path, modality: str) -> list[Path]:
    pattern = root / "data" / modality
    paths = [path for path in pattern.glob(f"{modality}_*/{modality}_*.json") if re.fullmatch(rf"{re.escape(modality)}_\d+", path.stem)]
    return sorted(paths, key=lambda p: sample_sort_key(p, modality))


def safe_model_name(model: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", model.strip()).strip("_")
    return value or "unknown_model"


def parse_id_filters(raw_values: list[str] | None, modality: str) -> set[str] | None:
    if not raw_values:
        return None
    prefix = f"{modality}_"
    selected: set[str] = set()
    for raw_value in raw_values:
        for part in str(raw_value).split(","):
            value = part.strip()
            if not value:
                continue
            if value.isdigit():
                value = f"{prefix}{value}"
            if not re.fullmatch(rf"{re.escape(prefix)}\d+", value):
                raise SystemExit(f"Invalid {modality} id: {value}")
            selected.add(value)
    return selected


def parse_id_bound(raw_value: str | None, modality: str, label: str) -> int | None:
    if raw_value is None or raw_value == "":
        return None
    value = str(raw_value).strip()
    prefix = f"{modality}_"
    if value.startswith(prefix):
        value = value[len(prefix) :]
    if not value.isdigit():
        raise SystemExit(f"Invalid {label}: {raw_value}")
    return int(value)


def parse_level_filters(raw_values: list[str] | None) -> list[int]:
    if not raw_values:
        return [0, 20, 40, 60, 80]
    levels: list[int] = []
    exported_levels = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90}
    for raw_value in raw_values:
        value = str(raw_value).strip()
        if not value:
            continue
        parts = value.split(",")
        if len(parts) == 1 and re.fullmatch(r"[0-9]+", value) and int(value) not in exported_levels:
            parts = list(value)
        for part in parts:
            part = part.strip()
            if not part:
                continue
            if not part.isdigit():
                raise SystemExit(f"Invalid level: {part}")
            level = int(part)
            if 1 <= level <= 9:
                level *= 10
            if level not in exported_levels:
                raise SystemExit(f"Invalid level: {part}")
            levels.append(level)
    deduped: list[int] = []
    seen: set[int] = set()
    for level in levels:
        if level not in seen:
            deduped.append(level)
            seen.add(level)
    if not deduped:
        raise SystemExit("No valid perturbation levels selected.")
    return deduped


def parse_model_filters(raw_values: list[str] | None, fallback_model: str) -> list[str]:
    values: list[str] = []
    if fallback_model:
        values.append(fallback_model)
    for raw_value in raw_values or []:
        for part in str(raw_value).split(","):
            value = part.strip()
            if value:
                values.append(value)
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            deduped.append(value)
            seen.add(value)
    return deduped


def select_samples(
    samples: list[Path],
    selected_ids: set[str] | None,
    id_start: str | None = None,
    id_end: str | None = None,
    modality: str | None = None,
) -> list[Path]:
    if selected_ids is None:
        selected = samples
    else:
        available = {path.stem: path for path in samples}
        missing = sorted(selected_ids - set(available))
        if missing:
            raise SystemExit(f"Requested id(s) not found: {', '.join(missing)}")
        selected = [path for path in samples if path.stem in selected_ids]
    if id_start is None and id_end is None:
        return selected
    if not modality:
        raise SystemExit("modality is required when using id bounds.")
    lower = parse_id_bound(id_start, modality, "--id-start")
    upper = parse_id_bound(id_end, modality, "--id-end")
    if lower is not None and upper is not None and lower > upper:
        raise SystemExit("--id-start cannot be greater than --id-end.")
    bounded: list[Path] = []
    for path in selected:
        number = int(path.stem.split("_")[1])
        if lower is not None and number < lower:
            continue
        if upper is not None and number > upper:
            continue
        bounded.append(path)
    return bounded


def make_run_dir(root: Path, results_dir: str, modality: str, model: str) -> Path:
    return root / results_dir / modality / safe_model_name(model)


def resolve_resume_run_dir(root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        root_relative = root / path
        path = root_relative if root_relative.exists() else Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        raise SystemExit(f"--resume path does not exist: {path}")
    if not path.is_dir():
        raise SystemExit(f"--resume must point to a run directory: {path}")
    return path


def row_job_key(row: dict[str, Any]) -> tuple[str, str, int] | None:
    try:
        return (str(row["id"]), str(row["method"]), int(row["level"]))
    except Exception:
        return None


def row_is_failure(row: dict[str, Any]) -> bool:
    """Return True when a stored row should be rerun with --resume --retry-failures.

    Model-side failures, LLM-judge infrastructure failures, and valid subjective
    answers whose LLM judge remained notsure are retryable.
    """
    failure_reason = str(row.get("failure_reason") or "")
    if failure_reason.startswith("runtime_error:"):
        return True
    if row.get("json_parse_ok") is False:
        return True
    if row.get("response_incomplete") is True:
        return True
    if failure_reason == "no_valid_json":
        return True
    # Legacy empty-answer rows are standard incorrect model responses, not
    # transport/format failures. Current rows no longer carry this reason.
    if failure_reason == "missing_answer":
        return not (row.get("json_parse_ok") is True and row.get("correct") == 0)
    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    judge_method = str(judge.get("judge_method") or row.get("judge_method") or "")
    if judge_answer_module.judge_method_is_infra_failure(judge_method):
        return True
    if row_needs_judge_retry(row):
        return True
    return False


def row_has_output_schema_violation(row: dict[str, Any]) -> bool:
    """Return True when a parsed model answer violates the formal output fields."""
    if row.get("json_parse_ok") is not True or row.get("answer_parse_ok") is False:
        return False
    if str(row.get("probe1_answer") or "").strip().lower() not in {"yes", "no"}:
        return True
    confidence = row.get("confidence")
    return not (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and 0.0 <= float(confidence) <= 1.0
    )


def row_has_missing_confidence(row: dict[str, Any]) -> bool:
    """Return True when a parsed model answer lacks valid task confidence."""
    if row.get("json_parse_ok") is not True or row.get("answer_parse_ok") is False:
        return False
    confidence = row.get("confidence")
    return not (
        isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and 0.0 <= float(confidence) <= 1.0
    )


def row_needs_judge_retry(row: dict[str, Any]) -> bool:
    """Return True when retry should reuse the model answer and only rerun judge."""
    if row.get("json_parse_ok") is not True or row.get("answer_parse_ok") is False:
        return False
    if not str(row.get("answer") or "").strip():
        return False
    judge = row.get("judge") if isinstance(row.get("judge"), dict) else {}
    judge_method = str(judge.get("judge_method") or row.get("judge_method") or "")
    verdict = str(judge.get("judge_verdict") or row.get("judge_verdict") or "").strip().lower()
    normalized_verdict = verdict.replace("_", "").replace("-", "").replace(" ", "")
    return (
        judge_method == "llm_as_judge"
        and row.get("correct") is None
        and normalized_verdict == "notsure"
    )


def result_job_keys(result: dict[str, Any]) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    sid = str(result.get("id") or "")
    for row in result.get("model_raw_outputs") or []:
        if not isinstance(row, dict):
            continue
        key = row_job_key({**row, "id": row.get("id") or sid})
        if key:
            keys.add(key)
    return keys


def count_result_failures(results: dict[str, dict[str, Any]], selected_keys: set[tuple[str, str, int]]) -> int:
    return len(result_failure_rows(results, selected_keys))


def result_failure_rows(
    results: dict[str, dict[str, Any]],
    selected_keys: set[tuple[str, str, int]],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    failures: dict[tuple[str, str, int], dict[str, Any]] = {}
    for sid, result in results.items():
        for row in result.get("model_raw_outputs") or []:
            if not isinstance(row, dict):
                continue
            row_with_id = {**row, "id": row.get("id") or sid}
            key = row_job_key(row_with_id)
            if key in selected_keys and row_is_failure(row_with_id):
                failures[key] = row_with_id
    return failures


def plan_resume_jobs(
    jobs: list[tuple[Path, str, int]],
    results: dict[str, dict[str, Any]],
    completed_keys: set[tuple[str, str, int]],
    retry_failures: bool,
    retry_schema_invalid: bool = False,
    retry_missing_confidence: bool = False,
) -> tuple[list[tuple[Path, str, int]], int, int, dict[tuple[str, str, int], dict[str, Any]]]:
    selected_keys = {(path.stem, method, int(level)) for path, method, level in jobs}
    existing_failures = result_failure_rows(results, selected_keys)
    retry_rows = dict(existing_failures) if retry_failures else {}
    if retry_schema_invalid:
        for sid, result in results.items():
            for row in result.get("model_raw_outputs") or []:
                if not isinstance(row, dict):
                    continue
                row_with_id = {**row, "id": row.get("id") or sid}
                key = row_job_key(row_with_id)
                if key in selected_keys and row_has_output_schema_violation(row_with_id):
                    retry_rows[key] = row_with_id
    if retry_missing_confidence:
        for sid, result in results.items():
            for row in result.get("model_raw_outputs") or []:
                if not isinstance(row, dict):
                    continue
                row_with_id = {**row, "id": row.get("id") or sid}
                key = row_job_key(row_with_id)
                if key in selected_keys and row_has_missing_confidence(row_with_id):
                    retry_rows[key] = row_with_id
    pending = [
        (path, method, level)
        for path, method, level in jobs
        if (path.stem, method, int(level)) not in completed_keys
        or (path.stem, method, int(level)) in retry_rows
    ]
    completed_rows = len(selected_keys & completed_keys) - len(retry_rows)
    failure_count = len(set(existing_failures) - set(retry_rows))
    return pending, completed_rows, failure_count, retry_rows


def load_resume_state(
    root: Path,
    run_dir: Path,
    modality: str,
    canonical_model: str,
    display_name: str,
) -> tuple[dict[str, dict[str, Any]], set[str], set[tuple[str, str, int]]]:
    results: dict[str, dict[str, Any]] = {}
    dirty_ids: set[str] = set()

    for path in sorted(run_dir.glob("*.json")):
        if path.name in {"run_manifest.json", "run_summary.json"}:
            continue
        try:
            result = load_json(path)
        except Exception:
            continue
        if not isinstance(result, dict):
            continue
        sid = str(result.get("id") or path.stem)
        if not sid:
            continue
        previous_model = str(result.get("model") or "")
        if previous_model and previous_model != canonical_model:
            result.setdefault("legacy_model", previous_model)
        result["model"] = canonical_model
        result["canonical_model"] = canonical_model
        result["display_name"] = display_name
        results[sid] = result

    raw_rows_path = run_dir / "raw_rows.jsonl"
    if raw_rows_path.exists():
        with raw_rows_path.open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                key = row_job_key(row)
                if not key:
                    continue
                sid = key[0]
                if sid not in results:
                    meta_path = root / "data" / modality / sid / f"{sid}.json"
                    if not meta_path.exists():
                        continue
                    meta = load_json(meta_path)
                    meta["modality"] = modality
                    results[sid] = empty_result(meta, canonical_model, display_name)
                before = len(result_job_keys(results[sid]))
                merge_result_row(results[sid], row)
                update_thresholds(results[sid])
                after = len(result_job_keys(results[sid]))
                if after >= before:
                    dirty_ids.add(sid)

    written_ids = set(results)
    completed_keys: set[tuple[str, str, int]] = set()
    for result in results.values():
        completed_keys.update(result_job_keys(result))

    for sid in dirty_ids:
        write_json(run_dir / f"{sid}.json", results[sid])

    return results, written_ids, completed_keys


def validate_resume_manifest(
    run_dir: Path,
    modality: str,
    canonical_model: str,
    inference_variant: str,
    model_spec: str,
) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise SystemExit(f"Resume manifest must be a JSON object: {manifest_path}")
    existing_modality = str(manifest.get("modality") or "")
    if existing_modality and existing_modality != modality:
        raise SystemExit(f"--resume modality mismatch: manifest has {existing_modality}, current run is {modality}.")
    existing_model = str(manifest.get("canonical_model") or manifest.get("model") or "")
    existing_request_model = str(manifest.get("request_model") or manifest.get("model") or "")
    existing_canonical = model_identity.canonicalize_request_model(existing_model or existing_request_model)
    if existing_canonical and existing_canonical != canonical_model:
        raise SystemExit(
            f"--resume model mismatch: manifest is {existing_canonical}, "
            f"current run is {canonical_model} ({model_spec})."
        )
    existing_variant = str(manifest.get("inference_variant") or "") or model_identity.inference_variant(existing_request_model)
    if existing_variant and inference_variant and existing_variant != inference_variant:
        raise SystemExit(
            f"--resume inference variant mismatch: manifest has {existing_variant}, current route has {inference_variant}."
        )
    return manifest


def image_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def image_payload_for_request(path: Path, max_base64_bytes: int = 0) -> tuple[str, dict[str, Any]]:
    raw = path.read_bytes()
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(raw)
    metadata: dict[str, Any] = {
        "source_path": str(path),
        "source_mime_type": mime,
        "source_bytes": len(raw),
        "transmitted_mime_type": mime,
        "transmitted_bytes": len(raw),
        "transmitted_base64_bytes": len(encoded),
        "compressed_for_provider_limit": False,
    }
    if max_base64_bytes <= 0 or len(encoded) <= max_base64_bytes:
        return f"data:{mime};base64,{encoded.decode('ascii')}", metadata

    from PIL import Image

    with Image.open(path) as source_image:
        image = source_image.convert("RGB")
        for quality in (95, 92, 90, 88, 85, 82, 80, 75, 70, 65, 60):
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=quality, optimize=True)
            payload = buffer.getvalue()
            encoded = base64.b64encode(payload)
            if len(encoded) <= max_base64_bytes:
                metadata.update(
                    {
                        "transmitted_mime_type": "image/jpeg",
                        "transmitted_bytes": len(payload),
                        "transmitted_base64_bytes": len(encoded),
                        "compressed_for_provider_limit": True,
                        "jpeg_quality": quality,
                        "pixel_dimensions": list(image.size),
                        "max_base64_bytes": max_base64_bytes,
                    }
                )
                return f"data:image/jpeg;base64,{encoded.decode('ascii')}", metadata
    raise ValueError(
        f"Cannot compress image below provider base64 limit: {path} "
        f"({metadata['transmitted_base64_bytes']} > {max_base64_bytes})"
    )


def chat_completions_url(base_url: str) -> str:
    return api_request_config.chat_completions_url(base_url)


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int | None,
    timeout: int,
    request_profile: dict[str, Any] | None = None,
    request_profile_name: str = "",
) -> dict[str, Any]:
    return api_request_config.chat_completion(
        base_url,
        api_key,
        model,
        messages,
        max_tokens=max_tokens,
        timeout=timeout,
        profile=request_profile,
        profile_name=request_profile_name,
    )


def extract_message_content(response: dict[str, Any]) -> str:
    return str(response.get("choices", [{}])[0].get("message", {}).get("content") or "")


def parse_confidence(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number < 0:
        number = 0.0
    if number > 1:
        number = 1.0
    return round(number, 1)


def first_present(obj: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in obj and obj[key] is not None:
            return obj[key]
    return None


def extract_json_object(content: str) -> dict[str, Any] | None:
    text = str(content or "").strip()
    candidates = [text]
    for match in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE):
        candidates.insert(0, match.strip())
    for start in [idx for idx, ch in enumerate(text) if ch == "{"]:
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            ch = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : end + 1])
                        break
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and start < end:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def parse_model_json(content: str) -> dict[str, Any]:
    """Parse the one-call two-probe JSON output.

    The target schema is:
      {"probe1_answer":"yes|no", "probe1_confidence":0.0,
       "answer":"...", "confidence":0.0}

    Valid partial JSON is retained. Only outputs without any valid JSON object
    are treated as no_valid_json.
    """
    try:
        obj = extract_json_object(content)
        if not obj:
            raise ValueError("model output JSON object not found")

        probe1_value = first_present(obj, ("probe1_answer", "probe_1_answer", "probe1", "report_answer", "report_choice", "perception_answer"))
        probe1_answer = str(probe1_value or "").strip().lower()
        if probe1_answer in {"y", "yes", "true", "1"}:
            probe1_answer = "yes"
        elif probe1_answer in {"n", "no", "false", "0"}:
            probe1_answer = "no"

        probe1_confidence = parse_confidence(first_present(obj, ("probe1_confidence", "probe_1_confidence", "report_confidence", "perception_confidence")))

        answer_value = first_present(obj, ("answer", "performance_answer", "final_answer", "task_answer", "response", "result"))
        answer = str(answer_value).strip() if answer_value is not None else ""

        confidence = parse_confidence(first_present(obj, ("confidence", "performance_confidence", "answer_confidence", "final_confidence", "task_confidence")))

        return {
            "answer": answer,
            "confidence": confidence,
            "probe1_answer": probe1_answer,
            "probe1_confidence": probe1_confidence,
            "json_parse_ok": True,
            "answer_parse_ok": True,
            "answer_parse_error": "",
        }
    except Exception as exc:
        return {
            "answer": "",
            "confidence": None,
            "probe1_answer": "",
            "probe1_confidence": None,
            "json_parse_ok": False,
            "answer_parse_ok": False,
            "answer_parse_error": str(exc),
        }


def sample_input_path(root: Path, meta: dict[str, Any], modality: str, method: str, level: int) -> Path:
    sid = meta["id"]
    if level == 0:
        if modality == "text":
            return root / meta["paths"]["original_text"]
        return root / meta["original_sample"]
    if modality == "text":
        return root / meta["paths"]["perturbations"][method] / f"{sid}_{method}_{level}.txt"
    return root / meta["perturbed_samples"][method] / f"{sid}_{method}_{level}.png"


def format_options(options: Any) -> str:
    if not options:
        return ""
    if isinstance(options, list) and options and all(isinstance(item, tuple) and len(item) == 2 for item in options):
        values = [f"{label}. {value}" for label, value in options]
    elif isinstance(options, dict):
        values = [f"{key}: {value}" for key, value in options.items()]
    elif isinstance(options, list):
        values = [str(item) for item in options]
    else:
        values = [str(options)]
    return "Options:\n" + "\n".join(f"- {item}" for item in values)


def localization_line_plan(clean_text: str) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for line in clean_text.splitlines():
        stripped = line.strip()
        if not stripped:
            plan.append({"kind": "blank"})
            continue
        match = EVIDENCE_LABEL_RE.match(line)
        if match:
            plan.append({"kind": "sentence", "title": match.group(1).strip(), "sent_id": int(match.group(2))})
        else:
            plan.append({"kind": "title", "title": stripped})
    return plan


def strip_evidence_label(line: str) -> str:
    match = EVIDENCE_LABEL_RE.match(line)
    if match:
        return match.group(3).strip()
    return line.strip()


def format_localization_input(clean_text: str, current_text: str) -> str:
    plan = localization_line_plan(clean_text)
    if not any(item.get("kind") == "sentence" for item in plan):
        return current_text
    clean_lines = clean_text.splitlines()
    current_lines = current_text.splitlines()
    rendered: list[str] = []
    for idx, item in enumerate(plan):
        kind = item.get("kind")
        if kind == "blank":
            rendered.append("")
        elif kind == "title":
            rendered.append(f"[{item['title']}]")
        elif kind == "sentence":
            current_line = current_lines[idx] if idx < len(current_lines) else ""
            clean_line = clean_lines[idx] if idx < len(clean_lines) else ""
            body = strip_evidence_label(current_line) or strip_evidence_label(clean_line)
            rendered.append(f"[{item['title']}#{item['sent_id']}] {body}")
    return "\n".join(rendered)


def build_messages(
    root: Path,
    meta: dict[str, Any],
    modality: str,
    input_path: Path,
    prompt_dir: Path,
    max_image_base64_bytes: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question = meta.get("question") or ""
    gt = meta.get("gt") if isinstance(meta.get("gt"), dict) else {}
    question_type = str(meta.get("question_type") or "")
    options = judge_answer_module.choice_options(meta) if question_type == "objective_choice_discrimination" else gt.get("options") or meta.get("options") or []
    if modality == "image" and question_type == "localization_evidence":
        raise SystemExit("Image localization tasks are not supported in this release. Remove image samples with question_type=localization_evidence.")
    prompt_filename = resolve_prompt_filename(question_type, prompt_dir)
    template = load_prompt(prompt_dir, prompt_filename)
    if modality == "text":
        text = input_path.read_text(encoding="utf-8", errors="replace")
        if question_type == "localization_evidence":
            clean_text = (root / meta["paths"]["original_text"]).read_text(encoding="utf-8", errors="replace")
            text = format_localization_input(clean_text, text)
        prompt = render_prompt(
            template,
            modality=modality,
            input_block=f"Input text:\n{text}",
            question=question,
            options_block=format_options(options),
        )
        return [{"role": "user", "content": prompt}], {}
    prompt = render_prompt(
        template,
        modality=modality,
        input_block="Input image: attached.",
        question=question,
        options_block=format_options(options),
    )
    image_url, input_transport = image_payload_for_request(input_path, max_image_base64_bytes)
    content: list[dict[str, Any]] = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": image_url}},
    ]
    return [{"role": "user", "content": content}], input_transport


def empty_result(meta: dict[str, Any], canonical_model: str, display_name: str = "") -> dict[str, Any]:
    modality = str(meta.get("modality") or ("text" if str(meta.get("id", "")).startswith("text_") else "image"))
    methods = TEXT_METHODS if modality == "text" else IMAGE_METHODS
    return {
        "id": meta["id"],
        "modality": modality,
        "model": canonical_model,
        "canonical_model": canonical_model,
        "display_name": display_name or model_identity.fallback_display_name(canonical_model),
        "threshold_definition": "first_failed_perturbation_level",
        "model_raw_outputs": [],
        "results_by_perturbation": {m: {} for m in methods},
        "probe1_by_perturbation": {m: {} for m in methods},
        "ignition_threshold_by_perturbation": {},
        "last_correct_level_by_perturbation": {},
        "ignition_threshold_status_by_perturbation": {},
    }


def update_thresholds(result: dict[str, Any]) -> None:
    threshold_by_method = result.setdefault("ignition_threshold_by_perturbation", {})
    last_correct_by_method = result.setdefault("last_correct_level_by_perturbation", {})
    status_by_method = result.setdefault("ignition_threshold_status_by_perturbation", {})
    for method, values in result.get("results_by_perturbation", {}).items():
        first_failure = None
        last_correct = None
        sorted_levels = sorted(int(k) for k in values)
        for level in sorted_levels:
            correct = values[str(level)]
            if correct == 1:
                last_correct = level
            elif correct == 0 and first_failure is None:
                first_failure = level
        threshold_by_method[method] = first_failure
        last_correct_by_method[method] = last_correct
        if not sorted_levels:
            status_by_method[method] = "no_judged_levels"
        elif first_failure is None:
            status_by_method[method] = "not_reached"
        else:
            status_by_method[method] = "first_failure"


def merge_result_row(result: dict[str, Any], row: dict[str, Any]) -> None:
    method = str(row["method"])
    level = str(row["level"])
    raw_outputs = result.setdefault("model_raw_outputs", [])
    replacement_index = None
    for idx, existing in enumerate(raw_outputs):
        if str(existing.get("method")) == method and str(existing.get("level")) == level:
            replacement_index = idx
            break
    if replacement_index is None:
        raw_outputs.append(row)
    else:
        raw_outputs[replacement_index] = row

    probe_levels = result.setdefault("probe1_by_perturbation", {}).setdefault(method, {})
    probe_levels.pop(level, None)
    if row.get("probe1_answer"):
        probe_levels[level] = {
            "answer": row.get("probe1_answer"),
            "confidence": row.get("probe1_confidence"),
        }

    result_levels = result.setdefault("results_by_perturbation", {}).setdefault(method, {})
    result_levels.pop(level, None)
    if row["correct"] is not None:
        result_levels[level] = int(row["correct"])


def runtime_failure_row(meta_path: Path, modality: str, method: str, level: int, exc: Exception) -> dict[str, Any]:
    reason = f"runtime_error:{type(exc).__name__}: {exc}"
    return {
        "id": meta_path.stem,
        "modality": modality,
        "method": method,
        "level": int(level),
        "input_path": "",
        "probe1_answer": "",
        "probe1_confidence": None,
        "answer": "",
        "confidence": None,
        "json_parse_ok": False,
        "answer_parse_ok": False,
        "answer_parse_error": reason,
        "failure_reason": reason,
        "finish_reason": "",
        "response_incomplete": True,
        "api_request": {},
        "correct": 0,
        "judge": {"correct": 0, "judge_method": "runtime_error", "error": repr(exc)},
        "raw_output": "",
        "raw_response": {"error": repr(exc)},
    }


def validate_sample_metadata(root: Path, meta_path: Path, modality: str, methods: list[str], levels: list[int], prompt_dir: Path) -> list[str]:
    errors: list[str] = []
    try:
        meta = load_json(meta_path)
    except Exception as exc:
        return [f"{meta_path}: cannot read JSON: {exc!r}"]
    sid = str(meta.get("id") or meta_path.stem)
    if sid != meta_path.stem:
        errors.append(f"{sid}: metadata id does not match filename {meta_path.stem}")
    question_type = str(meta.get("question_type") or "")
    if question_type not in QUESTION_TYPE_PROMPTS:
        errors.append(f"{sid}: unsupported or missing question_type: {question_type}")
    else:
        prompt_path = prompt_dir / QUESTION_TYPE_PROMPTS[question_type]
        if not prompt_path.exists():
            errors.append(f"{sid}: missing prompt file: {prompt_path}")
    if not str(meta.get("question") or "").strip():
        errors.append(f"{sid}: missing question")
    if modality == "text":
        original_rel = (meta.get("paths") or {}).get("original_text")
        if not original_rel:
            errors.append(f"{sid}: missing paths.original_text")
        elif not (root / str(original_rel)).exists():
            errors.append(f"{sid}: missing original text file: {original_rel}")
        perturb_paths = (meta.get("paths") or {}).get("perturbations") or {}
        for method in methods:
            if method not in perturb_paths and any(level != 0 for level in levels):
                errors.append(f"{sid}: missing perturbation path for method {method}")
            for level in levels:
                if level == 0:
                    continue
                rel_dir = perturb_paths.get(method)
                if rel_dir and not (root / str(rel_dir) / f"{sid}_{method}_{level}.txt").exists():
                    errors.append(f"{sid}: missing perturbed text: {rel_dir}/{sid}_{method}_{level}.txt")
    else:
        original_rel = meta.get("original_sample")
        if not original_rel:
            errors.append(f"{sid}: missing original_sample")
        elif not (root / str(original_rel)).exists():
            errors.append(f"{sid}: missing original image file: {original_rel}")
        if question_type == "localization_evidence":
            errors.append(f"{sid}: image localization_evidence is not supported")
        perturb_paths = meta.get("perturbed_samples") or {}
        for method in methods:
            if method not in perturb_paths and any(level != 0 for level in levels):
                errors.append(f"{sid}: missing perturbed_samples path for method {method}")
            for level in levels:
                if level == 0:
                    continue
                rel_dir = perturb_paths.get(method)
                if rel_dir and not (root / str(rel_dir) / f"{sid}_{method}_{level}.png").exists():
                    errors.append(f"{sid}: missing perturbed image: {rel_dir}/{sid}_{method}_{level}.png")
    return errors


def validate_experiment_inputs(root: Path, samples: list[Path], modality: str, methods: list[str], levels: list[int], prompt_dir: Path) -> None:
    errors: list[str] = []
    for meta_path in samples:
        errors.extend(validate_sample_metadata(root, meta_path, modality, methods, levels, prompt_dir))
        if len(errors) >= 50:
            break
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:50])
        suffix = "\n..." if len(errors) >= 50 else ""
        raise SystemExit(f"Preflight validation failed:\n{preview}{suffix}")


def judge_with_chain(answer: str, meta: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    chain = getattr(args, "judge_chain", None)
    if isinstance(chain, list) and chain:
        return judge_answer_module.judge_answer_with_chain(
            answer,
            meta,
            chain,
            timeout=args.timeout,
            prompt_dir=Path(args.prompt_dir),
            api_request_profile=getattr(args, "judge_api_request_profile", None),
            api_request_profile_name=getattr(args, "judge_api_request_profile_name", ""),
        )
    return judge_answer_module.judge_answer(
        answer,
        meta,
        judge_base_url=args.judge_base_url or None,
        judge_api_key=args.judge_api_key or None,
        judge_model=args.judge_model or None,
        judge_max_tokens=getattr(args, "judge_max_tokens", 256),
        timeout=args.timeout,
        prompt_dir=Path(args.prompt_dir),
        api_request_profile=getattr(args, "judge_api_request_profile", None),
        api_request_profile_name=getattr(args, "judge_api_request_profile_name", ""),
    )


def run_one(root: Path, meta_path: Path, modality: str, method: str, level: int, args: argparse.Namespace) -> dict[str, Any]:
    meta = load_json(meta_path)
    input_path = sample_input_path(root, meta, modality, method, level)
    messages, input_transport = build_messages(
        root,
        meta,
        modality,
        input_path,
        Path(args.prompt_dir),
        int(getattr(args, "max_image_base64_bytes", 0) or 0),
    )
    response = chat_completion(
        args.base_url,
        args.api_key,
        args.model,
        messages,
        args.max_tokens,
        args.timeout,
        getattr(args, "api_request_profile", None),
        getattr(args, "api_request_profile_name", ""),
    )
    raw_content = extract_message_content(response)
    parsed = parse_model_json(raw_content)
    answer = parsed["answer"]
    if not parsed.get("json_parse_ok"):
        judged = {
            "correct": 0,
            "judge_method": "model_output_json_parse_failed",
            "judge_skipped": True,
            "failure_reason": "no_valid_json",
            "parse_error": parsed.get("answer_parse_error", ""),
        }
    elif not str(answer or "").strip():
        judged = {
            "correct": 0,
            "judge_method": "empty_answer_scored_incorrect",
            "judge_skipped": True,
            "empty_answer_policy": "score_incorrect",
        }
    elif parsed.get("answer_parse_ok"):
        judged = judge_with_chain(answer, meta, args)
    else:
        judged = {
            "correct": 0,
            "judge_method": "model_output_missing_answer",
            "judge_skipped": True,
            "failure_reason": "missing_answer",
            "parse_error": parsed.get("answer_parse_error", ""),
        }
    return {
        "id": meta["id"],
        "modality": modality,
        "model": getattr(args, "canonical_model", args.model),
        "canonical_model": getattr(args, "canonical_model", args.model),
        "display_name": getattr(args, "display_name", args.model),
        "model_spec": getattr(args, "model_spec", args.model),
        "model_route_id": getattr(args, "model_route_id", args.model),
        "request_model": args.model,
        "provider": getattr(args, "provider", ""),
        "inference_variant": getattr(args, "inference_variant", "default"),
        "route_variant": getattr(args, "route_variant", "standard"),
        "method": method,
        "level": level,
        "input_path": str(input_path.relative_to(root)),
        "input_transport": input_transport,
        "probe1_answer": parsed.get("probe1_answer", ""),
        "probe1_confidence": parsed.get("probe1_confidence"),
        "answer": answer,
        "confidence": parsed.get("confidence"),
        "json_parse_ok": parsed.get("json_parse_ok", False),
        "answer_parse_ok": parsed.get("answer_parse_ok", False),
        "answer_parse_error": parsed.get("answer_parse_error", ""),
        "empty_answer": bool(parsed.get("json_parse_ok") and not str(answer or "").strip()),
        "empty_answer_policy": judged.get("empty_answer_policy", ""),
        "failure_reason": judged.get("failure_reason", ""),
        "finish_reason": api_request_config.first_choice_finish_reason(response),
        "response_incomplete": api_request_config.response_incomplete(response),
        "api_request": response.get("_gnwt_request", {}),
        "correct": judged.get("correct"),
        "judge": judged,
        "raw_output": raw_content,
        "raw_response": response,
    }


def retry_judge_for_existing_row(
    meta_path: Path,
    modality: str,
    method: str,
    level: int,
    args: argparse.Namespace,
    previous_row: dict[str, Any],
) -> dict[str, Any]:
    """Rerun the configured judge chain without requesting the evaluated model."""
    meta = load_json(meta_path)
    judged = judge_with_chain(str(previous_row.get("answer") or ""), meta, args)
    previous_judge = previous_row.get("judge") if isinstance(previous_row.get("judge"), dict) else {}
    row = dict(previous_row)
    row.update(
        {
            "id": str(meta["id"]),
            "modality": modality,
            "method": method,
            "level": int(level),
            "correct": judged.get("correct"),
            "judge": judged,
            "failure_reason": judged.get("failure_reason", ""),
            "retry_mode": "judge_only",
            "retry_of_failure": True,
            "previous_judge_method": previous_judge.get("judge_method", ""),
            "previous_judge_verdict": previous_judge.get("judge_verdict", ""),
            "previous_judge_model": previous_judge.get("judge_model", ""),
        }
    )
    return row


def run_model(
    root: Path,
    samples: list[Path],
    methods: list[str],
    args: argparse.Namespace,
    model: str,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_args = argparse.Namespace(**vars(args))
    request_model = runtime.get("request_model", model) if runtime else model
    canonical_model = runtime.get("canonical_model", model) if runtime else model_identity.canonicalize_request_model(model)
    display_name = runtime.get("display_name", canonical_model) if runtime else model_identity.fallback_display_name(canonical_model)
    inference_variant = runtime.get("inference_variant", "default") if runtime else model_identity.inference_variant(request_model)
    route_variant = runtime.get("route_variant", "standard") if runtime else model_identity.route_variant(request_model)
    run_args.model = request_model
    if runtime:
        run_args.provider = runtime.get("provider", run_args.provider)
        run_args.base_url = runtime.get("base_url", run_args.base_url)
        run_args.api_key = runtime.get("api_key", run_args.api_key)
        request_overrides = runtime.get("request_overrides")
        if isinstance(request_overrides, dict) and request_overrides:
            run_args.api_request_profile = copy.deepcopy(getattr(run_args, "api_request_profile", {}) or {})
            run_args.api_request_profile["request_overrides"] = copy.deepcopy(request_overrides)
    model_spec = runtime.get("model_spec", model) if runtime else model
    run_args.model_spec = model_spec
    run_args.model_route_id = model
    run_args.canonical_model = canonical_model
    run_args.display_name = display_name
    run_args.inference_variant = inference_variant
    run_args.route_variant = route_variant
    resume_run_dir = getattr(run_args, "resume_run_dir", None)
    resume_mode = bool(getattr(run_args, "resume_requested", False))
    if resume_mode:
        run_dir = resume_run_dir if isinstance(resume_run_dir, Path) else make_run_dir(root, run_args.results_dir, run_args.modality, canonical_model)
        run_dir.mkdir(parents=True, exist_ok=True)
        previous_manifest = validate_resume_manifest(run_dir, run_args.modality, canonical_model, inference_variant, model_spec)
    else:
        run_dir = make_run_dir(root, run_args.results_dir, run_args.modality, canonical_model)
        if run_dir.exists() and any(run_dir.iterdir()):
            rel_run_dir = run_dir.relative_to(root) if run_dir.is_relative_to(root) else run_dir
            raise SystemExit(f"Results directory already exists: {rel_run_dir}. Use --resume to continue it.")
        run_dir.mkdir(parents=True, exist_ok=True)
        previous_manifest = {}
    jobs = [(p, m, l) for p in samples for m in methods for l in run_args.levels]
    manifest_path = run_dir / "run_manifest.json"
    raw_rows_path = run_dir / "raw_rows.jsonl"
    if resume_mode:
        results, written_ids, completed_keys = load_resume_state(root, run_dir, run_args.modality, canonical_model, display_name)
    else:
        results = {}
        written_ids = set()
        completed_keys = set()
    retry_failures = bool(getattr(run_args, "retry_failures", False))
    retry_schema_invalid = bool(getattr(run_args, "retry_schema_invalid", False))
    retry_missing_confidence = bool(getattr(run_args, "retry_missing_confidence", False))
    pending_jobs, completed_rows, failure_count, retry_failure_rows = plan_resume_jobs(
        jobs,
        results,
        completed_keys,
        resume_mode and retry_failures,
        resume_mode and retry_schema_invalid,
        resume_mode and retry_missing_confidence,
    )
    now = datetime.now().isoformat(timespec="seconds")
    manifest: dict[str, Any] = dict(previous_manifest)
    previous_route_record = None
    if previous_manifest:
        previous_route_record = {
            "provider": previous_manifest.get("provider", ""),
            "model_spec": previous_manifest.get("model_spec", previous_manifest.get("model", "")),
            "model_route_id": previous_manifest.get("model_route_id", previous_manifest.get("model", "")),
            "request_model": previous_manifest.get("request_model", previous_manifest.get("model", "")),
            "inference_variant": previous_manifest.get("inference_variant", model_identity.inference_variant(str(previous_manifest.get("request_model") or previous_manifest.get("model") or ""))),
            "route_variant": previous_manifest.get("route_variant", model_identity.route_variant(str(previous_manifest.get("request_model") or previous_manifest.get("model") or ""))),
            "used_at": previous_manifest.get("updated_at", previous_manifest.get("created_at", "")),
        }
    manifest.update(
        {
            "run_id": run_dir.name,
            "status": "running",
            "updated_at": now,
            "root": str(root),
            "results_dir": str(Path(run_args.results_dir) / run_args.modality / safe_model_name(canonical_model)),
            "run_dir": str(run_dir.relative_to(root) if run_dir.is_relative_to(root) else run_dir),
            "modality": run_args.modality,
            "model": canonical_model,
            "canonical_model": canonical_model,
            "display_name": display_name,
            "model_route_id": model,
            "model_spec": model_spec,
            "request_model": request_model,
            "inference_variant": inference_variant,
            "route_variant": route_variant,
            "model_safe_name": safe_model_name(canonical_model),
            "threshold_definition": "first_failed_perturbation_level",
            "profile": run_args.profile,
            "provider": run_args.provider,
            "api_params_profile": getattr(run_args, "api_request_profile_name", ""),
            "api_params": api_request_config.safe_payload_params(
                api_request_config.build_chat_payload(
                    request_model,
                    [],
                    getattr(run_args, "api_request_profile", {}) or {},
                    max_tokens=run_args.max_tokens,
                )
            ),
            "judge_api_params_profile": getattr(run_args, "judge_api_request_profile_name", ""),
            "selected_ids": [path.stem for path in samples],
            "methods": methods,
            "levels": run_args.levels,
            "jobs_total": len(jobs),
            "completed_rows": completed_rows,
            "failure_count": failure_count,
            "existing_completed_rows": completed_rows if resume_mode else 0,
            "remaining_rows": len(pending_jobs),
            "resume_run": resume_mode,
            "retry_failures": retry_failures,
            "retry_schema_invalid": retry_schema_invalid,
            "retry_missing_confidence": retry_missing_confidence,
            "retry_failure_rows_scheduled": len(retry_failure_rows),
            "written_ids_count": len(written_ids),
            "result_files": [f"{sid}.json" for sid in sorted(written_ids)],
            "raw_rows_path": str(raw_rows_path.relative_to(root) if raw_rows_path.is_relative_to(root) else raw_rows_path),
        }
    )
    manifest.setdefault("created_at", now)
    route_record = {
        "provider": run_args.provider,
        "model_spec": model_spec,
        "model_route_id": model,
        "request_model": request_model,
        "inference_variant": inference_variant,
        "route_variant": route_variant,
        "used_at": now,
    }
    history = manifest.setdefault("provider_history", [])
    if isinstance(history, list) and previous_route_record and not history:
        history.append(previous_route_record)
    if isinstance(history, list) and not any(
        isinstance(item, dict)
        and item.get("provider") == route_record["provider"]
        and item.get("request_model") == route_record["request_model"]
        for item in history
    ):
        history.append(route_record)
    if resume_mode:
        manifest.pop("completed_at", None)
        events = manifest.setdefault("resume_events", [])
        if isinstance(events, list):
            events.append(
                {
                    "resumed_at": now,
                    "remaining_rows": len(pending_jobs),
                    "completed_rows": completed_rows,
                    "canonical_model": canonical_model,
                    "provider": run_args.provider,
                    "model_spec": model_spec,
                    "request_model": request_model,
                    "retry_failures": retry_failures,
                    "retry_schema_invalid": retry_schema_invalid,
                    "retry_missing_confidence": retry_missing_confidence,
                    "retry_failure_rows_scheduled": len(retry_failure_rows),
                }
            )
    write_json(manifest_path, manifest)

    progress = None
    if tqdm is not None and sys.stderr.isatty():
        progress = tqdm(
            total=len(jobs),
            initial=completed_rows,
            desc=f"{run_args.modality}:{safe_model_name(canonical_model)}",
            unit="row",
            dynamic_ncols=True,
        )
        progress.set_postfix_str(f"failures={failure_count}", refresh=False)
    try:
        with ThreadPoolExecutor(max_workers=run_args.workers) as ex:
            job_iter = iter(pending_jobs)
            in_flight: dict[Any, tuple[Path, str, int]] = {}
            max_in_flight = max(1, run_args.workers)

            def submit_next() -> bool:
                try:
                    p, m, l = next(job_iter)
                except StopIteration:
                    return False
                job_key = (p.stem, m, int(l))
                previous_row = retry_failure_rows.get(job_key)
                if previous_row is not None and row_needs_judge_retry(previous_row):
                    future = ex.submit(
                        retry_judge_for_existing_row,
                        p,
                        run_args.modality,
                        m,
                        l,
                        run_args,
                        previous_row,
                    )
                else:
                    future = ex.submit(run_one, root, p, run_args.modality, m, l, run_args)
                in_flight[future] = (p, m, int(l))
                return True

            for _ in range(min(max_in_flight, len(pending_jobs))):
                submit_next()

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for fut in done:
                    job = in_flight.pop(fut, None)
                    if job is None:
                        continue
                    job_path, job_method, job_level = job
                    try:
                        row = fut.result()
                    except Exception as exc:
                        row = runtime_failure_row(job_path, run_args.modality, job_method, job_level, exc)
                    row.setdefault("model", canonical_model)
                    row.setdefault("canonical_model", canonical_model)
                    row.setdefault("display_name", display_name)
                    row.setdefault("model_spec", model_spec)
                    row.setdefault("model_route_id", model)
                    row.setdefault("request_model", request_model)
                    row.setdefault("provider", run_args.provider)
                    row.setdefault("inference_variant", inference_variant)
                    row.setdefault("route_variant", route_variant)
                    job_key = (row["id"], job_method, job_level)
                    previous_failure = retry_failure_rows.get(job_key)
                    if previous_failure is not None:
                        row["retry_of_failure"] = True
                        row["previous_failure_reason"] = str(
                            previous_failure.get("failure_reason")
                            or previous_failure.get("answer_parse_error")
                            or (
                                "judge_notsure"
                                if row_needs_judge_retry(previous_failure)
                                else ""
                            )
                            or "failed_result"
                        )
                    append_jsonl(raw_rows_path, row)
                    result_path = run_dir / f"{row['id']}.json"
                    if row["id"] not in results:
                        meta = load_json(root / "data" / run_args.modality / row["id"] / f"{row['id']}.json")
                        meta["modality"] = run_args.modality
                        results[row["id"]] = empty_result(meta, canonical_model, display_name)
                    result = results[row["id"]]
                    merge_result_row(result, row)
                    update_thresholds(result)
                    write_json(result_path, result)
                    written_ids.add(row["id"])
                    completed_rows += 1
                    if row_is_failure(row):
                        failure_count += 1

                    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
                    manifest["completed_rows"] = completed_rows
                    manifest["failure_count"] = failure_count
                    manifest["remaining_rows"] = max(0, len(jobs) - completed_rows)
                    manifest["written_ids_count"] = len(written_ids)
                    manifest["result_files"] = [f"{sid}.json" for sid in sorted(written_ids)]
                    write_json(manifest_path, manifest)
                    if progress is not None:
                        progress.update(1)
                        progress.set_postfix_str(f"failures={failure_count}", refresh=False)
                    submit_next()
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["completed_rows"] = completed_rows
        manifest["failure_count"] = failure_count
        manifest["remaining_rows"] = max(0, len(jobs) - completed_rows)
        manifest["written_ids_count"] = len(written_ids)
        manifest["error"] = repr(exc)
        write_json(manifest_path, manifest)
        raise
    finally:
        if progress is not None:
            progress.close()

    summary = summarize_results.build_summary(root, run_dir.relative_to(root))
    summary_path = run_dir / "run_summary.json"
    write_json(summary_path, summary)
    manifest["status"] = "complete"
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["summary_path"] = str(summary_path.relative_to(root) if summary_path.is_relative_to(root) else summary_path)
    manifest["failure_count"] = failure_count
    write_json(manifest_path, manifest)
    return {
        "model": canonical_model,
        "display_name": display_name,
        "written_ids": len(written_ids),
        "completed_rows": completed_rows,
        "failure_count": failure_count,
        "run_dir": manifest["run_dir"],
        "summary_path": manifest["summary_path"],
    }


def apply_quick_cli_args(args: argparse.Namespace) -> argparse.Namespace:
    quick_values = [args.quick_provider, args.quick_model, args.quick_modality, args.quick_levels]
    if not any(quick_values):
        args.quick_mode = False
        return args
    if not all(quick_values):
        raise SystemExit(
            "Quick mode requires exactly four values: PROVIDER MODEL MODALITY LEVELS, "
            "for example: example_provider example-chat text 02468"
        )
    conflicts = []
    for option, value in (
        ("--profile", args.profile),
        ("--provider", args.provider),
        ("--modality", args.modality),
        ("--model", args.model),
        ("--models", args.models),
        ("--levels", args.levels),
    ):
        if value:
            conflicts.append(option)
    if conflicts:
        raise SystemExit("Quick positional mode cannot be combined with: " + ", ".join(conflicts))
    if args.quick_modality not in {"text", "image"}:
        raise SystemExit(f"Quick mode modality must be text or image: {args.quick_modality}")
    parse_level_filters([args.quick_levels])
    model_spec = str(args.quick_model)
    if not model_spec.startswith(f"{args.quick_provider}/"):
        model_spec = f"{args.quick_provider}/{model_spec}"
    args.profile = f"{args.quick_modality}_default_02468"
    args.provider = str(args.quick_provider)
    args.models = [model_spec]
    args.modality = str(args.quick_modality)
    args.levels = [str(args.quick_levels)]
    args.quick_mode = True
    return args


def apply_named_single_model_defaults(args: argparse.Namespace) -> argparse.Namespace:
    args.named_single_mode = bool(
        not args.quick_mode
        and args.provider
        and args.model
        and args.modality
        and args.levels
    )
    if args.named_single_mode and not args.profile:
        args.profile = f"{args.modality}_default_02468"
    return args


def validate_quick_runtime(runtime: dict[str, Any]) -> None:
    if runtime.get("identity_status") == "provider_catalog_only":
        raise SystemExit(
            "This command requires an official website or Hugging Face published identity, but "
            f"{runtime.get('model_spec')} is currently provider-catalog-only. "
            "Add or verify its identity mapping before a formal run."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Run GNWT packaged-data experiments.")
    ap.add_argument("quick_provider", nargs="?", help="Quick mode: provider name from my_api.json.")
    ap.add_argument("quick_model", nargs="?", help="Quick mode: provider model id or exact request_model.")
    ap.add_argument("quick_modality", nargs="?", help="Quick mode: text or image.")
    ap.add_argument("quick_levels", nargs="?", help="Quick mode: compact levels such as 02468.")
    ap.add_argument("--config", default=str(default_config_path()), help="Path to my_api.json.")
    ap.add_argument("--api-params", default=str(default_api_params_path()), help="Path to API request parameter profiles.")
    ap.add_argument("--api-params-profile", default="", help="API request parameter profile for formal model evaluation.")
    ap.add_argument("--env-file", default=str(default_env_path()), help="Path to a local .env file.")
    ap.add_argument("--prompt-dir", default=str(default_prompt_dir()), help="Directory containing prompt templates.")
    ap.add_argument("--profile", default="", help="Experiment profile from my_api.json.")
    ap.add_argument("--provider", default="", help="Provider name from my_api.json.")
    ap.add_argument("--root", default=str(default_root_path()))
    ap.add_argument("--modality", choices=["text", "image"], default=None)
    ap.add_argument("--model", default="")
    ap.add_argument("--models", nargs="+", default=None, help="One or more model ids; provider/model is accepted. Overrides --model/profile model.")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--ids", nargs="+", default=None, help="Sample ids to run, e.g. text_1 text_2 or 1,2.")
    ap.add_argument("--id-start", default="", help="First sample id to run, e.g. 1 or text_1.")
    ap.add_argument("--id-end", default="", help="Last sample id to run, e.g. 4000 or text_4000.")
    ap.add_argument("--levels", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=None)
    resume_group = ap.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        nargs="?",
        const="__default__",
        default="",
        help="Resume an existing run. Without a path, resume results/<modality>/<model>/.",
    )
    resume_group.add_argument(
        "--no-resume",
        action="store_true",
        help="Start only if the canonical results directory is empty; never overwrite existing results.",
    )
    ap.add_argument("--resume-run", default="", help=argparse.SUPPRESS)
    ap.add_argument(
        "--retry-failures",
        action="store_true",
        help="With --resume, rerun stored runtime/JSON/incomplete failures, missing answers, and LLM-judge infrastructure failures.",
    )
    ap.add_argument(
        "--retry-schema-invalid",
        action="store_true",
        help="With --resume, rerun parsed rows whose probe1_answer or confidence violates the formal output schema.",
    )
    ap.add_argument(
        "--retry-missing-confidence",
        action="store_true",
        help="With --resume, rerun parsed rows whose task confidence is missing or invalid.",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument(
        "--max-image-base64-bytes",
        type=int,
        default=0,
        help="Compress oversized images in memory so the transmitted base64 payload stays below this limit.",
    )
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--api-max-retries", type=int, default=None, help="Override eval API retry count for this run.")
    ap.add_argument("--judge-profile", default="", help="Judge profile from my_api.json.")
    ap.add_argument("--judge-provider", default="", help="Judge provider name from my_api.json.")
    ap.add_argument("--judge-base-url", default="")
    ap.add_argument("--judge-api-key", default="")
    ap.add_argument("--judge-model", default="")
    ap.add_argument("--judge-api-params-profile", default="", help="API request parameter profile for LLM judge calls.")
    ap.add_argument("--results-dir", default="results", help="Root-relative directory for experiment outputs.")
    args = apply_named_single_model_defaults(apply_quick_cli_args(ap.parse_args()))

    load_env_file(Path(args.env_file))
    config = load_config(Path(args.config))
    api_params_config = api_request_config.load_api_request_config(Path(args.api_params))
    args.api_request_profile_name, args.api_request_profile = api_request_config.resolve_api_profile(
        api_params_config,
        "eval",
        args.api_params_profile,
    )
    if args.api_max_retries is not None:
        args.api_request_profile = dict(args.api_request_profile)
        args.api_request_profile["max_retries"] = max(0, int(args.api_max_retries))
    args.judge_api_request_profile_name, args.judge_api_request_profile = api_request_config.resolve_api_profile(
        api_params_config,
        "judge",
        args.judge_api_params_profile,
    )
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    profile_name = args.profile or str(defaults.get("profile") or "")
    profile = {}
    if profile_name:
        profiles = config.get("experiments") or {}
        if not isinstance(profiles, dict) or profile_name not in profiles:
            raise SystemExit(f"Experiment profile not found in config: {profile_name}")
        if not isinstance(profiles[profile_name], dict):
            raise SystemExit(f"Experiment profile must be an object: {profile_name}")
        profile = profiles[profile_name]

    provider_name = args.provider or str(profile.get("provider") or defaults.get("provider") or "")

    args.modality = args.modality or profile.get("modality")
    profile_models = profile.get("models")
    if isinstance(profile_models, str):
        profile_model_values = [profile_models]
    elif isinstance(profile_models, list):
        profile_model_values = [str(value) for value in profile_models]
    else:
        profile_model_values = None
    fallback_model = str(profile.get("model") or defaults.get("model") or "")
    if args.models:
        models = parse_model_filters(args.models, "")
    elif args.model:
        models = parse_model_filters(None, args.model)
    elif profile_model_values:
        models = parse_model_filters(profile_model_values, "")
    else:
        models = parse_model_filters(None, fallback_model)
    args.base_url = args.base_url or ""
    args.api_key = args.api_key or ""
    configured_levels = args.levels if args.levels is not None else profile.get("levels")
    if isinstance(configured_levels, (str, int)):
        configured_level_values = [str(configured_levels)]
    else:
        configured_level_values = configured_levels
    args.levels = parse_level_filters(configured_level_values)
    args.methods = args.methods or profile.get("methods")
    args.workers = args.workers if args.workers is not None else int(profile.get("workers") or defaults.get("workers") or 8)
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    args.max_tokens = args.max_tokens if args.max_tokens is not None else None
    args.timeout = args.timeout if args.timeout is not None else int(profile.get("timeout") or defaults.get("timeout") or 90)
    args.judge_max_tokens = int(args.judge_api_request_profile.get("max_tokens") or 256)

    judge_profile_name = args.judge_profile or str(profile.get("judge_profile") or defaults.get("judge_profile") or "")
    args.judge_chain = resolve_judge_chain(
        config,
        judge_profile_name,
        args.judge_provider,
        args.judge_base_url,
        args.judge_api_key,
        args.judge_model,
        args.timeout,
        args.judge_max_tokens,
    )

    if not args.modality:
        raise SystemExit("Set --modality or choose a profile with a modality.")
    if args.modality not in {"text", "image"}:
        raise SystemExit(f"Unsupported modality: {args.modality}")
    if not models:
        raise SystemExit("Set --model/--models or choose a profile with a model.")
    model_runtimes = [resolve_model_runtime(config, model, provider_name, args.base_url, args.api_key) for model in models]
    if args.quick_mode or args.named_single_mode:
        runtime = model_runtimes[0]
        validate_quick_runtime(runtime)
        if (args.retry_failures or args.retry_schema_invalid or args.retry_missing_confidence) and (args.resume or args.resume_run):
            retry_modes = []
            if args.retry_failures:
                retry_modes.append("failures")
            if args.retry_schema_invalid:
                retry_modes.append("schema-invalid")
            if args.retry_missing_confidence:
                retry_modes.append("missing-confidence")
            mode = "resume+retry-" + "+".join(retry_modes)
        else:
            mode = "resume" if args.resume or args.resume_run else "no-resume"
        print(
            "resolved "
            f"provider={runtime['provider']} "
            f"route={runtime['model']} "
            f"request_model={runtime['request_model']} "
            f"published={runtime['display_name']} "
            f"canonical={runtime['canonical_model']} "
            f"modality={args.modality} "
            f"levels={','.join(str(level) for level in args.levels)} "
            f"mode={mode}",
            flush=True,
        )
    root = Path(args.root)
    if (args.resume and args.resume_run) or (args.no_resume and args.resume_run):
        raise SystemExit("Use only one of --resume, --no-resume, or deprecated --resume-run.")
    resume_value = args.resume_run or args.resume
    args.resume_requested = bool(resume_value)
    if args.retry_failures and not args.resume_requested:
        raise SystemExit("--retry-failures requires --resume.")
    if args.retry_schema_invalid and not args.resume_requested:
        raise SystemExit("--retry-schema-invalid requires --resume.")
    if args.retry_missing_confidence and not args.resume_requested:
        raise SystemExit("--retry-missing-confidence requires --resume.")
    args.resume_run_dir = None
    if resume_value and resume_value != "__default__":
        args.resume_run_dir = resolve_resume_run_dir(root, str(resume_value))
    if args.resume_run_dir is not None and len(model_runtimes) != 1:
        raise SystemExit("--resume PATH can only be used with exactly one model. Pass one --models provider/model value.")
    missing_endpoint = [runtime["model_spec"] for runtime in model_runtimes if not runtime.get("base_url") or not runtime.get("api_key")]
    if missing_endpoint:
        raise SystemExit(
            "Missing API endpoint/key for model(s): "
            + ", ".join(missing_endpoint)
            + ". Choose a configured provider or pass --base-url/--api-key explicitly."
        )
    if not (root / "data" / args.modality).exists():
        raise SystemExit(f"Missing GNWT {args.modality} data directory: {root / 'data' / args.modality}")
    if not Path(args.prompt_dir).exists():
        raise SystemExit(f"Missing prompt directory: {args.prompt_dir}")
    methods = args.methods or list(TEXT_METHODS if args.modality == "text" else IMAGE_METHODS)
    allowed_methods = set(TEXT_METHODS if args.modality == "text" else IMAGE_METHODS)
    unknown_methods = sorted(set(methods) - allowed_methods)
    if unknown_methods:
        raise SystemExit(f"Unsupported {args.modality} perturbation method(s): {', '.join(unknown_methods)}")
    selected_ids = parse_id_filters(args.ids, args.modality)
    samples = select_samples(sample_json_paths(root, args.modality), selected_ids, args.id_start, args.id_end, args.modality)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise SystemExit("No samples selected.")
    validate_experiment_inputs(root, samples, args.modality, methods, args.levels, Path(args.prompt_dir))

    reports = [run_model(root, samples, methods, args, runtime["model"], runtime) for runtime in model_runtimes]
    total_rows = sum(int(report.get("completed_rows") or 0) for report in reports)
    total_failures = sum(int(report.get("failure_count") or 0) for report in reports)
    print(f"complete rows={total_rows} failures={total_failures}", flush=True)


if __name__ == "__main__":
    main()
