#!/usr/bin/env python3
"""Judge GNWT answers by task type."""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path
from typing import Any

import api_request_config
import model_identity


SUBJECTIVE_QTYPES = {"subjective_generation_integration"}
SUBJECTIVE_SCORING = {"reference_summary_judge", "llm_judge", "semantic_judge"}


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8", errors="replace") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


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
    for model in provider.get("models") or []:
        if isinstance(model, dict) and str(model.get("id") or "").strip() == model_id:
            return model
        if isinstance(model, str) and model.strip() == model_id:
            return {"id": model_id}
    return {}


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
    if model not in provider_model_ids(provider):
        raise SystemExit(f"{label} model {model} is not registered under provider {resolved_provider}.")
    return resolved_provider, model


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
    request_model = str(model_entry.get("request_model") or model_entry.get("api_model") or model_entry.get("name") or model)
    identity = model_identity.identity_for_request(request_model, model_entry)
    return {
        "provider": provider_name,
        "base_url": base_url,
        "api_key": api_key,
        "model": request_model,
        "model_route_id": model,
        **identity,
        "max_tokens": int(profile.get("max_tokens") or default_max_tokens),
        "timeout": int(profile.get("timeout") or timeout),
        "request_overrides": request_overrides,
    }


def judge_request_profile(
    base_profile: dict[str, Any] | None,
    judge: dict[str, Any],
) -> dict[str, Any] | None:
    overrides = judge.get("request_overrides")
    if not isinstance(overrides, dict) or not overrides:
        return base_profile
    profile = copy.deepcopy(base_profile) if isinstance(base_profile, dict) else {}
    existing = profile.get("request_overrides")
    merged = copy.deepcopy(existing) if isinstance(existing, dict) else {}
    merged.update(copy.deepcopy(overrides))
    profile["request_overrides"] = merged
    return profile


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
        model_entry = provider_model_entry(provider_config(config, provider_name), model)
        request_model = str(model_entry.get("request_model") or model_entry.get("api_model") or model_entry.get("name") or model)
        return [
            {
                "provider": provider_name,
                "base_url": base_url_override or base_url,
                "api_key": api_key_override or api_key,
                "model": request_model,
                "model_route_id": model,
                **model_identity.identity_for_request(request_model, model_entry),
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


def load_prompt(prompt_dir: Path, filename: str) -> str:
    path = prompt_dir / filename
    if not path.exists():
        raise SystemExit(f"Missing prompt file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def render_prompt(template: str, **values: Any) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{" + key + "}", "" if value is None else str(value))
    return rendered


def normalize_answer(value: Any) -> str:
    text = str(value or "").casefold().strip()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


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


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
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


def chat_completions_url(base_url: str) -> str:
    return api_request_config.chat_completions_url(base_url)


def extract_message_content(response: dict[str, Any]) -> str:
    return str(response.get("choices", [{}])[0].get("message", {}).get("content") or "")


def gt_answer(meta: dict[str, Any]) -> str:
    gt = meta.get("gt")
    if isinstance(gt, dict):
        return str(gt.get("answer") or gt.get("answer_text") or gt.get("reference_answer") or "")
    return str(gt or "")


def aliases(meta: dict[str, Any]) -> list[str]:
    gt = meta.get("gt")
    values: list[str] = []
    if isinstance(gt, dict):
        for key in ("answer", "answer_text", "reference_answer", "answer_label"):
            if gt.get(key):
                values.append(str(gt[key]))
        for item in gt.get("aliases") or []:
            values.append(str(item))
    elif gt:
        values.append(str(gt))
    return values


def original_evidence(meta: dict[str, Any]) -> dict[str, Any]:
    evidence = meta.get("original_dataset_gt_evidence")
    return evidence if isinstance(evidence, dict) else {}


def choice_options(meta: dict[str, Any]) -> list[tuple[str, str]]:
    gt = meta.get("gt") if isinstance(meta.get("gt"), dict) else {}
    evidence = original_evidence(meta)
    raw_options = gt.get("options") or meta.get("options") or evidence.get("options") or []
    if isinstance(raw_options, dict):
        return [(str(k).strip().upper(), str(v).strip()) for k, v in raw_options.items()]
    if isinstance(raw_options, list):
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return [(labels[i], str(v).strip()) for i, v in enumerate(raw_options) if i < len(labels)]
    return []


def answer_label(answer: Any) -> str | None:
    text = str(answer or "").strip()
    if not text:
        return None
    normalized = normalize_answer(text).upper()
    if len(normalized) == 1 and "A" <= normalized <= "Z":
        return normalized
    match = re.match(r"^\s*(?:option|answer)?\s*[\(\[]?([A-Za-z])[\)\]]?\s*(?:[.)：:\-]|$)", text)
    if match:
        return match.group(1).upper()
    return None


def choice_gold_labels(meta: dict[str, Any]) -> set[str]:
    gt = meta.get("gt") if isinstance(meta.get("gt"), dict) else {}
    evidence = original_evidence(meta)
    labels = {
        str(value).strip().upper()
        for value in (
            gt.get("answer_label"),
            gt.get("label"),
            evidence.get("dataset_answer_label"),
            evidence.get("answer_label"),
        )
        if value
    }
    answer_texts = {normalize_answer(x) for x in aliases(meta) if str(x).strip()}
    for label, option_text in choice_options(meta):
        if normalize_answer(option_text) in answer_texts:
            labels.add(label)
    return {label for label in labels if len(label) == 1 and "A" <= label <= "Z"}


def choice_judge(answer: str, meta: dict[str, Any]) -> dict[str, Any]:
    pred = normalize_answer(answer)
    pred_label = answer_label(answer)
    gold_labels = choice_gold_labels(meta)
    golds = [normalize_answer(x) for x in aliases(meta) if str(x).strip()]
    option_text_by_label = {label: normalize_answer(text) for label, text in choice_options(meta)}
    gold_option_texts = {option_text_by_label[label] for label in gold_labels if label in option_text_by_label}
    if gold_labels:
        correct = int(bool((pred_label and pred_label in gold_labels) or (pred and pred in gold_option_texts) or (pred and pred in set(golds))))
        judge_method = "multiple_choice_label_or_option_text_exact"
    else:
        correct = int(bool(pred) and pred in set(golds))
        judge_method = "multiple_choice_exact_or_alias_no_label"
    return {
        "correct": correct,
        "judge_method": judge_method,
        "normalized_prediction": pred,
        "predicted_label": pred_label,
        "gold_labels": sorted(gold_labels),
        "gold_option_texts": sorted(gold_option_texts),
        "normalized_gt": golds,
    }


def supporting_fact_pairs(meta: dict[str, Any]) -> list[tuple[str, int]]:
    facts = original_evidence(meta).get("supporting_facts")
    if not isinstance(facts, dict):
        return []
    titles = facts.get("title") or []
    sent_ids = facts.get("sent_id") or []
    pairs: list[tuple[str, int]] = []
    for title, sent_id in zip(titles, sent_ids):
        try:
            pairs.append((str(title).strip(), int(sent_id)))
        except Exception:
            continue
    return pairs


def parse_exact_supporting_fact_pairs(
    answer: str,
    gold_pairs: list[tuple[str, int]],
) -> tuple[list[tuple[str, int]], bool]:
    text = str(answer or "").strip()
    remaining = list(gold_pairs)
    predicted: list[tuple[str, int]] = []
    while text:
        matches = [
            (index, pair, f"{pair[0]}#{pair[1]}")
            for index, pair in enumerate(remaining)
            if text.startswith(f"{pair[0]}#{pair[1]}")
        ]
        if not matches:
            return predicted, False
        index, pair, evidence_id = max(matches, key=lambda item: len(item[2]))
        predicted.append(pair)
        remaining.pop(index)
        text = text[len(evidence_id) :].lstrip()
        if not text:
            break
        if text[0] not in {",", ";"}:
            return predicted, False
        text = text[1:].lstrip()
        if not text:
            return predicted, False
    return predicted, bool(predicted) and not remaining


def localization_judge(answer: str, meta: dict[str, Any]) -> dict[str, Any]:
    gold_pairs = supporting_fact_pairs(meta)
    if gold_pairs:
        pred_pairs, exact_match = parse_exact_supporting_fact_pairs(answer, gold_pairs)
        return {
            "correct": int(exact_match),
            "judge_method": "supporting_fact_id_list_exact_once",
            "predicted_supporting_facts": pred_pairs,
            "gold_supporting_facts": gold_pairs,
        }
    pred = normalize_answer(answer)
    golds = [normalize_answer(x) for x in aliases(meta) if str(x).strip()]
    return {
        "correct": int(bool(pred) and pred in set(golds)),
        "judge_method": "localization_exact_or_alias",
        "normalized_prediction": pred,
        "normalized_gt": golds,
    }


def needs_llm_judge(meta: dict[str, Any]) -> bool:
    gt = meta.get("gt") if isinstance(meta.get("gt"), dict) else {}
    scoring = str(gt.get("scoring_method") or "").lower()
    return meta.get("question_type") in SUBJECTIVE_QTYPES or scoring in SUBJECTIVE_SCORING or "judge" in scoring


def automatic_judge(answer: str, meta: dict[str, Any]) -> dict[str, Any]:
    question_type = meta.get("question_type")
    if question_type == "objective_choice_discrimination":
        return choice_judge(answer, meta)
    if question_type == "localization_evidence":
        return localization_judge(answer, meta)
    pred = normalize_answer(answer)
    golds = [normalize_answer(x) for x in aliases(meta) if str(x).strip()]
    correct = int(bool(pred) and pred in set(golds))
    return {
        "correct": correct,
        "judge_method": "automatic_exact_or_alias",
        "normalized_prediction": pred,
        "normalized_gt": golds,
    }


def llm_judge(
    answer: str,
    meta: dict[str, Any],
    *,
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    timeout: int,
    prompt_dir: Path,
    api_request_profile: dict[str, Any] | None = None,
    api_request_profile_name: str = "",
) -> dict[str, Any]:
    reference = gt_answer(meta)
    prompt = render_prompt(
        load_prompt(prompt_dir, "judge.txt"),
        question=str(meta.get("question") or ""),
        reference_answer=reference,
        model_answer=answer,
    )
    response = chat_completion(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        timeout=timeout,
        request_profile=api_request_profile,
        request_profile_name=api_request_profile_name,
    )
    raw = extract_message_content(response)
    verdict = "notsure"
    judge_confidence: float | None = None
    judge_parse_ok = False
    parsed_obj = extract_json_object(raw)
    if parsed_obj:
        raw_verdict = str(parsed_obj.get("verdict", "notsure")).strip().lower()
        raw_confidence = parse_confidence(parsed_obj.get("confidence"))
        if raw_verdict in {"1", "0", "notsure"} and raw_confidence is not None:
            verdict = raw_verdict
            judge_confidence = raw_confidence
            judge_parse_ok = True
    if not judge_parse_ok:
        return {
            "correct": None,
            "judge_method": "llm_judge_invalid_json",
            "judge_model": model,
            "judge_verdict": "notsure",
            "judge_confidence": judge_confidence,
            "judge_parse_ok": False,
            "failure_reason": "no_valid_json",
            "judge_raw_output": raw,
            "judge_finish_reason": api_request_config.first_choice_finish_reason(response),
            "judge_response_incomplete": api_request_config.response_incomplete(response),
            "judge_response": response,
        }
    if judge_confidence is None or judge_confidence < 0.75:
        verdict = "notsure"
    return {
        "correct": 1 if verdict == "1" else 0 if verdict == "0" else None,
        "judge_method": "llm_as_judge",
        "judge_model": model,
        "judge_verdict": verdict,
        "judge_confidence": judge_confidence,
        "judge_parse_ok": judge_parse_ok,
        "judge_raw_output": raw,
        "judge_finish_reason": api_request_config.first_choice_finish_reason(response),
        "judge_response_incomplete": api_request_config.response_incomplete(response),
        "judge_response": response,
    }


JUDGE_INFRA_FAILURE_METHODS = frozenset(
    {
        "llm_judge_invalid_json",
        "llm_required_but_not_configured",
        "llm_judge_all_failed",
    }
)


def judge_method_is_infra_failure(method: Any) -> bool:
    return str(method or "") in JUDGE_INFRA_FAILURE_METHODS


def should_try_next_judge(judged: dict[str, Any]) -> bool:
    """Continue the fallback chain until a decisive 0/1 score is obtained.

    Only correct=0 or correct=1 ends the chain. Infrastructure failures and
    successfully parsed notsure verdicts both try the next judge when available.
    """
    if judged.get("correct") is not None:
        return False
    return True


def judge_answer(
    answer: str,
    meta: dict[str, Any],
    *,
    judge_base_url: str | None = None,
    judge_api_key: str | None = None,
    judge_model: str | None = None,
    judge_max_tokens: int = 256,
    timeout: int = 60,
    prompt_dir: Path | None = None,
    api_request_profile: dict[str, Any] | None = None,
    api_request_profile_name: str = "",
) -> dict[str, Any]:
    if needs_llm_judge(meta):
        if judge_base_url and judge_api_key and judge_model:
            return llm_judge(
                answer,
                meta,
                base_url=judge_base_url,
                api_key=judge_api_key,
                model=judge_model,
                max_tokens=judge_max_tokens,
                timeout=timeout,
                prompt_dir=prompt_dir or default_prompt_dir(),
                api_request_profile=api_request_profile,
                api_request_profile_name=api_request_profile_name,
            )
        return {"correct": None, "judge_method": "llm_required_but_not_configured"}
    return automatic_judge(answer, meta)


def judge_answer_with_chain(
    answer: str,
    meta: dict[str, Any],
    chain: list[dict[str, Any]],
    *,
    timeout: int,
    prompt_dir: Path,
    api_request_profile: dict[str, Any] | None = None,
    api_request_profile_name: str = "",
) -> dict[str, Any]:
    if not chain:
        return judge_answer(
            answer,
            meta,
            timeout=timeout,
            prompt_dir=prompt_dir,
            api_request_profile=api_request_profile,
            api_request_profile_name=api_request_profile_name,
        )
    errors: list[dict[str, str]] = []
    last_judged: dict[str, Any] | None = None
    last_notsure: dict[str, Any] | None = None
    for judge in chain:
        model_name = str(judge.get("request_model") or judge.get("model") or "")
        judge_api_request_profile = judge_request_profile(api_request_profile, judge)
        try:
            judged = judge_answer(
                answer,
                meta,
                judge_base_url=str(judge.get("base_url") or "") or None,
                judge_api_key=str(judge.get("api_key") or "") or None,
                judge_model=model_name or None,
                judge_max_tokens=int(judge.get("max_tokens") or 256),
                timeout=int(judge.get("timeout") or timeout),
                prompt_dir=prompt_dir,
                api_request_profile=judge_api_request_profile,
                api_request_profile_name=api_request_profile_name,
            )
        except Exception as exc:
            errors.append({"model": model_name, "error": repr(exc)})
            continue
        last_judged = judged
        if judged.get("judge_method") == "llm_as_judge" and judged.get("correct") is None:
            last_notsure = judged
        if should_try_next_judge(judged):
            errors.append(
                {
                    "model": model_name or str(judged.get("judge_model") or ""),
                    "error": str(
                        judged.get("failure_reason")
                        or judged.get("judge_verdict")
                        or judged.get("judge_method")
                        or "judge_failed"
                    ),
                }
            )
            continue
        if errors:
            judged["judge_fallback_errors"] = errors
        return judged
    # Prefer the last explicit notsure over a pure infrastructure-all-failed marker.
    if last_notsure is not None:
        if errors:
            last_notsure["judge_fallback_errors"] = errors
        return last_notsure
    if last_judged is not None and not should_try_next_judge(last_judged):
        if errors:
            last_judged["judge_fallback_errors"] = errors
        return last_judged
    return {"correct": None, "judge_method": "llm_judge_all_failed", "judge_fallback_errors": errors}


def main() -> None:
    ap = argparse.ArgumentParser(description="Judge one answer or a result JSON file.")
    ap.add_argument("--config", default=str(default_config_path()), help="Path to my_api.json.")
    ap.add_argument("--api-params", default=str(default_api_params_path()), help="Path to API request parameter profiles.")
    ap.add_argument("--api-params-profile", default="", help="API request parameter profile for LLM judge calls.")
    ap.add_argument("--env-file", default=str(default_env_path()), help="Path to a local .env file.")
    ap.add_argument("--prompt-dir", default=str(default_prompt_dir()), help="Directory containing prompt templates.")
    ap.add_argument("--meta", required=True, help="Path to data/<modality>/<id>/<id>.json")
    ap.add_argument("--answer", default="")
    ap.add_argument("--result-json", default="")
    ap.add_argument("--judge-profile", default="", help="Judge profile from my_api.json.")
    ap.add_argument("--judge-provider", default="", help="Judge provider name from my_api.json.")
    ap.add_argument("--judge-base-url", default="")
    ap.add_argument("--judge-api-key", default="")
    ap.add_argument("--judge-model", default="")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()
    load_env_file(Path(args.env_file))
    config = load_config(Path(args.config))
    api_params_config = api_request_config.load_api_request_config(Path(args.api_params))
    api_request_profile_name, api_request_profile = api_request_config.resolve_api_profile(
        api_params_config,
        "judge",
        args.api_params_profile,
    )
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    judge_max_tokens = int(api_request_profile.get("max_tokens") or 256)
    judge_profile_name = args.judge_profile or str(defaults.get("judge_profile") or "")
    judge_chain = resolve_judge_chain(
        config,
        judge_profile_name,
        args.judge_provider,
        args.judge_base_url,
        args.judge_api_key,
        args.judge_model,
        args.timeout,
        judge_max_tokens,
    )

    meta_path = Path(args.meta)
    if not meta_path.exists():
        raise SystemExit(f"Missing metadata file: {meta_path}")
    if not Path(args.prompt_dir).exists():
        raise SystemExit(f"Missing prompt directory: {args.prompt_dir}")
    meta = load_json(meta_path)
    if args.result_json:
        result_path = Path(args.result_json)
        result = load_json(result_path)
        for raw in result.get("model_raw_outputs", []):
            judged = judge_answer_with_chain(
                str(raw.get("answer") or ""),
                meta,
                judge_chain,
                timeout=args.timeout,
                prompt_dir=Path(args.prompt_dir),
                api_request_profile=api_request_profile,
                api_request_profile_name=api_request_profile_name,
            )
            raw.update(judged)
        write_json(result_path, result)
        print(json.dumps({"updated": str(result_path)}, ensure_ascii=False))
    else:
        print(
            json.dumps(
                judge_answer_with_chain(
                    args.answer,
                    meta,
                    judge_chain,
                    timeout=args.timeout,
                    prompt_dir=Path(args.prompt_dir),
                    api_request_profile=api_request_profile,
                    api_request_profile_name=api_request_profile_name,
                ),
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
