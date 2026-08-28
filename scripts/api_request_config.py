#!/usr/bin/env python3
"""Role-specific API request parameters for GNWT scripts."""

from __future__ import annotations

import copy
import http.client
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class ChatCompletionHTTPError(RuntimeError):
    def __init__(self, status_code: int, body: str, message: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body[:500] or message}")


def default_api_params_path() -> Path:
    return Path(__file__).with_name("api_request_params.json")


def load_api_request_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    obj = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(obj, dict):
        raise SystemExit(f"API request params must be a JSON object: {path}")
    return obj


def resolve_api_profile(config: dict[str, Any], role: str, profile_name: str = "") -> tuple[str, dict[str, Any]]:
    defaults = config.get("defaults") if isinstance(config.get("defaults"), dict) else {}
    name = profile_name or str(defaults.get(role) or "")
    if not name:
        return "", {}
    profiles = config.get("profiles") or {}
    if not isinstance(profiles, dict) or name not in profiles:
        raise SystemExit(f"API request profile not found for {role}: {name}")
    profile = profiles[name]
    if not isinstance(profile, dict):
        raise SystemExit(f"API request profile must be an object: {name}")
    return name, profile


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def build_chat_payload(
    model: str,
    messages: list[dict[str, Any]],
    profile: dict[str, Any],
    *,
    max_tokens: int | None = None,
    response_format: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "messages": messages}
    for key in (
        "temperature",
        "top_p",
        "n",
        "stream",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "logprobs",
        "top_logprobs",
        "tool_choice",
    ):
        if key in profile and profile[key] is not None:
            payload[key] = copy.deepcopy(profile[key])
    token_limit = effective_max_tokens(model, profile, max_tokens)
    if token_limit is not None:
        payload[effective_token_limit_field(model, profile)] = int(token_limit)
    selected_response_format = response_format
    if selected_response_format is None:
        selected_response_format = effective_response_format(model, profile)
    if isinstance(selected_response_format, dict):
        payload["response_format"] = copy.deepcopy(selected_response_format)
    if "tools" in profile:
        payload["tools"] = copy.deepcopy(profile["tools"])
    apply_reasoning_controls(payload, model, profile)
    apply_param_omissions(payload, model, profile)
    request_overrides = profile.get("request_overrides")
    if isinstance(request_overrides, dict):
        for key, value in request_overrides.items():
            if key not in {"model", "messages"}:
                payload[str(key)] = copy.deepcopy(value)
    return payload


def effective_response_format(model: str, profile: dict[str, Any]) -> dict[str, Any] | None:
    """Return a model-specific structured-output format when one is configured."""
    overrides = profile.get("response_format_for_model_substrings")
    if isinstance(overrides, dict):
        lower_model = model.lower()
        for marker, value in overrides.items():
            if str(marker).lower() in lower_model and isinstance(value, dict):
                return value
    value = profile.get("response_format")
    return value if isinstance(value, dict) else None


def effective_max_tokens(model: str, profile: dict[str, Any], requested_max_tokens: int | None = None) -> int | None:
    token_limit = requested_max_tokens if requested_max_tokens is not None else profile.get("max_tokens")
    overrides = profile.get("max_tokens_for_model_substrings")
    if isinstance(overrides, dict) and requested_max_tokens is None:
        lower_model = model.lower()
        for marker, value in overrides.items():
            if str(marker).lower() in lower_model:
                token_limit = value
                break
    return int(token_limit) if token_limit is not None else None


def effective_token_limit_field(model: str, profile: dict[str, Any]) -> str:
    field = str(profile.get("token_limit_field") or "max_tokens")
    overrides = profile.get("token_limit_field_for_model_substrings")
    if isinstance(overrides, dict):
        lower_model = model.lower()
        for marker, value in overrides.items():
            if str(marker).lower() in lower_model:
                return str(value)
    return field


def apply_reasoning_controls(payload: dict[str, Any], model: str, profile: dict[str, Any]) -> None:
    controls = profile.get("reasoning_controls")
    if not isinstance(controls, dict):
        return
    lower_model = model.lower()
    if any(str(marker).lower() in lower_model for marker in controls.get("enable_thinking_false_for") or []):
        payload["enable_thinking"] = False
    if any(str(marker).lower() in lower_model for marker in controls.get("thinking_disabled_for") or []):
        payload["thinking"] = {"type": "disabled"}
    if any(str(marker).lower() in lower_model for marker in controls.get("reasoning_effort_none_for") or []):
        payload["reasoning_effort"] = "none"
    elif any(str(marker).lower() in lower_model for marker in controls.get("reasoning_effort_low_for") or []):
        payload["reasoning_effort"] = "low"
    if any(str(marker).lower() in lower_model for marker in controls.get("extra_body_google_thinking_minimal_for") or []):
        extra_body = payload.get("extra_body")
        if not isinstance(extra_body, dict):
            extra_body = {}
            payload["extra_body"] = extra_body
        google = extra_body.get("google")
        if not isinstance(google, dict):
            google = {}
            extra_body["google"] = google
        thinking_config = google.get("thinking_config")
        if not isinstance(thinking_config, dict):
            thinking_config = {}
            google["thinking_config"] = thinking_config
        thinking_config["thinking_level"] = "minimal"


def apply_param_omissions(payload: dict[str, Any], model: str, profile: dict[str, Any]) -> None:
    rules = profile.get("omit_params_for_model_substrings")
    if not isinstance(rules, dict):
        return
    lower_model = model.lower()
    for marker, params in rules.items():
        if str(marker).lower() not in lower_model:
            continue
        for param in params or []:
            payload.pop(str(param), None)


def safe_payload_params(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: copy.deepcopy(value) for key, value in payload.items() if key not in {"model", "messages"}}


def is_schema_rejection(exc: Exception) -> bool:
    status = getattr(exc, "status_code", None)
    if status not in {400, 422}:
        return False
    text = f"{getattr(exc, 'body', '')} {exc}".lower()
    return any(marker in text for marker in ("response_format", "json_schema", "schema", "strict"))


def retryable_error(exc: Exception) -> bool:
    if isinstance(exc, (http.client.RemoteDisconnected, ConnectionResetError, TimeoutError, urllib.error.URLError)):
        return True
    status = getattr(exc, "status_code", None)
    if status in {408, 409, 429, 500, 502, 503, 504}:
        return True
    text = str(exc).lower()
    return "timed out" in text or "timeout" in text


def post_chat_completion(base_url: str, api_key: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    req = urllib.request.Request(
        chat_completions_url(base_url),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        raise ChatCompletionHTTPError(exc.code, body, str(exc)) from exc


def chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int | None,
    timeout: int,
    profile: dict[str, Any] | None = None,
    profile_name: str = "",
) -> dict[str, Any]:
    request_profile = profile or {}
    payload = build_chat_payload(model, messages, request_profile, max_tokens=max_tokens)
    fallback_format = request_profile.get("fallback_response_format")
    max_retries = max(0, int(request_profile.get("max_retries") or 0))
    attempts = 0
    fallback_used = False
    schema_fallback_error = ""
    errors: list[str] = []

    while True:
        attempts += 1
        try:
            response = post_chat_completion(base_url, api_key, payload, timeout)
            response["_gnwt_request"] = {
                "api_params_profile": profile_name,
                "request_params": safe_payload_params(payload),
                "response_format_fallback_used": fallback_used,
                "schema_fallback_error": schema_fallback_error,
                "attempts": attempts,
                "transient_errors": errors,
            }
            return response
        except Exception as exc:
            if isinstance(fallback_format, dict) and not fallback_used and is_schema_rejection(exc):
                schema_fallback_error = repr(exc)
                payload = build_chat_payload(model, messages, request_profile, max_tokens=max_tokens, response_format=fallback_format)
                fallback_used = True
                continue
            if attempts <= max_retries and retryable_error(exc):
                errors.append(repr(exc))
                continue
            raise


def first_choice_finish_reason(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    return str(choices[0].get("finish_reason") or "")


def response_incomplete(response: dict[str, Any]) -> bool:
    if first_choice_finish_reason(response) == "length":
        return True
    return str(response.get("status") or "").lower() == "incomplete"
