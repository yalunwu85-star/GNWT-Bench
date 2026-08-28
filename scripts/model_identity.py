#!/usr/bin/env python3
"""Shared model identity rules for provider routes, results, and resume.

Every model route that reaches the experiment runner is resolved to a stable
identity (canonical id, display name, published source) before any request is
sent, so results and resume state stay comparable across providers.

Resolution order, first match wins:

1. Explicit fields on the model entry in ``my_api.json``:
   ``canonical_model``, ``display_name``, ``identity_source``,
   ``identity_status``.
2. ``OFFICIAL_IDENTITIES`` below.
3. A Hugging Face repo URL derived from a known org namespace in the request
   model string (``org/model``).
4. A family catalog URL from ``FAMILY_SOURCES``.
5. Fallback: source ``provider_catalog`` with status
   ``provider_catalog_only``.

The lookup tables below ship empty on purpose. This package is a generic
runner and does not bundle any model catalog: register the models your own
``my_api.json`` defines, either by filling the tables or by setting the
identity fields on each model entry directly. Runs started through the
quick/named CLI reject ``provider_catalog_only`` identities, so unmapped
formal runs fail fast instead of silently reporting catalog names.
"""

from __future__ import annotations

import re
from typing import Any


# Publication names for canonical model ids, e.g.
#     "my-model-id": {
#         "display_name": "My Model",
#         "source": "https://provider.example.com/docs/models",
#     }
OFFICIAL_IDENTITIES: dict[str, dict[str, str]] = {}

# ``org/`` namespaces recognized and stripped in request model strings.
ORG_PREFIXES: set[str] = set()

# Org namespaces for which a request model of the form ``org/model`` resolves
# to a Hugging Face repo URL as its identity source.
HUGGINGFACE_ORGS: set[str] = set()

# Canonical-prefix -> public catalog URL rules used as identity sources, e.g.
#     ((("my-family", "my-family-pro"), "https://provider.example.com/models"),)
FAMILY_SOURCES: tuple[tuple[tuple[str, ...], str], ...] = ()

# Route markers some providers or gateways wrap around model ids. Add the
# prefixes/suffixes your endpoint uses so canonical ids stay stable.
ROUTE_PREFIXES: tuple[str, ...] = ()
ROUTE_SUFFIXES: tuple[str, ...] = ()

# Request-model alias rewrites applied before canonicalization, e.g.
#     (r"^acme-(big|small)-(.*)$", r"acme-\1-\2")
ALIAS_PATTERNS: tuple[tuple[str, str], ...] = ()


def _clean_route_name(request_model: str) -> str:
    value = request_model.strip()
    lower = value.lower()
    if "/" in value:
        namespace, suffix = value.split("/", 1)
        if namespace.lower() in ORG_PREFIXES and suffix:
            value = suffix
            lower = value.lower()
    for prefix in ROUTE_PREFIXES:
        if lower.startswith(prefix):
            value = value[len(prefix) :]
            lower = value.lower()
            break
    for suffix in ROUTE_SUFFIXES:
        if lower.endswith(suffix):
            value = value[: -len(suffix)]
            lower = value.lower()
            break
    return value


def snapshot_variant(request_model: str) -> str:
    value = _clean_route_name(request_model)
    match = re.search(r"-(20\d{2}-\d{2}-\d{2}|\d{8}|\d{6})(?:-(?:thinking|nothinking|non-thinking|reasoning|search))?$", value, flags=re.IGNORECASE)
    return f"snapshot-{match.group(1)}" if match else ""


def inference_variant(request_model: str) -> str:
    lower = request_model.strip().lower()
    snapshot = snapshot_variant(request_model)
    if "reasoner" in lower or re.search(r"(?:^|[-_/])(thinking|think|reasoning)(?:$|[-_/])", lower):
        mode = "thinking"
    elif "nothinking" in lower or "non-thinking" in lower:
        mode = "non-thinking"
    elif "search" in lower:
        mode = "search"
    else:
        mode = "default"
    return f"{mode}+{snapshot}" if snapshot else mode


def route_variant(request_model: str) -> str:
    lower = request_model.strip().lower()
    tags: list[str] = []
    if any(lower.startswith(prefix) for prefix in ROUTE_PREFIXES):
        tags.append("provider-prefix")
    if any(lower.endswith(suffix) for suffix in ROUTE_SUFFIXES):
        tags.append("provider-suffix")
    return ",".join(tags) or "standard"


def canonicalize_request_model(request_model: str) -> str:
    raw = request_model.strip()
    value = _clean_route_name(raw)
    lower_value = value.lower()
    for pattern, replacement in ALIAS_PATTERNS:
        if re.match(pattern, lower_value):
            value = re.sub(pattern, replacement, lower_value)
            value = value.rstrip("-")
            break
    for _ in range(2):
        value = re.sub(r"-(?:thinking|nothinking|non-thinking|reasoning|search)$", "", value, flags=re.IGNORECASE)
        value = re.sub(r"-(20\d{2}-\d{2}-\d{2}|\d{8}|\d{6})$", "", value)
    value = value.replace("@", "-")
    value = re.sub(r"(?<=\d)_(?=\d)", ".", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-_.").lower()
    return value or "unknown-model"


def fallback_display_name(canonical_model: str) -> str:
    """Return a readable fallback name; prefer display_name in my_api.json."""
    return canonical_model


def _huggingface_source(request_model: str) -> str:
    value = request_model.strip()
    if "/" not in value:
        return ""
    namespace, suffix = value.split("/", 1)
    if namespace.lower() not in HUGGINGFACE_ORGS or not suffix:
        return ""
    return f"https://huggingface.co/{value}"


def _family_source(canonical_model: str) -> str:
    for prefixes, source in FAMILY_SOURCES:
        if canonical_model.startswith(prefixes):
            return source
    return ""


def _catalog_target(model_entry: dict[str, Any]) -> str:
    description = str(model_entry.get("description") or model_entry.get("desc") or "")
    match = re.search(
        r"(?:指向|routes?\s+to|alias(?:es)?\s+to)\s*"
        r"([a-z0-9._-]+)",
        description,
        flags=re.IGNORECASE,
    )
    return match.group(1) if match else ""


def identity_for_request(request_model: str, model_entry: dict[str, Any] | None = None) -> dict[str, str]:
    entry = model_entry or {}
    catalog_target = _catalog_target(entry)
    canonical = str(entry.get("canonical_model") or "").strip() or canonicalize_request_model(catalog_target or request_model)
    official = OFFICIAL_IDENTITIES.get(canonical)
    display = str(entry.get("display_name") or "").strip()
    if not display:
        display = official["display_name"] if official else fallback_display_name(canonical)
    source = str(entry.get("identity_source") or "").strip()
    derived_status = ""
    if not source:
        if official:
            source = official["source"]
        else:
            source = _huggingface_source(request_model)
            if source:
                derived_status = "huggingface_repo_id"
            else:
                source = _family_source(canonical)
                if source:
                    derived_status = "official_family_catalog"
                else:
                    source = "provider_catalog"
    status = str(entry.get("identity_status") or "").strip()
    if not status:
        status = "official_verified" if official else (derived_status or "provider_catalog_only")
    derived_inference_variant = inference_variant(catalog_target or request_model)
    return {
        "canonical_model": canonical,
        "display_name": display,
        "identity_source": source,
        "identity_status": status,
        "inference_variant": str(entry.get("inference_variant") or "").strip() or derived_inference_variant,
        "route_variant": str(entry.get("route_variant") or "").strip() or route_variant(request_model),
    }


def enrich_model_entry(model_entry: dict[str, Any]) -> dict[str, Any]:
    entry = dict(model_entry)
    request_model = str(
        entry.get("request_model")
        or entry.get("api_model")
        or entry.get("name")
        or entry.get("id")
        or ""
    ).strip()
    entry["request_model"] = request_model
    entry.update(identity_for_request(request_model, entry))
    return entry
