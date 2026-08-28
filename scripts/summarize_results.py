#!/usr/bin/env python3
"""Summarize GNWT ignition thresholds."""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

import model_identity


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8", errors="replace") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sample_json_paths(root: Path, modality: str) -> list[Path]:
    def key(path: Path) -> tuple[int, str]:
        match = re.fullmatch(rf"{re.escape(modality)}_(\d+)", path.stem)
        if match:
            return int(match.group(1)), path.stem
        return 10**18, path.stem

    paths = [path for path in (root / "data" / modality).glob(f"{modality}_*/{modality}_*.json") if re.fullmatch(rf"{re.escape(modality)}_\d+", path.stem)]
    return sorted(paths, key=key)


def load_meta(root: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for modality in ("text", "image"):
        for path in sample_json_paths(root, modality):
            obj = load_json(path)
            obj["modality"] = modality
            out[obj["id"]] = obj
    return out


def compact(bucket: dict[str, list[float]]) -> dict[str, dict[str, float | int | None]]:
    result = {}
    for key, values in sorted(bucket.items()):
        result[key] = {"count": len(values), "average_ignition_threshold": mean(values) if values else None}
    return result


def normalize_probe_answer(value: Any) -> str:
    text = str(value or "").lower().strip()
    if text in {"yes", "y", "true", "1", "noticed", "understood", "clear"}:
        return "yes"
    if text in {"no", "n", "false", "0", "not noticed", "unclear"}:
        return "no"
    return text


def compact_probe(bucket: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for key, rows in sorted(bucket.items()):
        answers = [normalize_probe_answer(row.get("answer")) for row in rows]
        confidences = [float(row["confidence"]) for row in rows if isinstance(row.get("confidence"), (int, float))]
        yes_count = sum(1 for answer in answers if answer == "yes")
        no_count = sum(1 for answer in answers if answer == "no")
        result[key] = {
            "count": len(rows),
            "yes_count": yes_count,
            "no_count": no_count,
            "yes_rate": yes_count / len(rows) if rows else None,
            "average_probe1_confidence": mean(confidences) if confidences else None,
        }
    return result


def thresholds_from_result(result: dict[str, Any]) -> dict[str, Any]:
    if isinstance(result.get("ignition_threshold_by_perturbation"), dict):
        return result["ignition_threshold_by_perturbation"]
    if isinstance(result.get("ignition_threshold"), dict):
        return result["ignition_threshold"]
    if isinstance(result.get("judgments"), dict):
        out = {}
        for method, values in result["judgments"].items():
            if not isinstance(values, dict):
                continue
            first_failure = None
            for level in sorted(int(k) for k in values):
                if int(values[str(level)]) == 0:
                    first_failure = level
                    break
            out[method] = first_failure
        return out
    return {}


def result_json_paths(root: Path, results_root: Path) -> tuple[list[Path], str]:
    formal_root = (root / "results").resolve()
    if results_root.resolve() == formal_root:
        paths: list[Path] = []
        for modality in ("text", "image"):
            modality_root = results_root / modality
            if not modality_root.exists():
                continue
            for model_dir in sorted(path for path in modality_root.iterdir() if path.is_dir()):
                if not (model_dir / "run_manifest.json").exists():
                    continue
                paths.extend(sorted(model_dir.glob("*.json")))
        return paths, "active_formal_only"
    if (results_root / "run_manifest.json").exists():
        return sorted(results_root.glob("*.json")), "explicit_run_directory"
    return sorted(results_root.rglob("*.json")), "explicit_recursive_directory"


def build_summary(root: Path, results_dir: str | Path = "results") -> dict[str, Any]:
    if not (root / "data").exists():
        raise SystemExit(f"Missing GNWT data directory: {root / 'data'}")
    meta = load_meta(root)
    results_root = root / results_dir
    overall: list[float] = []
    by_modality: dict[str, list[float]] = collections.defaultdict(list)
    by_method: dict[str, list[float]] = collections.defaultdict(list)
    by_qtype: dict[str, list[float]] = collections.defaultdict(list)
    by_model: dict[str, list[float]] = collections.defaultdict(list)
    model_display_names: dict[str, str] = {}
    probe_overall: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    probe_by_modality: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    probe_by_method: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    probe_by_qtype: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    probe_by_model: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    files = 0

    result_paths, scan_scope = result_json_paths(root, results_root)
    for path in result_paths:
        if path.name in {"run_manifest.json", "run_summary.json"}:
            continue
        try:
            result = load_json(path)
        except Exception:
            continue
        sid = result.get("id")
        if sid not in meta:
            continue
        files += 1
        m = meta[sid]
        raw_model_name = str(result.get("canonical_model") or result.get("model") or "unknown")
        model_name = model_identity.canonicalize_request_model(raw_model_name)
        model_display_names.setdefault(
            model_name,
            str(result.get("display_name") or model_identity.identity_for_request(raw_model_name)["display_name"]),
        )
        for method, threshold in thresholds_from_result(result).items():
            if threshold is None:
                continue
            value = float(threshold)
            overall.append(value)
            by_modality[m.get("modality", "unknown")].append(value)
            by_method[method].append(value)
            by_qtype[m.get("question_type", "unknown")].append(value)
            by_model[model_name].append(value)
        for method, levels in (result.get("probe1_by_perturbation") or {}).items():
            if not isinstance(levels, dict):
                continue
            for probe in levels.values():
                if not isinstance(probe, dict):
                    continue
                probe_overall["overall"].append(probe)
                probe_by_modality[m.get("modality", "unknown")].append(probe)
                probe_by_method[str(method)].append(probe)
                probe_by_qtype[m.get("question_type", "unknown")].append(probe)
                probe_by_model[model_name].append(probe)

    return {
        "source_results_dir": str(results_root.relative_to(root) if results_root.is_relative_to(root) else results_root),
        "scan_scope": scan_scope,
        "threshold_definition": "first_failed_perturbation_level",
        "result_files_scanned": files,
        "threshold_count": len(overall),
        "average_ignition_threshold": mean(overall) if overall else None,
        "by_modality": compact(by_modality),
        "by_perturbation": compact(by_method),
        "by_question_type": compact(by_qtype),
        "by_model": compact(by_model),
        "model_display_names": dict(sorted(model_display_names.items())),
        "probe1_summary": {
            "overall": compact_probe(probe_overall),
            "by_modality": compact_probe(probe_by_modality),
            "by_perturbation": compact_probe(probe_by_method),
            "by_question_type": compact_probe(probe_by_qtype),
            "by_model": compact_probe(probe_by_model),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize GNWT result thresholds.")
    ap.add_argument("--root", required=True)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--out", default="result_summary.json")
    args = ap.parse_args()
    root = Path(args.root)
    summary = build_summary(root, args.results_dir)
    write_json(root / args.out, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
