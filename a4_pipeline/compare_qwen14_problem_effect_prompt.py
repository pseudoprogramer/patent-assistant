from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import requests


BASE = Path("/Volumes/외장 2TB/cpu2026")
PROBE_PATH = BASE / "common" / "runtime" / "reports" / "A4" / "problem_effect_probe.json"
OUT_PATH = BASE / "common" / "runtime" / "reports" / "A4" / "qwen14_problem_effect_prompt_compare.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:14b"

PROBLEM_LABELS = [
    "latency_issue",
    "power_consumption_issue",
    "data_loss_risk",
    "reliability_issue",
    "performance_degradation",
    "storage_management_overhead",
    "manufacturing_complexity",
]

EFFECT_LABELS = [
    "latency_reduction",
    "latency_improvement",
    "power_reduction",
    "reliability_improvement",
    "performance_improvement",
    "data_recovery",
    "data_loss_prevention",
    "throughput_increase",
    "write_amplification_reduction",
    "endurance_improvement",
]


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def extract_json(text: str) -> dict[str, Any]:
    text = normalize_ws(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return {"parse_error": "json object not found", "raw": text}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {"parse_error": str(exc), "raw": text}


def ask_qwen(item: dict[str, Any]) -> dict[str, Any]:
    snippets = [
        {
            "id": s.get("id"),
            "page_no": s.get("page_no"),
            "section": s.get("section"),
            "text": normalize_ws(s.get("text"))[:650],
        }
        for s in item.get("problem_effect_snippets", [])[:6]
    ]
    prompt = f"""
You label patent problem/effect fields for a searchable patent dictionary.
Use only the evidence snippets. Do not guess beyond the snippets.
Return only JSON with these fields:
problem_labels: array of allowed problem labels
effect_labels: array of allowed effect labels
why_needed: one short phrase in the patent source language
expected_effect: one short phrase in the patent source language
confidence: number from 0 to 1

Allowed problem labels: {PROBLEM_LABELS}
Allowed effect labels: {EFFECT_LABELS}

Patent:
patent_id: {item.get("patent_id")}
country: {item.get("country")}
title: {item.get("title")}

Evidence snippets:
{json.dumps(snippets, ensure_ascii=False, indent=2)}
""".strip()
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 700,
            },
        },
        timeout=180,
    )
    response.raise_for_status()
    body = response.json()
    parsed = extract_json(body.get("response", ""))
    return {
        "patent_id": item.get("patent_id"),
        "country": item.get("country"),
        "title": item.get("title"),
        "probe_problem_labels": item.get("problem_labels"),
        "probe_effect_labels": item.get("effect_labels"),
        "qwen_prompt_result": parsed,
        "raw_response": body.get("response", ""),
        "total_duration_sec": round(float(body.get("total_duration", 0)) / 1_000_000_000, 2),
    }


def main() -> None:
    data = json.loads(PROBE_PATH.read_text(encoding="utf-8"))
    results = [ask_qwen(item) for item in data["results"]]
    OUT_PATH.write_text(json.dumps({"model": MODEL, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(OUT_PATH), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
