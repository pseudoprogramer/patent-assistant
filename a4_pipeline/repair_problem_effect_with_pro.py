from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from llm_clients import LLMClient, json_from_text
from gemini_cli_client import generate_with_gemini_cli


BASE = Path("/Volumes/외장 2TB/cpu2026")
DEFAULT_PROBE = BASE / "common" / "runtime" / "reports" / "A4" / "problem_effect_probe_qc9_20260508.json"
DEFAULT_OUT = BASE / "common" / "runtime" / "reports" / "A4" / "problem_effect_repair_gemini_qc9_20260508.json"

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


def load_patent_ids(path: Path) -> List[str]:
    return [
        normalize_ws(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if normalize_ws(line) and not normalize_ws(line).startswith("#")
    ]


def compact_snippets(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        {
            "id": snippet.get("id"),
            "page_no": snippet.get("page_no"),
            "section": snippet.get("section"),
            "trigger": snippet.get("trigger"),
            "text": normalize_ws(snippet.get("text"))[:700],
        }
        for snippet in item.get("problem_effect_snippets", [])[:6]
    ]


def compact_claims(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    claims = item.get("independent_claims") or item.get("claims") or []
    out = []
    for claim in claims[:4]:
        claim_no = str(claim.get("claim_no") or claim.get("id") or "").strip()
        text = normalize_ws(claim.get("raw_text") or claim.get("text") or claim.get("norm_text"))
        if not claim_no or not text:
            continue
        out.append(
            {
                "id": f"claim_{claim_no}",
                "claim_no": claim_no,
                "text": text[:900],
            }
        )
    return out


def flatten_ids(value: Any) -> List[str]:
    if isinstance(value, dict):
        out: List[str] = []
        for nested in value.values():
            out.extend(flatten_ids(nested))
        return out
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, tuple):
        return [str(x) for x in value]
    if value:
        return [str(value)]
    return []


def repair_one(
    client: LLMClient | None,
    item: Dict[str, Any],
    provider: str,
    model: str,
    timeout_sec: int,
) -> Dict[str, Any]:
    snippets = compact_snippets(item)
    claims = compact_claims(item)
    valid_snippet_ids = {str(snippet.get("id")) for snippet in snippets if snippet.get("id")}
    valid_claim_ids = {str(claim.get("id")) for claim in claims if claim.get("id")}
    prompt = f"""
You are repairing only the problem/solution/effect fields of a patent dictionary card.
Use only the provided independent claims and evidence snippets.
If evidence is weak, return empty arrays and lower confidence.
Return exactly one JSON object with:
- problem_labels: array, only from allowed problem labels
- solution_labels: array of concise lower_snake_case labels derived only from the independent claims
- effect_labels: array, only from allowed effect labels
- supporting_snippet_ids: array of snippet ids that directly support the selected labels
- supporting_claim_ids: array of claim ids that directly support selected solution labels
- why_needed: one concise phrase in the patent source language
- expected_effect: one concise phrase in the patent source language
- confidence: number 0..1

Rules:
- Do not select a label unless it is directly supported by at least one snippet id.
- Do not create solution_labels unless directly supported by at least one independent claim id.
- If no snippet directly supports a field, leave that field empty.
- If no independent claim directly supports a solution field, leave solution_labels empty.
- solution_labels must describe concrete technical means, not broad goals.
- why_needed and expected_effect must be copied or tightly paraphrased from the snippets.

Allowed problem labels:
{json.dumps(PROBLEM_LABELS, ensure_ascii=False)}

Allowed effect labels:
{json.dumps(EFFECT_LABELS, ensure_ascii=False)}

Patent:
patent_id={item.get("patent_id")}
country={item.get("country")}
title={item.get("title")}

Independent claims:
{json.dumps(claims, ensure_ascii=False, indent=2)}

Evidence snippets:
{json.dumps(snippets, ensure_ascii=False, indent=2)}
""".strip()
    started = time.monotonic()
    try:
        if provider == "gemini-cli":
            text = generate_with_gemini_cli(
                "Return only valid JSON. Do not include markdown or explanations.\n\n" + prompt,
                model=model,
                timeout=timeout_sec,
            )
        else:
            if client is None:
                raise RuntimeError("LLM client is not initialized")
            text = client.generate(
                prompt,
                instructions="Return only valid JSON. Do not include markdown or explanations.",
                max_tokens=900,
                temperature=0,
            )
        parsed = json_from_text(text)
        status = "success"
        error = ""
    except Exception as exc:
        parsed = {}
        text = ""
        status = "failed"
        error = repr(exc)
    support_ids = [str(x) for x in flatten_ids(parsed.get("supporting_snippet_ids")) if str(x) in valid_snippet_ids]
    claim_support_ids = [str(x) for x in flatten_ids(parsed.get("supporting_claim_ids")) if str(x) in valid_claim_ids]
    quality_flags = []
    if (parsed.get("problem_labels") or parsed.get("effect_labels")) and not support_ids:
        quality_flags.append("missing_supporting_snippet_ids")
    if parsed.get("solution_labels") and claims and not claim_support_ids:
        quality_flags.append("missing_supporting_claim_ids")
    if parsed.get("problem_labels") and not normalize_ws(parsed.get("why_needed")):
        quality_flags.append("missing_why_needed")
    if parsed.get("effect_labels") and not normalize_ws(parsed.get("expected_effect")):
        quality_flags.append("missing_expected_effect")
    if support_ids != parsed.get("supporting_snippet_ids"):
        parsed["supporting_snippet_ids"] = support_ids
    if claim_support_ids != parsed.get("supporting_claim_ids"):
        parsed["supporting_claim_ids"] = claim_support_ids
    return {
        "patent_id": item.get("patent_id"),
        "country": item.get("country"),
        "title": item.get("title"),
        "probe_problem_labels": item.get("problem_labels"),
        "probe_effect_labels": item.get("effect_labels"),
        "repair": parsed,
        "status": status,
        "error": error,
        "quality_flags": quality_flags,
        "quality_pass": status == "success" and not quality_flags,
        "elapsed": round(time.monotonic() - started, 2),
        "raw": text[:2000],
    }


def build_summary(provider: str, model: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    elapsed_values = [float(item.get("elapsed") or 0) for item in results if item.get("elapsed")]
    return {
        "provider": provider,
        "model": model,
        "total": len(results),
        "success": sum(1 for item in results if item["status"] == "success"),
        "failed": sum(1 for item in results if item["status"] != "success"),
        "repair_problem_nonempty": sum(1 for item in results if item.get("repair", {}).get("problem_labels")),
        "repair_effect_nonempty": sum(1 for item in results if item.get("repair", {}).get("effect_labels")),
        "quality_pass": sum(1 for item in results if item.get("quality_pass")),
        "avg_elapsed_sec": round(sum(elapsed_values) / len(elapsed_values), 2) if elapsed_values else 0,
    }


def write_report(out: Path, provider: str, model: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = build_summary(provider, model, results)
    out.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair problem/effect labels with a pro model using extracted evidence.")
    parser.add_argument("--probe", default=str(DEFAULT_PROBE))
    parser.add_argument("--patent-list-file", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--request-delay-sec", type=float, default=0)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    probe_data = json.loads(Path(args.probe).read_text(encoding="utf-8"))
    probes = {item["patent_id"]: item for item in probe_data.get("results", [])}
    patent_ids = load_patent_ids(Path(args.patent_list_file))
    provider = args.provider.strip().lower()
    model = args.model or ("gemini-2.5-flash-lite" if provider == "gemini-cli" else "")
    client = None if provider == "gemini-cli" else LLMClient(provider=args.provider, model=args.model or None, timeout=args.timeout_sec)
    actual_model = model if provider == "gemini-cli" else (client.model if client else "")
    out = Path(args.out)
    report_provider = provider if provider == "gemini-cli" else (client.provider if client else provider)

    results: List[Dict[str, Any]] = []
    done_ids = set()
    if args.resume and out.exists():
        existing = json.loads(out.read_text(encoding="utf-8"))
        results = list(existing.get("results", []))
        done_ids = {str(item.get("patent_id")) for item in results if item.get("patent_id")}

    target_ids = [patent_id for patent_id in patent_ids if patent_id in probes and patent_id not in done_ids]
    for idx, patent_id in enumerate(target_ids, 1):
        print(f"[repair] {idx}/{len(target_ids)} patent_id={patent_id}", flush=True)
        result = repair_one(
            client,
            probes[patent_id],
            provider=provider,
            model=actual_model,
            timeout_sec=args.timeout_sec,
        )
        results.append(result)
        print(
            f"[repair]   status={result['status']} quality_pass={result['quality_pass']} elapsed={result['elapsed']}s",
            flush=True,
        )
        if args.checkpoint_every > 0 and (idx % args.checkpoint_every == 0):
            write_report(out, report_provider, actual_model, results)
        if args.request_delay_sec > 0 and idx < len(target_ids):
            print(f"[repair]   sleep={args.request_delay_sec}s", flush=True)
            time.sleep(args.request_delay_sec)

    summary = write_report(out, report_provider, actual_model, results)
    print(json.dumps({"out": str(out), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
