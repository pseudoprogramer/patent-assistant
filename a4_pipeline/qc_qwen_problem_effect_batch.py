from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


BASE = Path("/Volumes/외장 2TB/cpu2026")
DEFAULT_QWEN_DIR = BASE / "patent_hub" / "outputs" / "minimal_analysis" / "A4" / "qwen14_qc9_20260508"
DEFAULT_PROBE = BASE / "common" / "runtime" / "reports" / "A4" / "problem_effect_probe_qc9_20260508.json"
DEFAULT_OUT = BASE / "common" / "runtime" / "reports" / "A4" / "qwen14_qc9_20260508_report.json"


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def contamination_flags(card: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    title = normalize_ws(card.get("title_source"))
    if re.search(r"摘要|ABSTRACT|权利要求书|청구항|청구범위|\(\s*57\s*\)|Applicant:|Inventor:|Assignee:", title, flags=re.I):
        flags.append("title_contaminated")
    return flags


def qc_one(card: Dict[str, Any], probe: Dict[str, Any]) -> Dict[str, Any]:
    flags = contamination_flags(card)
    probe_snippet_count = int(probe.get("snippet_count") or len(probe.get("problem_effect_snippets", [])))
    if not card:
        flags.append("qwen_file_missing")
    if not card.get("problem_labels"):
        flags.append("qwen_problem_empty")
    if not card.get("effect_labels"):
        flags.append("qwen_effect_empty")
    if not card.get("core_elements"):
        flags.append("qwen_core_elements_empty")
    if not card.get("solution_labels"):
        flags.append("qwen_solution_empty")
    if not card.get("evidence_ids"):
        flags.append("qwen_evidence_empty")
    if float(card.get("confidence") or 0) < 0.7:
        flags.append("qwen_low_confidence")
    if probe_snippet_count == 0:
        flags.append("problem_effect_evidence_missing")
    if probe.get("problem_labels") and not card.get("problem_labels"):
        flags.append("gemini_candidate_problem_recoverable")
    if probe.get("effect_labels") and not card.get("effect_labels"):
        flags.append("gemini_candidate_effect_recoverable")
    needs_gemini = any(
        flag in flags
        for flag in [
            "qwen_problem_empty",
            "qwen_effect_empty",
            "qwen_low_confidence",
            "problem_effect_evidence_missing",
            "gemini_candidate_problem_recoverable",
            "gemini_candidate_effect_recoverable",
        ]
    )
    return {
        "flags": flags,
        "needs_gemini": needs_gemini,
        "qwen": {
            "title_source": card.get("title_source"),
            "core_subject": card.get("core_subject"),
            "problem_labels": card.get("problem_labels"),
            "effect_labels": card.get("effect_labels"),
            "solution_labels": card.get("solution_labels"),
            "confidence": card.get("confidence"),
            "evidence_ids": card.get("evidence_ids"),
        },
        "probe": {
            "title": probe.get("title"),
            "problem_labels": probe.get("problem_labels"),
            "effect_labels": probe.get("effect_labels"),
            "snippet_count": probe_snippet_count,
            "snippet_pages": [s.get("page_no") for s in probe.get("problem_effect_snippets", [])],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="QC Qwen minimal outputs against problem/effect evidence probes.")
    parser.add_argument("--qwen-dir", default=str(DEFAULT_QWEN_DIR))
    parser.add_argument("--probe", default=str(DEFAULT_PROBE))
    parser.add_argument("--patent-list-file", required=True)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    qwen_dir = Path(args.qwen_dir)
    probe_data = load_json(Path(args.probe))
    probes = {item["patent_id"]: item for item in probe_data.get("results", [])}
    patent_ids = [
        normalize_ws(line)
        for line in Path(args.patent_list_file).read_text(encoding="utf-8").splitlines()
        if normalize_ws(line) and not normalize_ws(line).startswith("#")
    ]

    items = []
    for patent_id in patent_ids:
        card = load_json(qwen_dir / f"{patent_id}.minimal.json")
        item = {"patent_id": patent_id, "country": patent_id[:2].upper(), **qc_one(card, probes.get(patent_id, {}))}
        items.append(item)

    summary = {
        "total": len(items),
        "qwen_files": sum(1 for item in items if "qwen_file_missing" not in item["flags"]),
        "qwen_problem_empty": sum(1 for item in items if "qwen_problem_empty" in item["flags"]),
        "qwen_effect_empty": sum(1 for item in items if "qwen_effect_empty" in item["flags"]),
        "problem_effect_evidence_missing": sum(1 for item in items if "problem_effect_evidence_missing" in item["flags"]),
        "needs_gemini": sum(1 for item in items if item["needs_gemini"]),
        "needs_gemini_by_country": {},
    }
    for country in sorted({item["country"] for item in items}):
        summary["needs_gemini_by_country"][country] = sum(
            1 for item in items if item["country"] == country and item["needs_gemini"]
        )

    report = {"summary": summary, "items": items}
    out = Path(args.out)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(out), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
