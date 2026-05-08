from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from llm_clients import LLMClient, json_from_text  # noqa: E402
from patent_dictionary_ask import DEFAULT_MODEL, build_prompt_cards  # noqa: E402
from patent_local_triage import DEFAULT_PACK_DB, search_packs, triage_question  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("/Volumes/외장 2TB/cpu2026/patent_hub/outputs/missions/A4")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,}|[가-힣]{2,}|[\u4e00-\u9fff]{2,}")


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = normalize_ws(item)
        if not value or value.lower() in seen:
            continue
        out.append(value)
        seen.add(value.lower())
    return out


def safe_slug(text: str, limit: int = 48) -> str:
    tokens = TOKEN_RE.findall(text.lower())
    slug = "_".join(tokens[:8]) or "mission"
    slug = re.sub(r"[^0-9a-zA-Z_\uac00-\ud7a3]+", "_", slug).strip("_")
    return slug[:limit] or "mission"


def fallback_queries(goal: str, max_queries: int) -> List[str]:
    tokens = TOKEN_RE.findall(goal)
    phrases = [goal]
    if tokens:
        phrases.append(" ".join(tokens[:8]))
        phrases.append(" ".join(tokens[:4]))
    return unique_keep_order(phrases)[:max_queries]


def plan_queries(goal: str, model: str, timeout: int, max_queries: int) -> Dict[str, Any]:
    prompt = f"""You are planning a local patent-dictionary research mission.

Goal:
{goal}

Return strict JSON only:
{{
  "queries": ["short keyword query 1", "short keyword query 2"],
  "success_criteria": ["what evidence would answer the goal"],
  "notes": "one short Korean note"
}}

Rules:
- Make {max_queries} or fewer queries.
- Queries should be short keyword searches, not long questions.
- Include original technical terms when useful.
- Do not invent patent ids.
"""
    try:
        client = LLMClient(provider="ollama", model=model, timeout=timeout)
        text = client.generate(prompt, max_tokens=700, temperature=0.0)
        data = json_from_text(text)
    except Exception as exc:
        data = {"queries": fallback_queries(goal, max_queries), "success_criteria": [], "notes": f"planner fallback: {exc}"}
    queries = unique_keep_order([str(item) for item in data.get("queries", [])])
    if not queries:
        queries = fallback_queries(goal, max_queries)
    data["queries"] = queries[:max_queries]
    data["success_criteria"] = unique_keep_order([str(item) for item in data.get("success_criteria", [])])[:6]
    data["notes"] = normalize_ws(data.get("notes"))
    return data


def merge_candidates(goal: str, queries: Sequence[str], per_query_limit: int, max_candidates: int) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for query in queries:
        result = triage_question(query, limit=per_query_limit)
        for rank, pack in enumerate(result.get("candidates", []), 1):
            patent_id = pack["patent_id"]
            entry = merged.get(patent_id)
            contribution = float(pack.get("score") or 0.0) + max(0.0, 2.0 - rank * 0.15)
            if not entry:
                pack = dict(pack)
                pack["mission_queries"] = [query]
                pack["mission_score"] = contribution
                pack["mission_hits"] = 1
                merged[patent_id] = pack
            else:
                entry["mission_queries"].append(query)
                entry["mission_score"] += contribution * 0.35
                entry["mission_hits"] += 1
                if float(pack.get("score") or 0.0) > float(entry.get("score") or 0.0):
                    for key in pack:
                        if key not in {"mission_queries", "mission_score", "mission_hits"}:
                            entry[key] = pack[key]

    if not merged:
        con = sqlite3.connect(DEFAULT_PACK_DB)
        try:
            for pack in search_packs(con, goal, limit=max_candidates):
                pack["mission_queries"] = [goal]
                pack["mission_score"] = float(pack.get("score") or 0.0)
                pack["mission_hits"] = 1
                merged[pack["patent_id"]] = pack
        finally:
            con.close()

    candidates = list(merged.values())
    for pack in candidates:
        flags = set(pack.get("quality_flags", []))
        pack["mission_queries"] = unique_keep_order(pack.get("mission_queries", []))
        pack["mission_score"] = round(float(pack.get("mission_score") or 0.0) + 0.8 * int(pack.get("mission_hits", 1)), 3)
        if "missing_strong_independent_claim_text" in flags:
            pack["mission_score"] -= 2.0
        if "contaminated_independent_claim_text_excluded" in flags:
            pack["mission_score"] -= 1.5
        if "low_minimal_confidence" in flags:
            pack["mission_score"] -= 0.8
    candidates.sort(key=lambda item: item["mission_score"], reverse=True)
    return candidates[:max_candidates]


def compact_mission_pack(pack: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "patent_id": pack["patent_id"],
        "language": pack.get("language"),
        "title": pack.get("title"),
        "title_quality": pack.get("title_quality"),
        "primary_claim_type": pack.get("primary_claim_type"),
        "independent_claim_nos": pack.get("independent_claim_nos", [])[:8],
        "core_subject": pack.get("core_subject"),
        "core_elements": pack.get("core_elements", [])[:10],
        "problem_labels": pack.get("problem_labels", [])[:8],
        "solution_labels": pack.get("solution_labels", [])[:10],
        "effect_labels": pack.get("effect_labels", [])[:8],
        "strong_evidence_ids": pack.get("strong_evidence_ids", [])[:8],
        "weak_evidence_ids": pack.get("weak_evidence_ids", [])[:8],
        "quality_flags": pack.get("quality_flags", [])[:8],
        "mission_score": pack.get("mission_score"),
        "mission_queries": pack.get("mission_queries", [])[:5],
        "why_selected": pack.get("why_selected", [])[:4],
    }


def write_report(
    goal: str,
    plan: Dict[str, Any],
    candidates: Sequence[Dict[str, Any]],
    model: str,
    timeout: int,
    output_dir: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"{stamp}_{safe_slug(goal)}"
    json_path = base.with_suffix(".json")
    md_path = base.with_suffix(".md")

    evidence = build_prompt_cards([{"patent_id": c["patent_id"]} for c in candidates])
    mission_cards = [compact_mission_pack(c) for c in candidates]
    prompt = f"""You are running an overnight patent research mission for a local patent dictionary.

Answer in Korean. Preserve original patent titles and technical terms.
Use only the provided candidate cards and cleaned evidence packs.
Be conservative: clearly mark weak evidence, missing independent claim text, OCR/title repairs, and low confidence.

Mission goal:
{goal}

Planner:
{json.dumps(plan, ensure_ascii=False, indent=2)}

Ranked candidate cards:
{json.dumps(mission_cards, ensure_ascii=False, indent=2)}

Cleaned evidence packs:
{json.dumps(evidence, ensure_ascii=False, indent=2)}

Write a practical Markdown report with these exact sections:
## 결론
## 핵심 후보
## 기술축별 묶음
## 근거가 강한 부분
## 불확실하거나 재검증할 부분
## 내일 이어서 할 일
"""
    started = time.monotonic()
    try:
        client = LLMClient(provider="ollama", model=model, timeout=timeout)
        report = client.generate(prompt, max_tokens=2200, temperature=0.1)
    except Exception as exc:
        report = fallback_report(goal, plan, candidates, exc)
    elapsed = round(time.monotonic() - started, 1)

    payload = {
        "goal": goal,
        "model": model,
        "elapsed_sec": elapsed,
        "planner": plan,
        "candidates": mission_cards,
        "report_path": str(md_path),
        "json_path": str(json_path),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(report.strip() + "\n", encoding="utf-8")
    payload["report"] = report.strip()
    return payload


def fallback_report(goal: str, plan: Dict[str, Any], candidates: Sequence[Dict[str, Any]], exc: Exception) -> str:
    lines = [
        "## 결론",
        f"로컬 LLM 리포트 생성은 실패했지만 후보 수집은 완료됐어. 오류: `{exc}`",
        "",
        "## 핵심 후보",
    ]
    for i, pack in enumerate(candidates[:12], 1):
        lines.extend(
            [
                f"{i}. `{pack['patent_id']}` - {pack.get('title')}",
                f"   - score={pack.get('mission_score')}, strong={', '.join(pack.get('strong_evidence_ids', [])[:5]) or '-'}",
                f"   - flags={', '.join(pack.get('quality_flags', [])[:5]) or '-'}",
            ]
        )
    lines.extend(["", "## 내일 이어서 할 일", "- 위 후보의 strong evidence 원문을 우선 확인하기"])
    return "\n".join(lines)


def run_mission(
    goal: str,
    model: str = DEFAULT_MODEL,
    max_queries: int = 5,
    per_query_limit: int = 8,
    max_candidates: int = 16,
    timeout: int = 360,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Dict[str, Any]:
    goal = normalize_ws(goal)
    if not goal:
        raise ValueError("Mission goal is empty")
    started = time.monotonic()
    plan = plan_queries(goal, model=model, timeout=timeout, max_queries=max_queries)
    candidates = merge_candidates(goal, plan["queries"], per_query_limit=per_query_limit, max_candidates=max_candidates)
    result = write_report(goal, plan, candidates, model=model, timeout=timeout, output_dir=output_dir)
    result["total_elapsed_sec"] = round(time.monotonic() - started, 1)
    return result


def format_mission_summary(result: Dict[str, Any], max_chars: int = 3200) -> str:
    candidates = result.get("candidates", [])
    lines = [
        "오토 미션 완료",
        f"- 목표: {result.get('goal')}",
        f"- 모델: {result.get('model')}",
        f"- 후보: {len(candidates)}건",
        f"- 총 소요: {result.get('total_elapsed_sec', result.get('elapsed_sec'))}초",
        f"- 리포트: {result.get('report_path')}",
        "",
        (result.get("report") or "").strip(),
    ]
    text = "\n".join(lines).strip()
    if len(text) > max_chars:
        text = text[: max_chars - 120].rstrip() + "\n\n...전체 리포트는 저장된 md 파일에서 이어서 보면 돼."
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an autonomous local patent research mission.")
    parser.add_argument("goal")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-queries", type=int, default=5)
    parser.add_argument("--per-query-limit", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=16)
    parser.add_argument("--timeout", type=int, default=360)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = run_mission(
        args.goal,
        model=args.model,
        max_queries=max(1, min(10, args.max_queries)),
        per_query_limit=max(1, min(30, args.per_query_limit)),
        max_candidates=max(1, min(40, args.max_candidates)),
        timeout=args.timeout,
        output_dir=Path(args.output_dir),
    )
    if args.json:
        print(json.dumps({k: v for k, v in result.items() if k != "report"}, ensure_ascii=False, indent=2))
    else:
        print(format_mission_summary(result, max_chars=8000))


if __name__ == "__main__":
    main()
