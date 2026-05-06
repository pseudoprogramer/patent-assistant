from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from llm_clients import LLMClient, json_from_text  # noqa: E402
from patent_dictionary_ask import infer_search_query  # noqa: E402
from patent_dictionary_search import DEFAULT_DB, lookup, search  # noqa: E402

try:
    from build_evidence_units import DEFAULT_UNITS_DB  # noqa: E402
    from evidence_reranker import rank_evidence  # noqa: E402
except Exception:
    DEFAULT_UNITS_DB = Path("/Volumes/외장 2TB/cpu2026/patent_hub/outputs/indexes/A4/patent_evidence_units.sqlite")
    rank_evidence = None  # type: ignore[assignment]


DEFAULT_EVIDENCE_DB = Path("/Volumes/외장 2TB/cpu2026/common/runtime/db/patent_A4.sqlite")
PATENT_ID_RE = re.compile(r"\b(?:us|cn|kr)[a-z0-9]{6,}p\b", re.IGNORECASE)
PATENT_NUMBER_RE = re.compile(r"\d{7,}")


def unique_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        item = str(item or "").strip()
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def clip(value: Any, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:max_chars]


def extract_patent_id(text: str) -> str:
    match = PATENT_ID_RE.search(text or "")
    return match.group(0).lower() if match else ""


def fuzzy_lookup_cards(con: sqlite3.Connection, text: str, limit: int) -> List[Dict[str, Any]]:
    patent_id = extract_patent_id(text)
    if patent_id:
        card = lookup(con, patent_id)
        return [card] if card else []

    fragments = PATENT_NUMBER_RE.findall(text or "")
    if not fragments:
        return []
    lower = (text or "").lower()
    prefix = ""
    if "us" in lower or "미국" in lower:
        prefix = "us"
    elif "cn" in lower or "중국" in lower:
        prefix = "cn"
    elif "kr" in lower or "한국" in lower:
        prefix = "kr"

    out: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for fragment in fragments:
        stripped = fragment.lstrip("0") or fragment
        for pattern in unique_keep_order([f"%{fragment.lower()}%", f"%{stripped.lower()}%"]):
            params: List[Any] = [pattern]
            where = "patent_id LIKE ?"
            if prefix:
                where += " AND patent_id LIKE ?"
                params.append(f"{prefix}%")
            rows = con.execute(
                f"SELECT patent_id FROM minimal_index WHERE {where} ORDER BY length(patent_id), patent_id LIMIT ?",
                [*params, limit],
            ).fetchall()
            for (candidate_id,) in rows:
                if candidate_id in seen:
                    continue
                card = lookup(con, candidate_id)
                if card:
                    seen.add(candidate_id)
                    out.append(card)
                if len(out) >= limit:
                    return out
    return out


def fallback_plan(question: str, limit: int) -> Dict[str, Any]:
    inferred = infer_search_query(question)
    queries = unique_keep_order([inferred, question]) if inferred else [question]
    return {
        "intent": "answer",
        "search_queries": queries,
        "patent_numbers": PATENT_NUMBER_RE.findall(question or ""),
        "languages": [],
        "claim_types": [],
        "labels": [],
        "max_results": limit,
        "notes": "fallback plan",
    }


def make_query_plan(question: str, client: Optional[LLMClient], limit: int = 8) -> Dict[str, Any]:
    if client is None:
        return fallback_plan(question, limit)
    instructions = """You convert patent questions into precise local search plans.
Return JSON only. Do not answer the patent question.
Prefer short source-language technical terms and exact patent number fragments when present."""
    prompt = f"""User question:
{question}

Return this JSON schema:
{{
  "intent": "answer|compare|verify|cluster|strategy|lookup",
  "search_queries": ["short keyword query 1", "short keyword query 2"],
  "patent_numbers": ["number fragments or patent ids explicitly mentioned"],
  "languages": ["zh|en|ko"],
  "claim_types": ["device|method|system|process"],
  "labels": ["snake_case label if obvious"],
  "max_results": {limit},
  "notes": "brief Korean note about retrieval intent"
}}"""
    try:
        data = json_from_text(client.generate(prompt, instructions=instructions, max_tokens=700, temperature=0.0))
    except Exception as exc:
        data = fallback_plan(question, limit)
        data["planner_error"] = repr(exc)
    queries = data.get("search_queries") if isinstance(data.get("search_queries"), list) else []
    if not queries:
        queries = fallback_plan(question, limit)["search_queries"]
    data["search_queries"] = unique_keep_order([str(q) for q in queries])[:5]
    data["patent_numbers"] = unique_keep_order([str(x) for x in data.get("patent_numbers", [])] + PATENT_NUMBER_RE.findall(question or ""))[:8]
    try:
        data["max_results"] = max(1, min(20, int(data.get("max_results") or limit)))
    except Exception:
        data["max_results"] = limit
    return data


def compact_card(card: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "patent_id": card["patent_id"],
        "language": card["language"],
        "title": card["title"],
        "primary_claim_type": card["primary_claim_type"],
        "secondary_claim_types": card["secondary_claim_types"],
        "independent_claim_nos": card["independent_claim_nos"],
        "core_subject": card["core_subject"],
        "core_elements": card["core_elements"][:12],
        "problem_labels": card["problem_labels"][:8],
        "solution_labels": card["solution_labels"][:12],
        "effect_labels": card["effect_labels"][:8],
        "evidence_ids": card["evidence_ids"][:16],
        "confidence": card["confidence"],
        "qc_flags": card["qc_flags"],
        "score": card.get("score"),
    }


def retrieve_cards(plan: Dict[str, Any], index_db: Path = DEFAULT_DB) -> List[Dict[str, Any]]:
    con = sqlite3.connect(index_db)
    cards: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    limit = int(plan.get("max_results") or 8)

    for number in plan.get("patent_numbers", []) or []:
        for card in fuzzy_lookup_cards(con, str(number), limit=limit):
            if card["patent_id"] not in seen:
                cards.append(card)
                seen.add(card["patent_id"])

    language = ""
    languages = plan.get("languages") if isinstance(plan.get("languages"), list) else []
    if len(languages) == 1 and languages[0] in {"zh", "en", "ko"}:
        language = languages[0]

    for query in plan.get("search_queries", []) or []:
        for card in search(con, str(query), limit, language=language):
            if card["patent_id"] not in seen:
                cards.append(card)
                seen.add(card["patent_id"])
            if len(cards) >= limit:
                break
        if len(cards) >= limit:
            break

    con.close()
    return cards[:limit]


def evidence_ids_to_claim_nos(card: Dict[str, Any]) -> List[str]:
    claim_nos = []
    for eid in card.get("evidence_ids", []) or []:
        m = re.match(r"claim_(.+)", str(eid))
        if m:
            claim_nos.append(m.group(1))
    claim_nos.extend(card.get("independent_claim_nos", []) or [])
    return unique_keep_order(claim_nos)[:8]


def evidence_ids_to_fig_nos(card: Dict[str, Any]) -> List[str]:
    fig_nos = []
    for eid in card.get("evidence_ids", []) or []:
        m = re.match(r"fig_(.+)", str(eid))
        if m:
            fig_nos.append(m.group(1))
    return unique_keep_order(fig_nos)[:8]


def fetch_evidence_for_card(con: sqlite3.Connection, card: Dict[str, Any]) -> Dict[str, Any]:
    patent_id = card["patent_id"]
    out: Dict[str, Any] = {"patent_id": patent_id, "claims": [], "figures": []}

    for claim_no in evidence_ids_to_claim_nos(card):
        row = con.execute(
            "SELECT claim_no, claim_type, raw_text, norm_text, page_start, page_end FROM claims WHERE patent_id=? AND claim_no=?",
            (patent_id, claim_no),
        ).fetchone()
        if row:
            out["claims"].append({
                "claim_no": row[0],
                "claim_type": row[1],
                "text": clip(row[2] or row[3], 1200),
                "page_start": row[4],
                "page_end": row[5],
            })

    for fig_no in evidence_ids_to_fig_nos(card):
        rows = con.execute(
            "SELECT figure_no, caption_raw, caption_norm, page_no FROM figure_captions WHERE patent_id=? AND figure_no=? LIMIT 3",
            (patent_id, fig_no),
        ).fetchall()
        for row in rows:
            out["figures"].append({
                "figure_no": row[0],
                "caption": clip(row[1] or row[2], 400),
                "page_no": row[3],
            })
    return out


def build_evidence_pack(
    question: str,
    planner_client: Optional[LLMClient] = None,
    index_db: Path = DEFAULT_DB,
    evidence_db: Path = DEFAULT_EVIDENCE_DB,
    units_db: Path = DEFAULT_UNITS_DB,
    limit: int = 8,
) -> Dict[str, Any]:
    plan = make_query_plan(question, planner_client, limit=limit)
    if rank_evidence is not None and Path(units_db).exists():
        ranked = rank_evidence(question, plan, index_db=Path(index_db), units_db=Path(units_db), limit=limit)
        cards = [item["card"] for item in ranked]
        return {
            "question": question,
            "query_plan": plan,
            "retrieved_cards": [compact_card(card) for card in cards],
            "ranked_candidates": [
                {
                    "patent_id": item["patent_id"],
                    "score": item["score"],
                    "why_selected": item["why_selected"],
                    "weaknesses": item["weaknesses"],
                    "top_units": item["top_units"],
                }
                for item in ranked
            ],
            "evidence": [
                {
                    "patent_id": item["patent_id"],
                    "units": item["top_units"],
                    "why_selected": item["why_selected"],
                    "weaknesses": item["weaknesses"],
                }
                for item in ranked
            ],
            "limits": {"max_results": limit, "claim_chars": 1200, "reranker": "evidence_units"},
        }

    cards = retrieve_cards(plan, index_db=index_db)
    evidence: List[Dict[str, Any]] = []
    if evidence_db.exists():
        con = sqlite3.connect(evidence_db)
        for card in cards:
            evidence.append(fetch_evidence_for_card(con, card))
        con.close()
    return {
        "question": question,
        "query_plan": plan,
        "retrieved_cards": [compact_card(card) for card in cards],
        "evidence": evidence,
        "limits": {"max_results": limit, "claim_chars": 1200},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an evidence pack for pro-model patent judgment.")
    parser.add_argument("question")
    parser.add_argument("--provider", default="none", choices=["none", "auto", "openai", "gemini", "ollama"])
    parser.add_argument("--model", default="")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--units-db", default=str(DEFAULT_UNITS_DB))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    client = None if args.provider == "none" else LLMClient(args.provider, model=args.model or None)
    pack = build_evidence_pack(args.question, planner_client=client, units_db=Path(args.units_db), limit=args.limit)
    print(json.dumps(pack, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
