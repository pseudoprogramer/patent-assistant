from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_evidence_units import DEFAULT_UNITS_DB  # noqa: E402
from patent_dictionary_search import DEFAULT_DB, expand_query, lookup, search  # noqa: E402


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,}|[가-힣]{2,}|[\u4e00-\u9fff]{2,}")
NOISY_TITLE_RE = re.compile(r"(onfrouler|g1iic|ceeee|wees|o\.\.|frorn|vaive)", re.I)
GENERIC_TITLE_RE = re.compile(r"^(memory device|memory system|method|system|device|program operation memory)$", re.I)


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_json(value: Any) -> Any:
    try:
        return json.loads(value or "[]")
    except Exception:
        return []


def unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        item = normalize_ws(item)
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def query_terms(question: str, plan: Dict[str, Any]) -> List[str]:
    values: List[str] = [question]
    values.extend([str(x) for x in plan.get("search_queries", []) or []])
    values.extend([str(x) for x in plan.get("labels", []) or []])
    expanded: List[str] = []
    for value in values:
        expanded.extend(expand_query(value))
        expanded.extend(TOKEN_RE.findall(value.lower()))
    stop = {
        "관련", "특허", "후보", "비교", "정리", "알려줘", "대해서", "기술",
        "patent", "candidate", "compare", "method", "device", "system",
    }
    return [x for x in unique_keep_order(expanded) if x.lower() not in stop and len(x) >= 2][:40]


def fts_query_for_terms(terms: List[str]) -> str:
    phrases = []
    for term in terms[:16]:
        escaped = term.replace('"', '""')
        phrases.append(f'"{escaped}"')
    return " OR ".join(phrases)


def row_to_unit(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "unit_id": row["unit_id"],
        "patent_id": row["patent_id"],
        "source_language": row["source_language"],
        "unit_type": row["unit_type"],
        "unit_ref": row["unit_ref"],
        "claim_no": row["claim_no"],
        "claim_type": row["claim_type"],
        "is_independent_claim": bool(row["is_independent_claim"]),
        "page_no": row["page_no"],
        "text": row["text"],
        "source_weight": float(row["source_weight"] or 1.0),
        "quality_flags": load_json(row["quality_flags_json"]),
        "minimal_labels": load_json(row["minimal_labels_json"]),
        "minimal_elements": load_json(row["minimal_elements_json"]),
        "title_source": row["title_source"],
        "primary_claim_type": row["primary_claim_type"],
        "confidence": float(row["confidence"] or 0.0),
        "qc_flags": load_json(row["qc_flags_json"]),
        "bm25": row["bm25"] if "bm25" in row.keys() else None,
    }


def candidate_cards(plan: Dict[str, Any], index_db: Path, limit: int) -> Dict[str, Dict[str, Any]]:
    con = sqlite3.connect(index_db)
    cards: Dict[str, Dict[str, Any]] = {}
    patent_numbers = [str(x) for x in plan.get("patent_numbers", []) or []]
    for value in patent_numbers:
        exact = lookup(con, value.lower())
        if exact:
            cards[exact["patent_id"]] = exact
        rows = con.execute(
            "SELECT patent_id FROM minimal_index WHERE patent_id LIKE ? LIMIT ?",
            (f"%{value.lstrip('0') or value.lower()}%", limit),
        ).fetchall()
        for (patent_id,) in rows:
            card = lookup(con, patent_id)
            if card:
                cards[patent_id] = card

    language = ""
    languages = plan.get("languages") if isinstance(plan.get("languages"), list) else []
    if len(languages) == 1 and languages[0] in {"zh", "en", "ko"}:
        language = languages[0]
    for query in plan.get("search_queries", []) or []:
        for card in search(con, str(query), limit * 3, language=language):
            cards.setdefault(card["patent_id"], card)
            if len(cards) >= limit * 4:
                break
    con.close()
    return cards


def fetch_units_by_fts(units_con: sqlite3.Connection, terms: List[str], limit: int) -> List[Dict[str, Any]]:
    match = fts_query_for_terms(terms)
    if not match:
        return []
    units_con.row_factory = sqlite3.Row
    sql = """
        SELECT eu.*, bm25(evidence_units_fts) AS bm25
        FROM evidence_units_fts fts
        JOIN evidence_units eu ON eu.unit_id = fts.unit_id
        WHERE evidence_units_fts MATCH ?
        ORDER BY bm25(evidence_units_fts) ASC, eu.source_weight DESC
        LIMIT ?
    """
    return [row_to_unit(row) for row in units_con.execute(sql, (match, limit)).fetchall()]


def fetch_units_for_patents(units_con: sqlite3.Connection, patent_ids: Iterable[str], limit_per_patent: int = 10) -> List[Dict[str, Any]]:
    units_con.row_factory = sqlite3.Row
    out: List[Dict[str, Any]] = []
    for patent_id in patent_ids:
        rows = units_con.execute(
            """
            SELECT *, 0.0 AS bm25
            FROM evidence_units
            WHERE patent_id=?
            ORDER BY
                CASE unit_type WHEN 'claim' THEN 0 WHEN 'minimal_summary' THEN 1 WHEN 'title' THEN 2 ELSE 3 END,
                is_independent_claim DESC,
                source_weight DESC
            LIMIT ?
            """,
            (patent_id, limit_per_patent),
        ).fetchall()
        out.extend(row_to_unit(row) for row in rows)
    return out


def score_unit(unit: Dict[str, Any], terms: List[str]) -> Tuple[float, List[str], List[str]]:
    text = " ".join(
        [
            unit.get("text", ""),
            " ".join(unit.get("minimal_labels", [])),
            " ".join(unit.get("minimal_elements", [])),
            unit.get("title_source", ""),
        ]
    ).lower()
    why: List[str] = []
    weaknesses: List[str] = []
    hits = [term for term in terms if term.lower() in text]
    score = 0.0

    if hits:
        score += min(5.0, len(hits) * 0.7)
        why.append(f"{unit['unit_ref']} matches terms: " + ", ".join(hits[:6]))
    if unit["unit_type"] == "claim":
        score += 2.0
        why.append(f"{unit['unit_ref']} is claim evidence")
    quality_flags = unit.get("quality_flags", [])
    dependent_like = "dependent_claim_reference" in quality_flags or unit.get("claim_type") == "dependent_inferred"
    if unit.get("is_independent_claim") and not dependent_like:
        score += 1.4
        why.append(f"{unit['unit_ref']} is an independent claim")
    elif unit["unit_type"] == "claim" and dependent_like:
        score -= 1.6
        weaknesses.append(f"{unit['unit_ref']} appears dependent despite parser claim_type")
    if unit["unit_type"] == "minimal_summary":
        score += 0.7
    if unit["unit_type"] == "figure":
        score -= 0.5
        weaknesses.append(f"{unit['unit_ref']} is figure/caption evidence, weaker than claim text")

    bm25 = unit.get("bm25")
    if bm25 is not None:
        try:
            score += max(0.0, min(2.0, -float(bm25) / 4.0))
        except Exception:
            pass
    score += unit.get("source_weight", 1.0) * 0.4
    score += min(0.8, max(0.0, unit.get("confidence", 0.0) - 0.5))

    qc_flags = unit.get("qc_flags", [])
    if quality_flags:
        score -= 0.6 * len(quality_flags)
        weaknesses.append("unit quality flags: " + ", ".join(quality_flags))
    if "publication_line_only" in quality_flags:
        score -= 2.0
    if "short_claim_text" in quality_flags:
        score -= 1.2
    if qc_flags:
        score -= 0.8 * len(qc_flags)
        weaknesses.append("minimal QC flags: " + ", ".join(qc_flags))
    title = unit.get("title_source", "")
    if NOISY_TITLE_RE.search(title):
        score -= 1.0
        weaknesses.append("title appears OCR-noisy")
    if GENERIC_TITLE_RE.search(title):
        score -= 0.4
        weaknesses.append("title/core subject appears generic")
    if not hits:
        weaknesses.append(f"{unit['unit_ref']} has no direct query-term hit")

    return score, why, weaknesses


def rank_evidence(
    question: str,
    plan: Dict[str, Any],
    index_db: Path = DEFAULT_DB,
    units_db: Path = DEFAULT_UNITS_DB,
    limit: int = 8,
    units_per_patent: int = 4,
) -> List[Dict[str, Any]]:
    terms = query_terms(question, plan)
    cards = candidate_cards(plan, index_db=index_db, limit=limit)
    units_con = sqlite3.connect(units_db)
    units = fetch_units_by_fts(units_con, terms, limit=max(100, limit * 20))
    units.extend(fetch_units_for_patents(units_con, cards.keys(), limit_per_patent=8))
    units_con.close()

    by_patent: Dict[str, List[Tuple[float, Dict[str, Any], List[str], List[str]]]] = defaultdict(list)
    for unit in units:
        score, why, weaknesses = score_unit(unit, terms)
        by_patent[unit["patent_id"]].append((score, unit, why, weaknesses))

    results: List[Dict[str, Any]] = []
    index_con = sqlite3.connect(index_db)
    for patent_id, scored_units in by_patent.items():
        scored_units.sort(key=lambda item: item[0], reverse=True)
        top = scored_units[:units_per_patent]
        card = cards.get(patent_id)
        if card is None:
            card = lookup(index_con, patent_id)
        if card is None:
            # Build a light card from unit metadata when this patent came only from unit FTS.
            unit0 = top[0][1]
            card = {
                "patent_id": patent_id,
                "language": unit0.get("source_language", ""),
                "title": unit0.get("title_source", ""),
                "primary_claim_type": unit0.get("primary_claim_type", ""),
                "secondary_claim_types": [],
                "independent_claim_nos": [],
                "core_subject": unit0.get("title_source", ""),
                "core_elements": unit0.get("minimal_elements", []),
                "problem_labels": [],
                "solution_labels": unit0.get("minimal_labels", []),
                "effect_labels": [],
                "evidence_ids": [u[1]["unit_ref"] for u in top],
                "confidence": unit0.get("confidence", 0.0),
                "qc_flags": unit0.get("qc_flags", []),
                "json_path": "",
                "score": None,
            }
        patent_score = sum(item[0] for item in top) + math.log1p(len(scored_units)) * 0.3
        why_all = unique_keep_order(reason for item in top for reason in item[2])[:8]
        weak_all = unique_keep_order(reason for item in top for reason in item[3])[:8]
        results.append(
            {
                "patent_id": patent_id,
                "score": round(patent_score, 3),
                "card": card,
                "why_selected": why_all,
                "weaknesses": weak_all,
                "top_units": [
                    {
                        "unit_id": item[1]["unit_id"],
                        "unit_type": item[1]["unit_type"],
                        "unit_ref": item[1]["unit_ref"],
                        "claim_no": item[1]["claim_no"],
                        "claim_type": item[1]["claim_type"],
                        "page_no": item[1]["page_no"],
                        "score": round(item[0], 3),
                        "text": normalize_ws(item[1]["text"])[:1200],
                        "quality_flags": item[1]["quality_flags"],
                    }
                    for item in top
                ],
            }
        )

    index_con.close()
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank local patent evidence candidates with explainable features.")
    parser.add_argument("question")
    parser.add_argument("--plan-json", default="")
    parser.add_argument("--index-db", default=str(DEFAULT_DB))
    parser.add_argument("--units-db", default=str(DEFAULT_UNITS_DB))
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    plan = json.loads(args.plan_json) if args.plan_json else {"search_queries": [args.question], "max_results": args.limit}
    ranked = rank_evidence(args.question, plan, Path(args.index_db), Path(args.units_db), limit=args.limit)
    print(json.dumps(ranked, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
