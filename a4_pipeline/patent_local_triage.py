from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from patent_dictionary_search import expand_query


DEFAULT_PACK_DB = Path(
    "/Volumes/외장 2TB/cpu2026/patent_hub/outputs/indexes/A4/patent_evidence_pack_index.sqlite"
)
PATENT_ID_RE = re.compile(r"\b(?:us|cn|kr)[a-z0-9]{6,}p\b", re.I)
PATENT_NUMBER_RE = re.compile(r"\d{7,}")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+-]{2,}|[가-힣]{2,}|[\u4e00-\u9fff]{2,}")

GEMINI_WORTHY_TERMS = {
    "비교", "차이", "더 좋은", "우선", "전략", "검증", "맞는지", "리스크", "회피",
    "compare", "difference", "better", "strategy", "verify", "risk",
}
LOW_RISK_FLAGS = {"missing_effect_labels", "missing_some_independent_claim_text"}
HIGH_RISK_FLAGS = {
    "missing_strong_independent_claim_text",
    "contaminated_independent_claim_text_excluded",
    "low_minimal_confidence",
    "source_language_mismatch",
}


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_json(value: Any) -> Any:
    try:
        return json.loads(value or "[]")
    except Exception:
        return []


def unique_keep_order(items: Iterable[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = normalize_ws(item)
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def fts_phrase(term: str) -> str:
    return f'"{term.replace(chr(34), chr(34) * 2)}"'


def fts_query(query: str) -> str:
    terms = []
    for value in expand_query(query):
        terms.append(value)
        terms.extend(TOKEN_RE.findall(value))
    terms = [term for term in unique_keep_order(terms) if len(term) >= 2]
    return " OR ".join(fts_phrase(term) for term in terms[:24])


def extract_patent_prefix(text: str) -> str:
    lower = text.lower()
    if "us" in lower or "미국" in lower:
        return "us"
    if "cn" in lower or "중국" in lower:
        return "cn"
    if "kr" in lower or "한국" in lower:
        return "kr"
    return ""


def row_to_pack(row: sqlite3.Row, score: float = 0.0) -> Dict[str, Any]:
    strong_claims = load_json(row["strong_claims_json"])
    support_claims = load_json(row["support_claims_json"])
    return {
        "patent_id": row["patent_id"],
        "language": row["source_language"],
        "title": row["title"],
        "title_quality": row["title_quality"],
        "core_subject": row["core_subject"],
        "primary_claim_type": row["primary_claim_type"],
        "independent_claim_nos": load_json(row["independent_claim_nos_json"]),
        "core_elements": load_json(row["core_elements_json"]),
        "problem_labels": load_json(row["problem_labels_json"]),
        "solution_labels": load_json(row["solution_labels_json"]),
        "effect_labels": load_json(row["effect_labels_json"]),
        "strong_evidence_ids": load_json(row["strong_evidence_ids_json"]),
        "weak_evidence_ids": load_json(row["weak_evidence_ids_json"]),
        "quality_flags": load_json(row["quality_flags_json"]),
        "confidence": float(row["confidence"] or 0.0),
        "strong_claims": strong_claims,
        "support_claims": support_claims,
        "score": round(score, 3),
    }


def direct_patent_matches(con: sqlite3.Connection, question: str, limit: int) -> List[Dict[str, Any]]:
    con.row_factory = sqlite3.Row
    match = PATENT_ID_RE.search(question or "")
    candidates: List[str] = []
    if match:
        candidates.append(match.group(0).lower())
    prefix = extract_patent_prefix(question)
    for fragment in PATENT_NUMBER_RE.findall(question or ""):
        stripped = fragment.lstrip("0") or fragment
        candidates.extend([fragment.lower(), stripped.lower()])
    seen = set()
    packs: List[Dict[str, Any]] = []
    for candidate in unique_keep_order(candidates):
        if PATENT_ID_RE.fullmatch(candidate):
            rows = con.execute("SELECT *, 0.0 AS bm25_score FROM evidence_pack_index WHERE patent_id=?", (candidate,)).fetchall()
        else:
            where = "patent_id LIKE ?"
            params: List[Any] = [f"%{candidate}%"]
            if prefix:
                where += " AND patent_id LIKE ?"
                params.append(f"{prefix}%")
            rows = con.execute(
                f"SELECT *, 0.0 AS bm25_score FROM evidence_pack_index WHERE {where} ORDER BY length(patent_id), patent_id LIMIT ?",
                [*params, limit],
            ).fetchall()
        for row in rows:
            if row["patent_id"] in seen:
                continue
            seen.add(row["patent_id"])
            pack = row_to_pack(row, score=100.0)
            pack["why_selected"] = ["explicit patent number match"]
            packs.append(pack)
            if len(packs) >= limit:
                return packs
    return packs


def search_packs(con: sqlite3.Connection, question: str, limit: int, pool: int = 40) -> List[Dict[str, Any]]:
    con.row_factory = sqlite3.Row
    direct = direct_patent_matches(con, question, limit)
    if direct:
        return direct[:limit]
    match = fts_query(question)
    if not match:
        return []
    rows = con.execute(
        """
        SELECT epi.*, bm25(evidence_pack_fts) AS bm25_score
        FROM evidence_pack_fts fts
        JOIN evidence_pack_index epi ON epi.patent_id = fts.patent_id
        WHERE evidence_pack_fts MATCH ?
        ORDER BY bm25(evidence_pack_fts) ASC, epi.confidence DESC
        LIMIT ?
        """,
        (match, max(limit, pool)),
    ).fetchall()
    packs = []
    for row in rows:
        bm25_score = float(row["bm25_score"] or 0.0)
        score = max(0.0, min(20.0, -bm25_score))
        score += term_fit_score(question, row)
        score += float(row["confidence"] or 0.0)
        flags = set(load_json(row["quality_flags_json"]))
        score -= 0.8 * len(flags & HIGH_RISK_FLAGS)
        score -= 0.2 * len(flags & LOW_RISK_FLAGS)
        pack = row_to_pack(row, score=score)
        pack["why_selected"] = explain_match(question, pack)
        packs.append(pack)
    packs.sort(key=lambda item: item["score"], reverse=True)
    return packs[:limit]


def term_fit_score(question: str, row: sqlite3.Row) -> float:
    query_terms = [
        term.lower()
        for term in unique_keep_order([*expand_query(question), *TOKEN_RE.findall(question.lower())])
        if len(term) >= 2
    ]
    if not query_terms:
        return 0.0
    labels = " ".join(
        [
            " ".join(load_json(row["problem_labels_json"])),
            " ".join(load_json(row["solution_labels_json"])),
            " ".join(load_json(row["effect_labels_json"])),
        ]
    ).lower()
    elements = " ".join(load_json(row["core_elements_json"])).lower()
    title = normalize_ws(row["title"]).lower()
    claims = " ".join(claim.get("text", "") for claim in load_json(row["strong_claims_json"])).lower()
    score = 0.0
    for term in query_terms:
        if term in title:
            score += 1.6
        if term in labels:
            score += 1.3
        if term in elements:
            score += 1.0
        if term in claims:
            score += 0.45
    phrase_hits = [
        "page buffer",
        "bit line",
        "program verify",
        "evaluation operation",
        "plane address",
        "pad group",
        "garbage collection",
    ]
    q_lower = question.lower()
    for phrase in phrase_hits:
        if phrase in q_lower and (phrase in title or phrase.replace(" ", "_") in labels or phrase in claims):
            score += 2.0
    return min(score, 12.0)


def explain_match(question: str, pack: Dict[str, Any]) -> List[str]:
    text = " ".join(
        [
            pack.get("title", ""),
            pack.get("core_subject", ""),
            " ".join(pack.get("core_elements", [])),
            " ".join(pack.get("solution_labels", [])),
            " ".join(pack.get("strong_evidence_ids", [])),
        ]
    ).lower()
    terms = []
    for term in unique_keep_order([*expand_query(question), *TOKEN_RE.findall(question.lower())]):
        if len(term) >= 2 and term.lower() in text:
            terms.append(term)
    reasons = []
    if terms:
        reasons.append("matched: " + ", ".join(terms[:6]))
    if pack.get("strong_evidence_ids"):
        reasons.append("has strong independent-claim evidence")
    if pack.get("quality_flags"):
        reasons.append("quality flags: " + ", ".join(pack["quality_flags"][:4]))
    return reasons[:4] or ["matched evidence pack search"]


def gemini_recommendation(question: str, packs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    text = question.lower()
    reasons: List[str] = []
    if any(term in text for term in GEMINI_WORTHY_TERMS):
        reasons.append("question asks for comparison/verification/strategy")
    if len(packs) >= 4:
        reasons.append("many plausible candidates")
    risky = [pack for pack in packs if set(pack.get("quality_flags", [])) & HIGH_RISK_FLAGS]
    if risky:
        reasons.append("some top candidates have high-risk quality flags")
    close_scores = len(packs) >= 2 and abs(float(packs[0]["score"]) - float(packs[1]["score"])) < 0.8
    if close_scores:
        reasons.append("top candidate scores are close")
    need = bool(reasons)
    return {
        "needed": need,
        "level": "recommended" if need else "not_needed",
        "reasons": reasons or ["local evidence is narrow enough for a first answer"],
    }


def triage_question(question: str, limit: int = 8, db_path: Path = DEFAULT_PACK_DB) -> Dict[str, Any]:
    con = sqlite3.connect(db_path)
    packs = search_packs(con, question, limit=limit)
    con.close()
    return {
        "question": question,
        "limit": limit,
        "candidates": packs,
        "gemini": gemini_recommendation(question, packs),
    }


def format_triage(result: Dict[str, Any]) -> str:
    candidates = result.get("candidates", [])
    if not candidates:
        return "로컬 triage 결과: 후보를 찾지 못했어. 키워드를 더 짧게 넣어줘."
    gemini = result.get("gemini", {})
    lines = [
        f"로컬 triage 결과: {len(candidates)}건",
        f"Gemini 필요도: {gemini.get('level', 'unknown')} - " + "; ".join(gemini.get("reasons", [])[:3]),
    ]
    for i, pack in enumerate(candidates, 1):
        labels = ", ".join(pack.get("solution_labels", [])[:5])
        strong = ", ".join(pack.get("strong_evidence_ids", [])[:5])
        weak = ", ".join(pack.get("weak_evidence_ids", [])[:4])
        flags = ", ".join(pack.get("quality_flags", [])[:4])
        reasons = "; ".join(pack.get("why_selected", [])[:3])
        lines.extend(
            [
                "",
                f"[{i}] {pack['patent_id']} ({pack['language']}, {pack['primary_claim_type']}, score={pack['score']})",
                f"Title: {pack['title']}",
                f"Core: {pack['core_subject']}",
                f"Independent: {', '.join(pack.get('independent_claim_nos', [])) or '-'}",
            ]
        )
        if labels:
            lines.append(f"Labels: {labels}")
        if strong:
            lines.append(f"Strong: {strong}")
        if weak:
            lines.append(f"Weak: {weak}")
        if flags:
            lines.append(f"Flags: {flags}")
        if reasons:
            lines.append(f"Why: {reasons}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local pre-triage against cleaned A4 patent evidence packs.")
    parser.add_argument("question")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--db", default=str(DEFAULT_PACK_DB))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = triage_question(args.question, limit=max(1, min(30, args.limit)), db_path=Path(args.db))
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_triage(result))


if __name__ == "__main__":
    main()
