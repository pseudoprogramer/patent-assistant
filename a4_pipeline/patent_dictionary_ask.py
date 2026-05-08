from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List

import requests


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from patent_dictionary_search import DEFAULT_DB, lookup, search  # noqa: E402


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:14b"
DEFAULT_PACK_DB = Path(
    "/Volumes/외장 2TB/cpu2026/patent_hub/outputs/indexes/A4/patent_evidence_pack_index.sqlite"
)


QUESTION_QUERY_TERMS = [
    ("page buffer", ["page buffer", "페이지 버퍼", "페이지버퍼"]),
    ("bit line", ["bit line", "비트 라인", "비트라인"]),
    ("word line", ["word line", "워드 라인", "워드라인"]),
    ("erase voltage", ["erase voltage", "소거 전압", "소거전압"]),
    ("read voltage", ["read voltage", "읽기 전압", "읽기전압", "판독 전압"]),
    ("firmware", ["firmware", "펌웨어", "固件"]),
    ("ssd", ["ssd", "solid state drive", "솔리드 스테이트", "固态硬盘"]),
    ("plane address", ["plane address", "평면 주소", "평면주소", "平面地址"]),
    ("pad group", ["pad group", "패드 그룹", "패드그룹"]),
    ("garbage collection", ["garbage collection", "가비지 컬렉션", "垃圾回收"]),
    ("encryption", ["encryption", "decryption", "암호화", "복호화", "加密", "解密"]),
]


def normalize_ws(value: Any) -> str:
    return " ".join(str(value or "").split())


def infer_search_query(question: str) -> str:
    lower = question.lower()
    terms: List[str] = []
    for canonical, aliases in QUESTION_QUERY_TERMS:
        if any(alias.lower() in lower for alias in aliases):
            terms.append(canonical)
    return " ".join(dict.fromkeys(terms))


def compact_card(card: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "patent_id": card["patent_id"],
        "language": card["language"],
        "title": card["title"],
        "primary_claim_type": card["primary_claim_type"],
        "secondary_claim_types": card["secondary_claim_types"],
        "independent_claim_nos": card["independent_claim_nos"],
        "core_subject": card["core_subject"],
        "core_elements": card["core_elements"][:8],
        "problem_labels": card["problem_labels"][:6],
        "solution_labels": card["solution_labels"][:8],
        "effect_labels": card["effect_labels"][:6],
        "evidence_ids": card["evidence_ids"][:10],
        "confidence": card["confidence"],
        "qc_flags": card["qc_flags"],
    }


def load_json(value: Any) -> Any:
    try:
        return json.loads(value or "[]")
    except Exception:
        return []


def compact_claims(claims: List[Dict[str, Any]], limit: int, chars: int) -> List[Dict[str, Any]]:
    out = []
    for claim in claims[:limit]:
        out.append(
            {
                "claim_no": claim.get("claim_no"),
                "claim_type": claim.get("claim_type"),
                "text": normalize_ws(claim.get("text"))[:chars],
            }
        )
    return out


def pack_lookup(con: sqlite3.Connection, patent_id: str) -> Dict[str, Any] | None:
    con.row_factory = sqlite3.Row
    row = con.execute(
        """
        SELECT *
        FROM evidence_pack_index
        WHERE patent_id=?
        """,
        (patent_id,),
    ).fetchone()
    if not row:
        return None
    strong_claims = load_json(row["strong_claims_json"])
    support_claims = load_json(row["support_claims_json"])
    return {
        "patent_id": row["patent_id"],
        "language": row["source_language"],
        "title": row["title"],
        "title_quality": row["title_quality"],
        "minimal_title": row["title_source"],
        "primary_claim_type": row["primary_claim_type"],
        "secondary_claim_types": load_json(row["secondary_claim_types_json"]),
        "independent_claim_nos": load_json(row["independent_claim_nos_json"]),
        "core_subject": row["core_subject"],
        "core_elements": load_json(row["core_elements_json"])[:10],
        "problem_labels": load_json(row["problem_labels_json"])[:6],
        "solution_labels": load_json(row["solution_labels_json"])[:8],
        "effect_labels": load_json(row["effect_labels_json"])[:6],
        "strong_evidence_ids": load_json(row["strong_evidence_ids_json"])[:10],
        "weak_evidence_ids": load_json(row["weak_evidence_ids_json"])[:10],
        "strong_claims": compact_claims(strong_claims, limit=3, chars=900),
        "support_claims": compact_claims(support_claims, limit=2, chars=450),
        "confidence": row["confidence"],
        "quality_flags": load_json(row["quality_flags_json"]),
    }


def build_prompt_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not DEFAULT_PACK_DB.exists():
        return [compact_card(card) for card in cards]
    con = sqlite3.connect(DEFAULT_PACK_DB)
    out = []
    for card in cards:
        pack = pack_lookup(con, card["patent_id"])
        out.append(pack if pack else compact_card(card))
    con.close()
    return out


def build_prompt(question: str, cards: List[Dict[str, Any]]) -> str:
    evidence = json.dumps(build_prompt_cards(cards), ensure_ascii=False, indent=2)
    return f"""You are helping analyze a local patent dictionary.

Answer in Korean, but preserve original patent titles and key technical terms in their source language.
Use only the provided cleaned evidence packs. Do not invent patents, claims, or evidence.
Treat strong_claims and strong_evidence_ids as primary evidence.
Treat weak_evidence_ids and support_claims as supporting evidence only.
If quality_flags mention missing text, OCR repair, low confidence, or missing labels, say so clearly.

User question:
{question}

Retrieved cleaned evidence packs:
{evidence}

Write a concise answer. Keep it practical and avoid long background.
Use short bullets under these headings only:
1. 핵심 요약
2. 관련 특허 후보, with patent_id, why each matters, and key evidence_ids
3. 공통 기술축 / 차이점
"""


def ask_llm(prompt: str, model: str, timeout: int, num_predict: int) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 12000,
            "num_predict": num_predict,
        },
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return str(data.get("response") or data.get("thinking") or "").strip()


def print_context(cards: List[Dict[str, Any]]) -> None:
    print(f"Retrieved patents: {len(cards)}")
    for i, card in enumerate(cards, 1):
        print(f"\n[{i}] {card['patent_id']} ({card['language']}, {card['primary_claim_type']}, conf={card['confidence']})")
        print(f"Title: {card['title']}")
        print(f"Core: {card['core_subject']}")
        if card["core_elements"]:
            print("Elements: " + ", ".join(card["core_elements"][:6]))
        if card["solution_labels"]:
            print("Solution labels: " + ", ".join(card["solution_labels"][:6]))
        if card["evidence_ids"]:
            print("Evidence: " + ", ".join(card["evidence_ids"][:8]))
        if card["qc_flags"]:
            print("QC flags: " + ", ".join(card["qc_flags"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a Korean question against the local patent dictionary.")
    parser.add_argument("question", help="Question or search query")
    parser.add_argument("--query", default="", help="Optional shorter retrieval query. Useful when question is natural language.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--lang", choices=["zh", "en", "ko"], default="")
    parser.add_argument("--claim-type", choices=["device", "system", "method", "process", "unknown"], default="")
    parser.add_argument("--label", default="")
    parser.add_argument("--include-qc", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--num-predict", type=int, default=1000)
    parser.add_argument("--no-llm", action="store_true", help="Only print retrieved patent cards")
    parser.add_argument("--json", action="store_true", help="Emit retrieved cards as JSON")
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    retrieval_query = args.query or args.question
    exact = lookup(con, retrieval_query)
    cards = [exact] if exact else search(
        con,
        retrieval_query,
        args.limit,
        language=args.lang,
        claim_type=args.claim_type,
        label=args.label,
        include_qc=args.include_qc,
    )
    cards = [card for card in cards if card]
    if not cards and not args.query:
        fallback_query = infer_search_query(args.question)
        if fallback_query and fallback_query != retrieval_query:
            retrieval_query = fallback_query
            cards = search(
                con,
                retrieval_query,
                args.limit,
                language=args.lang,
                claim_type=args.claim_type,
                label=args.label,
                include_qc=args.include_qc,
            )
            cards = [card for card in cards if card]

    if args.json:
        print(json.dumps([compact_card(card) for card in cards], ensure_ascii=False, indent=2))
        return

    if retrieval_query != args.question:
        print(f"Retrieval query: {retrieval_query}")
    print_context(cards)
    if args.no_llm:
        return
    if not cards:
        print("\nLLM answer skipped: no retrieved patents.")
        return

    prompt = build_prompt(args.question, cards)
    print("\n--- LLM answer ---")
    print(ask_llm(prompt, args.model, args.timeout, args.num_predict))


if __name__ == "__main__":
    main()
