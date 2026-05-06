from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Sequence


DEFAULT_DB = Path(
    "/Volumes/외장 2TB/cpu2026/patent_hub/outputs/indexes/A4/patent_minimal_index.sqlite"
)

QUERY_ALIASES = {
    "메모리 평면": ["memory plane", "multiple memory planes", "multi_plane_architecture", "多个存储器平面"],
    "평면 주소": ["plane address", "no plane address", "no_plane_address_input", "不接收平面地址", "平面地址"],
    "plane address": ["plane address", "no plane address", "no_plane_address_input", "不接收平面地址", "平面地址"],
    "독립 명령": ["independent command", "command and address", "独立接收命令和地址", "independent command address"],
    "independent command": ["independent command", "command and address", "独立接收命令和地址", "independent command address"],
    "패드 그룹": ["pad group", "plane dedicated pad", "plane_dedicated_pad_group", "平面专用焊盘组"],
    "pad group": ["pad group", "plane dedicated pad", "plane_dedicated_pad_group", "平面专用焊盘组"],
    "데이터 경로": ["data path", "separate_data_path", "数据路径", "데이터 경로"],
    "페이지 버퍼": ["page buffer", "page_buffer_circuit", "页缓冲器", "페이지 버퍼"],
    "워드 라인": ["word line", "word_line_control", "字线", "워드 라인"],
    "비트 라인": ["bit line", "bit_line_connection", "位线", "비트 라인"],
    "읽기 전압": ["read voltage", "read_reference_voltage_control", "读取电压", "읽기 전압"],
    "garbage collection": ["garbage collection", "flash_garbage_collection", "垃圾回收"],
    "가비지 컬렉션": ["garbage collection", "flash_garbage_collection", "垃圾回收"],
    "ssd": ["SSD", "solid state drive", "solid_state_drive_data_operation", "固态硬盘"],
    "ssd accelerator": ["SSD accelerator", "accelerator", "token manager", "non-volatile memory array"],
    "펌웨어": ["firmware", "firmware_upgrade_control", "固件", "펌웨어"],
    "암호화": ["encryption", "decryption", "encryption_decryption_processing", "암호화"],
}


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_json(value: str) -> Any:
    try:
        return json.loads(value or "[]")
    except Exception:
        return []


def expand_query(query: str) -> List[str]:
    query = normalize_ws(query)
    terms = [query] if query else []
    lower = query.lower()
    for key, aliases in QUERY_ALIASES.items():
        if key.lower() in lower:
            terms.extend(aliases)
    return list(dict.fromkeys([t for t in terms if t]))


def fts_phrase(term: str) -> str:
    escaped = term.replace('"', '""')
    return f'"{escaped}"'


def fts_query(query: str) -> str:
    terms = expand_query(query)
    if not terms:
        return ""
    return " OR ".join(fts_phrase(t) for t in terms)


def row_to_card(row: sqlite3.Row) -> Dict[str, Any]:
    return {
        "patent_id": row["patent_id"],
        "language": row["source_language"],
        "title": row["title_source"],
        "primary_claim_type": row["primary_claim_type"],
        "secondary_claim_types": load_json(row["secondary_claim_types_json"]),
        "independent_claim_nos": load_json(row["independent_claim_nos_json"]),
        "core_subject": row["core_subject"],
        "core_elements": load_json(row["core_elements_json"]),
        "problem_labels": load_json(row["problem_labels_json"]),
        "solution_labels": load_json(row["solution_labels_json"]),
        "effect_labels": load_json(row["effect_labels_json"]),
        "evidence_ids": load_json(row["evidence_ids_json"]),
        "confidence": row["confidence"],
        "qc_flags": load_json(row["qc_flags_json"]),
        "json_path": row["json_path"],
        "score": row["score"] if "score" in row.keys() else None,
    }


def search(
    con: sqlite3.Connection,
    query: str,
    limit: int,
    language: str = "",
    claim_type: str = "",
    label: str = "",
    include_qc: bool = False,
) -> List[Dict[str, Any]]:
    con.row_factory = sqlite3.Row
    params: List[Any] = []
    where = []
    score_expr = "0.0 AS score"
    from_expr = "minimal_index mi"

    match = fts_query(query)
    if match:
        from_expr = "minimal_index_fts fts JOIN minimal_index mi ON mi.patent_id = fts.patent_id"
        where.append("minimal_index_fts MATCH ?")
        params.append(match)
        score_expr = "bm25(minimal_index_fts) AS score"

    if language:
        where.append("mi.source_language = ?")
        params.append(language)
    if claim_type:
        where.append("mi.primary_claim_type = ?")
        params.append(claim_type)
    if label:
        where.append("EXISTS (SELECT 1 FROM minimal_labels ml WHERE ml.patent_id = mi.patent_id AND ml.label = ?)")
        params.append(label)
    if not include_qc:
        where.append("(mi.qc_flags_json = '[]' OR mi.qc_flags_json IS NULL)")

    sql = f"""
        SELECT mi.*, {score_expr}
        FROM {from_expr}
        {'WHERE ' + ' AND '.join(where) if where else ''}
        ORDER BY score ASC, mi.confidence DESC, mi.patent_id ASC
        LIMIT ?
    """
    params.append(limit)
    return [row_to_card(row) for row in con.execute(sql, params).fetchall()]


def lookup(con: sqlite3.Connection, patent_id: str) -> Dict[str, Any] | None:
    con.row_factory = sqlite3.Row
    row = con.execute(
        "SELECT *, 0.0 AS score FROM minimal_index WHERE patent_id = ?",
        (patent_id,),
    ).fetchone()
    return row_to_card(row) if row else None


def print_card(card: Dict[str, Any], rank: int) -> None:
    print(f"\n[{rank}] {card['patent_id']} ({card['language']}, {card['primary_claim_type']}, conf={card['confidence']})")
    print(f"Title: {card['title']}")
    print(f"Core: {card['core_subject']}")
    if card["core_elements"]:
        print("Elements: " + ", ".join(card["core_elements"][:8]))
    labels = card["solution_labels"][:8]
    if labels:
        print("Solution labels: " + ", ".join(labels))
    if card["problem_labels"]:
        print("Problem labels: " + ", ".join(card["problem_labels"][:6]))
    if card["effect_labels"]:
        print("Effect labels: " + ", ".join(card["effect_labels"][:6]))
    if card["evidence_ids"]:
        print("Evidence: " + ", ".join(card["evidence_ids"][:10]))
    if card["qc_flags"]:
        print("QC flags: " + ", ".join(card["qc_flags"]))
    print(f"JSON: {card['json_path']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the local A4 patent minimal dictionary.")
    parser.add_argument("query", nargs="?", default="", help="Keyword, phrase, Korean alias, or patent_id")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Path to patent_minimal_index.sqlite")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--lang", choices=["zh", "en", "ko"], default="")
    parser.add_argument("--claim-type", choices=["device", "system", "method", "process", "unknown"], default="")
    parser.add_argument("--label", default="", help="Exact label filter, e.g. page_buffer_circuit")
    parser.add_argument("--include-qc", action="store_true", help="Include rows with QC flags")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of readable cards")
    args = parser.parse_args()

    con = sqlite3.connect(args.db)
    exact = lookup(con, args.query) if args.query else None
    cards = [exact] if exact else search(
        con,
        args.query,
        args.limit,
        language=args.lang,
        claim_type=args.claim_type,
        label=args.label,
        include_qc=args.include_qc,
    )
    cards = [c for c in cards if c]

    if args.json:
        print(json.dumps(cards, ensure_ascii=False, indent=2))
        return

    print(f"Patent dictionary results: {len(cards)}")
    for i, card in enumerate(cards, 1):
        print_card(card, i)


if __name__ == "__main__":
    main()
