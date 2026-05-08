from __future__ import annotations

import argparse
import json
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


BASE = Path("/Volumes/외장 2TB/cpu2026")
DEFAULT_DB = BASE / "common" / "runtime" / "db" / "patent_A4.sqlite"
DEFAULT_OUT = BASE / "common" / "runtime" / "reports" / "A4" / "problem_effect_probe.json"


SECTION_PATTERNS = {
    "CN": [
        ("background", r"背景技术"),
        ("summary", r"发明内容|本发明"),
        ("problem", r"要解决的技术问题|技术问题|问题是|存在.*问题|不足"),
        ("effect", r"有益效果|技术效果|提高|提升|降低|减少|改善|避免|防止"),
    ],
    "US": [
        ("background", r"\bBACKGROUND\b|Description of Related Art"),
        ("summary", r"\bSUMMARY\b|BRIEF SUMMARY"),
        ("problem", r"\bproblem\b|drawback|disadvantage|need(?:ed)?|difficult|undesirable|latency|power consumption|damage(?:d)?|stability|reliability"),
        ("effect", r"\badvantage\b|effect|improv(?:e|es|ed|ing)|reduc(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|increas(?:e|es|ed|ing)|prevent(?:s|ed|ing)?"),
    ],
    "KR": [
        ("background", r"배경기술|종래기술"),
        ("summary", r"발명의 내용|과제의 해결 수단"),
        ("problem", r"해결하고자 하는 과제|문제점|필요|어려움|한계"),
        ("effect", r"발명의 효과|효과|향상|감소|저감|개선|방지|억제|증가"),
    ],
}


LABEL_RULES = {
    "problem": [
        ("latency_issue", r"延迟|latency|레이턴시|지연"),
        ("power_consumption_issue", r"功耗|power consumption|전력|소모"),
        ("data_loss_risk", r"数据丢失|data loss|데이터 손실"),
        ("reliability_issue", r"可靠性|reliability|stability|damage(?:d)?|신뢰성|안정성|손상"),
        ("performance_degradation", r"性能|performance|성능|throughput|처리량"),
        ("storage_management_overhead", r"垃圾回收|garbage collection|地址映射|address mapping|가비지 컬렉션|주소 매핑"),
        ("manufacturing_complexity", r"制造|manufactur|공정|제조|复杂|complex"),
    ],
    "effect": [
        ("latency_reduction", r"降低.{0,12}延迟|减少.{0,12}延迟|reduce.{0,20}latency|tail latency|레이턴시.{0,12}(감소|저감)"),
        ("power_reduction", r"降低.{0,12}功耗|减少.{0,12}功耗|reduce.{0,20}power|전력.{0,12}(감소|저감)"),
        ("reliability_improvement", r"提高.{0,12}可靠性|improv.{0,20}reliability|maintain.{0,30}(stability|reliability)|prevent.{0,40}damage|신뢰성.{0,12}향상|안정성.{0,12}(향상|유지)"),
        ("performance_improvement", r"提高.{0,12}性能|提升.{0,12}性能|improv.{0,20}performance|성능.{0,12}향상"),
        ("data_recovery", r"数据恢复|data recovery|rebuild|복구|재구성"),
        ("data_loss_prevention", r"防止.{0,12}数据丢失|prevent.{0,20}data loss|데이터 손실.{0,12}방지"),
        ("throughput_increase", r"提高.{0,12}吞吐|increase.{0,20}throughput|처리량.{0,12}증가"),
        ("latency_improvement", r"改善.{0,12}(延迟|时延)|improv.{0,20}latency|tail latency.{0,40}improv|레이턴시.{0,20}개선|테일 레이턴시.{0,20}개선"),
        ("write_amplification_reduction", r"write amplification.{0,40}(reduc|lower)|reduc.{0,40}write amplification"),
        ("endurance_improvement", r"increas.{0,40}lifetime|lifetime.{0,40}increas|endurance.{0,40}improv"),
    ],
}


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def clean_title(value: Any) -> str:
    title = normalize_ws(value)
    title = re.split(
        r"(?:\(\s*57\s*\)|摘要|ABSTRACT|Abstract|权利要求书|청구범위|청구항|Claims?:|What is claimed is)",
        title,
        maxsplit=1,
        flags=re.I,
    )[0]
    title = re.sub(r"^\s*\(?\s*54\s*\)?\s*(发明名称|Title)?\s*", "", title, flags=re.I)
    title = re.split(r"\(\s*(?:71|72|73)\s*\)|Applicant:|Inventor:|Assignee:", title, maxsplit=1, flags=re.I)[0]
    return normalize_ws(title)[:240]


def is_claim_like_text(text: str) -> bool:
    return bool(
        re.search(
            r"(权利要求书|权利要求\s*\d+|청구범위|청구항\s*\d+|What is claimed is|Claims?\s*:?\s*1[\).])",
            text,
            flags=re.I,
        )
    )


def snippet_score(section: str, text: str) -> int:
    priority = {"effect": 40, "problem": 35, "summary": 22, "background": 15}.get(section, 0)
    label_hits = 0
    for rules in LABEL_RULES.values():
        for _label, pattern in rules:
            if re.search(pattern, text, flags=re.I):
                label_hits += 1
    return priority + label_hits * 6


def connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    return con


def claim_page_set(con: sqlite3.Connection, patent_id: str) -> set[int]:
    rows = con.execute(
        """
        SELECT page_start, page_end
        FROM claims
        WHERE patent_id=?
        """,
        (patent_id,),
    ).fetchall()
    pages: set[int] = set()
    for row in rows:
        start = row["page_start"]
        end = row["page_end"]
        if start:
            pages.add(int(start))
        if end:
            pages.add(int(end))
        if start and end:
            for page_no in range(int(start), int(end) + 1):
                pages.add(page_no)
    return pages


def page_texts(
    con: sqlite3.Connection,
    patent_id: str,
    max_pages: Optional[int] = None,
    skip_pages: set[int] | None = None,
) -> Dict[int, str]:
    rows = con.execute(
        """
        SELECT page_no, block_no, line_no, span_no, raw_text
        FROM text_spans
        WHERE patent_id=?
        ORDER BY page_no, block_no, line_no, span_no, id
        """,
        (patent_id,),
    ).fetchall()
    pages: Dict[int, List[str]] = {}
    current_key: Tuple[int, int, int] | None = None
    current_line: List[str] = []

    def flush() -> None:
        nonlocal current_key, current_line
        if current_key and current_line:
            page_no = current_key[0]
            pages.setdefault(page_no, []).append(normalize_ws(" ".join(current_line)))
        current_line = []

    for row in rows:
        page_no = int(row["page_no"])
        if (max_pages is not None and page_no > max_pages) or page_no in (skip_pages or set()):
            continue
        key = (page_no, int(row["block_no"] or 0), int(row["line_no"] or 0))
        if current_key is not None and key != current_key:
            flush()
        current_key = key
        text = str(row["raw_text"] or "").strip()
        if text:
            current_line.append(text)
    flush()
    return {page_no: "\n".join(lines) for page_no, lines in pages.items()}


def window(text: str, start: int, size: int = 900) -> str:
    left = max(0, start - 160)
    right = min(len(text), start + size)
    return normalize_ws(text[left:right])


def collect_snippets(country: str, pages: Dict[int, str], max_snippets: int = 8) -> List[Dict[str, Any]]:
    patterns = SECTION_PATTERNS.get(country, SECTION_PATTERNS["US"])
    candidates: List[Dict[str, Any]] = []
    seen = set()
    for page_no, text in pages.items():
        flat = normalize_ws(text)
        if country == "KR":
            marker = re.search(r"발명의 설명|기\s*술\s*분\s*야", flat)
            if marker:
                flat = flat[marker.start():]
        for section, pattern in patterns:
            for match in re.finditer(pattern, flat, flags=re.I):
                snippet = window(flat, match.start())
                if is_claim_like_text(snippet):
                    continue
                key = snippet[:120]
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "id": "",
                        "page_no": page_no,
                        "section": section,
                        "trigger": match.group(0),
                        "text": snippet[:900],
                        "_score": snippet_score(section, snippet),
                    }
                )
    candidates.sort(key=lambda item: (-int(item["_score"]), int(item["page_no"])))
    snippets = candidates[:max_snippets]
    snippets.sort(key=lambda item: (int(item["page_no"]), -int(item["_score"])))
    for index, snippet in enumerate(snippets, 1):
        snippet["id"] = f"pe_{index}"
        snippet.pop("_score", None)
    return snippets


def infer_labels(snippets: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    text = "\n".join(snippet["text"] for snippet in snippets)
    problem_labels = []
    effect_labels = []
    for label, pattern in LABEL_RULES["problem"]:
        if re.search(pattern, text, flags=re.I):
            problem_labels.append(label)
    for label, pattern in LABEL_RULES["effect"]:
        if re.search(pattern, text, flags=re.I):
            effect_labels.append(label)
    return {
        "problem_labels": unique_keep_order(problem_labels)[:6],
        "effect_labels": unique_keep_order(effect_labels)[:6],
    }


def run_one(con: sqlite3.Connection, patent_id: str, max_pages: Optional[int] = None) -> Dict[str, Any]:
    meta = con.execute(
        "SELECT patent_id, country, title_raw FROM patents WHERE patent_id=?",
        (patent_id,),
    ).fetchone()
    if not meta:
        raise RuntimeError(f"patent not found: {patent_id}")
    claim_pages = claim_page_set(con, patent_id)
    pages = page_texts(con, patent_id, max_pages=max_pages, skip_pages=claim_pages)
    snippets = collect_snippets(str(meta["country"] or "US"), pages)
    if not snippets:
        pages = page_texts(con, patent_id, max_pages=max_pages, skip_pages=set())
        snippets = collect_snippets(str(meta["country"] or "US"), pages)
    labels = infer_labels(snippets)
    return {
        "patent_id": patent_id,
        "country": meta["country"],
        "title": clean_title(meta["title_raw"]),
        "claim_pages_skipped": sorted(claim_pages),
        "problem_effect_snippets": snippets,
        **labels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe problem/effect evidence extraction for sample patents.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--patent-id", action="append", default=[])
    parser.add_argument("--patent-list-file", default="")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--max-pages", type=int, default=0, help="0 means scan all pages")
    args = parser.parse_args()

    patent_ids = list(args.patent_id)
    if args.patent_list_file:
        patent_ids.extend(
            normalize_ws(line)
            for line in Path(args.patent_list_file).read_text(encoding="utf-8").splitlines()
            if normalize_ws(line) and not normalize_ws(line).startswith("#")
        )
    patent_ids = unique_keep_order(patent_ids) or [
        "cn202010106592bp",
        "us0000re50315e1p",
        "kr00002538126b1p",
    ]
    con = connect(Path(args.db))
    try:
        max_pages = args.max_pages if args.max_pages > 0 else None
        results = [run_one(con, patent_id, max_pages=max_pages) for patent_id in patent_ids]
    finally:
        con.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    compact = {
        "out": str(out),
        "results": [
            {
                "patent_id": item["patent_id"],
                "country": item["country"],
                "title": item["title"],
                "problem_labels": item["problem_labels"],
                "effect_labels": item["effect_labels"],
                "snippet_count": len(item["problem_effect_snippets"]),
                "snippet_pages": [s["page_no"] for s in item["problem_effect_snippets"]],
            }
            for item in results
        ],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
