from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from config import A4_DB


DEFAULT_REPORT_DIR = Path("/Volumes/외장 2TB/cpu2026/common/runtime/reports/A4")

CERTIFICATE_RE = re.compile(
    r"(?:\*\s*){3,}\s*(?:\d+\s*)?UNITED STATES PATENT AND TRADEMARK OFFICE CERTIFICATE OF CORRECTION|"
    r"UNITED STATES PATENT AND TRADEMARK OFFICE CERTIFICATE OF CORRECTION",
    re.I,
)
FRONTMATTER_RE = re.compile(
    r"(References Cited|Primary Examiner|cited by examiner|U\.S\.\s+Patent|"
    r"Sheet\s+\d+\s+of\s+\d+|ABSTRACT|Foreign Application Priority Data|\(51\)\s*Int\. Cl\.)",
    re.I,
)
CN_CLAIM_START_RE = re.compile(
    r"^\s*\d+[\.、]\s*(?:一种|一種|根据权利要求|如权利要求|权利要求|根据|如|"
    r"基于|用于|一种用于|一种基于|一种应用|一种存储|一种半导体|一种控制|一种方法|"
    r"一种装置|一种系统|一种设备)",
    re.I,
)
US_CLAIM_START_RE = re.compile(
    r"^\s*\d*\.?\s*(?:A|An|The|That|In|Method|Device|System|Apparatus|Circuit|Memory|Non-transitory|One or more)\b.{0,220}\b"
    r"(?:claim|comprising|comprises|wherein|configured|including|method|device|system|"
    r"apparatus|circuit|memory|medium|controller|processor|storage)\b",
    re.I,
)
US_DESCRIPTION_RE = re.compile(
    r"\b(?:may|example|embodiment|shown|described|referring|can comprise|can be|in some embodiments)\b",
    re.I,
)
US_CLAIM_MARKER_RE = re.compile(r"\b(?:of claim\s+\d+|wherein|comprising|configured to)\b", re.I)
US_SUBSTEP_FRAGMENT_RE = re.compile(r"^\s*\d+\)")
US_DESCRIPTION_START_RE = re.compile(r"^\s*(?:Specifically|While|In some embodiments|Referring|As described)\b", re.I)
US_GLUE_CLAIM_START_RE = re.compile(
    r"^\s*\d*\.?\s*(?:Anon-transitory|Amethod|Adevice|Asystem|Aapparatus|Acircuit|Amemory|"
    r"Themethod|Thesystem|Thedevice|Thememory|Theflash|Themouse|Theapparatus|Thecircuit)\b",
    re.I,
)
US_GLUE_FIXES = (
    (re.compile(r"\bAnon-transitory\b", re.I), "A non-transitory"),
    (re.compile(r"\bAmethod\b", re.I), "A method"),
    (re.compile(r"\bAdevice\b", re.I), "A device"),
    (re.compile(r"\bAsystem\b", re.I), "A system"),
    (re.compile(r"\bAapparatus\b", re.I), "An apparatus"),
    (re.compile(r"\bAcircuit\b", re.I), "A circuit"),
    (re.compile(r"\bAmemory\b", re.I), "A memory"),
    (re.compile(r"\bThemethod\b", re.I), "The method"),
    (re.compile(r"\bThesystem\b", re.I), "The system"),
    (re.compile(r"\bThedevice\b", re.I), "The device"),
    (re.compile(r"\bThememory\b", re.I), "The memory"),
    (re.compile(r"\bTheflash\b", re.I), "The flash"),
    (re.compile(r"\bThemouse\b", re.I), "The mouse"),
    (re.compile(r"\bTheapparatus\b", re.I), "The apparatus"),
    (re.compile(r"\bThecircuit\b", re.I), "The circuit"),
    (re.compile(r"\bofclaim\b", re.I), "of claim"),
)
CN_PAGE_MARK_RE = re.compile(r"\s*\b\d+/\d+\s+页\b\s*")
CN_SECTION_HEADER_RE = re.compile(r"(技术领域|背景技术|发明内容|说明书|摘要|附图说明)")
KR_PUBLICATION_MARK_RE = re.compile(r"\s*등록특허\s+\d{1,3}-\d+\s*-\s*\d+\s*-\s*")
KR_DESCRIPTION_TAIL_RE = re.compile(r"\s+발명의\s+설명.*$", re.S)
KR_DESCRIPTION_START_RE = re.compile(r"^\s*(?:삭제\s+)?발명의\s+설명|^\s*기\s*술\s*분\s*야|^\s*배\s*경\s*기\s*술")
CN_SUBSTEP_FRAGMENT_RE = re.compile(r"^\s*\d+\s*[\.、]?\s*\d+(?:[\).]|的公式|[A-Za-z]|[\u4e00-\u9fff])")
CN_SHORT_BACKGROUND_RE = re.compile(
    r"^\s*\d+\s*[\.、]\s*(?:发明目的|CPU|由于|增加|提高|报告|软解码|使用默认|本地连接|"
    r"主机Host|上述方法|当日度实时|先对测试盘)",
    re.I,
)
CN_GENERIC_CLAIM_TERMS_RE = re.compile(
    r"(其特征在于|包括|包含|配置|配置成|用于|所述|至少一种|机器可读|介质|装置|系统|方法|"
    r"电路|设备|处理方法|存储器|计算机|车辆|电子装置|控制电路)"
)


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def claim_no_int(claim_no: Any) -> int:
    value = str(claim_no or "")
    return int(value) if value.isdigit() else -1


def looks_like_claim_start(patent_id: str, text: str) -> bool:
    if patent_id.lower().startswith("cn"):
        if CN_CLAIM_START_RE.search(text):
            return True
        head = text[:240]
        if re.match(r"^\s*\d+[\.、]\s*[\u4e00-\u9fffA-Za-z]", text) and CN_GENERIC_CLAIM_TERMS_RE.search(head):
            return True
        return bool(CN_CLAIM_START_RE.search(text))
    if patent_id.lower().startswith("us"):
        return bool(US_CLAIM_START_RE.search(text) or (US_GLUE_CLAIM_START_RE.search(text) and US_CLAIM_MARKER_RE.search(text)))
    if patent_id.lower().startswith("kr"):
        return bool(
            re.search(r"(제\s*\d+\s*항|청구항|상기|포함|있어서|단계|장치|방법|시스템|회로|메모리)", text[:220])
        )
    return True


def cleaned_text(text: str) -> str:
    text = CERTIFICATE_RE.split(text, maxsplit=1)[0]
    text = KR_DESCRIPTION_TAIL_RE.sub("", text)
    text = KR_PUBLICATION_MARK_RE.sub(" ", text)
    for pattern, replacement in US_GLUE_FIXES:
        text = pattern.sub(replacement, text)
    for match in CN_PAGE_MARK_RE.finditer(text):
        tail = text[match.end() : match.end() + 220]
        if CN_SECTION_HEADER_RE.search(tail):
            text = text[: match.start()]
            break
    text = CN_PAGE_MARK_RE.sub(" ", text)
    return normalize_ws(text).strip(" *")


def classify_action(patent_id: str, claim_no: str, text: str) -> Dict[str, Any]:
    text = normalize_ws(text)
    no = claim_no_int(claim_no)
    starts_like_claim = looks_like_claim_start(patent_id, text)
    reasons: List[str] = []

    if claim_no == "0":
        reasons.append("delete:claim_no_zero_fragment")
    if FRONTMATTER_RE.search(text) and not starts_like_claim:
        reasons.append("delete:frontmatter_not_claim")
    if patent_id.lower().startswith("cn") and no >= 40 and not starts_like_claim:
        reasons.append("delete:cn_large_claim_no_not_claim")
    if (
        patent_id.lower().startswith("cn")
        and not starts_like_claim
        and (CN_SUBSTEP_FRAGMENT_RE.search(text) or CN_SHORT_BACKGROUND_RE.search(text) or "[000" in text)
    ):
        reasons.append("delete:cn_short_or_substep_fragment")
    if no >= 80 and not starts_like_claim:
        reasons.append("delete:large_claim_no_not_claim")
    if patent_id.lower().startswith("us") and not starts_like_claim and len(text) < 140:
        reasons.append("delete:us_short_fragment_not_claim")
    if (
        patent_id.lower().startswith("us")
        and not starts_like_claim
        and (US_SUBSTEP_FRAGMENT_RE.search(text) or US_DESCRIPTION_START_RE.search(text) or text.count("|") >= 4)
    ):
        reasons.append("delete:us_nonclaim_fragment")
    if patent_id.lower().startswith("kr") and KR_DESCRIPTION_START_RE.search(text):
        reasons.append("delete:kr_description_not_claim")
    if (
        patent_id.lower().startswith("us")
        and no >= 80
        and US_DESCRIPTION_RE.search(text)
        and not US_CLAIM_MARKER_RE.search(text)
    ):
        reasons.append("delete:us_description_paragraph_as_claim")

    if reasons:
        return {"action": "delete", "reasons": reasons}

    new_text = cleaned_text(text)
    if new_text != text:
        update_reasons = []
        if CERTIFICATE_RE.search(text):
            update_reasons.append("truncate_certificate_tail")
        cn_tail = any(CN_SECTION_HEADER_RE.search(text[match.end() : match.end() + 220]) for match in CN_PAGE_MARK_RE.finditer(text))
        if cn_tail:
            update_reasons.append("truncate_cn_description_tail")
        elif CN_PAGE_MARK_RE.search(text):
            update_reasons.append("remove_page_marker")
        if KR_DESCRIPTION_TAIL_RE.search(text):
            update_reasons.append("truncate_kr_description_tail")
        if KR_PUBLICATION_MARK_RE.search(text):
            update_reasons.append("remove_kr_publication_marker")
        if any(pattern.search(text) for pattern, _ in US_GLUE_FIXES):
            update_reasons.append("repair_us_glued_claim_words")
        if len(new_text) >= 30:
            return {"action": "update", "reasons": update_reasons, "new_text": new_text}

    return {"action": "keep", "reasons": []}


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON;")
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def ensure_tables(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS claim_text_cleanup_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT,
            target_count INTEGER,
            update_count INTEGER,
            delete_count INTEGER,
            dry_run INTEGER,
            report_json TEXT
        );

        CREATE TABLE IF NOT EXISTS claim_text_cleanup_backup_claims (
            run_id TEXT NOT NULL,
            patent_id TEXT NOT NULL,
            claim_no TEXT NOT NULL,
            parent_claim_no TEXT,
            claim_type TEXT,
            raw_text TEXT,
            norm_text TEXT,
            page_start INTEGER,
            page_end INTEGER,
            action TEXT,
            reasons_json TEXT,
            PRIMARY KEY (run_id, patent_id, claim_no)
        );
        """
    )
    con.commit()


def unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = normalize_ws(item).lower()
        if value and value not in seen:
            out.append(value)
            seen.add(value)
    return out


def iter_claim_rows(con: sqlite3.Connection, patent_ids: Sequence[str]) -> List[sqlite3.Row]:
    if patent_ids:
        rows: List[sqlite3.Row] = []
        for patent_id in patent_ids:
            rows.extend(
                con.execute(
                    """
                    SELECT patent_id, claim_no, parent_claim_no, claim_type, raw_text, norm_text, page_start, page_end
                    FROM claims
                    WHERE patent_id=?
                    ORDER BY CAST(claim_no AS INT), claim_no
                    """,
                    (patent_id,),
                ).fetchall()
            )
        return rows
    return con.execute(
        """
        SELECT patent_id, claim_no, parent_claim_no, claim_type, raw_text, norm_text, page_start, page_end
        FROM claims
        ORDER BY patent_id, CAST(claim_no AS INT), claim_no
        """
    ).fetchall()


def backup_row(con: sqlite3.Connection, run_id: str, row: sqlite3.Row, action: Dict[str, Any]) -> None:
    con.execute(
        """
        INSERT OR REPLACE INTO claim_text_cleanup_backup_claims (
            run_id, patent_id, claim_no, parent_claim_no, claim_type,
            raw_text, norm_text, page_start, page_end, action, reasons_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            row["patent_id"],
            row["claim_no"],
            row["parent_claim_no"],
            row["claim_type"],
            row["raw_text"],
            row["norm_text"],
            row["page_start"],
            row["page_end"],
            action["action"],
            json.dumps(action["reasons"], ensure_ascii=False),
        ),
    )


def run_cleanup(db_path: Path, patent_ids: Sequence[str], dry_run: bool, report_dir: Path) -> Dict[str, Any]:
    started = time.monotonic()
    run_id = datetime.now().strftime("claim_text_cleanup_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    con = connect(db_path)
    ensure_tables(con)
    rows = iter_claim_rows(con, unique_keep_order(patent_ids))
    actions: List[Dict[str, Any]] = []

    try:
        for row in rows:
            text = normalize_ws(row["raw_text"] or row["norm_text"])
            action = classify_action(str(row["patent_id"]), str(row["claim_no"]), text)
            if action["action"] == "keep":
                continue
            item = {
                "patent_id": row["patent_id"],
                "claim_no": row["claim_no"],
                "claim_type": row["claim_type"],
                "action": action["action"],
                "reasons": action["reasons"],
                "old_preview": text[:350],
            }
            if action["action"] == "update":
                item["new_preview"] = action["new_text"][:350]
            actions.append(item)
            if dry_run:
                continue
            backup_row(con, run_id, row, action)
            if action["action"] == "delete":
                con.execute(
                    "DELETE FROM claim_ref_map WHERE patent_id=? AND claim_no=?",
                    (row["patent_id"], row["claim_no"]),
                )
                con.execute(
                    "DELETE FROM claims WHERE patent_id=? AND claim_no=?",
                    (row["patent_id"], row["claim_no"]),
                )
            elif action["action"] == "update":
                con.execute(
                    """
                    UPDATE claims
                    SET raw_text=?, norm_text=?
                    WHERE patent_id=? AND claim_no=?
                    """,
                    (action["new_text"], action["new_text"], row["patent_id"], row["claim_no"]),
                )
        update_count = sum(1 for item in actions if item["action"] == "update")
        delete_count = sum(1 for item in actions if item["action"] == "delete")
        report = {
            "run_id": run_id,
            "db_path": str(db_path),
            "dry_run": dry_run,
            "target_count": len(actions),
            "update_count": update_count,
            "delete_count": delete_count,
            "elapsed_sec": round(time.monotonic() - started, 1),
            "actions": actions,
        }
        report_path = report_dir / f"{run_id}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not dry_run:
            con.execute(
                """
                INSERT INTO claim_text_cleanup_runs (
                    run_id, started_at, target_count, update_count, delete_count, dry_run, report_json
                )
                VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?)
                """,
                (run_id, len(actions), update_count, delete_count, int(dry_run), str(report_path)),
            )
            con.commit()
        con.close()
    except Exception:
        con.rollback()
        con.close()
        raise
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean residual contaminated claim rows without reparsing PDFs.")
    parser.add_argument("--db", default=str(A4_DB))
    parser.add_argument("--patent-id", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()
    report = run_cleanup(
        db_path=Path(args.db),
        patent_ids=args.patent_id,
        dry_run=args.dry_run,
        report_dir=Path(args.report_dir),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "actions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
