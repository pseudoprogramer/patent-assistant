from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from build_evidence_db import (  # noqa: E402
    PARSER_VERSION,
    detect_country,
    extract_pages_and_spans,
    extract_references_and_figures,
    insert_claim_ref_map,
    insert_claims,
    insert_drawing_ref_map,
    insert_figure_captions,
    insert_ref_entities,
    parse_claims,
)
from build_evidence_pack_index import is_contaminated_claim_text, normalize_ws  # noqa: E402
from config import A4_DB, A4_LOGS  # noqa: E402


DEFAULT_REPORT_DIR = Path("/Volumes/외장 2TB/cpu2026/common/runtime/reports/A4")


def unique_keep_order(items: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = normalize_ws(item).lower()
        if not value or value in seen:
            continue
        out.append(value)
        seen.add(value)
    return out


def connect(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON;")
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def ensure_repair_tables(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS claim_repair_runs (
            run_id TEXT PRIMARY KEY,
            started_at TEXT,
            parser_version TEXT,
            target_count INTEGER,
            repaired_count INTEGER,
            skipped_count INTEGER,
            failed_count INTEGER,
            dry_run INTEGER,
            report_json TEXT
        );

        CREATE TABLE IF NOT EXISTS claim_repair_backup_claims (
            run_id TEXT NOT NULL,
            patent_id TEXT NOT NULL,
            claim_no TEXT NOT NULL,
            parent_claim_no TEXT,
            claim_type TEXT,
            raw_text TEXT,
            norm_text TEXT,
            page_start INTEGER,
            page_end INTEGER,
            PRIMARY KEY (run_id, patent_id, claim_no)
        );
        """
    )
    con.commit()


def find_contaminated_patents(con: sqlite3.Connection, limit: int = 0) -> List[str]:
    rows = con.execute(
        """
        SELECT patent_id, claim_no, raw_text, norm_text
        FROM claims
        ORDER BY patent_id, CAST(claim_no AS INT), claim_no
        """
    ).fetchall()
    targets: List[str] = []
    for row in rows:
        if is_contaminated_claim_text(row["raw_text"] or row["norm_text"]):
            targets.append(str(row["patent_id"]))
    targets = unique_keep_order(targets)
    return targets[:limit] if limit else targets


def backup_claims(con: sqlite3.Connection, run_id: str, patent_id: str) -> None:
    con.execute("DELETE FROM claim_repair_backup_claims WHERE run_id=? AND patent_id=?", (run_id, patent_id))
    con.execute(
        """
        INSERT INTO claim_repair_backup_claims (
            run_id, patent_id, claim_no, parent_claim_no, claim_type,
            raw_text, norm_text, page_start, page_end
        )
        SELECT ?, patent_id, claim_no, parent_claim_no, claim_type,
               raw_text, norm_text, page_start, page_end
        FROM claims
        WHERE patent_id=?
        """,
        (run_id, patent_id),
    )


def replace_claim_artifacts(
    con: sqlite3.Connection,
    patent_id: str,
    claims: Sequence[Dict[str, Any]],
    refs: Sequence[Dict[str, Any]],
    claim_ref_map: Sequence[Dict[str, Any]],
    figure_caps: Sequence[Dict[str, Any]],
    drawing_ref_map: Sequence[Dict[str, Any]],
) -> None:
    con.execute("DELETE FROM drawing_ref_map WHERE patent_id=?", (patent_id,))
    con.execute("DELETE FROM figure_captions WHERE patent_id=?", (patent_id,))
    con.execute("DELETE FROM claim_ref_map WHERE patent_id=?", (patent_id,))
    con.execute("DELETE FROM ref_entities WHERE patent_id=?", (patent_id,))
    con.execute("DELETE FROM claims WHERE patent_id=?", (patent_id,))
    insert_claims(con, patent_id, list(claims))
    insert_ref_entities(con, patent_id, list(refs))
    insert_claim_ref_map(con, patent_id, list(claim_ref_map))
    insert_figure_captions(con, patent_id, list(figure_caps))
    insert_drawing_ref_map(con, patent_id, list(drawing_ref_map))
    con.execute(
        """
        UPDATE patents
        SET parser_version=?, updated_at=CURRENT_TIMESTAMP
        WHERE patent_id=?
        """,
        (f"{PARSER_VERSION}_claim_repair", patent_id),
    )
    row = con.execute("SELECT pdf_path FROM patents WHERE patent_id=?", (patent_id,)).fetchone()
    pdf_path = row["pdf_path"] if row else ""
    con.execute(
        """
        INSERT INTO jobs (patent_id, pdf_path, status, retry_count, last_error, updated_at)
        VALUES (?, ?, 'evidence_claim_repaired', 0, NULL, CURRENT_TIMESTAMP)
        ON CONFLICT(patent_id) DO UPDATE SET
            status='evidence_claim_repaired',
            last_error=NULL,
            updated_at=CURRENT_TIMESTAMP
        """,
        (patent_id, pdf_path),
    )


def repair_one(con: sqlite3.Connection, patent_id: str, run_id: str, dry_run: bool) -> Dict[str, Any]:
    patent = con.execute("SELECT patent_id, country, pdf_path FROM patents WHERE patent_id=?", (patent_id,)).fetchone()
    if not patent:
        return {"patent_id": patent_id, "status": "skipped", "reason": "missing_patent_row"}
    pdf_path = Path(patent["pdf_path"] or "")
    if not pdf_path.exists():
        return {"patent_id": patent_id, "status": "skipped", "reason": f"missing_pdf:{pdf_path}"}

    old_claims = con.execute("SELECT COUNT(*) FROM claims WHERE patent_id=?", (patent_id,)).fetchone()[0]
    old_contaminated = con.execute(
        "SELECT claim_no, raw_text, norm_text FROM claims WHERE patent_id=? ORDER BY CAST(claim_no AS INT), claim_no",
        (patent_id,),
    ).fetchall()
    old_bad_nos = [
        str(row["claim_no"])
        for row in old_contaminated
        if is_contaminated_claim_text(row["raw_text"] or row["norm_text"])
    ]

    pages, full_text = extract_pages_and_spans(pdf_path)
    country = detect_country(pdf_path.name, full_text[:10000])
    claims, claim_pages = parse_claims(country, pages)
    if not claims:
        return {
            "patent_id": patent_id,
            "status": "skipped",
            "reason": "reparse_found_no_claims",
            "old_claims": old_claims,
            "old_bad_claim_nos": old_bad_nos,
        }
    new_bad_nos = [
        claim["claim_no"]
        for claim in claims
        if is_contaminated_claim_text(claim.get("raw_text") or claim.get("norm_text"))
    ]
    independent_count = sum(1 for claim in claims if claim.get("claim_type") == "independent")
    if independent_count == 0:
        return {
            "patent_id": patent_id,
            "status": "skipped",
            "reason": "reparse_found_no_independent_claims",
            "old_claims": old_claims,
            "new_claims": len(claims),
            "old_bad_claim_nos": old_bad_nos,
            "new_bad_claim_nos": new_bad_nos,
        }
    refs, claim_ref_map, figure_caps, drawing_ref_map, drawing_pages = extract_references_and_figures(pages, claims, claim_pages)
    if not dry_run:
        backup_claims(con, run_id, patent_id)
        replace_claim_artifacts(con, patent_id, claims, refs, claim_ref_map, figure_caps, drawing_ref_map)
    return {
        "patent_id": patent_id,
        "status": "repaired" if not dry_run else "would_repair",
        "country": country,
        "pdf_path": str(pdf_path),
        "old_claims": old_claims,
        "new_claims": len(claims),
        "old_bad_claim_nos": old_bad_nos,
        "new_bad_claim_nos": new_bad_nos,
        "independent_claims": [claim["claim_no"] for claim in claims if claim.get("claim_type") == "independent"],
        "claim_pages": sorted(claim_pages),
    }


def run_repair(
    db_path: Path,
    patent_ids: Sequence[str],
    limit: int,
    dry_run: bool,
    report_dir: Path,
) -> Dict[str, Any]:
    started = time.monotonic()
    run_id = datetime.now().strftime("claim_repair_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    con = connect(db_path)
    ensure_repair_tables(con)
    targets = unique_keep_order(patent_ids) if patent_ids else find_contaminated_patents(con, limit=limit)
    if limit and patent_ids:
        targets = targets[:limit]
    results: List[Dict[str, Any]] = []
    try:
        for index, patent_id in enumerate(targets, 1):
            try:
                result = repair_one(con, patent_id, run_id, dry_run=dry_run)
                results.append(result)
                con.commit()
                print(
                    f"[claim-repair] {index}/{len(targets)} {patent_id} {result['status']} "
                    f"{result.get('old_claims', '-') }->{result.get('new_claims', '-')}",
                    flush=True,
                )
            except Exception as exc:
                con.rollback()
                results.append({"patent_id": patent_id, "status": "failed", "error": repr(exc)})
                print(f"[claim-repair] {index}/{len(targets)} {patent_id} failed {exc}", flush=True)
    finally:
        repaired = sum(1 for result in results if result["status"] in {"repaired", "would_repair"})
        skipped = sum(1 for result in results if result["status"] == "skipped")
        failed = sum(1 for result in results if result["status"] == "failed")
        report = {
            "run_id": run_id,
            "db_path": str(db_path),
            "parser_version": PARSER_VERSION,
            "dry_run": dry_run,
            "target_count": len(targets),
            "repaired_count": repaired,
            "skipped_count": skipped,
            "failed_count": failed,
            "elapsed_sec": round(time.monotonic() - started, 1),
            "results": results,
        }
        report_path = report_dir / f"{run_id}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        con.execute(
            """
            INSERT INTO claim_repair_runs (
                run_id, started_at, parser_version, target_count, repaired_count,
                skipped_count, failed_count, dry_run, report_json
            ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, PARSER_VERSION, len(targets), repaired, skipped, failed, int(dry_run), str(report_path)),
        )
        con.commit()
        con.close()
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair contaminated claim rows in the original A4 evidence DB.")
    parser.add_argument("--db", default=str(A4_DB))
    parser.add_argument("--patent-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    args = parser.parse_args()
    report = run_repair(
        db_path=Path(args.db),
        patent_ids=args.patent_id,
        limit=max(0, args.limit),
        dry_run=args.dry_run,
        report_dir=Path(args.report_dir),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
