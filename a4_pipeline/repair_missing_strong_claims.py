from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from build_evidence_pack_index import DEFAULT_PACK_DB, json_loads, normalize_ws
from config import A4_DB
import repair_evidence_claims
from repair_evidence_claims import connect, ensure_repair_tables, repair_one


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


def missing_strong_patent_ids(pack_db: Path, limit: int = 0) -> List[str]:
    con = sqlite3.connect(pack_db)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT patent_id, quality_flags_json
        FROM evidence_pack_index
        ORDER BY patent_id
        """
    ).fetchall()
    con.close()
    targets = []
    for row in rows:
        flags = set(json_loads(row["quality_flags_json"]))
        if "missing_strong_independent_claim_text" in flags:
            targets.append(str(row["patent_id"]))
    targets = unique_keep_order(targets)
    return targets[:limit] if limit else targets


def run_missing_strong_repair(
    db_path: Path,
    pack_db: Path,
    report_dir: Path,
    dry_run: bool,
    limit: int,
) -> Dict[str, Any]:
    started = time.monotonic()
    run_id = time.strftime("missing_strong_repair_%Y%m%d_%H%M%S")
    report_dir.mkdir(parents=True, exist_ok=True)
    targets = missing_strong_patent_ids(pack_db, limit=limit)
    con = connect(db_path)
    ensure_repair_tables(con)
    results: List[Dict[str, Any]] = []
    try:
        for index, patent_id in enumerate(targets, 1):
            try:
                result = repair_one(con, patent_id, run_id, dry_run=dry_run)
                results.append(result)
                con.commit()
                print(
                    f"[missing-strong-repair] {index}/{len(targets)} {patent_id} "
                    f"{result['status']} {result.get('old_claims', '-')}->{result.get('new_claims', '-')}",
                    flush=True,
                )
            except Exception as exc:
                con.rollback()
                result = {"patent_id": patent_id, "status": "failed", "error": repr(exc)}
                results.append(result)
                print(f"[missing-strong-repair] {index}/{len(targets)} {patent_id} failed {exc}", flush=True)
    finally:
        repaired = sum(1 for result in results if result["status"] in {"repaired", "would_repair"})
        skipped = sum(1 for result in results if result["status"] == "skipped")
        failed = sum(1 for result in results if result["status"] == "failed")
        report = {
            "run_id": run_id,
            "db_path": str(db_path),
            "pack_db": str(pack_db),
            "parser_version": repair_evidence_claims.PARSER_VERSION,
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
        con.close()
    report["report_path"] = str(report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Reparse patents missing strong independent claim evidence.")
    parser.add_argument("--db", default=str(A4_DB))
    parser.add_argument("--pack-db", default=str(DEFAULT_PACK_DB))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = run_missing_strong_repair(
        db_path=Path(args.db),
        pack_db=Path(args.pack_db),
        report_dir=Path(args.report_dir),
        dry_run=args.dry_run,
        limit=max(0, args.limit),
    )
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
