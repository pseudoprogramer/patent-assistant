from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import audit_evidence_quality
import build_evidence_db as builder
import db_schema
import repair_evidence_claims
from build_evidence_pack_index import DEFAULT_MINIMAL_DB, build_pack_index
from config import A4_INBOX, A4_LOGS, A4_PARSED_JSON
from repair_missing_strong_claims import run_missing_strong_repair
from repair_residual_claim_text import run_cleanup


BASE = Path("/Volumes/외장 2TB/cpu2026")
DEFAULT_DB_DIR = BASE / "common" / "runtime" / "db"
DEFAULT_INDEX_DIR = BASE / "patent_hub" / "outputs" / "indexes" / "A4"
DEFAULT_REPORT_DIR = BASE / "common" / "runtime" / "reports" / "A4"
PARSER_VERSION_V2 = "evidence_db_v2.0"


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def normalize_ws(value: Any) -> str:
    return " ".join(str(value or "").split())


def set_builder_runtime(db_path: Path, parsed_json_dir: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    parsed_json_dir.mkdir(parents=True, exist_ok=True)
    db_schema.DB_PATH = db_path
    builder.A4_DB = db_path
    builder.A4_PARSED_JSON = parsed_json_dir
    builder.PARSER_VERSION = PARSER_VERSION_V2
    repair_evidence_claims.PARSER_VERSION = PARSER_VERSION_V2


def list_target_pdfs(args: argparse.Namespace) -> List[Path]:
    if args.pdf:
        pdfs = [Path(item) for item in args.pdf]
    elif args.folder:
        pdfs = builder.list_pdfs(Path(args.folder), recursive=args.recursive)
    elif args.image_folders:
        image_folders = sorted(d for d in A4_INBOX.glob("Image_*") if d.is_dir())
        pdfs = []
        for folder in image_folders:
            pdfs.extend(builder.list_pdfs(folder, recursive=args.recursive))
    elif args.all:
        pdfs = builder.list_pdfs(A4_INBOX, recursive=args.recursive)
    else:
        pdfs = []
    if args.limit and args.limit > 0:
        return pdfs[: args.limit]
    return pdfs


def table_count(db_path: Path, table: str) -> int:
    con = sqlite3.connect(db_path)
    try:
        return int(con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        con.close()


def build_from_pdfs(args: argparse.Namespace, db_path: Path, parsed_json_dir: Path) -> Dict[str, Any]:
    set_builder_runtime(db_path, parsed_json_dir)
    builder.ensure_runtime_dirs()
    builder.ensure_output_dirs()
    db_schema.ensure_db()
    builder.init_log_file()

    pdfs = list_target_pdfs(args)
    total = len(pdfs)
    if total == 0:
        return {"target_pdfs": 0, "success": 0, "failed": 0, "results": []}

    builder.log(f"[v2 시작] 대상 PDF 수: {total}")
    builder.log(f"[v2 설정] DB: {db_path}")
    builder.log(f"[v2 설정] parsed_json_dir: {parsed_json_dir}")
    builder.log(f"[v2 설정] parser_version: {PARSER_VERSION_V2}")

    started = time.monotonic()
    success = 0
    failed = 0
    results: List[Dict[str, Any]] = []

    for index, pdf_path in enumerate(pdfs, 1):
        item_start = time.monotonic()
        builder.log("")
        builder.log(builder.progress_bar(index, total))
        builder.log(f"[v2 처리중] {index}/{total} - {pdf_path.name}")
        try:
            result = builder.process_one_pdf(pdf_path, move_processed=args.move_processed)
            result["status"] = "success"
            result["elapsed_sec"] = round(time.monotonic() - item_start, 1)
            success += 1
            builder.log(
                "    ✓ 완료: "
                f"{result['patent_id']} claims={result['claims_count']}, "
                f"refs={result['refs_count']}, figures={result['figures_count']}, "
                f"{result['elapsed_sec']}초"
            )
        except Exception as exc:
            failed += 1
            result = {
                "status": "failed",
                "pdf": str(pdf_path),
                "patent_id": pdf_path.stem,
                "error": repr(exc),
                "elapsed_sec": round(time.monotonic() - item_start, 1),
            }
            builder.log(f"    ✗ 실패: {pdf_path.name}")
            builder.log(f"      오류: {exc}")
        results.append(result)

    elapsed = round(time.monotonic() - started, 1)
    builder.log(f"[v2 PDF 종료] 성공: {success}, 실패: {failed}, 총 소요: {elapsed}초")
    return {
        "target_pdfs": total,
        "success": success,
        "failed": failed,
        "elapsed_sec": elapsed,
        "log_path": str(builder.LOG_FILE_PATH) if builder.LOG_FILE_PATH else "",
        "results": results,
    }


def write_audit(evidence_db: Path, pack_db: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "evidence_db": str(evidence_db),
        "pack_db": str(pack_db),
        "claims": audit_evidence_quality.audit_claims(evidence_db, sample_limit=40),
        "packs": audit_evidence_quality.audit_packs(pack_db, sample_limit=40),
    }
    json_path = out_dir / "evidence_quality_audit.json"
    md_path = out_dir / "evidence_quality_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    audit_evidence_quality.write_markdown(report, md_path)
    return {
        "json_path": str(json_path),
        "md_path": str(md_path),
        "claim_bucket_counts": report["claims"]["bucket_counts"],
        "high_or_severe_claim_patents": report["claims"]["high_or_severe_patents"],
        "affected_packs_high_or_severe": report["packs"]["affected_packs_high_or_severe"],
        "packs_without_claim_text": report["packs"]["packs_without_claim_text"],
    }


def run_postprocess(
    db_path: Path,
    minimal_db: Path,
    pack_db: Path,
    report_dir: Path,
    skip_missing_strong: bool,
) -> Dict[str, Any]:
    stages: Dict[str, Any] = {}

    stages["residual_cleanup_1"] = {
        k: v
        for k, v in run_cleanup(db_path=db_path, patent_ids=[], dry_run=False, report_dir=report_dir).items()
        if k != "actions"
    }

    stages["pack_index_1"] = build_pack_index(
        minimal_db=minimal_db,
        evidence_db=db_path,
        pack_db=pack_db,
        evidence_only=True,
    )

    if skip_missing_strong:
        stages["missing_strong_repair"] = {"skipped": True}
    else:
        stages["missing_strong_repair"] = {
            k: v
            for k, v in run_missing_strong_repair(
                db_path=db_path,
                pack_db=pack_db,
                report_dir=report_dir,
                dry_run=False,
                limit=0,
            ).items()
            if k != "results"
        }
        stages["residual_cleanup_2"] = {
            k: v
            for k, v in run_cleanup(db_path=db_path, patent_ids=[], dry_run=False, report_dir=report_dir).items()
            if k != "actions"
        }

    stages["pack_index_final"] = build_pack_index(
        minimal_db=minimal_db,
        evidence_db=db_path,
        pack_db=pack_db,
        evidence_only=True,
    )
    stages["audit"] = write_audit(evidence_db=db_path, pack_db=pack_db, out_dir=report_dir)
    stages["db_counts"] = {
        "patents": table_count(db_path, "patents"),
        "claims": table_count(db_path, "claims"),
        "claim_ref_map": table_count(db_path, "claim_ref_map"),
        "text_spans": table_count(db_path, "text_spans"),
    }
    return stages


def copy_if_requested(src: Path, dst: Path | None) -> str:
    if not dst:
        return ""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    run_id = f"evidence_v2_{now_id()}"
    parser = argparse.ArgumentParser(
        description="Build an A4 evidence DB with the v2 parse-clean-repair-audit workflow."
    )
    parser.add_argument("--all", action="store_true", help="Process all PDFs in the A4 inbox.")
    parser.add_argument("--pdf", action="append", default=[], help="Process one PDF path. May be repeated.")
    parser.add_argument("--folder", help="Process PDFs in one folder.")
    parser.add_argument("--image-folders", action="store_true", help="Process PDFs under Image_* folders in inbox.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders.")
    parser.add_argument("--limit", type=int, default=0, help="Limit PDFs for smoke tests.")
    parser.add_argument("--move-processed", action="store_true", help="Move PDFs to processed after success.")
    parser.add_argument("--post-only", action="store_true", help="Run v2 cleanup/index/audit on an existing DB.")
    parser.add_argument("--skip-missing-strong", action="store_true", help="Skip PDF reparse repair for missing strong claims.")
    parser.add_argument("--db", default=str(DEFAULT_DB_DIR / f"{run_id}.sqlite"))
    parser.add_argument("--minimal-db", default=str(DEFAULT_MINIMAL_DB))
    parser.add_argument("--pack-db", default=str(DEFAULT_INDEX_DIR / f"patent_evidence_pack_index_{run_id}.sqlite"))
    parser.add_argument("--parsed-json-dir", default=str(A4_PARSED_JSON / run_id))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR / run_id))
    parser.add_argument(
        "--copy-final-db-to",
        default="",
        help="Optional explicit destination for the final DB copy. Never used unless provided.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    db_path = Path(args.db)
    minimal_db = Path(args.minimal_db)
    pack_db = Path(args.pack_db)
    parsed_json_dir = Path(args.parsed_json_dir)
    report_dir = Path(args.report_dir)

    started = time.monotonic()
    report: Dict[str, Any] = {
        "pipeline": "evidence_db_v2",
        "parser_version": PARSER_VERSION_V2,
        "db_path": str(db_path),
        "minimal_db": str(minimal_db),
        "pack_db": str(pack_db),
        "parsed_json_dir": str(parsed_json_dir),
        "report_dir": str(report_dir),
        "post_only": bool(args.post_only),
    }

    if not args.post_only:
        report["build"] = build_from_pdfs(args, db_path=db_path, parsed_json_dir=parsed_json_dir)
    else:
        if not db_path.exists():
            raise SystemExit(f"--post-only DB does not exist: {db_path}")

    report["postprocess"] = run_postprocess(
        db_path=db_path,
        minimal_db=minimal_db,
        pack_db=pack_db,
        report_dir=report_dir,
        skip_missing_strong=bool(args.skip_missing_strong),
    )
    copied_to = copy_if_requested(db_path, Path(args.copy_final_db_to) if args.copy_final_db_to else None)
    if copied_to:
        report["copied_final_db_to"] = copied_to

    report["elapsed_sec"] = round(time.monotonic() - started, 1)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "evidence_db_v2_run.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "report_path": str(report_path),
        "db_path": str(db_path),
        "pack_db": str(pack_db),
        "parser_version": PARSER_VERSION_V2,
        "elapsed_sec": report["elapsed_sec"],
        "audit": report["postprocess"]["audit"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
