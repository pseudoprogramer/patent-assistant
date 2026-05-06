from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List


BASE = Path("/Volumes/외장 2TB/cpu2026")
INDEX_DIR = BASE / "patent_hub" / "outputs" / "indexes" / "A4"
DEFAULT_MINIMAL_DB = INDEX_DIR / "patent_minimal_index.sqlite"
DEFAULT_EVIDENCE_DB = BASE / "common" / "runtime" / "db" / "patent_A4.sqlite"
DEFAULT_UNITS_DB = INDEX_DIR / "patent_evidence_units.sqlite"


OCR_NOISE_RE = re.compile(r"(ceeee|wees|o\.\.|frorn|vaive|g1iic|onfrouler|[A-Z]{2,}\d[A-Z]{2,})", re.I)
DEPENDENT_CLAIM_RE = re.compile(
    r"^\s*(?:the\s+.+?\s+of\s+claim\s+\d+|claim\s+\d+|제\s*\d+\s*항\s*에\s*있어서|청구항\s*\d+)",
    re.I,
)
PUBLICATION_LINE_RE = re.compile(r"^(?:공개특허|등록특허)\s+\d{2}-\d{4}-\d+\s+-\s+\d+\s+-$")


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def load_json(value: Any) -> Any:
    try:
        return json.loads(value or "[]")
    except Exception:
        return []


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def quality_flags(text: str, unit_type: str) -> List[str]:
    flags: List[str] = []
    clean = normalize_ws(text)
    if not clean:
        flags.append("empty_text")
    if len(clean) < 12 and unit_type not in {"title"}:
        flags.append("very_short_text")
    if unit_type == "claim" and len(clean) < 80:
        flags.append("short_claim_text")
    if OCR_NOISE_RE.search(clean):
        flags.append("ocr_noise")
    if unit_type == "figure" and re.fullmatch(r"fig\.?\s*\d+[a-z]?", clean, re.I):
        flags.append("caption_only")
    if unit_type == "claim" and DEPENDENT_CLAIM_RE.search(clean):
        flags.append("dependent_claim_reference")
    if unit_type == "claim" and PUBLICATION_LINE_RE.search(clean):
        flags.append("publication_line_only")
    return flags


def source_weight(unit_type: str, claim_type: str = "") -> float:
    if unit_type == "claim":
        return 3.0 if claim_type == "independent" else 2.1
    if unit_type == "minimal_summary":
        return 1.5
    if unit_type == "title":
        return 1.2
    if unit_type == "figure":
        return 0.7
    return 1.0


def iter_minimal_rows(con: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    con.row_factory = sqlite3.Row
    yield from con.execute(
        """
        SELECT patent_id, source_language, title_source, primary_claim_type,
               secondary_claim_types_json, independent_claim_nos_json,
               core_subject, core_elements_json, problem_labels_json,
               solution_labels_json, effect_labels_json, evidence_ids_json,
               confidence, qc_flags_json, search_text
        FROM minimal_index
        ORDER BY patent_id
        """
    )


def create_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        DROP TABLE IF EXISTS evidence_units;
        DROP TABLE IF EXISTS evidence_units_fts;

        CREATE TABLE evidence_units (
            unit_id TEXT PRIMARY KEY,
            patent_id TEXT NOT NULL,
            source_language TEXT,
            unit_type TEXT NOT NULL,
            unit_ref TEXT NOT NULL,
            claim_no TEXT,
            claim_type TEXT,
            is_independent_claim INTEGER DEFAULT 0,
            page_no INTEGER,
            text TEXT,
            source_weight REAL DEFAULT 1.0,
            quality_flags_json TEXT DEFAULT '[]',
            minimal_labels_json TEXT DEFAULT '[]',
            minimal_elements_json TEXT DEFAULT '[]',
            title_source TEXT,
            primary_claim_type TEXT,
            confidence REAL,
            qc_flags_json TEXT DEFAULT '[]'
        );

        CREATE INDEX idx_evidence_units_patent ON evidence_units(patent_id);
        CREATE INDEX idx_evidence_units_type ON evidence_units(unit_type);
        CREATE INDEX idx_evidence_units_claim_type ON evidence_units(claim_type);

        CREATE VIRTUAL TABLE evidence_units_fts USING fts5(
            unit_id UNINDEXED,
            patent_id UNINDEXED,
            unit_type UNINDEXED,
            unit_ref UNINDEXED,
            text,
            labels,
            elements,
            title
        );
        """
    )


def insert_unit(
    con: sqlite3.Connection,
    *,
    patent_id: str,
    source_language: str,
    unit_type: str,
    unit_ref: str,
    text: str,
    title_source: str,
    primary_claim_type: str,
    confidence: float,
    qc_flags: List[str],
    labels: List[str],
    elements: List[str],
    claim_no: str = "",
    claim_type: str = "",
    page_no: int | None = None,
) -> None:
    clean = normalize_ws(text)
    if not clean:
        return
    flags = quality_flags(clean, unit_type)
    if unit_type == "claim" and "dependent_claim_reference" in flags:
        claim_type = "dependent_inferred"
    unit_id = f"{patent_id}:{unit_ref}"
    con.execute(
        """
        INSERT OR REPLACE INTO evidence_units (
            unit_id, patent_id, source_language, unit_type, unit_ref,
            claim_no, claim_type, is_independent_claim, page_no, text,
            source_weight, quality_flags_json, minimal_labels_json,
            minimal_elements_json, title_source, primary_claim_type,
            confidence, qc_flags_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            unit_id,
            patent_id,
            source_language,
            unit_type,
            unit_ref,
            claim_no,
            claim_type,
            1 if claim_type == "independent" else 0,
            page_no,
            clean,
            source_weight(unit_type, claim_type),
            json_dumps(flags),
            json_dumps(labels),
            json_dumps(elements),
            title_source,
            primary_claim_type,
            confidence,
            json_dumps(qc_flags),
        ),
    )
    con.execute(
        """
        INSERT INTO evidence_units_fts (unit_id, patent_id, unit_type, unit_ref, text, labels, elements, title)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            unit_id,
            patent_id,
            unit_type,
            unit_ref,
            clean,
            " ".join(labels),
            " ".join(elements),
            title_source,
        ),
    )


def build_units(minimal_db: Path, evidence_db: Path, units_db: Path, limit: int = 0) -> Dict[str, int]:
    if units_db.exists():
        units_db.unlink()
    units_db.parent.mkdir(parents=True, exist_ok=True)

    min_con = sqlite3.connect(minimal_db)
    ev_con = sqlite3.connect(evidence_db)
    out_con = sqlite3.connect(units_db)
    out_con.execute("PRAGMA journal_mode=WAL;")
    out_con.execute("PRAGMA synchronous=NORMAL;")
    create_schema(out_con)

    counts = {"patents": 0, "units": 0, "claims": 0, "figures": 0}
    for row in iter_minimal_rows(min_con):
        patent_id = row["patent_id"]
        labels = [
            *load_json(row["problem_labels_json"]),
            *load_json(row["solution_labels_json"]),
            *load_json(row["effect_labels_json"]),
        ]
        elements = load_json(row["core_elements_json"])
        qc_flags = load_json(row["qc_flags_json"])
        confidence = float(row["confidence"] or 0.0)
        title = normalize_ws(row["title_source"])
        core = normalize_ws(row["core_subject"])
        source_language = row["source_language"] or ""
        primary_claim_type = row["primary_claim_type"] or ""

        insert_unit(
            out_con,
            patent_id=patent_id,
            source_language=source_language,
            unit_type="title",
            unit_ref="title",
            text=title,
            title_source=title,
            primary_claim_type=primary_claim_type,
            confidence=confidence,
            qc_flags=qc_flags,
            labels=labels,
            elements=elements,
        )
        insert_unit(
            out_con,
            patent_id=patent_id,
            source_language=source_language,
            unit_type="minimal_summary",
            unit_ref="minimal_summary",
            text=" ".join([title, core, " ".join(elements), " ".join(labels)]),
            title_source=title,
            primary_claim_type=primary_claim_type,
            confidence=confidence,
            qc_flags=qc_flags,
            labels=labels,
            elements=elements,
        )

        for claim in ev_con.execute(
            """
            SELECT claim_no, claim_type, raw_text, norm_text, page_start
            FROM claims
            WHERE patent_id=?
            ORDER BY CAST(claim_no AS INTEGER), claim_no
            """,
            (patent_id,),
        ):
            claim_no, claim_type, raw_text, norm_text, page_start = claim
            insert_unit(
                out_con,
                patent_id=patent_id,
                source_language=source_language,
                unit_type="claim",
                unit_ref=f"claim_{claim_no}",
                claim_no=str(claim_no),
                claim_type=str(claim_type or ""),
                page_no=page_start,
                text=raw_text or norm_text,
                title_source=title,
                primary_claim_type=primary_claim_type,
                confidence=confidence,
                qc_flags=qc_flags,
                labels=labels,
                elements=elements,
            )
            counts["claims"] += 1

        for fig in ev_con.execute(
            """
            SELECT figure_no, caption_raw, caption_norm, page_no
            FROM figure_captions
            WHERE patent_id=?
            ORDER BY figure_no, page_no
            """,
            (patent_id,),
        ):
            figure_no, caption_raw, caption_norm, page_no = fig
            insert_unit(
                out_con,
                patent_id=patent_id,
                source_language=source_language,
                unit_type="figure",
                unit_ref=f"fig_{figure_no}",
                page_no=page_no,
                text=caption_raw or caption_norm or f"FIG. {figure_no}",
                title_source=title,
                primary_claim_type=primary_claim_type,
                confidence=confidence,
                qc_flags=qc_flags,
                labels=labels,
                elements=elements,
            )
            counts["figures"] += 1

        counts["patents"] += 1
        if counts["patents"] % 500 == 0:
            out_con.commit()
            print(f"[units] patents={counts['patents']}", flush=True)
        if limit and counts["patents"] >= limit:
            break

    out_con.commit()
    counts["units"] = out_con.execute("SELECT COUNT(*) FROM evidence_units").fetchone()[0]
    min_con.close()
    ev_con.close()
    out_con.close()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build claim/figure evidence-unit search index.")
    parser.add_argument("--minimal-db", default=str(DEFAULT_MINIMAL_DB))
    parser.add_argument("--evidence-db", default=str(DEFAULT_EVIDENCE_DB))
    parser.add_argument("--out", default=str(DEFAULT_UNITS_DB))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    started = time.monotonic()
    counts = build_units(Path(args.minimal_db), Path(args.evidence_db), Path(args.out), limit=args.limit)
    elapsed = time.monotonic() - started
    print(f"[units] wrote {args.out}")
    print(f"[units] counts={counts}, elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
