from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


BASE = Path("/Volumes/외장 2TB/cpu2026")
INDEX_DIR = BASE / "patent_hub" / "outputs" / "indexes" / "A4"
DEFAULT_MINIMAL_DB = INDEX_DIR / "patent_minimal_index.sqlite"
DEFAULT_EVIDENCE_DB = BASE / "common" / "runtime" / "db" / "patent_A4.sqlite"
DEFAULT_PACK_DB = INDEX_DIR / "patent_evidence_pack_index.sqlite"

PATENT_ID_RE = re.compile(r"\b(?:us|cn|kr)[a-z0-9]{6,}p\b", re.I)
EVIDENCE_CLAIM_RE = re.compile(r"^claim_(.+)$", re.I)
EVIDENCE_FIG_RE = re.compile(r"^fig_(.+)$", re.I)
OCR_NOISE_RE = re.compile(r"(onfrouler|g1iic|ceeee|wees|o\.\.|frorn|vaive|[A-Z]{2,}\d[A-Z]{2,})", re.I)
DEPENDENT_CLAIM_RE = re.compile(
    r"^\s*(?:the\s+.+?\s+of\s+claim\s+\d+|claim\s+\d+|제\s*\d+\s*항\s*에\s*있어서|청구항\s*\d+)",
    re.I,
)
TITLE_CONTAMINATION_RE = re.compile(
    r"(ABSTRACT|摘要|요약|权利要求|청구항|\(\s*(?:30|57|60|71|72|73|86)\s*\)|"
    r"Applicant|Inventor|Assignee|Appl\.?\s*No\.?|Filed:|Prior Publication Data|"
    r"Foreign Application Priority Data|Int\. Cl\.|U\.S\. Cl\.|References Cited|"
    r"Primary Examiner|Attorney, Agent, or Firm|U\.S\. Patent)",
    re.I,
)
TITLE_STOP_RE = re.compile(
    r"\s+(?:\(\d{2}\)|Applicant:|Applicants:|Inventors?:|Assignee:|Notice:|"
    r"Appl\.?\s*No\.?:|Filed:|Prior Publication Data|Foreign Application Priority Data|"
    r"Int\. Cl\.|U\.S\. Cl\.|Field of Classification|References Cited|"
    r"Primary Examiner|Attorney, Agent, or Firm|\(57\)\s*ABSTRACT|ABSTRACT|U\.S\. Patent)",
    re.I,
)
CLAIM_FRONTMATTER_RE = re.compile(
    r"(U\.S\.\s+Patent|Sheet\s+\d+\s+of\s+\d+|cited by examiner|References Cited|"
    r"Foreign Application Priority Data|Primary Examiner)",
    re.I,
)
CLAIM_FIGURE_RE = re.compile(r"\bFIG\.?\s*\d+[A-Z]?\b", re.I)
CLAIM_START_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(?:A|An|The|That|In|Method|Device|System|Apparatus|Circuit|Memory|Non-transitory|One or more|"
    r"What is claimed is|一种|一種|根据|如权利要求|基于|"
    r"제\s*\d+\s*항|청구항|장치|방법|시스템|메모리|반도체)",
    re.I,
)
CLAIM_KEYWORD_RE = re.compile(
    r"\b(comprising|wherein|configured|including|method|device|system|apparatus|circuit|memory|claim|medium|controller|processor|storage)\b|"
    r"(包括|包含|其中|配置|方法|装置|系统|电路|存储器|权利要求|청구항|제\s*\d+\s*항|상기|있어서|포함|구비|방법|장치|시스템|회로|메모리)",
    re.I,
)
OCR_GIBBERISH_RE = re.compile(r"\b(?:Memary|Mernory|EONTRELLER|CENERATOR|Whas|Unesiacted|desver|Aysueg)\b", re.I)


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def json_loads(value: Any, fallback: Any = None) -> Any:
    try:
        return json.loads(value or "[]")
    except Exception:
        return [] if fallback is None else fallback


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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


def clip(value: Any, max_chars: int) -> str:
    return normalize_ws(value)[:max_chars]


def claim_text_quality_flags(text: Any) -> List[str]:
    text = normalize_ws(text)
    if not text:
        return ["empty_claim_text"]
    flags: List[str] = []
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 80:
        flags.append("very_short_claim_text")
    if CLAIM_FRONTMATTER_RE.search(text):
        flags.append("frontmatter_in_claim_text")
    if CLAIM_FIGURE_RE.search(text):
        flags.append("figure_text_in_claim_text")
    if OCR_GIBBERISH_RE.search(text):
        flags.append("ocr_gibberish_in_claim_text")
    if compact:
        letters = sum(ch.isalpha() or "\u4e00" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7a3" for ch in compact)
        symbols = sum((not ch.isalnum()) and not ("\u4e00" <= ch <= "\u9fff") and not ("\uac00" <= ch <= "\ud7a3") for ch in compact)
        if len(compact) >= 120 and letters / len(compact) < 0.42:
            flags.append("low_letter_ratio_claim_text")
        if len(compact) >= 120 and symbols / len(compact) > 0.28:
            flags.append("high_symbol_ratio_claim_text")
    if not CLAIM_START_RE.search(text) and (
        "frontmatter_in_claim_text" in flags or "figure_text_in_claim_text" in flags
    ):
        flags.append("does_not_start_like_claim")
    return flags


def looks_like_substantive_claim(text: Any) -> bool:
    text = normalize_ws(text)
    return bool(CLAIM_START_RE.search(text) and CLAIM_KEYWORD_RE.search(text))


def is_contaminated_claim_text(text: Any) -> bool:
    flags = set(claim_text_quality_flags(text))
    substantive_claim = looks_like_substantive_claim(text)
    if "empty_claim_text" in flags:
        return True
    if "ocr_gibberish_in_claim_text" in flags:
        return True
    if "frontmatter_in_claim_text" in flags and "does_not_start_like_claim" in flags:
        return True
    if "frontmatter_in_claim_text" in flags and "figure_text_in_claim_text" in flags and not substantive_claim:
        return True
    if "figure_text_in_claim_text" in flags and "does_not_start_like_claim" in flags:
        return True
    if {"low_letter_ratio_claim_text", "high_symbol_ratio_claim_text"} <= flags and not substantive_claim:
        return True
    return False


def clean_db_title(raw_title: Any) -> str:
    title = clip(raw_title, 700)
    title = TITLE_STOP_RE.split(title, maxsplit=1)[0]
    title = normalize_ws(title).strip(" -:;,.")
    return title if len(title) >= 8 else ""


def title_needs_repair(title: Any) -> bool:
    title = normalize_ws(title)
    if not title:
        return True
    if len(title) > 180:
        return True
    return bool(OCR_NOISE_RE.search(title) or TITLE_CONTAMINATION_RE.search(title))


def expected_language(patent_id: str) -> str:
    return {"cn": "zh", "us": "en", "kr": "ko"}.get(patent_id[:2].lower(), "")


def evidence_claim_nos(evidence_ids: Sequence[str]) -> List[str]:
    out: List[str] = []
    for evidence_id in evidence_ids:
        match = EVIDENCE_CLAIM_RE.match(str(evidence_id))
        if match:
            out.append(match.group(1))
    return unique_keep_order(out)


def evidence_fig_nos(evidence_ids: Sequence[str]) -> List[str]:
    out: List[str] = []
    for evidence_id in evidence_ids:
        match = EVIDENCE_FIG_RE.match(str(evidence_id))
        if match:
            out.append(match.group(1))
    return unique_keep_order(out)


def iter_minimal_cards(con: sqlite3.Connection) -> Iterable[sqlite3.Row]:
    con.row_factory = sqlite3.Row
    yield from con.execute(
        """
        SELECT patent_id, source_language, summary_language, title_source, title_ko,
               primary_claim_type, secondary_claim_types_json, independent_claim_nos_json,
               protected_terms_json, core_subject, core_elements_json, problem_labels_json,
               solution_labels_json, effect_labels_json, evidence_ids_json, confidence,
               json_path, qc_flags_json, search_text
        FROM minimal_index
        ORDER BY patent_id
        """
    )


def fetch_claims(con: sqlite3.Connection, patent_id: str) -> List[Dict[str, Any]]:
    rows = con.execute(
        """
        SELECT claim_no, parent_claim_no, claim_type, raw_text, norm_text, page_start, page_end
        FROM claims
        WHERE patent_id=?
        ORDER BY CAST(claim_no AS INT), claim_no
        """,
        (patent_id,),
    ).fetchall()
    claims: List[Dict[str, Any]] = []
    for row in rows:
        text = clip(row[3] or row[4], 1400)
        quality_flags = claim_text_quality_flags(text)
        claims.append(
            {
                "claim_no": str(row[0] or ""),
                "parent_claim_no": str(row[1] or ""),
                "claim_type": str(row[2] or ""),
                "text": text,
                "page_start": row[5],
                "page_end": row[6],
                "text_quality_flags": quality_flags,
                "text_contaminated": is_contaminated_claim_text(text),
            }
        )
    return claims


def fetch_figures(con: sqlite3.Connection, patent_id: str, fig_nos: Sequence[str]) -> List[Dict[str, Any]]:
    if not fig_nos:
        return []
    out: List[Dict[str, Any]] = []
    for fig_no in fig_nos[:12]:
        rows = con.execute(
            """
            SELECT figure_no, caption_raw, caption_norm, page_no
            FROM figure_captions
            WHERE patent_id=? AND figure_no=?
            LIMIT 3
            """,
            (patent_id, fig_no),
        ).fetchall()
        for row in rows:
            caption = clip(row[1] or row[2], 500)
            if caption:
                out.append({"figure_no": str(row[0]), "caption": caption, "page_no": row[3]})
    return out


def choose_claims(
    claims: List[Dict[str, Any]],
    minimal_claim_nos: Sequence[str],
    evidence_ids: Sequence[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str], List[str], List[str]]:
    by_no = {claim["claim_no"]: claim for claim in claims}
    parser_independent_nos = unique_keep_order(
        claim["claim_no"] for claim in claims if claim.get("claim_type") == "independent"
    )
    candidate_independent_nos = unique_keep_order([*minimal_claim_nos, *parser_independent_nos])
    independent_nos = []
    for no in candidate_independent_nos:
        claim = by_no.get(no)
        if claim and (claim.get("claim_type") == "dependent" or DEPENDENT_CLAIM_RE.search(claim.get("text", ""))):
            continue
        independent_nos.append(no)
    contaminated_nos = [
        no for no in independent_nos
        if by_no.get(no) and by_no[no].get("text_contaminated")
    ]
    evidence_nos = evidence_claim_nos(evidence_ids)
    strong_nos = independent_nos
    support_nos = [
        no for no in unique_keep_order([*evidence_nos, *minimal_claim_nos])
        if no not in strong_nos
    ]

    strong = [by_no[no] for no in strong_nos if no in by_no and not by_no[no].get("text_contaminated")][:8]
    support = [by_no[no] for no in support_nos if no in by_no and not by_no[no].get("text_contaminated")][:8]
    return strong, support, independent_nos, support_nos, contaminated_nos


def build_quality_flags(
    *,
    patent_id: str,
    source_language: str,
    minimal_title: str,
    clean_title: str,
    minimal_independent_nos: Sequence[str],
    actual_independent_nos: Sequence[str],
    strong_claims: Sequence[Dict[str, Any]],
    support_claims: Sequence[Dict[str, Any]],
    missing_independent_claim_text_nos: Sequence[str],
    contaminated_independent_claim_text_nos: Sequence[str],
    solution_labels: Sequence[str],
    effect_labels: Sequence[str],
    confidence: float,
    minimal_qc_flags: Sequence[str],
) -> List[str]:
    flags: List[str] = []
    if title_needs_repair(minimal_title):
        flags.append("minimal_title_repaired")
    if not clean_title:
        flags.append("missing_clean_title")
    if not strong_claims:
        flags.append("missing_strong_independent_claim_text")
    if missing_independent_claim_text_nos:
        flags.append("missing_some_independent_claim_text")
    if contaminated_independent_claim_text_nos:
        flags.append("contaminated_independent_claim_text_excluded")
    if actual_independent_nos and list(actual_independent_nos) != list(minimal_independent_nos):
        flags.append("minimal_independent_claim_nos_corrected")
    if any(claim.get("claim_type") != "independent" for claim in strong_claims):
        flags.append("strong_claim_not_confirmed_independent")
    if support_claims:
        flags.append("has_dependent_support_claims")
    if not solution_labels:
        flags.append("missing_solution_labels")
    if not effect_labels:
        flags.append("missing_effect_labels")
    if confidence < 0.7:
        flags.append("low_minimal_confidence")
    expected = expected_language(patent_id)
    if expected and source_language != expected:
        flags.append("source_language_mismatch")
    flags.extend(f"minimal_qc:{flag}" for flag in minimal_qc_flags)
    return unique_keep_order(flags)


def create_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        DROP TABLE IF EXISTS evidence_pack_index;
        DROP TABLE IF EXISTS evidence_pack_fts;
        DROP TABLE IF EXISTS evidence_pack_labels;

        CREATE TABLE evidence_pack_index (
            patent_id TEXT PRIMARY KEY,
            source_language TEXT,
            title TEXT,
            title_source TEXT,
            title_quality TEXT,
            core_subject TEXT,
            primary_claim_type TEXT,
            secondary_claim_types_json TEXT,
            independent_claim_nos_json TEXT,
            dependent_support_claim_nos_json TEXT,
            core_elements_json TEXT,
            problem_labels_json TEXT,
            solution_labels_json TEXT,
            effect_labels_json TEXT,
            strong_evidence_ids_json TEXT,
            weak_evidence_ids_json TEXT,
            strong_claims_json TEXT,
            support_claims_json TEXT,
            figures_json TEXT,
            confidence REAL,
            quality_flags_json TEXT,
            minimal_json_path TEXT,
            search_text TEXT
        );

        CREATE TABLE evidence_pack_labels (
            patent_id TEXT NOT NULL,
            label TEXT NOT NULL,
            label_type TEXT NOT NULL
        );

        CREATE INDEX idx_pack_lang ON evidence_pack_index(source_language);
        CREATE INDEX idx_pack_claim_type ON evidence_pack_index(primary_claim_type);
        CREATE INDEX idx_pack_labels ON evidence_pack_labels(label);

        CREATE VIRTUAL TABLE evidence_pack_fts USING fts5(
            patent_id UNINDEXED,
            title,
            core_subject,
            elements,
            labels,
            claims,
            figures,
            search_text
        );
        """
    )


def insert_pack(con: sqlite3.Connection, pack: Dict[str, Any]) -> None:
    con.execute(
        """
        INSERT INTO evidence_pack_index (
            patent_id, source_language, title, title_source, title_quality, core_subject,
            primary_claim_type, secondary_claim_types_json, independent_claim_nos_json,
            dependent_support_claim_nos_json, core_elements_json, problem_labels_json,
            solution_labels_json, effect_labels_json, strong_evidence_ids_json,
            weak_evidence_ids_json, strong_claims_json, support_claims_json,
            figures_json, confidence, quality_flags_json, minimal_json_path, search_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pack["patent_id"],
            pack["source_language"],
            pack["title"],
            pack["title_source"],
            pack["title_quality"],
            pack["core_subject"],
            pack["primary_claim_type"],
            json_dumps(pack["secondary_claim_types"]),
            json_dumps(pack["independent_claim_nos"]),
            json_dumps(pack["dependent_support_claim_nos"]),
            json_dumps(pack["core_elements"]),
            json_dumps(pack["problem_labels"]),
            json_dumps(pack["solution_labels"]),
            json_dumps(pack["effect_labels"]),
            json_dumps(pack["strong_evidence_ids"]),
            json_dumps(pack["weak_evidence_ids"]),
            json_dumps(pack["strong_claims"]),
            json_dumps(pack["support_claims"]),
            json_dumps(pack["figures"]),
            pack["confidence"],
            json_dumps(pack["quality_flags"]),
            pack["minimal_json_path"],
            pack["search_text"],
        ),
    )
    con.execute(
        """
        INSERT INTO evidence_pack_fts (
            patent_id, title, core_subject, elements, labels, claims, figures, search_text
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            pack["patent_id"],
            pack["title"],
            pack["core_subject"],
            " ".join(pack["core_elements"]),
            " ".join([*pack["problem_labels"], *pack["solution_labels"], *pack["effect_labels"]]),
            " ".join(claim["text"] for claim in [*pack["strong_claims"], *pack["support_claims"]]),
            " ".join(fig["caption"] for fig in pack["figures"]),
            pack["search_text"],
        ),
    )
    for label_type in ("problem_labels", "solution_labels", "effect_labels"):
        for label in pack[label_type]:
            con.execute(
                "INSERT INTO evidence_pack_labels (patent_id, label, label_type) VALUES (?, ?, ?)",
                (pack["patent_id"], label, label_type),
            )


def build_pack_for_row(ev_con: sqlite3.Connection, row: sqlite3.Row) -> Dict[str, Any]:
    patent_id = row["patent_id"]
    minimal_title = normalize_ws(row["title_source"])
    core_subject = normalize_ws(row["core_subject"])
    source_language = row["source_language"] or ""
    secondary_claim_types = json_loads(row["secondary_claim_types_json"])
    minimal_independent_nos = unique_keep_order(json_loads(row["independent_claim_nos_json"]))
    core_elements = unique_keep_order(json_loads(row["core_elements_json"]))
    problem_labels = unique_keep_order(json_loads(row["problem_labels_json"]))
    solution_labels = unique_keep_order(json_loads(row["solution_labels_json"]))
    effect_labels = unique_keep_order(json_loads(row["effect_labels_json"]))
    evidence_ids = unique_keep_order(json_loads(row["evidence_ids_json"]))
    minimal_qc_flags = unique_keep_order(json_loads(row["qc_flags_json"]))
    confidence = float(row["confidence"] or 0.0)

    patent_row = ev_con.execute(
        "SELECT title_raw FROM patents WHERE patent_id=? LIMIT 1",
        (patent_id,),
    ).fetchone()
    db_title = clean_db_title(patent_row[0]) if patent_row else ""
    if title_needs_repair(minimal_title) and db_title:
        title = db_title
        title_quality = "db_cleaned"
    else:
        title = minimal_title or db_title
        title_quality = "minimal"
    if title_needs_repair(core_subject) and title:
        core_subject = title

    claims = fetch_claims(ev_con, patent_id)
    strong_claims, support_claims, actual_independent_nos, dependent_support_nos, contaminated_independent_nos = choose_claims(
        claims, minimal_independent_nos, evidence_ids
    )
    independent_claim_nos = actual_independent_nos or minimal_independent_nos
    strong_claim_text_nos = {claim["claim_no"] for claim in strong_claims}
    missing_independent_claim_text_nos = [no for no in independent_claim_nos if no not in strong_claim_text_nos]
    strong_evidence_ids = [f"claim_{no}" for no in independent_claim_nos]
    weak_evidence_ids = unique_keep_order(
        [
            *(f"claim_{claim['claim_no']}" for claim in support_claims),
            *(f"fig_{fig_no}" for fig_no in evidence_fig_nos(evidence_ids)),
        ]
    )
    figures = fetch_figures(ev_con, patent_id, evidence_fig_nos(evidence_ids))
    quality_flags = build_quality_flags(
        patent_id=patent_id,
        source_language=source_language,
        minimal_title=minimal_title,
        clean_title=title,
        minimal_independent_nos=minimal_independent_nos,
        actual_independent_nos=actual_independent_nos,
        strong_claims=strong_claims,
        support_claims=support_claims,
        missing_independent_claim_text_nos=missing_independent_claim_text_nos,
        contaminated_independent_claim_text_nos=contaminated_independent_nos,
        solution_labels=solution_labels,
        effect_labels=effect_labels,
        confidence=confidence,
        minimal_qc_flags=minimal_qc_flags,
    )
    search_text = normalize_ws(
        " ".join(
            [
                patent_id,
                title,
                core_subject,
                " ".join(core_elements),
                " ".join(problem_labels),
                " ".join(solution_labels),
                " ".join(effect_labels),
                " ".join(claim["text"] for claim in strong_claims[:3]),
                " ".join(fig["caption"] for fig in figures[:4]),
            ]
        )
    )
    return {
        "patent_id": patent_id,
        "source_language": source_language,
        "title": title,
        "title_source": minimal_title,
        "title_quality": title_quality,
        "core_subject": core_subject,
        "primary_claim_type": row["primary_claim_type"] or "",
        "secondary_claim_types": secondary_claim_types,
        "independent_claim_nos": independent_claim_nos,
        "dependent_support_claim_nos": dependent_support_nos,
        "core_elements": core_elements,
        "problem_labels": problem_labels,
        "solution_labels": solution_labels,
        "effect_labels": effect_labels,
        "strong_evidence_ids": strong_evidence_ids,
        "weak_evidence_ids": weak_evidence_ids,
        "strong_claims": strong_claims,
        "support_claims": support_claims,
        "figures": figures,
        "confidence": confidence,
        "quality_flags": quality_flags,
        "minimal_json_path": row["json_path"] or "",
        "search_text": search_text,
    }


def evidence_patent_ids(con: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in con.execute("SELECT patent_id FROM patents ORDER BY patent_id").fetchall()
        if row[0]
    }


def build_pack_index(
    minimal_db: Path,
    evidence_db: Path,
    pack_db: Path,
    limit: int = 0,
    evidence_only: bool = False,
) -> Dict[str, Any]:
    started = time.monotonic()
    pack_db.parent.mkdir(parents=True, exist_ok=True)
    if pack_db.exists():
        pack_db.unlink()

    min_con = sqlite3.connect(minimal_db)
    ev_con = sqlite3.connect(evidence_db)
    out_con = sqlite3.connect(pack_db)
    out_con.execute("PRAGMA journal_mode=WAL;")
    out_con.execute("PRAGMA synchronous=NORMAL;")
    create_schema(out_con)

    allowed_patents = evidence_patent_ids(ev_con) if evidence_only else set()
    counts: Dict[str, Any] = {
        "patents": 0,
        "skipped_not_in_evidence_db": 0,
        "title_repaired": 0,
        "claim_nos_corrected": 0,
        "missing_strong": 0,
    }
    for row in iter_minimal_cards(min_con):
        if allowed_patents and str(row["patent_id"]) not in allowed_patents:
            counts["skipped_not_in_evidence_db"] += 1
            continue
        pack = build_pack_for_row(ev_con, row)
        insert_pack(out_con, pack)
        counts["patents"] += 1
        flags = set(pack["quality_flags"])
        if "minimal_title_repaired" in flags:
            counts["title_repaired"] += 1
        if "minimal_independent_claim_nos_corrected" in flags:
            counts["claim_nos_corrected"] += 1
        if "missing_strong_independent_claim_text" in flags:
            counts["missing_strong"] += 1
        if counts["patents"] % 1000 == 0:
            out_con.commit()
            print(f"[pack-index] processed={counts['patents']}")
        if limit and counts["patents"] >= limit:
            break

    out_con.commit()
    min_con.close()
    ev_con.close()
    out_con.close()
    counts["elapsed_sec"] = round(time.monotonic() - started, 1)
    counts["pack_db"] = str(pack_db)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build cleaned evidence pack index for A4 patent QA.")
    parser.add_argument("--minimal-db", default=str(DEFAULT_MINIMAL_DB))
    parser.add_argument("--evidence-db", default=str(DEFAULT_EVIDENCE_DB))
    parser.add_argument("--out-db", default=str(DEFAULT_PACK_DB))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--evidence-only", action="store_true", help="Index only patents present in the evidence DB.")
    args = parser.parse_args()

    counts = build_pack_index(
        minimal_db=Path(args.minimal_db),
        evidence_db=Path(args.evidence_db),
        pack_db=Path(args.out_db),
        limit=args.limit,
        evidence_only=args.evidence_only,
    )
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
