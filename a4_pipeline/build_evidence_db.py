from __future__ import annotations
import argparse
import json
import re
import shutil
import time
from pathlib import Path

import fitz

from db_schema import ensure_db, get_connection, reset_patent_artifacts, upsert_job, increment_job_retry

try:
    from config import (
        A4_INBOX,
        A4_PROCESSED,
        A4_PARSED_JSON,
        A4_DB,
        A4_LOGS,
        A4_QUARANTINE,
        A4_TMP,
        A4_CACHE,
        A4_RAW_INVALID,
        ensure_runtime_dirs,
    )
except Exception:
    BASE = Path("/Volumes/외장 2TB/cpu2026")
    HUB = BASE / "patent_hub"
    COMMON = BASE / "common"
    A4_INBOX = HUB / "raw_patents" / "inbox" / "A4"
    A4_PROCESSED = HUB / "raw_patents" / "processed" / "A4"
    A4_PARSED_JSON = HUB / "outputs" / "parsed_json" / "A4"
    A4_DB = COMMON / "runtime" / "db" / "patent_A4.sqlite"
    A4_LOGS = COMMON / "runtime" / "logs" / "A4"
    A4_QUARANTINE = COMMON / "runtime" / "quarantine" / "A4"
    A4_TMP = COMMON / "runtime" / "tmp" / "A4"
    A4_CACHE = COMMON / "runtime" / "cache" / "A4"
    A4_RAW_INVALID = COMMON / "runtime" / "raw_invalid" / "A4"

    def ensure_runtime_dirs():
        for p in [A4_LOGS, A4_QUARANTINE, A4_TMP, A4_CACHE, A4_RAW_INVALID]:
            Path(p).mkdir(parents=True, exist_ok=True)

PARSER_VERSION = "evidence_db_v1.2"
LOG_FILE_PATH = None


def init_log_file() -> None:
    global LOG_FILE_PATH
    A4_LOGS.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    LOG_FILE_PATH = A4_LOGS / f"evidence_build_{ts}.log"


def log(msg: str) -> None:
    line = f"[EVIDENCE] {msg}"
    print(line, flush=True)
    if LOG_FILE_PATH:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def ensure_output_dirs() -> None:
    A4_INBOX.mkdir(parents=True, exist_ok=True)
    A4_PROCESSED.mkdir(parents=True, exist_ok=True)
    A4_PARSED_JSON.mkdir(parents=True, exist_ok=True)
    A4_QUARANTINE.mkdir(parents=True, exist_ok=True)


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def progress_bar(done: int, total: int, width: int = 30) -> str:
    ratio = 0 if total == 0 else done / total
    filled = int(width * ratio)
    return "[" + "-" * filled + "-" * (width - filled) + f"] {done}/{total}"


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def list_pdfs(root: Path = A4_INBOX, recursive: bool = False) -> list[Path]:
    pattern = "**/*.pdf" if recursive else "*.pdf"
    return sorted(root.glob(pattern))


def detect_country(pdf_name: str, first_text: str) -> str:
    name = pdf_name.lower()
    if name.startswith("cn"):
        return "CN"
    if name.startswith("kr"):
        return "KR"
    if name.startswith("us"):
        return "US"

    if "国家知识产权局" in first_text or "发明专利" in first_text:
        return "CN"
    if "대한민국특허청" in first_text:
        return "KR"
    if "United States Patent" in first_text or "What is claimed is" in first_text:
        return "US"
    return "UNKNOWN"


def extract_meta(country: str, pdf_path: Path, full_text: str) -> dict:
    text_head = full_text[:20000]
    title = ""
    assignee = ""
    application_no = ""
    publication_no = ""

    if country == "CN":
        title = _first_group(text_head, [r"\(54\)发明名称\s*(.+)"])
        assignee = _first_group(text_head, [r"\(73\)专利权人\s*(.+?)(?:地址|\(72\)|\(74\)|\(51\))"])
        application_no = _first_group(text_head, [r"\(21\)申请号\s*([A-Za-z0-9\.\s]+)"])
        publication_no = _first_group(text_head, [r"申请公布号\s*(CN\s*[A-Z0-9\s]+)", r"(CN\s*\d+\s*[ABU])"])
    elif country == "KR":
        title = _first_group(text_head, [r"\(54\)\s*발명의 명칭\s*(.+)"])
        assignee = _first_group(text_head, [r"\(73\)\s*특허권자\s*(.+?)(?:\(72\)|\(74\)|전체 청구항 수)"])
        application_no = _first_group(text_head, [r"\(21\)\s*출원번호\s*([0-9\-]+)"])
        publication_no = _first_group(text_head, [r"\(11\)\s*등록번호\s*([0-9\-]+)"])
    elif country == "US":
        title = _first_group(text_head, [r"\(54\)\s*(.+)", r"Title\s*(.+)"])
        assignee = _first_group(text_head, [r"Assignee[:\s]*(.+?)(?:Inventors|Appl\.|Filed)"])
        application_no = _first_group(text_head, [r"Appl\.\s*No\.\s*([0-9/\,]+)"])
        publication_no = _first_group(text_head, [r"United States Patent\s*([0-9\,]+)"])

    patent_id = pdf_path.stem
    return {
        "patent_id": patent_id,
        "country": country,
        "title_raw": normalize_ws(title) or patent_id,
        "assignee_raw": normalize_ws(assignee),
        "application_no": normalize_ws(application_no),
        "publication_no": normalize_ws(publication_no),
    }


def _first_group(text: str, patterns: list[str]) -> str:
    """
    Safe metadata extractor:
    - prefer capture group 1 when present
    - fall back to the whole match when the pattern has no capture group
    """
    for pat in patterns:
        m = re.search(pat, text, flags=re.S | re.M)
        if m:
            if m.lastindex and m.lastindex >= 1:
                return normalize_ws(m.group(1))
            return normalize_ws(m.group(0))
    return ""


def extract_pages_and_spans(pdf_path: Path) -> tuple[list[dict], str]:
    doc = fitz.open(pdf_path)
    pages = []
    full_text_parts = []
    for page_index, page in enumerate(doc):
        page_no = page_index + 1
        page_dict = page.get_text("dict")
        page_text = page.get_text("text")
        full_text_parts.append(page_text)
        spans = []
        for block_no, block in enumerate(page_dict.get("blocks", [])):
            if block.get("type") != 0:
                continue
            for line_no, line in enumerate(block.get("lines", [])):
                for span_no, span in enumerate(line.get("spans", [])):
                    raw = span.get("text", "")
                    if not raw or not raw.strip():
                        continue
                    x0, y0, x1, y1 = span.get("bbox", (None, None, None, None))
                    spans.append(
                        {
                            "page_no": page_no,
                            "span_id": f"p{page_no}_b{block_no}_l{line_no}_s{span_no}",
                            "block_no": block_no,
                            "line_no": line_no,
                            "span_no": span_no,
                            "raw_text": raw,
                            "norm_text": normalize_ws(raw),
                            "bbox": (x0, y0, x1, y1),
                        }
                    )
        pages.append(
            {
                "page_no": page_no,
                "width": float(page.rect.width),
                "height": float(page.rect.height),
                "text": page_text,
                "spans": spans,
            }
        )
    doc.close()
    return pages, "\n".join(full_text_parts)


def guess_section(page_no: int, claim_pages: set[int], drawing_pages: set[int]) -> str:
    if page_no in claim_pages:
        return "claim"
    if page_no in drawing_pages:
        return "drawing"
    return "description"


def _strip_cn_page_artifacts(text: str) -> str:
    """
    Remove common CN patent page headers/footers that break claim extraction.
    Conservative cleanup only.
    """
    if not text:
        return ""

    t = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = t.split("\n")
    cleaned = []

    header_patterns = [
        re.compile(r"^\s*权\s*利\s*要\s*求\s*书.*页\s*$"),
        re.compile(r"^\s*说\s*明\s*书.*页\s*$"),
        re.compile(r"^\s*说\s*明\s*书\s*附\s*图.*页\s*$"),
        re.compile(r"^\s*CN\s*[A-Z]?\s*[\d\s]+[ABU]?\s*$", re.I),
        re.compile(r"^\s*\d+\s*$"),
    ]

    for line in lines:
        s = line.strip()
        if not s:
            cleaned.append("")
            continue
        if any(p.match(s) for p in header_patterns):
            continue
        cleaned.append(line)

    t = "\n".join(cleaned)
    t = re.sub(r"CN\s+(\d)\s+(\d)\s+(\d)\s+(\d)\s+(\d+)\s+([ABU])", r"CN \1\2\3\4\5 \6", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t


def _find_claim_region_cn(full_text: str) -> str:
    t = _strip_cn_page_artifacts(full_text)

    start_candidates = [
        r"权\s*利\s*要\s*求\s*书",
        r"(?m)^\s*1\s*[\.．、]\s*",
    ]
    end_candidates = [
        r"说\s*明\s*书",
        r"技\s*术\s*领\s*域",
        r"发\s*明\s*内\s*容",
        r"附\s*图\s*说\s*明",
    ]

    start_pos = None
    start_matches = []
    for p in start_candidates:
        start_matches.extend(m.start() for m in re.finditer(p, t, re.M))
    if start_matches:
        start_pos = min(start_matches)
    if start_pos is None:
        start_pos = 0

    end_pos = None
    for p in end_candidates:
        m = re.search(p, t[start_pos:], re.M)
        if m:
            end_pos = start_pos + m.start()
            break
    if end_pos is None:
        end_pos = min(len(t), start_pos + 120000)

    return t[start_pos:end_pos].strip()


def _compile_claim_start_pattern_cn() -> re.Pattern:
    return re.compile(r"(?m)(?:^|\n)\s*(\d{1,3})\s*[\.．、]\s*(?=\S)")


def _slice_claim_blocks_cn(region_text: str) -> list[tuple[str, str]]:
    region = region_text.replace("\r\n", "\n").replace("\r", "\n")
    pat = _compile_claim_start_pattern_cn()
    matches = list(pat.finditer(region))
    out = []
    if not matches:
        return out

    for i, m in enumerate(matches):
        claim_no = m.group(1)
        block_start = m.start()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(region)
        block = region[block_start:block_end].strip()
        out.append((claim_no, block))
    return out


def infer_parent_claim_no(country: str, claim_text: str) -> str | None:
    patterns = []
    if country == "CN":
        patterns = [
            r"根据权利要求\s*(\d+)\s*所述",
            r"根据权利要求\s*([0-9]+)",
        ]
    elif country == "KR":
        patterns = [r"청구항\s*(\d+)에\s*있어서", r"청구항\s*(\d+)"]
    else:
        patterns = [r"claim\s*(\d+)"]
    for pat in patterns:
        m = re.search(pat, claim_text, flags=re.I)
        if m:
            return m.group(1)
    return None


def find_first_page_for_snippet(pages: list[dict], snippet: str, allowed_pages: set[int] | None = None) -> int | None:
    key = normalize_ws(snippet)[:30]
    if not key:
        return None
    for p in pages:
        if allowed_pages is not None and p["page_no"] not in allowed_pages:
            continue
        if key in normalize_ws(p["text"]):
            return p["page_no"]
    return None


def find_last_page_for_snippet(pages: list[dict], snippet: str, allowed_pages: set[int] | None = None) -> int | None:
    key = normalize_ws(snippet)[:30]
    if not key:
        return None
    found = None
    for p in pages:
        if allowed_pages is not None and p["page_no"] not in allowed_pages:
            continue
        if key in normalize_ws(p["text"]):
            found = p["page_no"]
    return found


def parse_claims(country: str, pages: list[dict]) -> tuple[list[dict], set[int]]:
    text_by_page = {p["page_no"]: p["text"] for p in pages}
    claim_pages = set()

    if country == "CN":
        # CN front pages often contain "权利要求书1页" as bibliographic metadata.
        # Exclude the cover page first so that metadata does not mask the real claims.
        claim_source_pages = pages[1:] if len(pages) > 1 else pages
        full_text = "\n".join([p["text"] for p in claim_source_pages])
        region = _find_claim_region_cn(full_text)
        blocks = _slice_claim_blocks_cn(region)
        page_scan_source = claim_source_pages
        if not blocks and claim_source_pages is not pages:
            full_text = "\n".join([p["text"] for p in pages])
            region = _find_claim_region_cn(full_text)
            blocks = _slice_claim_blocks_cn(region)
            page_scan_source = pages

        if not blocks:
            return [], set()

        for p in page_scan_source:
            txt_norm = normalize_ws(p["text"])
            for _, block in blocks[:3]:
                probe = normalize_ws(block)[:20]
                if probe and probe in txt_norm:
                    claim_pages.add(p["page_no"])
            if "权利要求书" in p["text"].replace(" ", ""):
                claim_pages.add(p["page_no"])

        claims = []
        seen = set()
        for claim_no, raw_block in blocks:
            if claim_no in seen:
                continue
            seen.add(claim_no)

            raw_text = raw_block
            raw_text = re.sub(r"(?m)^\s*权\s*利\s*要\s*求\s*书.*$", "", raw_text)
            raw_text = re.sub(r"(?m)^\s*CN\s*[A-Z]?\s*[\d\s]+[ABU]?\s*$", "", raw_text, flags=re.I)
            raw_text = re.sub(r"(?m)^\s*\d+\s*$", "", raw_text)
            raw_text = re.sub(r"\n{3,}", "\n\n", raw_text).strip()

            if not raw_text:
                continue

            parent_claim_no = infer_parent_claim_no(country, raw_text)
            claim_type = "dependent" if parent_claim_no else "independent"
            start_page = find_first_page_for_snippet(pages, raw_text[:80], claim_pages if claim_pages else None)
            end_page = find_last_page_for_snippet(pages, raw_text[-80:], claim_pages if claim_pages else None) or start_page

            claims.append(
                {
                    "claim_no": claim_no,
                    "parent_claim_no": parent_claim_no,
                    "claim_type": claim_type,
                    "raw_text": raw_text,
                    "norm_text": normalize_ws(raw_text),
                    "page_start": start_page,
                    "page_end": end_page,
                }
            )

        if not claim_pages:
            for c in claims:
                if c.get("page_start"):
                    claim_pages.add(c["page_start"])
                if c.get("page_end"):
                    claim_pages.add(c["page_end"])

        return claims, claim_pages

    # KR / US fallback
    description_start_page = None
    claim_start_page = None

    for p in pages:
        txt = p["text"]
        if claim_start_page is None:
            if country == "KR" and "청구범위" in txt:
                claim_start_page = p["page_no"]
            elif country == "US" and re.search(
                r"(?i)what\s+is\s+claimed\s+is|the\s+invention\s+claimed\s+is|(?:^|\n)\s*(?:i|we)\s+claim\s*:?",
                txt,
            ):
                claim_start_page = p["page_no"]
        if description_start_page is None:
            if (country == "KR" and "발명의 설명" in txt) or (country == "US" and "Detailed Description" in txt):
                description_start_page = p["page_no"]

    if claim_start_page is None:
        fallback_pages = pages[:8]
        if country == "US":
            fallback_pages = pages[1:]
        for p in fallback_pages:
            if country == "US":
                page_no = p["page_no"]
                window = "\n".join(text_by_page.get(n, "") for n in range(page_no, min(len(pages) + 1, page_no + 3)))
                starts = re.findall(r"(?m)^\s*(\d{1,3})\s*[\.:]\s+\S", p["text"])
                unique_starts = {int(n) for n in starts}
                has_claim_language = re.search(
                    r"(?im)^\s*\d{1,3}\s*[\.:]\s+(?:A|An|The|In)\b.*(?:comprising|claim|wherein)",
                    p["text"],
                )
                has_canceled_range = re.search(r"(?im)^\s*\d{1,3}\s*-\s*\d{1,3}\s*[\.:]\s*\(canceled\)", p["text"])
                has_numbered_claims = (
                    re.search(r"(?m)^\s*1\s*[\.:]\s+\S", p["text"])
                    and re.search(r"(?m)^\s*2\s*[\.:]\s+\S", window)
                    and re.search(r"(?m)^\s*3\s*[\.:]\s+\S", window)
                )
                has_late_claims = len(unique_starts) >= 3 and has_claim_language
                has_terminal_claim = page_no >= len(pages) - 1 and has_claim_language and (1 in unique_starts or has_canceled_range)
                if has_numbered_claims or has_late_claims or has_terminal_claim:
                    claim_start_page = page_no
                    break
            elif re.search(r"(^|\n)\s*1\s*[\.．]", p["text"]):
                claim_start_page = p["page_no"]
                break

    if claim_start_page is None:
        return [], set()

    if description_start_page is None or description_start_page <= claim_start_page:
        if country == "US":
            description_start_page = len(pages) + 1
        else:
            description_start_page = min(len(pages) + 1, claim_start_page + 6)

    claim_text_parts = []
    for page_no in range(claim_start_page, description_start_page):
        if page_no in text_by_page:
            claim_pages.add(page_no)
            claim_text_parts.append(text_by_page[page_no])
    claim_text = "\n".join(claim_text_parts)

    if country == "US":
        header = re.search(
            r"(?is)what\s+is\s+claimed\s+is\s*:|the\s+invention\s+claimed\s+is\s*:|(?:^|\n)\s*(?:i|we)\s+claim\s*:?",
            claim_text,
        )
        if header:
            claim_text = claim_text[header.end():]

    if country == "KR":
        split_pat = r"(?m)^\s*청구항\s*(\d+)\s*"
    else:
        split_pat = r"(?m)^\s*(\d+)\s*[\.:]\s*"

    parts = re.split(split_pat, claim_text)
    claims = []
    if len(parts) >= 3:
        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):
                break
            claim_no = normalize_ws(parts[i])
            raw = normalize_ws(parts[i + 1])
            if not claim_no or not raw:
                continue
            parent_claim_no = infer_parent_claim_no(country, raw)
            claim_type = "dependent" if parent_claim_no else "independent"
            start_page = find_first_page_for_snippet(pages, raw[:80], claim_pages)
            end_page = find_last_page_for_snippet(pages, raw[-80:], claim_pages) or start_page
            claims.append(
                {
                    "claim_no": claim_no,
                    "parent_claim_no": parent_claim_no,
                    "claim_type": claim_type,
                    "raw_text": raw,
                    "norm_text": normalize_ws(raw),
                    "page_start": start_page,
                    "page_end": end_page,
                }
            )

    return claims, claim_pages


REF_TOKEN_RE = re.compile(r"\b(?:\d+[A-Za-z]+|[A-Za-z]{1,8}\d+[A-Za-z]*|W\d+|SGMC\d+|MS\d+|IF\d+|CS|DR|CR|FG|CG|SG|EG)\b")
FIGURE_RE = re.compile(r"图\s*(\d+)|FIG\.?\s*(\d+)|도\s*(\d+)", re.I)


def extract_references_and_figures(
    pages: list[dict], claims: list[dict], claim_pages: set[int]
) -> tuple[list[dict], list[dict], list[dict], list[dict], set[int]]:
    claim_text_map = {c["claim_no"]: c for c in claims}
    references = []
    claim_ref_map = []
    figure_captions = []
    drawing_ref_map = []
    drawing_pages = set()

    # find explicit figure pages/captions
    for p in pages:
        lines = [normalize_ws(x) for x in p["text"].splitlines() if normalize_ws(x)]
        for line in lines[:8]:
            m = FIGURE_RE.search(line)
            if m:
                figure_no = next(g for g in m.groups() if g)
                drawing_pages.add(p["page_no"])
                figure_captions.append(
                    {
                        "figure_no": figure_no,
                        "caption_raw": line,
                        "caption_norm": normalize_ws(line),
                        "page_no": p["page_no"],
                    }
                )
                break

    # also parse descriptive captions from description pages
    for p in pages:
        if p["page_no"] in drawing_pages:
            continue
        for line in [normalize_ws(x) for x in p["text"].splitlines() if normalize_ws(x)]:
            m = re.match(r"^(图\s*\d+[^。；\n]*[。；]?)", line)
            if m:
                fig_no = re.search(r"图\s*(\d+)", m.group(1))
                if fig_no:
                    figure_captions.append(
                        {
                            "figure_no": fig_no.group(1),
                            "caption_raw": m.group(1),
                            "caption_norm": normalize_ws(m.group(1)),
                            "page_no": p["page_no"],
                        }
                    )

    figure_by_page = {}
    for cap in figure_captions:
        figure_by_page.setdefault(cap["page_no"], cap["figure_no"])

    for p in pages:
        section = guess_section(p["page_no"], claim_pages, drawing_pages)
        current_figure = figure_by_page.get(p["page_no"])
        for span in p["spans"]:
            raw = span["raw_text"]
            if not raw:
                continue
            tokens = list(REF_TOKEN_RE.finditer(raw))
            if not tokens:
                continue
            for m in tokens:
                ref = m.group(0)
                if ref.isdigit() and len(ref) <= 2:
                    continue
                x0, y0, x1, y1 = span["bbox"]
                entity = {
                    "ref_no_raw": ref,
                    "ref_no_norm": ref.upper(),
                    "label_raw": None,
                    "label_norm": None,
                    "source_section": section,
                    "page_no": p["page_no"],
                    "bbox": (x0, y0, x1, y1),
                }
                references.append(entity)
                if section == "drawing" and current_figure:
                    drawing_ref_map.append(
                        {
                            "figure_no": current_figure,
                            "ref_no_raw": ref,
                            "page_no": p["page_no"],
                            "bbox": (x0, y0, x1, y1),
                        }
                    )

    for claim in claims:
        refs = set(REF_TOKEN_RE.findall(claim["raw_text"]))
        for ref in refs:
            claim_ref_map.append(
                {
                    "claim_no": claim["claim_no"],
                    "ref_no_raw": ref,
                    "mention_text": ref,
                    "page_no": claim.get("page_start"),
                    "bbox": (None, None, None, None),
                }
            )

    return references, claim_ref_map, figure_captions, drawing_ref_map, drawing_pages


def upsert_patent(con, meta: dict, pdf_path: Path, page_count: int) -> None:
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO patents (
            patent_id, country, title_raw, assignee_raw, application_no,
            publication_no, pdf_path, page_count, parser_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(patent_id) DO UPDATE SET
            country=excluded.country,
            title_raw=excluded.title_raw,
            assignee_raw=excluded.assignee_raw,
            application_no=excluded.application_no,
            publication_no=excluded.publication_no,
            pdf_path=excluded.pdf_path,
            page_count=excluded.page_count,
            parser_version=excluded.parser_version,
            updated_at=CURRENT_TIMESTAMP
        """,
        (
            meta["patent_id"],
            meta["country"],
            meta["title_raw"],
            meta["assignee_raw"],
            meta["application_no"],
            meta["publication_no"],
            str(pdf_path),
            page_count,
            PARSER_VERSION,
        ),
    )
    con.commit()


def insert_pages(con, patent_id: str, pages: list[dict]) -> None:
    cur = con.cursor()
    cur.executemany(
        "INSERT OR REPLACE INTO pages (patent_id, page_no, width, height) VALUES (?, ?, ?, ?)",
        [(patent_id, p["page_no"], p["width"], p["height"]) for p in pages],
    )
    con.commit()


def insert_text_spans(con, patent_id: str, pages: list[dict]) -> None:
    cur = con.cursor()
    rows = []
    for p in pages:
        for s in p["spans"]:
            x0, y0, x1, y1 = s["bbox"]
            rows.append(
                (
                    patent_id,
                    s["page_no"],
                    s["span_id"],
                    s["block_no"],
                    s["line_no"],
                    s["span_no"],
                    s["raw_text"],
                    s["norm_text"],
                    x0,
                    y0,
                    x1,
                    y1,
                )
            )
    cur.executemany(
        """
        INSERT OR REPLACE INTO text_spans (
            patent_id, page_no, span_id, block_no, line_no, span_no,
            raw_text, norm_text, x0, y0, x1, y1
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()


def insert_claims(con, patent_id: str, claims: list[dict]) -> None:
    cur = con.cursor()
    cur.executemany(
        """
        INSERT OR REPLACE INTO claims (
            patent_id, claim_no, parent_claim_no, claim_type,
            raw_text, norm_text, page_start, page_end
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                patent_id,
                c["claim_no"],
                c["parent_claim_no"],
                c["claim_type"],
                c["raw_text"],
                c["norm_text"],
                c["page_start"],
                c["page_end"],
            )
            for c in claims
        ],
    )
    con.commit()


def insert_ref_entities(con, patent_id: str, refs: list[dict]) -> None:
    cur = con.cursor()
    rows = []
    for r in refs:
        x0, y0, x1, y1 = r["bbox"]
        rows.append(
            (
                patent_id,
                r["ref_no_raw"],
                r["ref_no_norm"],
                r["label_raw"],
                r["label_norm"],
                r["source_section"],
                r["page_no"],
                x0,
                y0,
                x1,
                y1,
            )
        )
    cur.executemany(
        """
        INSERT OR IGNORE INTO ref_entities (
            patent_id, ref_no_raw, ref_no_norm, label_raw, label_norm,
            source_section, page_no, x0, y0, x1, y1
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()


def insert_claim_ref_map(con, patent_id: str, rows_in: list[dict]) -> None:
    cur = con.cursor()
    rows = []
    for r in rows_in:
        x0, y0, x1, y1 = r["bbox"]
        rows.append((patent_id, r["claim_no"], r["ref_no_raw"], r["mention_text"], r["page_no"], x0, y0, x1, y1))
    cur.executemany(
        """
        INSERT OR IGNORE INTO claim_ref_map (
            patent_id, claim_no, ref_no_raw, mention_text,
            page_no, x0, y0, x1, y1
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()


def insert_figure_captions(con, patent_id: str, rows_in: list[dict]) -> None:
    cur = con.cursor()
    cur.executemany(
        """
        INSERT OR IGNORE INTO figure_captions (
            patent_id, figure_no, caption_raw, caption_norm, page_no
        ) VALUES (?, ?, ?, ?, ?)
        """,
        [(patent_id, r["figure_no"], r["caption_raw"], r["caption_norm"], r["page_no"]) for r in rows_in],
    )
    con.commit()


def insert_drawing_ref_map(con, patent_id: str, rows_in: list[dict]) -> None:
    cur = con.cursor()
    rows = []
    for r in rows_in:
        x0, y0, x1, y1 = r["bbox"]
        rows.append((patent_id, r["figure_no"], r["ref_no_raw"], r["page_no"], x0, y0, x1, y1))
    cur.executemany(
        """
        INSERT OR IGNORE INTO drawing_ref_map (
            patent_id, figure_no, ref_no_raw, page_no,
            x0, y0, x1, y1
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    con.commit()


def save_summary_json(meta: dict, pages: list[dict], claims: list[dict], refs: list[dict], figures: list[dict]) -> Path:
    out = {
        "patent_id": meta["patent_id"],
        "country": meta["country"],
        "title_raw": meta["title_raw"],
        "assignee_raw": meta["assignee_raw"],
        "application_no": meta["application_no"],
        "publication_no": meta["publication_no"],
        "page_count": len(pages),
        "claims_count": len(claims),
        "references_count": len(refs),
        "figures_count": len(figures),
        "parser_version": PARSER_VERSION,
    }
    path = A4_PARSED_JSON / f"{safe_name(meta['patent_id'])}.evidence_summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    return path


def move_to_processed(pdf_path: Path, patent_id: str) -> Path:
    target = A4_PROCESSED / f"{safe_name(patent_id)}{pdf_path.suffix.lower()}"
    if target.exists():
        target = A4_PROCESSED / pdf_path.name
    shutil.move(str(pdf_path), str(target))
    return target


def move_to_quarantine(pdf_path: Path) -> Path:
    A4_QUARANTINE.mkdir(parents=True, exist_ok=True)
    target = A4_QUARANTINE / pdf_path.name
    if target.exists():
        target = A4_QUARANTINE / f"{pdf_path.stem}_{int(time.time())}{pdf_path.suffix}"
    shutil.move(str(pdf_path), str(target))
    return target


def process_one_pdf(pdf_path: Path, move_processed: bool) -> dict:
    pages, full_text = extract_pages_and_spans(pdf_path)
    country = detect_country(pdf_path.name, full_text[:10000])
    meta = extract_meta(country, pdf_path, full_text)
    claims, claim_pages = parse_claims(country, pages)
    refs, claim_ref_map, figure_caps, drawing_ref_map, drawing_pages = extract_references_and_figures(pages, claims, claim_pages)

    con = get_connection()
    try:
        upsert_job(con, meta["patent_id"], str(pdf_path), "evidence_running")
        reset_patent_artifacts(con, meta["patent_id"])
        upsert_patent(con, meta, pdf_path, len(pages))
        insert_pages(con, meta["patent_id"], pages)
        insert_text_spans(con, meta["patent_id"], pages)
        insert_claims(con, meta["patent_id"], claims)
        insert_ref_entities(con, meta["patent_id"], refs)
        insert_claim_ref_map(con, meta["patent_id"], claim_ref_map)
        insert_figure_captions(con, meta["patent_id"], figure_caps)
        insert_drawing_ref_map(con, meta["patent_id"], drawing_ref_map)
        upsert_job(con, meta["patent_id"], str(pdf_path), "evidence_done")
    finally:
        con.close()

    summary_path = save_summary_json(meta, pages, claims, refs, figure_caps)
    moved_path = None
    if move_processed:
        moved_path = move_to_processed(pdf_path, meta["patent_id"])

    return {
        "patent_id": meta["patent_id"],
        "country": country,
        "summary_json": str(summary_path),
        "claims_count": len(claims),
        "refs_count": len(refs),
        "figures_count": len(figure_caps),
        "moved_pdf": str(moved_path) if moved_path else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="Process all PDFs in inbox")
    parser.add_argument("--pdf", type=str, help="Process one PDF path")
    parser.add_argument("--folder", type=str, help="Process PDFs in one folder")
    parser.add_argument("--image-folders", action="store_true", help="Process PDFs under Image_* folders in inbox")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subfolders for --all or --folder")
    parser.add_argument("--move-processed", action="store_true", help="Move original PDF to processed after success")
    parser.add_argument("--no-quarantine", action="store_true", help="Do not move failed PDFs to quarantine")
    args = parser.parse_args()

    ensure_runtime_dirs()
    ensure_output_dirs()
    ensure_db()
    init_log_file()

    if args.pdf:
        pdfs = [Path(args.pdf)]
    elif args.folder:
        pdfs = list_pdfs(Path(args.folder), recursive=args.recursive)
    elif args.image_folders:
        image_folders = sorted(d for d in A4_INBOX.glob("Image_*") if d.is_dir())
        pdfs = []
        for folder in image_folders:
            pdfs.extend(list_pdfs(folder, recursive=args.recursive))
    elif args.all:
        pdfs = list_pdfs(A4_INBOX, recursive=args.recursive)
    else:
        parser.error("Use --all, --folder, --image-folders, or --pdf")
        return

    total = len(pdfs)
    if total == 0:
        log("inbox 안에 처리할 PDF가 없습니다.")
        return

    log(f"[시작] 대상 PDF 수: {total}")
    log(f"[설정] DB: {A4_DB}")
    log(f"[설정] inbox: {A4_INBOX}")
    log(f"[설정] parser_version: {PARSER_VERSION}")

    success = 0
    failed = 0
    start_all = time.time()

    for idx, pdf_path in enumerate(pdfs, start=1):
        log("")
        log(progress_bar(idx, total))
        log(f"[처리중] {idx}/{total} - {pdf_path.name}")
        t0 = time.time()
        patent_id = pdf_path.stem
        try:
            result = process_one_pdf(pdf_path, move_processed=args.move_processed)
            success += 1
            elapsed = time.time() - t0
            log(f"    ✓ 완료: {result['patent_id']}")
            log(f"      · claims={result['claims_count']}, refs={result['refs_count']}, figures={result['figures_count']}")
            log(f"      · summary_json: {result['summary_json']}")
            if result['moved_pdf']:
                log(f"      · moved_pdf: {result['moved_pdf']}")
            log(f"      · 소요 시간: {elapsed:.1f}초")
        except Exception as e:
            failed += 1
            elapsed = time.time() - t0
            moved = pdf_path if args.no_quarantine else move_to_quarantine(pdf_path)
            con = get_connection()
            try:
                increment_job_retry(con, patent_id, str(moved), str(e))
            finally:
                con.close()
            log(f"    ✗ 실패: {pdf_path.name}")
            log(f"      오류: {e}")
            if args.no_quarantine:
                log(f"      · quarantine 이동 안 함: {moved}")
            else:
                log(f"      · quarantine 이동: {moved}")
            log(f"      · 소요 시간: {elapsed:.1f}초")

    total_elapsed = time.time() - start_all
    log("")
    log(f"[종료] 성공: {success}, 실패: {failed}, 총 소요: {total_elapsed:.1f}초")


if __name__ == "__main__":
    main()
