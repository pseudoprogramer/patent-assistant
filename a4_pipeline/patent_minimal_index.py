from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

BASE = Path("/Volumes/외장 2TB/cpu2026")
HUB = BASE / "patent_hub"
COMMON = BASE / "common"

A4_DB = COMMON / "runtime" / "db" / "patent_A4.sqlite"
A4_LOGS = COMMON / "runtime" / "logs" / "A4"
A4_RAW_INVALID = COMMON / "runtime" / "raw_invalid" / "A4" / "minimal"
MINIMAL_DIR = HUB / "outputs" / "minimal_analysis" / "A4"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:14b"
KEEP_ALIVE = -1
TEMPERATURE = 0.02
SEED = 42
TIMEOUT = (20, 900)
NUM_CTX = 6144
NUM_PREDICT = 380

LOG_FILE_PATH = None


# ---------------- basic utils ----------------

def ensure_dirs() -> None:
    for p in [A4_LOGS, A4_RAW_INVALID, MINIMAL_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def init_log_file() -> None:
    global LOG_FILE_PATH
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE_PATH = A4_LOGS / f"minimal_analysis_v2_{ts}.log"


def log(msg: str) -> None:
    line = "[MINIMAL] " + msg
    print(line, flush=True)
    if LOG_FILE_PATH is not None:
        with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def human_seconds(sec: float) -> str:
    sec = float(sec)
    m = int(sec // 60)
    s = sec - (m * 60)
    return f"{m}분 {s:.1f}초" if m > 0 else f"{s:.1f}초"


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def open_db() -> sqlite3.Connection:
    con = sqlite3.connect(A4_DB)
    con.row_factory = sqlite3.Row
    return con


def claim_sort_key(v: Any) -> Tuple[int, str]:
    text = str(v or "")
    m = re.match(r"(\d+)", text)
    return (int(m.group(1)) if m else 10**9, text)


def normalize_claim_no(v: Any) -> str:
    m = re.match(r"^\s*(\d+)", str(v or ""))
    if not m:
        return ""
    raw = m.group(1)
    if len(raw) > 1 and raw.startswith("0"):
        return ""
    n = int(raw)
    return str(n) if 0 < n <= 80 else ""


def is_valid_claim_no(v: Any) -> bool:
    return bool(normalize_claim_no(v))


def looks_dependent_claim(raw_text: str) -> bool:
    head = normalize_ws(raw_text)[:260].lower()
    head = re.sub(r"^\s*\d+\s*[.．]\s*", "", head)
    return bool(re.search(
        r"^(根据|依照|如).{0,12}权利要求|"
        r"^the .{0,80}\b(of|according to)\s+claim\s+\d+|"
        r"^the .{0,80}\bof\s+any\s+(?:preceding|one)\s+claim|"
        r"^청구항\s*\d+.{0,20}(있어서|따른|의)|"
        r"^제\s*\d+\s*항에\s*(있어서|따른|의)",
        head,
        flags=re.I,
    ))


def unique_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for x in items:
        s = normalize_ws(x)
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def is_generic_subject(text: str) -> bool:
    s = normalize_ws(text)
    generic_values = {
        "非易失性存储器装置",
        "非易失性存储器件",
        "存储器装置",
        "存储器设备",
        "半导体装置",
        "半导体存储器设备",
        "存储器系统",
        "memory device",
        "memory system",
        "semiconductor device",
        "storage device",
        "method",
        "system",
        "device",
        "nonvolatile memory device",
        "non-volatile memory device",
        "메모리 장치",
        "메모리 시스템",
        "반도체 장치",
        "방법",
        "시스템",
        "장치",
    }
    return s in generic_values or s.lower() in generic_values


def normalize_tag(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_tags(values: List[str], max_items: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for v in values or []:
        tag = normalize_tag(v)
        if not tag or tag in seen:
            continue
        seen.add(tag)
        out.append(tag)
        if len(out) >= max_items:
            break
    return out


def source_language_for_patent(patent_id: str, country: str = "") -> str:
    prefix = (patent_id or country or "").strip().lower()[:2]
    if prefix == "cn":
        return "zh"
    if prefix == "us":
        return "en"
    if prefix == "kr":
        return "ko"
    return "unknown"


def clean_us_title_noise(title: str) -> str:
    title = re.sub(r"^(?:\(\s*(?:21|22|30|51|52|57|58|60|63|65|71|72|73|\*)\s*\)\s*)+", "", title)
    title = re.sub(r"\bUSPC\b.*?\bSee application file for complete search history\.?", " ", title, flags=re.I)
    title = re.sub(r"\(\s*56\s*\)\s*References Cited.*?\bU\.S\. PATENT DOCUMENTS\b", " ", title, flags=re.I)
    title = re.sub(r"\(\s*52\s*\)\s*U\.?\s*S\.?\s*Cl\.?", " ", title, flags=re.I)
    title = re.sub(r"\(\s*58\s*\)\s*Field of Classification Search", " ", title, flags=re.I)
    title = re.sub(r"\bCPC\b\s*(?:\.|v|w|e|s)*", " ", title, flags=re.I)
    title = re.sub(
        r"\b(?:[A-HY][O0]?[0-9I]{1,2}[A-Z]?|GO6F|HO[AI]N|HOIL|HOLL)\s*[0-9I]+/[0-9A-Z.]+(?:\s*\(\d{4}\.\d{2}\))?",
        " ",
        title,
        flags=re.I,
    )
    title = re.sub(r"\(\d{4}\.\d{2}\)", " ", title)
    title = re.sub(r"\b(?:oe|we+|ve+c?e*s+e*|ces|veces|cepabe)\b", " ", title, flags=re.I)
    title = re.sub(r"\b\d{1,2},\d{3},\d{3}\s+(?:A|B\d|Bl|Al)\b.*$", "", title, flags=re.I)
    return normalize_ws(title)


US_TITLE_CONTAMINATION_RE = re.compile(
    r"(Date of Patent|Publication Date|Pub\.? Date|Applicant[:!]?|Applicants[:!]?|Inventor[:!]?|"
    r"Inventors[:!]?|Assignee[:!]?|Appl\.?\s*No\.?|Filed:|Related U\.?S\.? Application|"
    r"Publication Classification|References Cited|See application file|U\.?\s*S\.?\s*Cl\.?|"
    r"\bCPC\b|\bUSPC\b|\bH[I1][O0]B\b|\bet al\.)",
    re.I,
)


def _looks_like_us_date_only_title(text: str) -> bool:
    return bool(re.fullmatch(
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}",
        normalize_ws(text),
        flags=re.I,
    ))


def _us_title_quality_penalty(text: str) -> int:
    text = normalize_ws(text)
    upper = text.upper()
    penalty = 0
    if not text:
        return 100
    if US_TITLE_CONTAMINATION_RE.search(text) or _looks_like_us_date_only_title(text):
        penalty += 40
    if re.search(r"\b(?:GO6F|GO06F|GUC|GIIC|GHC|GLIC|G1IC|G1I1C|HOB|HOLS|HOLT|GC|HOIL|HOLL|H01L|H10B)\b", text, flags=re.I):
        penalty += 12
    if re.search(r"\b(?:oe|we|ve|ces|veces|cepabe|teer|eots|indoe|cese+|ce+|o\.\.|woe|esses)\b", text, flags=re.I):
        penalty += 12
    if re.search(r"\(\s*(?:30|56|60|65|71|72|73|86|87)\s*\)", text):
        penalty += 20
    if re.search(r"\b[A-Z][a-z]{2,}\s+(?:et\s+al\.|AL\*|A1\*|B1\*)", text):
        penalty += 10
    if re.search(r"\b\d{1,2},\d{3},\d{3}\b|\b\d{4}/\d{6,}\b|\b\d{1,3}/\d{3,6}\b", text):
        penalty += 10
    if re.search(r"\b(?:FOR|WITH|INCLUDING|HAVING|USING|AND|OF|TO)\s*$", text, flags=re.I):
        penalty += 12
    if len(text) > 130:
        penalty += 5
    if re.search(r"\b(?:SOLID STATE|NON-VOLATILE|READ|FIRMWARE|ENCRYPTION|DECRYPTION)\b", upper):
        penalty -= 3
    return penalty


def _clean_us_title_candidate(text: str) -> str:
    text = normalize_ws(text)
    text = re.sub(r"^\(\s*(?:86|60)\s*\)\s*", "", text)
    text = re.sub(r"^.*?\bDate of Patent:\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\s+", "", text, flags=re.I)
    text = re.sub(r"^.*?\b(?:Publication Date|Pub\.? Date):\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\s+", "", text, flags=re.I)
    text = re.sub(
        r"\(\s*30\s*\)\s*Foreign Application Priority Data.*?"
        r"(?=\b[A-Z][A-Z0-9 ,/&()'\\-]{8,}\b|$)",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\bForeign Application Priority Data\b.*?"
        r"(?=\b[A-Z][A-Z0-9 ,/&()'\\-]{8,}\b|$)",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\bprovisional application No\..*?\bfiled on\s+[A-Z][a-z]{2}\.?\s*", " ", text, flags=re.I)
    text = re.sub(r"\bprovisional application No\..*?(?=\b[A-Z][A-Z0-9 ,/&()'\\-]{8,}\b|$)", " ", text, flags=re.I)
    text = re.sub(r"^(?:[A-Z]?[0-9IO]{1,4}\s+){1,8}[-–]?\s*[A-Z][A-Za-z]+\s+et\s+al\.\s*", "", text, flags=re.I)
    text = re.sub(r"^\d{1,2},\d{3},\d{3}\s*[:;]?\s*[A-Z][A-Za-z]+\s+et\s+al\.\s*", "", text, flags=re.I)
    text = re.sub(r"\b\d(?:\s+\d{3}){2}\s*[:;]?\s*[A-Z][A-Za-z]+\s+et\s+al\.\s*", "", text, flags=re.I)
    text = re.sub(r"\b\d{4}/\d{6,}\s+[A-Z0-9*]+\s+\d{1,2}/\d{4}\s+[A-Z][A-Za-z]+(?:\s+et\s+al\.?)?", " ", text, flags=re.I)
    text = re.sub(r"\b\d{1,2},\d{3},\d{3}\s+[A-Z0-9*]+\s+\d{1,2}/\d{4}\s+[A-Z][A-Za-z]+(?:\s+et\s+al\.?)?", " ", text, flags=re.I)
    text = re.sub(r"\b\d{4}/\s*A[lI1]\*?®?\s+[A-Z][A-Za-z]+(?:\s+et\s+al\.?)?", " ", text, flags=re.I)
    text = re.sub(r"\b\d{4}/\d*\s*A[lI1]\*?®?\s*(?:[A-Z][A-Za-z]+(?:\s+et\s+al\.?)?)?", " ", text, flags=re.I)
    text = re.sub(r"\bA[lI1]\*?®?\b", " ", text, flags=re.I)
    text = re.sub(r"\b\d{1,2}\s+\d{3}\s+\d{3}\s+[A-Z][A-Za-z]+\b", " ", text)
    text = re.sub(r"\b[A-Z0-9]{2,}\s+\d{1,4}\s+B\d\*?\s+(?:\d{1,2}/\d{4}\s+)?[A-Z][A-Za-z]+(?:\s+\w+\.?)?", " ", text, flags=re.I)
    text = re.sub(r"\bB\d\*?\s+\d{1,2}/\d{4}\s+[A-Z][A-Za-z]+(?:\s+\w+)?", " ", text, flags=re.I)
    text = re.sub(r"\bB\d\*?\s+[A-Z][A-Za-z]+(?:\s+\w+)?", " ", text, flags=re.I)
    text = re.sub(r"B\d\*\s+[A-Z][A-Za-z]+\b", " ", text, flags=re.I)
    text = re.sub(r"B\d\S*\s+[A-Z][A-Za-z]+\b", " ", text, flags=re.I)
    text = re.sub(r"\bB\d\s+[A-Z][A-Za-z]+\b", " ", text, flags=re.I)
    text = re.sub(r"\bIN\s+B\d\S*\s+Kim\s+SPACE\b", "IN SPACE", text, flags=re.I)
    text = re.sub(r"\b\d{1,2},\d{3},\d{3}\s+[A-Z0-9*]+\b", " ", text, flags=re.I)
    text = re.sub(r"^(?:\(\s*(?:\d+|\*)\s*\)\s*)+", "", text)
    text = re.sub(r"\(\s*(?:15|30|56|60|65|71|72|73|86)\s*\)", " ", text)
    text = re.sub(r"^(?:Filed:\s*)+", "", text, flags=re.I)
    text = re.sub(r"^(?:\d{1,2},\d{3},\d{3}\s+B\d\s*)+", "", text, flags=re.I)
    text = re.sub(r"^(?:US\s*)?\d{4}/\d+\s*A\d\s*", "", text, flags=re.I)
    text = re.sub(r"\bH[I1][O0]B\s*\d{1,4}(?:/\d{1,4})?\s*\(\d{4}\.\d{2}\)", "", text, flags=re.I)
    text = re.sub(r"\bH[I1][O0]B\s*\d{1,4}(?:/\d{1,4})?\b", "", text, flags=re.I)
    text = re.sub(r"\bH[I1][O0]B\b", "", text, flags=re.I)
    text = re.sub(r"\b(?:GUC|GIIC|GIUIC|GHUIC|GHC|GLIC|G1IC|G1I1C|G11C|G06F|GO6F|GO6Fr|GOG6F|GOGF|GOGE|GO06F|GC|HOIL|HOLL|HO1L|H01L|H10B|HOB|HOLS|HOLT|TA28A05|TLL)\s*\d{1,4}/[0-9A-Z.]+", "", text, flags=re.I)
    text = re.sub(r"\b(?:GUC|GIIC|GIUIC|GHUIC|GHC|GLIC|G1IC|G1I1C|G11C|G06F|GO6F|GO6Fr|GOG6F|GOGF|GOGE|GO06F|GC|HOIL|HOLL|HO1L|H01L|H10B|HOB|HOLS|HOLT|TA28A05|TLL)\b", "", text, flags=re.I)
    text = re.sub(r"\b\d{1,3}/\d{3,6}\b", "", text)
    text = re.sub(r"\bA\s+\d{1,3}/\d+\b", " ", text, flags=re.I)
    text = re.sub(r"(?<![A-Z0-9])/\d{2,4}\b", " ", text, flags=re.I)
    text = re.sub(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2}\s+\d{4}\s+\([A-Z]{2}\)\s+\d{2}-\d{4}-\d{6,}", " ", text, flags=re.I)
    text = re.sub(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\([A-Z]{2}\)\s+\d{2}-\d{4}-\d{6,}", " ", text, flags=re.I)
    text = re.sub(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\([A-Z]{2}\)\s*", " ", text, flags=re.I)
    text = re.sub(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\([A-Z]{2}\)\b", " ", text, flags=re.I)
    text = re.sub(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s*", " ", text, flags=re.I)
    text = re.sub(r"\([A-Z]{2}\)", " ", text)
    text = re.sub(r"\b\d{2}-\d{4}-\d{6,}\b", " ", text)
    text = re.sub(r"\b\d{1,2}\s+\d{4}\b", " ", text)
    text = re.sub(r"\b20\d\s+1/\b", " ", text)
    text = re.sub(r"\.{3,}\d*\.?", " ", text)
    text = re.sub(r"\bo\.\.", " ", text, flags=re.I)
    text = re.sub(r"\(\s*Continued\s*\)", " ", text, flags=re.I)
    text = re.sub(r"\b(?:oe|we|wees|oer|ces|veces|cepabe|ce+|c{2,}e+|oece+|ooeie+c+e+|vce+c+ss+e+s+|e{4,}|o\.\.|o\.oo|occ|ice|frorn|vaive|ala|ou|ne|escessceeeens|ne\s+Eo)\b\.?", " ", text, flags=re.I)
    text = re.sub(r"«0\.0\.0\.", " ", text)
    text = re.sub(r"©\s*[A-Za-z]+", " ", text)
    text = re.sub(r"\b\d{6,}\b", " ", text)
    text = re.sub(r"\bl\*\s*\.\.", " ", text, flags=re.I)
    text = re.sub(r"=\s*[A-Z][a-z]+", " ", text)
    text = re.sub(r"\b(?:Tanak|anaka)\b\.?", " ", text, flags=re.I)
    text = re.sub(r"\s+:\s*[‘'`.]?\s*\d+\)?", " ", text)
    text = re.sub(r"\b(?:TEER|eots|indoe)\b.*$", "", text, flags=re.I)
    text = re.sub(r"\(\s*\d+\s*\)\s*U\.?\s*S\.?\s*CL\.?\s*", "", text, flags=re.I)
    text = re.sub(r"\bU\.?\s*S\.?\s*CL\.?\b", "", text, flags=re.I)
    text = re.sub(r"\bCPC\b.*$", "", text, flags=re.I)
    text = re.sub(r"\b(?:TECHNICAL FIELD|TECHNICAL|FIELD|BACKGROUND ART|BACKGROUND|CROSS-REFERENCE TO RELATED APPLICATION|BRIEF DESCRIPTION OF DRAWINGS|DESCRIPTION OF RELATED ART)\b.*$", "", text, flags=re.I)
    text = clean_us_title_noise(text)
    plasma_idx = text.upper().find("PLASMA ENHANCED ATOMIC LAYER DEPOSITION")
    if plasma_idx > 0:
        text = text[plasma_idx:]
    text = re.sub(r"\b(?:KIOXIA|KIOXTA|SAMSUNG|SK\s+hynix|RESONAC|MICRON|WESTERN DIGITAL)\s+(?:CORPORATION|Inc\.?|Co\.?|Ltd\.?).*$", "", text, flags=re.I)
    text = re.sub(r"\bCO\.?\s+LTD\.?.*$", "", text, flags=re.I)
    text = re.sub(r"\b(?:Applicant|Applicants|Inventor|Inventors|Assignee|Appl\.?\s*No\.?|Application No\.?|Filed|Related U\.?S\.? Application|Publication Classification)\b.*$", "", text, flags=re.I)
    text = re.sub(r"\b(?:FOR|WITH|INCLUDING|HAVING|USING|AND|OF|TO)\s*$", "", text, flags=re.I)
    text = re.sub(r"\s+\.\s+", " ", text)
    text = normalize_ws(re.sub(r"\s*[;,]\s*", " ", text))
    return text.strip(" -_:;,. ")[:180].strip()


def _usable_us_title_candidate(text: str) -> bool:
    text = normalize_ws(text)
    if len(text) < 6 or len(text) > 180:
        return False
    bad = (
        "PATENT DOCUMENT", "FOREIGN PATENT", "OTHER PUBLICATION", "REFERENCES CITED",
        "CLASSIFICATION", "APPLICATION DATA", "DRAWING SHEET", "CLAIMS",
        "INPUT/OUTPUT CIRCUIT", "MEMORY DEVICE U.S. PATENT", "TECHNICAL FIELD", "ABSTRACT",
        "USPC", "SEE APPLICATION FILE", "PRIMARY EXAMINER",
    )
    if any(x in text.upper() for x in bad):
        return False
    if US_TITLE_CONTAMINATION_RE.search(text) or _looks_like_us_date_only_title(text):
        return False
    return bool(re.search(
        r"(MEMORY|SEMICONDUCTOR|DEVICE|SYSTEM|METHOD|STORAGE|FLASH|NAND|ADHESIVE|FILM|CIRCUIT|ARRAY|"
        r"ACCELERATOR|ACCELERATORS|SERVICE|SERVICES|RESOURCE|MANAGEMENT|ENCRYPTION|DECRYPTION|"
        r"HAZARD|MITIGATION|AVOIDANCE|RADIO|PLASMA|DEPOSITION|VERIFY|QLC|TLC|FAILURE|RECOVERY|"
        r"CONTROLLER|OPERATION|TECHNOLOGY|DESCRIPTOR|TRIGGERING|"
        r"SSD|DRIVE|BUFFER|FIRMWARE|BMC|BIOS|MICROELECTRONIC|CONDUCTIVE|RAIL|FUNCTION|BLOCK|UNCLONABLE)",
        text,
        flags=re.I,
    ))


def extract_us_title_from_front_matter(raw: str) -> str:
    raw = normalize_ws(raw)
    if not raw:
        return ""
    head = re.split(
        r"\b(?:Applicant|Applicants|Inventor|Inventors|Assignee|Appl\.?\s*No\.?|Filed|Related U\.?S\.? Application|Publication Classification)\b[:!]?|"
        r"\(\s*57\s*\)\s*ABSTRACT",
        raw,
        maxsplit=1,
        flags=re.I,
    )[0]
    head = re.sub(r"^(?:\(\s*(?:\d+|\*)\s*\)\s*)+", " ", head)
    head = re.sub(r"\(\s*(?:21|22|30|51|52|57|58|60|63|65|71|72|73|\*)\s*\)", " ", head)
    head = re.sub(
        r"\b\d{1,2},\d{3},\d{3}\s+[A-Z0-9*]+(?:\s+\d{1,2}/\d{4})?\s+[A-Z][A-Za-z]+\.{0,3}"
        r"(?:\s+[A-Z0-9]+(?:\s+\d{1,4}/\d{1,4})?)?",
        " ",
        head,
        flags=re.I,
    )
    head = re.sub(r"\b\d{4}/\d{6,}\s+[A-Z0-9*]+.*$", " ", head, flags=re.I)
    head = re.sub(r"\b(?:CPC|USPC)\b\s*(?:\.|v|w|e|s|o|c|a|p|b)*", " ", head, flags=re.I)
    head = re.sub(
        r"\b(?:GUC|GIIC|GHC|G06F|GO6F|GO06F|HOIL|HOLL|HO1L|H01L|H10B)\s*\d{1,4}/[0-9A-Z.]+"
        r"(?:\s*\(\d{4}\.\d{2}\))?",
        " ",
        head,
        flags=re.I,
    )
    head = re.sub(r"\(\d{4}\.\d{2}\)", " ", head)
    head = re.sub(r"\b(?:oe|we|ces|veces|cepabe)\b\.?", " ", head, flags=re.I)
    head = clean_us_title_noise(head)

    upper = head.upper()
    anchors = [
        "SOLID STATE DRIVE",
        "READ VOLTAGE",
        "DEVICE OF ENCRYPTION",
        "STORAGE BIASING",
        "CACHE TO RECEIVE",
        "HINTING MECHANISM",
        "RESOURCE MANAGEMENT",
    ]
    for anchor in anchors:
        idx = upper.find(anchor)
        if idx > 0:
            prefix = head[:idx]
            if re.search(r"[a-z]", prefix):
                head = head[idx:]
                break

    candidate = _clean_us_title_candidate(head)
    if _usable_us_title_candidate(candidate):
        return candidate
    return ""


def extract_us_title_fallback(raw: str) -> str:
    raw = normalize_ws(raw)
    if not raw:
        return ""
    patterns = [
        r"\bentitled:\s*[“\"]([^”\"]{8,220})[”\"]",
        r"\([“\"]([A-Z][A-Z0-9 ,/\-&()]{8,220}?)[”\"]\)",
        r"^(?:\(\s*(?:\d+|\*)\s*\)\s*)+(?:Filed:\s*)?(?:\d{1,2},\d{3},\d{3}\s+B\d\s*){0,5}([A-Z][A-Z0-9 ,/\-&()]{8,220}?)\s+(?:Applicant:|Applicants:|Inventors?:|Assignee:|Appl\.?\s*No\.?:|Related U\.S\. Application|Publication Classification)",
        r"\bUS\s+\d{1,2},\d{3},\d{3}\s+B\d\s+\d+\s+([A-Z][A-Z0-9 ,/\-&()]{8,220}?)\s+(?:BACKGROUND|FIELD|CROSS-REFERENCE)",
        r"\bUS\s+\d{4}/\d+\s*A\d\s+([A-Z][A-Z0-9 ,/\-&()]{8,220}?)\s+(?:TECHNICAL FIELD|FIELD|BACKGROUND ART|BACKGROUND|CROSS-REFERENCE TO RELATED APPLICATION)",
        r"\bFiled:\s*([A-Z][A-Z0-9 ,/\-&()]{6,140}?)(?:\s+(?:Applicant:|KIOXIA|KIOXTA|SAMSUNG|SK\s+hynix|RESONAC|Related U\.S\. Application|Publication Classification))",
        r"\bDate of Patent:\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}\s+([A-Z][A-Z0-9 ,/\-&()]{8,220}?)(?:\s+(?:Applicant:|Applicants:|Inventors?:|Assignee:|Appl\.?\s*No\.?:|Related U\.S\. Application|Publication Classification|TECHNICAL FIELD|BACKGROUND))",
    ]
    for pat in patterns:
        m = re.search(pat, raw, flags=re.I)
        if m:
            candidate = _clean_us_title_candidate(m.group(1))
            if _usable_us_title_candidate(candidate):
                return candidate

    m = re.search(
        r"\(\s*57\s*\)\s*ABSTRACT\s+(?:There (?:are|is) provided|Provided (?:are|is)|Disclosed (?:are|is))\s+(?:a|an|the)?\s*([^.;]{8,150})",
        raw,
        flags=re.I,
    )
    if m:
        candidate = _clean_us_title_candidate(m.group(1))
        candidate = re.sub(r"\b(?:and|or)\s+(?:a|an|the)\b", "and", candidate, flags=re.I)
        if _usable_us_title_candidate(candidate):
            return candidate

    caps = re.findall(r"(?:[A-Z][A-Z0-9/,&()\-]+(?:\s+|$)){2,15}", raw)
    for item in caps:
        candidate = _clean_us_title_candidate(item)
        if _usable_us_title_candidate(candidate):
            return candidate
    return ""


def clean_title_source(meta: Dict[str, Any]) -> str:
    raw_title = normalize_ws(meta.get("title_raw", ""))
    title = raw_title
    country = str(meta.get("country", "") or "").upper()
    markers = [
        r"\(57\)\s*摘要",
        r"\(57\)\s*Abstract",
        r"\(57\)\s*초록",
        r"\(57\)\s*요\s*약",
        r"\bABSTRACT\b",
        r"\bAbstract\b",
        r"摘要",
        r"요\s*약",
        r"权利要求书",
        r"청구범위",
        r"청구항",
    ]
    if country == "US":
        title = clean_us_title_noise(title)
        markers.extend([
            r"\(\s*71\s*\)",
            r"\(\s*72\s*\)",
            r"\(\s*73\s*\)",
            r"\(\s*21\s*\)",
            r"\(\s*22\s*\)",
            r"\(\s*51\s*\)",
            r"\(\s*52\s*\)",
            r"\(\s*56\s*\)",
            r"\(\s*58\s*\)",
            r"\bApplicant:",
            r"\bApplicants:",
            r"\bInventor:",
            r"\bInventors:",
            r"\bAssignee:",
            r"\bAppl\.?\s*No\.?:",
            r"\bFiled:",
            r"\bReferences Cited\b",
            r"\bSee application file\b",
        ])
    cut_at = len(title)
    for pat in markers:
        m = re.search(pat, title, flags=re.I)
        if m:
            cut_at = min(cut_at, m.start())
    title = normalize_ws(title[:cut_at])
    if country == "US":
        title = _clean_us_title_candidate(title)
        front_title = extract_us_title_from_front_matter(raw_title)
        split_abstract_title = ""
        m = re.search(
            r"^([A-Z][A-Z0-9 ,/()&-]{6,120}?)\s*\(\s*57\s*\)\s*ABSTRACT\s+"
            r"([A-Z][A-Z0-9 ,/()&-]{6,120}?)(?:\s*[.;]|\s+Solid\s+State|\s+\(\s*71\s*\))",
            raw_title,
            flags=re.I,
        )
        if m:
            split_abstract_title = _clean_us_title_candidate(f"{m.group(1)} {m.group(2)}")
        fallback = extract_us_title_fallback(raw_title)
        candidates = [
            (front_title, 0),
            (split_abstract_title, 1),
            (title, 2),
            (fallback, 3),
        ]
        ranked = [
            (_us_title_quality_penalty(candidate), priority, candidate)
            for candidate, priority in candidates
            if candidate and _usable_us_title_candidate(candidate)
        ]
        if ranked:
            title = sorted(ranked, key=lambda x: (x[0], x[1], len(x[2])))[0][2]
        elif fallback and ("USPC" in raw_title or not title or US_TITLE_CONTAMINATION_RE.search(title) or _looks_like_us_date_only_title(title)):
            title = fallback
        elif not title or not _usable_us_title_candidate(title):
            title = fallback or title
    if country == "CN":
        title = re.sub(r"\s+", "", title)
    return title[:180]


def extract_abstract_text(meta: Dict[str, Any]) -> str:
    raw = normalize_ws(meta.get("title_raw", ""))
    if not raw:
        return ""
    start_patterns = [
        r"\(57\)\s*摘要",
        r"\(57\)\s*Abstract",
        r"\(57\)\s*초록",
        r"\bABSTRACT\b",
        r"\bAbstract\b",
        r"摘要",
    ]
    start = -1
    for pat in start_patterns:
        m = re.search(pat, raw)
        if m:
            start = m.end()
            break
    if start < 0:
        return ""
    tail = raw[start:]
    end_patterns = [
        r"权利要求书",
        r"权\s*利\s*要\s*求\s*书",
        r"\bCLAIMS\b",
        r"\bClaims\b",
        r"청구범위",
        r"청구항",
        r"说明书\d*页",
        r"說明書\d*頁",
    ]
    end = len(tail)
    for pat in end_patterns:
        m = re.search(pat, tail)
        if m:
            end = min(end, m.start())
    return normalize_ws(tail[:end])[:1600]


def _claim_text_end(raw: str, start: int, patterns: List[str]) -> int:
    end = len(raw)
    for pat in patterns:
        m = re.search(pat, raw[start:], flags=re.I)
        if m:
            end = min(end, start + m.start())
    return end


def _make_recovered_claim(claim_no: str, text: str, source: str) -> Dict[str, Any]:
    no = normalize_claim_no(claim_no)
    return {
        "claim_no": no,
        "parent_claim_no": None,
        "claim_type": "independent",
        "raw_text": f"{no}. {normalize_ws(text)[:1800]}",
        "norm_text": "",
        "page_start": None,
        "page_end": None,
        "_recovered_from": source,
    }


def recover_front_claims_from_meta(meta: Dict[str, Any], existing_claim_nos: List[str]) -> List[Dict[str, Any]]:
    raw = normalize_ws(meta.get("title_raw", ""))
    country = str(meta.get("country", "") or "").upper()
    source_language = source_language_for_patent(str(meta.get("patent_id", "")), country)
    existing = {normalize_claim_no(x) for x in existing_claim_nos}
    recovered: List[Dict[str, Any]] = []
    if not raw:
        return recovered

    if source_language == "ko":
        claim_start = re.search(r"(청구범위|청구항\s*1)", raw)
        if claim_start:
            end = _claim_text_end(raw, claim_start.start(), [r"발명의\s*설명", r"기\s*술\s*분\s*야", r"배\s*경\s*기\s*술"])
            section = raw[claim_start.start():end]
            for m in re.finditer(r"청구항\s*(\d+)\s*(.+?)(?=청구항\s*\d+|$)", section):
                no = normalize_claim_no(m.group(1))
                if no and no not in existing:
                    recovered.append(_make_recovered_claim(no, m.group(2), "patents.title_raw_claims"))
            return recovered[:6]

    if source_language == "zh":
        claim_start = re.search(r"(权利要求书|权\s*利\s*要\s*求\s*书|(?:^|\s)1[.．]\s*)", raw)
        if claim_start:
            end = _claim_text_end(raw, claim_start.start(), [r"说\s*明\s*书", r"说明书", r"发明内容", r"技术领域"])
            section = raw[claim_start.start():end]
            for m in re.finditer(r"(?:^|\s)(\d{1,2})[.．]\s*(.+?)(?=\s+\d{1,2}[.．]\s*|$)", section):
                no = normalize_claim_no(m.group(1))
                if no and no not in existing:
                    recovered.append(_make_recovered_claim(no, m.group(2), "patents.title_raw_claims"))
            if recovered:
                return recovered[:6]

    if source_language == "en":
        claim_start = re.search(r"(what is claimed is:?|claims?:?\s+1[.])", raw, flags=re.I)
        if claim_start:
            end = _claim_text_end(raw, claim_start.start(), [r"\bdescription\b", r"\bbackground\b", r"\bbrief description\b"])
            section = raw[claim_start.start():end]
            for m in re.finditer(r"(?:^|\s)(\d{1,2})[.]\s*(.+?)(?=\s+\d{1,2}[.]\s*|$)", section, flags=re.I):
                no = normalize_claim_no(m.group(1))
                if no and no not in existing:
                    recovered.append(_make_recovered_claim(no, m.group(2), "patents.title_raw_claims"))
            if recovered:
                return recovered[:6]

        summary = ""
        m = re.search(r"\bSUMMARY\b(.+?)(?:\bBRIEF DESCRIPTION\b|\bDETAILED DESCRIPTION\b|$)", raw, flags=re.I)
        if m:
            summary = normalize_ws(m.group(1))
        if not summary:
            summary = extract_abstract_text(meta)
        summary_bits = re.split(r"(?<=\.)\s+(?=According to|In an? example|The present disclosure|A[n]? )", summary)
        for i, bit in enumerate([b for b in summary_bits if len(b) >= 80][:3], start=1):
            no = str(i)
            if no not in existing:
                recovered.append(_make_recovered_claim(no, bit, "patents.title_raw_summary"))
        if recovered:
            return recovered[:3]

    if "1" not in existing:
        fallback = extract_abstract_text(meta) or clean_title_source(meta)
        if fallback:
            recovered.append(_make_recovered_claim("1", fallback, "patents.title_raw_fallback"))
    return recovered


def technical_term_roles_for_language(source_language: str, texts: List[str]) -> Tuple[List[str], List[str]]:
    joined = "\n".join(texts)
    core_candidates = {
        "zh": [
            "平面专用焊盘组",
            "多个存储器平面",
            "数据路径",
            "页缓冲器电路",
            "数据输入输出电路",
            "不接收平面地址",
        ],
        "en": [
            "plane dedicated pad group",
            "memory plane",
            "data path",
            "page buffer circuit",
            "data input/output circuit",
            "no plane address",
        ],
        "ko": [
            "평면 전용 패드 그룹",
            "메모리 평면",
            "데이터 경로",
            "페이지 버퍼 회로",
            "데이터 입출력 회로",
            "평면 주소",
        ],
    }
    context_candidates = {
        "zh": ["数据多路复用", "信号路由", "复用器"],
        "en": ["data multiplexing", "signal routing", "multiplexer"],
        "ko": ["데이터 다중화", "신호 라우팅", "멀티플렉서"],
    }

    core_terms = [t for t in core_candidates.get(source_language, []) if t.lower() in joined.lower()]
    context_terms = [t for t in context_candidates.get(source_language, []) if t.lower() in joined.lower()]

    if source_language == "zh" and re.search(r"独立地?接收.{0,20}命令.{0,10}地址|命令.{0,10}地址.{0,20}独立", joined):
        core_terms.append("独立接收命令和地址")
    elif source_language == "en" and re.search(r"independent(?:ly)?.{0,30}(?:receive|receiving).{0,20}command.{0,10}address", joined, flags=re.I):
        core_terms.append("independently receive command and address")
    elif source_language == "ko" and re.search(r"독립.{0,20}(명령|커맨드).{0,10}주소", joined):
        core_terms.append("독립적으로 명령 및 주소 수신")

    return unique_keep_order(core_terms)[:12], unique_keep_order(context_terms)[:8]


# ---------------- DB reads ----------------

def get_job_counts(con: sqlite3.Connection) -> Dict[str, int]:
    cur = con.cursor()
    rows = cur.execute("SELECT status, COUNT(*) AS c FROM jobs GROUP BY status").fetchall()
    return {r["status"]: int(r["c"]) for r in rows}


def fetch_next_patent_id(con: sqlite3.Connection, explicit_patent_id: Optional[str] = None) -> Optional[str]:
    cur = con.cursor()
    if explicit_patent_id:
        row = cur.execute(
            "SELECT patent_id FROM patents WHERE patent_id=?",
            (explicit_patent_id,),
        ).fetchone()
        return row["patent_id"] if row else None

    row = cur.execute(
        """
        SELECT patent_id
        FROM jobs
        WHERE status IN ('evidence_done', 'brief_done', 'profile_done', 'analysis_failed')
        ORDER BY updated_at ASC, patent_id ASC
        LIMIT 1
        """
    ).fetchone()
    return row["patent_id"] if row else None


def fetch_patent_ids(con: sqlite3.Connection, explicit_patent_id: Optional[str], limit: int) -> List[str]:
    cur = con.cursor()
    if explicit_patent_id:
        row = cur.execute(
            "SELECT patent_id FROM patents WHERE patent_id=?",
            (explicit_patent_id,),
        ).fetchone()
        return [row["patent_id"]] if row else []

    rows = cur.execute(
        """
        SELECT j.patent_id
        FROM jobs j
        JOIN patents p ON p.patent_id = j.patent_id
        WHERE j.status IN ('evidence_done', 'brief_done', 'profile_done', 'analysis_failed')
        ORDER BY j.updated_at ASC, j.patent_id ASC
        """
    ).fetchall()
    return [r["patent_id"] for r in rows]


def minimal_output_exists(patent_id: str) -> bool:
    return (MINIMAL_DIR / f"{safe_name(patent_id)}.minimal.json").exists()


def get_patent_meta(con: sqlite3.Connection, patent_id: str) -> Dict[str, Any]:
    cur = con.cursor()
    row = cur.execute(
        """
        SELECT patent_id, country, title_raw, assignee_raw, application_no, publication_no, pdf_path, page_count
        FROM patents
        WHERE patent_id=?
        """,
        (patent_id,),
    ).fetchone()
    if not row:
        raise RuntimeError(f"patents row not found: {patent_id}")
    return dict(row)


def get_claims(con: sqlite3.Connection, patent_id: str) -> List[Dict[str, Any]]:
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT claim_no, parent_claim_no, claim_type, raw_text, norm_text, page_start, page_end
        FROM claims
        WHERE patent_id=?
        ORDER BY claim_no
        """,
        (patent_id,),
    ).fetchall()
    out = [dict(r) for r in rows]
    out.sort(key=lambda x: claim_sort_key(x.get("claim_no")))
    return out


def get_claim_ref_counts(con: sqlite3.Connection, patent_id: str) -> List[Dict[str, Any]]:
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT claim_no, ref_no_raw, COUNT(*) AS freq
        FROM claim_ref_map
        WHERE patent_id=?
        GROUP BY claim_no, ref_no_raw
        ORDER BY claim_no ASC, freq DESC
        """
        ,
        (patent_id,),
    ).fetchall()
    out = [dict(r) for r in rows]
    out.sort(key=lambda x: (claim_sort_key(x.get("claim_no")), -int(x.get("freq", 0))))
    return out


def get_figures(con: sqlite3.Connection, patent_id: str, limit: int = 12) -> List[Dict[str, Any]]:
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT figure_no, caption_raw, caption_norm, page_no
        FROM figure_captions
        WHERE patent_id=?
        ORDER BY page_no ASC, figure_no ASC
        LIMIT ?
        """,
        (patent_id, limit),
    ).fetchall()
    return [dict(r) for r in rows]


def get_description_snippets(con: sqlite3.Connection, patent_id: str, claims: List[Dict[str, Any]], limit: int = 12) -> List[Dict[str, Any]]:
    cur = con.cursor()
    claim_end_pages = [c["page_end"] for c in claims if c.get("page_end")]
    min_claim_end = max(claim_end_pages) if claim_end_pages else 0

    rows = cur.execute(
        """
        SELECT page_no, raw_text
        FROM text_spans
        WHERE patent_id=?
          AND page_no > ?
          AND raw_text IS NOT NULL
          AND LENGTH(TRIM(raw_text)) BETWEEN 40 AND 800
        ORDER BY page_no ASC, block_no ASC, line_no ASC, id ASC
        LIMIT ?
        """,
        (patent_id, min_claim_end, limit * 4),
    ).fetchall()

    snippets = []
    seen = set()
    for r in rows:
        txt = normalize_ws(r["raw_text"])
        if not txt:
            continue
        key = txt[:180]
        if key in seen:
            continue
        seen.add(key)
        snippets.append({"page_no": r["page_no"], "text": txt})
        if len(snippets) >= limit:
            break
    return snippets


def get_front_matter_text(con: sqlite3.Connection, patent_id: str, limit: int = 2) -> str:
    rows = con.execute(
        """
        SELECT page_no, raw_text
        FROM text_spans
        WHERE patent_id=?
          AND page_no <= ?
          AND raw_text IS NOT NULL
          AND LENGTH(TRIM(raw_text)) > 0
        ORDER BY page_no ASC
        """,
        (patent_id, limit),
    ).fetchall()
    return normalize_ws(" ".join([r["raw_text"] for r in rows]))[:5000]


# ---------------- heuristics ----------------

def choose_primary_claim_type(independent_claims: List[Dict[str, Any]]) -> str:
    types = [classify_claim_type(c.get("raw_text", "")) for c in independent_claims]
    for preferred in ["device", "system", "method", "process"]:
        if preferred in types:
            return preferred
    return "unknown"


def infer_claim_type_from_text(text: str) -> str:
    head = normalize_ws(text).lower()
    if not head:
        return "unknown"
    if re.search(r"(方法|制备工艺|制作方法|制造方法|操作方法|控制方法|工艺|构建|应用|"
                 r"\bmethod\b|\bprocess\b|\balgorithm\b|\bestimation\b|\ballocation\b|"
                 r"\bprediction\b|\bavoidance\b|\bparticipation\b|방법|공정|단계)", head, flags=re.I):
        return "method"
    if re.search(r"(系统|子系统|基站|\bsystem\b|base station|시스템)", head, flags=re.I):
        return "system"
    if re.search(r"(装置|设备|元件|器件|结构|阵列|电路|芯片|控制器|存储器|闪存|"
                 r"晶体管|组合物|传感器|对讲机|产房|菌株|存储|密码卡|分析仪|计算机|调制器|通信模块|平板|薄膜|"
                 r"\bdevice\b|\bapparatus\b|\bcircuit\b|\bcontroller\b|\bmemory\b|"
                 r"\bcell\b|\barray\b|\bcomposition\b|\bsensor\b|\btransceiver\b|"
                 r"\bdrive\b|\bssd\b|\bnand\b|\binterface\b|\bframe\b|\bmotor\b|\btransistor\b|"
                 r"장치|회로|메모리|제어기|조립체|어셈블리|컨테이너|용기|검출기)", head, flags=re.I):
        return "device"
    return "unknown"


def classify_claim_type(raw_text: str) -> str:
    head = normalize_ws(raw_text)[:900].lower()
    if re.search(r"(一种|an?|the)?\s*(存储器系统|系统|基站|system|base station|시스템)", head, flags=re.I):
        return "system"
    if re.search(r"(一种|an?|the)?\s*(方法|制造方法|工艺|步骤|method|process|방법|공정|단계)", head, flags=re.I):
        return "method"
    if re.search(r"(装置|器件|结构|部件|固态硬盘|分析仪|计算机|调制器|通信模块|平板|薄膜|"
                 r"device|apparatus|structure|component|transistor|detector|assembly|container|"
                 r"장치|구조|메모리 장치|조립체|어셈블리|컨테이너|용기|검출기|存储器装置|半导体装置)", head, flags=re.I):
        return "device"
    return infer_claim_type_from_text(head)


def collect_secondary_claim_types(independent_claims: List[Dict[str, Any]]) -> List[str]:
    out = []
    seen = set()
    for c in independent_claims:
        label = classify_claim_type(c.get("raw_text", ""))
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def infer_candidate_problem_labels(snippets: List[Dict[str, Any]], extra_texts: List[str]) -> List[str]:
    # Problem labels must be tied to an explicit problem/limitation/improvement cue.
    # Claims contain many neutral words like "address" and "delay", so keep this
    # deliberately stricter than solution/effect label extraction.
    text = " ".join(extra_texts + [s["text"] for s in snippets[:10]])
    labels = []
    if re.search(r"(功耗|电力|전력|power).{0,24}(高|大|增加|消耗|浪费|降低|减少|问题|限制|limit|reduce|reduction|감소|저감)|"
                 r"(高|大|增加|消耗|浪费|降低|减少|问题|限制|limit|reduce|reduction|감소|저감).{0,24}(功耗|电力|전력|power)",
                 text, flags=re.I):
        labels.append("power_consumption")
    if re.search(r"(延迟|时延|지연|latency|delay).{0,24}(高|长|增加|降低|减少|问题|限制|limit|reduce|reduction|감소|저감)|"
                 r"(高|长|增加|降低|减少|问题|限制|limit|reduce|reduction|감소|저감).{0,24}(延迟|时延|지연|latency|delay)",
                 text, flags=re.I):
        labels.append("data_transfer_latency")
    if re.search(r"(带宽|bandwidth|대역폭).{0,24}(不足|有限|瓶颈|增加|提高|改善|限制|limit|bottleneck|increase|improve|향상|증가)|"
                 r"(不足|有限|瓶颈|增加|提高|改善|限制|limit|bottleneck|increase|improve|향상|증가).{0,24}(带宽|bandwidth|대역폭)",
                 text, flags=re.I):
        labels.append("bandwidth_limit")
    if re.search(r"(多路复用|复用器|mux|multiplex).{0,24}(复杂|延迟|功耗|降低|减少|消除|去除|问题|overhead|complex|eliminate|remove|감소|제거)|"
                 r"(复杂|延迟|功耗|降低|减少|消除|去除|问题|overhead|complex|eliminate|remove|감소|제거).{0,24}(多路复用|复用器|mux|multiplex)",
                 text, flags=re.I):
        labels.append("mux_complexity")
    if re.search(r"(平面地址|地址译码|地址解码|plane address|address decoding).{0,24}(复杂|无需|不需要|不接收|省略|问题|complex|without|no |not receive|불필요|미수신)|"
                 r"(复杂|无需|不需要|不接收|省略|问题|complex|without|no |not receive|불필요|미수신).{0,24}(平面地址|地址译码|地址解码|plane address|address decoding)",
                 text, flags=re.I):
        labels.append("address_handling_complexity")
    if re.search(r"(可靠性|稳定性|寿命|耐久|reliability|reliable|stability|lifetime|endurance|수명|신뢰성).{0,24}(问题|降低|不足|低|差|劣化|개선|향상|improv|increas|enhanc|degrad|issue|risk)|"
                 r"(问题|降低|不足|低|差|劣化|개선|향상|improv|increas|enhanc|degrad|issue|risk).{0,24}(可靠性|稳定性|寿命|耐久|reliability|reliable|stability|lifetime|endurance|수명|신뢰성)",
                 text, flags=re.I):
        labels.append("reliability_issue")
    if re.search(r"(性能|效率|速度|吞吐|读写|存取|performance|efficiency|speed|throughput|read|write|access|성능|효율).{0,24}(低|慢|不足|降低|瓶颈|提高|提升|改善|加快|优化|improv|increas|enhanc|accelerat|optimi|bottleneck)|"
                 r"(低|慢|不足|降低|瓶颈|提高|提升|改善|加快|优化|improv|increas|enhanc|accelerat|optimi|bottleneck).{0,24}(性能|效率|速度|吞吐|读写|存取|performance|efficiency|speed|throughput|read|write|access|성능|효율)",
                 text, flags=re.I):
        labels.append("performance_limit")
    if re.search(r"(错误|误差|失效|故障|短路|干扰|纠错|error|failure|fault|short circuit|interference|disturb|오류|고장|단락|간섭).{0,24}(减少|降低|防止|避免|校正|抑制|改进|reduce|prevent|avoid|correct|suppress|mitigate|방지)|"
                 r"(减少|降低|防止|避免|校正|抑制|改进|reduce|prevent|avoid|correct|suppress|mitigate|방지).{0,24}(错误|误差|失效|故障|短路|干扰|纠错|error|failure|fault|short circuit|interference|disturb|오류|고장|단락|간섭)",
                 text, flags=re.I):
        labels.append("error_risk")
    if re.search(r"丢失数据|数据丢失|数据恢复|恢复.{0,12}数据|data loss|lost data|data recover|데이터 손실|데이터 복구", text, flags=re.I):
        labels.append("data_loss_risk")
    if re.search(r"(成本|面积|尺寸|空间|资源|复杂度|cost|area|size|space|resource|complexity|비용|면적|크기|공간|복잡도).{0,24}(降低|减少|节省|缩小|简化|优化|reduce|save|shrink|simplify|optimi|감소|절감|축소|단순화)|"
                 r"(降低|减少|节省|缩小|简化|优化|reduce|save|shrink|simplify|optimi|감소|절감|축소|단순화).{0,24}(成本|面积|尺寸|空间|资源|复杂度|cost|area|size|space|resource|complexity|비용|면적|크기|공간|복잡도)",
                 text, flags=re.I):
        labels.append("resource_overhead")
    return unique_keep_order(labels)


GENERIC_SOLUTION_LABELS = {
    "claimed_method_process",
    "claimed_system_architecture",
    "claimed_device_structure",
    "claimed_technical_subject",
    "generic_memory_operation",
    "general_data_processing",
}

DENIED_SOLUTION_LABELS = {
    "ect_data_labeling",
    "ect_data_labeling_platform",
    "composite_electrode_preparation",
}

LOW_VALUE_SOLUTION_LABELS = {
    "generic_memory_operation",
    "general_data_processing",
}


def clean_solution_labels(labels: List[str]) -> List[str]:
    cleaned = [x for x in unique_keep_order(labels) if x not in DENIED_SOLUTION_LABELS]
    specific = [
        x for x in cleaned
        if x not in LOW_VALUE_SOLUTION_LABELS and not x.startswith(("core_", "claimed_", "safe_"))
    ]
    return specific if specific else cleaned


def infer_replacement_solution_labels_from_context(
    text: str,
    claim_types: Optional[List[str]] = None,
) -> List[str]:
    text = normalize_ws(text)
    claim_types = claim_types or []
    labels: List[str] = []
    patterns = [
        ("memory_read_operation", r"读取|读操作|read operation|read command|read request|read data|판독|읽기"),
        ("memory_program_operation", r"编程|写入|program operation|programming|write operation|write data|프로그램|쓰기"),
        ("memory_erase_operation", r"擦除|erase operation|erase command|소거"),
        ("memory_control_operation", r"控制器|控制电路|控制逻辑|memory controller|control circuit|controller|제어기|제어 회로"),
        ("storage_device_management", r"存储装置|存储设备|存储系统|storage device|storage system|solid state drive|SSD|저장 장치"),
        ("semiconductor_memory_structure", r"半导体存储器|半导体装置|存储器结构|semiconductor memory|semiconductor device|memory structure|반도체 메모리"),
        ("flash_memory_operation", r"闪存|快闪|flash memory|NAND|NOR flash|플래시|낸드"),
        ("address_mapping_management", r"地址映射|逻辑地址|物理地址|address mapping|logical address|physical address|주소 매핑"),
        ("cache_management", r"缓存|cache|캐시"),
        ("firmware_update_control", r"固件|firmware|펌웨어"),
        ("data_analysis_model", r"数据分析|分析方法|模型|拟合|评估|data analysis|analysis model|fitting|evaluation|분석|모델|평가"),
        ("data_collection_control", r"数据采集|数据收集|采集模块|acquisition|data collection|collecting data|데이터 수집"),
        ("data_prediction_model", r"预测|预警|forecast|prediction|predictive|예측"),
        ("data_classification_recognition", r"分类|识别|辨识|classification|recognition|identify|인식|분류"),
        ("monitoring_signal_processing", r"监测|监控|信号处理|signal processing|monitoring|모니터링|신호 처리"),
        ("data_security_processing", r"加密|解密|认证|鉴权|encryption|decryption|authentication|security|암호화|복호화|인증"),
        ("image_data_processing", r"图像|影像|视觉|image|vision|이미지|영상"),
        ("communication_data_processing", r"通信|传输|总线|communication|transmission|bus|통신|전송|버스"),
        ("test_validation_control", r"测试|验证|校验|test|testing|validation|verify|검증|테스트"),
        ("power_quality_analysis", r"电能质量|power quality|전력 품질"),
    ]
    for label, pat in patterns:
        if re.search(pat, text, flags=re.I):
            labels.append(label)
    if not labels:
        if "method" in claim_types or "process" in claim_types:
            labels.append("method_feature_extraction")
        elif "system" in claim_types:
            labels.append("system_feature_extraction")
        elif "device" in claim_types:
            labels.append("device_feature_extraction")
    return unique_keep_order(labels)[:6]


def normalize_solution_labels_for_quality(
    labels: List[str],
    context_text: str = "",
    claim_types: Optional[List[str]] = None,
) -> List[str]:
    cleaned = [x for x in unique_keep_order(labels) if x not in DENIED_SOLUTION_LABELS]
    specific = [
        x for x in cleaned
        if x not in LOW_VALUE_SOLUTION_LABELS and not x.startswith(("core_", "claimed_", "safe_"))
    ]
    if specific:
        return specific
    replacements = infer_replacement_solution_labels_from_context(context_text, claim_types)
    if replacements:
        return replacements
    return cleaned


def infer_specific_solution_labels_from_text(text: str) -> List[str]:
    labels: List[str] = []
    patterns = [
        ("plane_dedicated_pad_group", r"平面专用焊盘组|plane dedicated pad|전용 패드"),
        ("multi_plane_architecture", r"多个存储器平面|memory plane|메모리 평면"),
        ("separate_data_path", r"数据路径|data path|데이터 경로"),
        ("page_buffer_circuit", r"页缓冲器|page buffer|페이지 버퍼"),
        ("no_plane_address_input", r"不接收平面地址|no plane address|평면 주소"),
        ("nonvolatile_memory_device", r"非易失性存储器件|非易失性存储器装置|non-volatile memory|nonvolatile memory|비휘발성 메모리"),
        ("memory_cell_array", r"存储单元阵列|memory cell array|메모리 셀 어레이"),
        ("vertical_device_structure", r"垂直方向|vertical direction|수직 방향|垂直NAND|vertical nand"),
        ("stacked_semiconductor_layers", r"半导体层|semiconductor layer|반도체층"),
        ("vertical_metal_contacts", r"金属接触|metal contact|금속 콘택트"),
        ("capacitor_electrode_structure", r"电容器电极|capacitor electrode|커패시터 전극"),
        ("channel_capacitor_structure", r"沟道电容器|channel capacitor|채널 커패시터"),
        ("voltage_generator_integration", r"电压发生器|voltage generator|전압 발생기"),
        ("gate_pattern_structure", r"栅极图案|gate pattern|게이트 패턴"),
        ("active_pattern_structure", r"有源图案|active pattern|액티브 패턴"),
        ("nand_screening", r"NAND.{0,12}筛选|筛选.{0,12}NAND|NAND.{0,12}screen|screen.{0,12}NAND|NAND.{0,12}선별"),
        ("board_connector_interconnect", r"母板|子板|连接器|motherboard|daughterboard|connector|보드|커넥터"),
        ("screening_module_control", r"筛选模块|screening module|선별 모듈"),
        ("memory_controller_control", r"存储器控制器|memory controller|메모리 컨트롤러|메모리 제어기"),
        ("power_module_integration", r"电源模块|power module|전원 모듈"),
        ("current_based_type_detection", r"电流值|NAND类型|current value|NAND type|전류값|NAND 타입"),
        ("semiconductor_gate_structure", r"半导体结构|衬底|栅|semiconductor structure|substrate|gate|반도체 구조|기판|게이트"),
        ("tunnel_oxide_layer", r"隧穿氧化层|tunnel oxide|터널 산화"),
        ("floating_gate_control_gate", r"浮栅|控制栅|floating gate|control gate|플로팅 게이트|제어 게이트"),
        ("interlocking_convex_concave_structure", r"凹凸结构|互相嵌入|concave|convex|interlock|요철|맞물"),
        ("word_line_control", r"字线|word line|워드 라인"),
        ("bit_line_connection", r"位线|bit line|비트 라인"),
        ("erase_voltage_control", r"擦除电压|erase voltage|소거 전압"),
        ("program_verify_control", r"编程验证|program verify|프로그램 검증"),
        ("sense_amplifier_circuit", r"感测放大器|sense amplifier|센스 앰프"),
        ("peripheral_circuit_layout", r"外围电路|peripheral circuit|주변 회로"),
        ("sod_film_processing", r"SOD膜|SOD film|SOD 막"),
        ("annealing_sequence_control", r"低温退火|高温退火|退火顺序|annealing|어닐"),
        ("mixed_gas_annealing", r"氧气气体|氮气气体|氢气|混合气体|oxygen gas|nitrogen gas|hydrogen gas|mixed gas"),
        ("file_system_porting", r"文件系统.{0,12}移植|移植.{0,12}文件系统|file system.{0,20}port|파일 시스템.{0,12}이식"),
        ("configuration_interface_generation", r"配置界面|配置组件|configuration interface|configuration component|설정 인터페이스|구성 컴포넌트"),
        ("source_code_file_generation", r"源代码文件|source code file|소스 코드 파일"),
        ("target_os_porting", r"目标操作系统|target operating system|대상 운영체제"),
        ("storage_partition_configuration", r"存储分区|storage partition|저장 파티션"),
        ("converter_control_loop", r"转换器|converter|컨버터"),
        ("digital_analog_control_loop", r"数字控制回路|模拟控制回路|digital control loop|analog control loop|디지털 제어 루프|아날로그 제어 루프"),
        ("mode_switching_circuit", r"模式切换电路|mode switching circuit|모드 전환 회로"),
        ("compensation_comparator_circuit", r"补偿电路|比较器|compensation circuit|comparator|보상 회로|비교기"),
        ("flash_page_data_operation", r"闪存数据页|数据页存储结构|flash data page|flash page|플래시 데이터 페이지"),
        ("solid_state_drive_data_operation", r"固态硬盘|solid state drive|ssd|솔리드 스테이트 드라이브"),
        ("segmented_data_voltage_mapping", r"分段.{0,12}原始数据|十六进制数据.{0,12}电压值|segmented data|hexadecimal data.{0,20}voltage"),
        ("memory_cell_voltage_storage", r"存储单元.{0,12}电压值|memory cell.{0,20}voltage value|메모리 셀.{0,12}전압값"),
        ("flash_erase_method", r"闪存.{0,12}擦除|擦除方法|flash.{0,20}erase|플래시.{0,12}소거"),
        ("erase_duration_control", r"擦除时长|第一时长|第二时长|erase duration|erase time|소거 시간"),
        ("target_page_erase_control", r"目标页面|目标存储单元|目标比值|target page|target memory cell|target ratio"),
        ("storage_medium_read_method", r"存储介质.{0,12}读取|读取方法|storage medium.{0,20}read|저장 매체.{0,12}읽"),
        ("read_voltage_management_table", r"读电压管理表|读电压索引|read voltage management table|read voltage index"),
        ("physical_address_page_type_read", r"目标物理地址|目标页类型|target physical address|target page type"),
        ("adjacent_cell_interference_suppression", r"相邻存储器单元.{0,12}干扰|相邻单元状态|adjacent memory cell.{0,20}interference|인접 메모리 셀.{0,12}간섭"),
        ("read_reference_voltage_control", r"读取参考电压|read reference voltage|읽기 기준 전압"),
        ("read_pass_voltage_control", r"读取通过电压|read pass voltage|읽기 패스 전압"),
        ("sensing_compensation", r"感测.{0,12}补偿|补偿.{0,12}感测|sensing.{0,20}compensation|감지.{0,12}보상"),
        ("secure_spi_flash_access", r"安全.{0,12}(spi|串行外围接口).{0,12}闪存|密码引擎|数字签名|块锁定|secure.{0,20}spi.{0,20}flash|cryptographic.{0,20}flash"),
        ("nand_block_architecture", r"NAND.{0,16}(块架构|block architecture)|块提升|block lift|block boosting"),
        ("boost_circuit_integration", r"升压电路|boost circuit|boost converter|승압 회로"),
        ("address_mapping_cache", r"地址映射|逻辑地址.{0,12}物理地址|address mapping|logical address.{0,20}physical address"),
        ("pcie_virtual_machine_system", r"PCIE.{0,20}虚拟机|virtual machine.{0,20}PCIE|PCI-E.{0,20}接口"),
        ("hierarchical_error_detection", r"分层错误检测|错误校正|error correction|error detection|ecc"),
        ("defect_table_recording", r"缺陷记录|块缺陷记录表|列缺陷记录表|defect record|defect table"),
        ("storage_preallocation", r"预分配|preallocat|第一存储单元.{0,20}第二存储单元"),
        ("low_power_management", r"低功耗|睡眠唤醒|功耗控制|low power|sleep wake|power management"),
        ("diagnostic_data_processing", r"诊断|diagnostic|ECU|状态数据|fault diagnosis|故障诊断"),
        ("adaptive_voltage_control", r"电压控制|电压调整|偏移电压|重读表|自适应调节|voltage control|voltage adjust|adaptive voltage"),
        ("bios_spi_switch_detection", r"BIOS.{0,12}闪存|多工器|串行外设接口|spi.{0,12}switch|switch detection"),
        ("imaging_resource_reallocation", r"血流成像|成像装置|计算资源|imaging device|resource reallocation"),
        ("navigation_object_recognition", r"导航辅助|对象辨识|视力受损|object recognition|navigation assist"),
        ("write_protection_interception", r"写保护|拦截写入|保护分区|write protection|intercept.{0,20}write"),
        ("transaction_log_mapping", r"事务数据|日志映射表|数据回滚|transaction data|log mapping|rollback"),
        ("image_super_resolution_learning", r"图像超分|加权损失函数|神经网络|super.?resolution|neural network"),
        ("film_forming_material_process", r"成膜材料|成膜组合物|无机前体|反应气体|film forming|inorganic precursor"),
        ("vessel_structure_enhancement", r"血管结构增强|Hessian矩阵|高斯混合模型|vessel enhancement|hessian"),
        ("zynq_reconfiguration", r"ZYNQ|在轨加载|重构管理|reconfiguration"),
        ("machine_learning_analysis", r"机器学习|SHAP|高斯聚类|MARS模型|machine learning|model training"),
        ("circular_dichroism_enhancement", r"圆二色|手性微结构|相变材料|circular dichroism|chiral"),
        ("battery_diagnostic_system", r"电池.{0,12}(维修|检测|健康)|battery.{0,20}(diagnostic|repair|health)"),
        ("document_storage_generation", r"文档存储|文档.{0,12}生成|使用手册|document storage|document generation"),
        ("display_panel_register_drive", r"液晶面板|寄存器值|显示设备|display panel|register value"),
        ("multimodal_risk_prediction", r"多模态影像|风险预测|肿瘤转移|risk prediction|multimodal"),
        ("spi_nand_performance_test", r"SPI.{0,4}NAND.{0,20}测试|待测芯片|测试电压|测试频率|performance test"),
        ("servo_controller_program_storage", r"伺服马达|微控制器|程序代码|servo motor|program code"),
        ("baseband_program_loading", r"通信基带芯片|物理层程序|片内存储器|baseband|physical layer program"),
        ("wireless_sensor_module", r"无线传感器|供电机构|软制按键|wireless sensor"),
        ("noise_reduction_transceiver", r"降噪对讲机|麦克风|扬声器|环境传感|noise reduction|transceiver"),
        ("engineered_strain_release", r"工程菌株|自裂解|虾青素|yeast|engineered strain"),
        ("enhanced_io_interface", r"增强.{0,8}IO|enhanced.{0,8}io|program-suspend-resume|suspend-resume"),
        ("word_line_shift_control", r"time tag.{0,20}word line|word line shift|字线.{0,12}偏移|failed bit count"),
        ("pre_boosting_program_strategy", r"pre-boost|preboost|预升压|negative word line"),
        ("configurable_computational_storage", r"computational storage|configurable drive|FPGA|可配置驱动器"),
        ("smart_verify_algorithm", r"smart verify|verify algorithm|智能验证"),
        ("proof_of_space_storage", r"proof of space|solid state drives.{0,20}activities"),
        ("code_rate_ecc_allocation", r"code.?rate|adaptive ecc|wordline noise|代码率"),
        ("building_area_estimation", r"building area|aerial image|canopy height|建筑面积"),
        ("data_disintegration_mechanism", r"data disintegrator|support frame|motor assembly|drive shaft|电子数据销毁"),
        ("memory_manufacturing_process", r"制作方法|制造方法|制备工艺|形成方法|蚀刻|通道孔|掺杂膜层|manufactur|fabricat|etch"),
        ("three_dimensional_memory_structure", r"三维存储器|3D\s*NAND|3DNAND|堆叠结构|垂直堆叠|通道结构|3d memory"),
        ("flash_font_configuration", r"闪存字库|字符编码|字体数据|character data|font data"),
        ("spi_command_interaction", r"串行外围接口|串行外设接口|SPI总线|操作命令|spi bus|spi command"),
        ("flash_interface_reset_mitigation", r"线路重置|重置请求|异常事件|interface reset|line reset"),
        ("flash_issue_source_location", r"问题源定位|异常能重现|累计写入次数|issue source|failure source"),
        ("data_collection_by_type", r"数据类型|数据搜集|来源块|目的块|data collection|data type"),
        ("vehicle_data_transmission_control", r"车联网|拥塞窗口|信道容量|服务质量|traffic control|congestion"),
        ("bad_block_marking", r"坏块标记|坏块检测|bad block|block marking"),
        ("firmware_upgrade_control", r"固件升级|固件程序|firmware upgrade|firmware program"),
        ("insar_landslide_detection", r"InSAR|滑坡识别|冻土区|landslide"),
        ("battery_gauge_learning", r"电量计|充放电循环|剩余电量|battery gauge"),
        ("biometric_registration", r"身份注册|生物识别|登入|biometric|identity registration"),
        ("flash_read_write_temperature_control", r"数据读写|温度区间|数据读取异常|read.?write|temperature interval"),
        ("flash_erase_cycle_control", r"编程抹除循环|抹除验证电压|擦除验证电压|erase verify|program erase cycle"),
        ("flash_validation_test", r"Flash存储器测试|写入验证|读取验证|flash test|read verify|write verify"),
        ("flash_control_switching", r"控制开关|基本输入输出系统控制器|主控制器|从控制器|control switch"),
        ("decoding_schedule_optimization", r"LDPC|RAID|解码调度|解码计划表|decoding schedule"),
        ("storage_thermal_management", r"散热外壳|热传导|高温度电子部件|thermal|heat dissipation"),
        ("photovoltaic_power_prediction", r"光伏出力|太阳辐射|卫星云图|photovoltaic|solar radiation"),
        ("pharmaceutical_colloid_composition", r"胶体药物组合物|纳米晶体|药代动力学|drug composition|nanocrystal"),
        ("memory_read_voltage_recording", r"读取电压|记录读取电压|read voltage"),
        ("mold_sensor_monitoring", r"模具主体|传感器盒|长度传感器|NFC天线|mold|sensor box"),
        ("lithography_inspection", r"光刻系统|周期性结构|相位掩模|测量辐射|lithography"),
        ("ufs_sanitize_operation", r"通用闪存|UFS|清除操作|紧急命令|sanitize"),
        ("bayesian_control_model", r"贝叶斯网络|集成学习|控制方法|Bayesian network"),
        ("boot_security_verification", r"启动安全性|引导加载程序|内核镜像|security verification|boot"),
        ("logical_physical_address_lookup", r"逻辑到物理地址|逻辑至物理映射|查找加速|logical.?physical"),
        ("user_profile_service_configuration", r"用户画像|在途服务|偏好结果|user profile"),
        ("erosion_mapping_system", r"水土流失|图斑落地|侵蚀模数|erosion"),
        ("lidar_debug_calibration", r"单光子激光雷达|地面调试|调试校正|lidar"),
        ("serial_encoder_decoder", r"串行编码器|解码器电路|并行信号|串行信号|serial encoder"),
        ("memory_initialization", r"初始化过程|唤醒状态|唤醒过程|initialization|wake"),
        ("arduino_audio_control", r"Arduino|音乐控制器|声音采集|音频输出|audio control"),
        ("semiconductor_imaging_recipe", r"影像配方|关注样本|电子逃逸|主束参数|imaging recipe"),
        ("gas_interference_compensation", r"可燃气体|油烟干扰|补偿方法|gas monitoring"),
        ("deep_learning_weather_model", r"天气预报|深度学习|物理因子|weather forecast"),
        ("embedded_flash_automated_test", r"嵌入式闪存芯片|自动化测试|测试平台|automated test"),
        ("computational_storage_optimization", r"计算存储设备|数据库引擎|语义提示|computational storage"),
        ("power_line_fault_location", r"并网线路|故障定位|全场景适应|fault location"),
        ("nand_flash_data_transfer", r"NAND\s*FLASH.{0,16}数据传输|校验码|缓存区|data transfer"),
        ("equipment_life_prediction", r"设备寿命预测|因果发现|多变量时序|life prediction"),
        ("reconfigurable_production_control", r"可重构模块化生产|过程管控|预紧力补偿|production control"),
        ("firmware_upgrade_link_test", r"固件升级链路|测试链路|测试闪存|upgrade link"),
        ("remote_sensing_change_monitoring", r"遥感数据|森林变化|反转点检测|remote sensing"),
        ("fault_code_processing", r"故障码|故障记录表|车辆网关|fault code"),
        ("flash_error_injection_test", r"闪存错误注入|错误比例|预设测试参数|error injection"),
        ("nand_flash_programming", r"NAND\s*FLASH.{0,16}烧录|烧录效率|不良品|programming"),
        ("spaceborne_storage_cache", r"星载固态存储|并行缓存|载荷数据|spaceborne storage"),
        ("emmc_read_performance", r"嵌入式多媒体卡|读取效能|开放端读取|eMMC|embedded multimedia card"),
        ("firmware_loading_validation", r"固件加载|固件镜像校验|片选信号|firmware loading|firmware image"),
        ("flash_datasheet_matching", r"数据手册|制造商信息匹配|指令适配|datasheet"),
        ("flash_screening_test", r"闪存筛选|目标温度|擦写操作|screening.{0,20}flash"),
        ("spatiotemporal_copula_modeling", r"时间空间相关性|自回归积分滑动平均|Copula|GARCH|ARIMA"),
        ("watershed_pollution_clustering", r"流域径流污染|聚类分区|景观生态|watershed"),
        ("flash_upgrade_controller", r"闪存.{0,12}升级|总线升级控制器|重试控制器|flash upgrade"),
        ("sdnand_flash_management", r"SDNAND|访问热度|高热度分区|缓存分配"),
        ("nor_flash_driver_status_query", r"非门闪存|状态查询|错误处理|执行频率|NOR flash"),
        ("read_reference_voltage_determination", r"读参考电压|阈值电压分布|电压区间|reference voltage"),
        ("multimodal_trace_detection", r"多模联检|电化学阻抗|表面增强拉曼|质谱|multimodal detection"),
        ("earthquake_motion_fitting", r"震源特征|场地特征|地震动|ground motion"),
        ("ecological_restoration_monitoring", r"生态修复|无人机巡检|多模态融合|ecological restoration"),
        ("flash_aging_test", r"闪存芯片老化测试|平均读取速率|aging test"),
        ("demand_response_uncertainty", r"需求响应|不确定性量化|广义线性模型|demand response"),
        ("credit_overdue_warning", r"信贷逾期|智能预警|实时数据分析|credit overdue"),
        ("stray_current_vibration_analysis", r"杂散电流|变压器振动|关联分析|stray current"),
        ("spiflash_embedded_file_system", r"SPIFlash|嵌入式文件系统|超级块|文件记录区"),
        ("wetland_partitioning", r"湿地分区|地表水|地下水|随机森林|wetland"),
        ("mcu_flash_execution", r"微控制单元|内嵌闪存|外设闪存|设备运行"),
        ("bios_log_transfer", r"BIOS日志|异步收发器|日志传输|UART"),
        ("vehicle_trajectory_tracking", r"车辆轨迹|模糊逻辑|驾驶模式控制器|trajectory tracking"),
        ("communication_bus_controller", r"高速通信总线|片选管理|状态转换模块|bus controller"),
        ("radar_deformation_monitoring", r"毫米波雷达|形变监测|多模态数据融合|radar"),
        ("flash_memory_control_operation", r"半导体装置|快闪存储器|输入/输出部件|串行时钟|flash memory control"),
        ("general_data_processing", r"数据处理方法|数据采集模块|数据清洗|处理器|processor"),
        ("visual_colorimetric_detection", r"比色检测|恒温扩增|颜色变化曲线|colorimetric"),
        ("nand_access_frequency_remapping", r"访问频率|重新映射|数据重新映射|remap"),
        ("read_disturb_mitigation", r"读干扰|读取次数|热数据|冷数据|read disturb"),
        ("flash_storage_tiering", r"动态分级|存储块资源|块内页|tiered storage"),
        ("queued_flash_write", r"二级队列表|候选队列|数据写入|write queue"),
        ("visual_badge_inspection", r"徽章粘贴检测|相机位|物料载盘|visual inspection"),
        ("metallurgical_process_monitoring", r"冶金|层次化监测|图注意力|metallurgical"),
        ("server_boot_firmware_recovery", r"服务器开机固件|异常处理|通道切换指令|boot firmware"),
        ("slc_qlc_data_migration", r"SLC模式|QLC模式|源物理块|目标物理块|single-bit programming"),
        ("lakehouse_tiered_storage", r"湖仓一体|动态分级存储|多工作负载|lakehouse"),
        ("flash_garbage_collection", r"垃圾回收|无效页|可用块|garbage collection"),
        ("flash_data_recovery", r"数据恢复|硬解码|重读列表|读校准|data recovery"),
        ("authentication_mask_encryption", r"鉴权|随机掩码|密文|安全元件|authentication"),
        ("blockchain_recommendation_model", r"区块链|交易对象推荐|链上数据|blockchain"),
        ("lifetime_data_distribution_analysis", r"多源寿命数据|分布模型|寿命数据|lifetime data"),
        ("host_request_cache_protection", r"主机请求|缓存保护|日志空间|host request"),
        ("nand_temperature_patrol", r"宽温|温度等级|巡检|temperature patrol"),
        ("student_behavior_analysis", r"学生行为|混合教学|线上与线下|student behavior"),
        ("pollution_risk_assessment", r"污染风险|镉|风险评估|预警方法|pollution risk"),
        ("wind_farm_maintenance_optimization", r"风电场|智能运维|风险量化|维护动作|wind farm"),
        ("thermal_power_frequency_regulation", r"火电机组|调频性能|出清模型|frequency regulation"),
        ("nand_auto_test_script", r"NAND闪存.{0,12}自动测试|测试脚本|有序命令链"),
        ("flash_lifetime_monitoring", r"Flash存储实时监控|块擦除次数|文件更新频率|寿命预测"),
        ("power_market_supervision", r"电力交易市场|市场出清价|输电断面|market supervision"),
        ("ocr_image_preprocessing", r"图像预处理|文字方向校正|光学字符识别|OCR"),
        ("signal_acquisition_control", r"信号收发|信号采集|CAN通信|signal acquisition"),
        ("aerosol_activity_monitoring", r"气溶胶|活度监测|涂层试验|Po-210"),
        ("medical_communication_evaluation", r"医患沟通|虚拟患者|沟通效果|patient communication"),
        ("equipment_reliability_evaluation", r"设备可靠性|多源退化数据|可靠度函数|reliability"),
        ("power_system_state_evolution", r"电力系统运行状态|隐马尔可夫|state evolution"),
        ("flash_sampling_test", r"抽样测试|擦写读操作|sampling test"),
        ("nand_encoding_decoding", r"编码和解码|译码方法|有限域|encoding|decoding"),
        ("flash_permutation_code", r"置换码|编码方法|译码方法|permutation code"),
        ("storage_backend_loopback_test", r"后端回环测试|循环队列|loopback test"),
        ("multispectral_sediment_signal_fusion", r"多光谱|测沙仪|含沙量|signal fusion"),
        ("flash_particle_grading", r"闪存颗粒筛选|筛选分级|并行读取"),
        ("composite_electrode_preparation", r"复合电极材料|预碳化|混合碱活化|electrode"),
        ("clearing_path_recommendation", r"跨境清算|清算路径|知识图谱|clearing path"),
        ("causal_structure_discovery", r"因果结构|有向无环图|强化学习|causal"),
        ("flash_plane_binding", r"闪存绑定|plane|绑定块"),
        ("flash_boot_sequence", r"系统启动|初始引导代码|静态存储器|boot code"),
        ("meter_flash_sector_management", r"电能表|外部Flash|扇区划分|meter"),
        ("sediment_transport_capture", r"水槽泥沙|推移特征|高清摄像机|sediment"),
        ("bayesian_power_forecast", r"贝叶斯模型平均|功率预测|后验模型概率"),
        ("storage_device_test_control", r"存储装置.{0,12}测试|第一控制器|第一接口"),
        ("dmx512_signal_recording", r"DMX512|信号录制|灯具控制"),
        ("biosensor_metasurface", r"BIC超表面|生物传感器|m-SWCNT"),
        ("esd_protection_optimization", r"ESD保护|电荷成像|高密度电荷"),
        ("storage_capacity_recovery", r"容量回收|标准容量表|多余容量"),
        ("coal_mine_sensor_monitoring", r"煤矿|塌孔监测|全孔深|sensor"),
        ("phototherapy_signal_system", r"光疗系统|生理感知|信号处理"),
        ("cubesat_platform", r"立方星|卫星总线|CubeSat"),
        ("storage_control_chip_validation", r"存储控制芯片|验证装置|插座"),
        ("audio_wakeup_path_switching", r"音频唤醒|音频通路|智能穿戴"),
        ("soil_degradation_simulation", r"土壤退化|残膜|贝叶斯流"),
        ("descriptor_queue_management", r"描述符队列|任务调度|descriptor"),
        ("firmware_update_fault_tolerance", r"固件更新|标志寄存器|旧固件|新固件"),
        ("green_development_path_modeling", r"绿色发展路径|生态足迹|城市群"),
        ("nand_circular_storage_management", r"NANDFlash循环存储|索引区域|数据采集器"),
        ("tourism_economy_monitoring", r"文化旅游经济|价值雷达|数据监测"),
        ("nvm_rom_loading", r"NVM固态盘ROM|ROM加载|主动搜索"),
        ("flash_programming_circuit", r"闪存烧录电路|排线座|可编程逻辑"),
        ("nand_test_decision_model", r"NAND闪存芯片.{0,12}测试决策|功能合格性预测"),
        ("incremental_compile_upgrade", r"增量编译|动态函数接口表|软件快速升级"),
        ("soft_information_read", r"软信息|判读电压|电压偏移"),
        ("wireless_charger_lifetime_test", r"无线充电器|寿命测试|健康指数"),
        ("pxie_distributed_board", r"PXIe|分布式板卡|信号调理"),
        ("eeg_mood_disorder_assessment", r"脑电|心境障碍|症状导向"),
        ("parcel_risk_identification", r"寄递网点|风险识别|因果关系建模"),
        ("substation_anomaly_detection", r"YOLOv11|变电站|异常行为"),
        ("llr_dynamic_quantization", r"对数似然比|动态量化|LLR"),
        ("power_supply_test_emulation", r"电源测试|负载控制|模拟闪存控制芯片"),
        ("aquaculture_communication", r"无人投食船|养殖数据|抗干扰通讯"),
        ("bridge_cable_force_identification", r"桥梁索力|自回归功率谱|cable force"),
        ("nand_write_pulse_adjustment", r"写入脉冲强度|擦写循环次数|物理特性劣化"),
        ("seepage_factor_modeling", r"地下厂房渗流|脉冲响应|ReliefF"),
        ("multimodal_tamper_detection", r"图文篡改|多模态|tamper"),
        ("lithium_battery_life_prediction", r"锂电池剩余寿命|容量退化|lithium"),
        ("psoc_online_upgrade", r"PSOC|程序在线升级|升级程序"),
        ("pest_peak_detection", r"水稻螟虫|测报灯|多峰高斯"),
        ("temperature_compensated_read_threshold", r"温度补偿|读取阈值|实时温度"),
        ("ecc_read_voltage_threshold", r"闪存纠错|读电压阈值|待纠错页"),
        ("flexible_production_simulation", r"柔性可重构产线|虚实交互|蒙特卡洛"),
        ("firmware_forced_update", r"强制更新模式|引导程序运行|主程序"),
        ("sewage_leakage_prediction", r"污水处理|漏损量|增广迪基"),
        ("day_ahead_grid_trading", r"日前多电网交易|蒙特卡罗|场景缩减|grid trading"),
        ("digital_twin_component_data_collection", r"数字孪生模型|关键部件数据采集|边缘计算模块|digital twin"),
        ("multi_plane_independent_read", r"다중 평면|독립적 판독|multi plane"),
        ("digital_content_ideation", r"디지털 컨텐츠|구상 가이드|아이디어"),
        ("ect_data_labeling", r"와전류탐상|데이터 라벨링|\bECT\b"),
        ("neuromorphic_memory_system", r"뉴로모픽|낸드 셀 스트링|neuromorphic"),
        ("predictive_prefetch_system", r"미리 읽기|결정트리|prefetch"),
        ("microelectronic_device_structure", r"마이크로전자|유전체 슬롯|전도성 패드|microelectronic"),
        ("string_select_threshold_control", r"string select|threshold voltage|문턱 전압"),
        ("continuous_read_performance", r"continuous read|연속 읽기"),
        ("encryption_decryption_processing", r"encryption|decryption|암호화|복호화"),
        ("power_loss_protection_allocation", r"power loss protection|전력 손실 보호"),
        ("reset_operation_control", r"reset operation|리셋 동작"),
        ("unified_memory_architecture", r"unified memory architecture|multiple memory types"),
        ("gray_code_counter_circuit", r"gray code counter|TCLK|TRSTN"),
        ("erase_block_management", r"erase blocks|same string|소거 블록"),
        ("in_memory_bitwise_operation", r"in-memory bitwise|pop-count|bitwise"),
        ("parallel_incremental_writes", r"parallel incremental writes|hyperscaler"),
        ("device_driven_erasure_coding", r"erasure coding|device-driven"),
        ("activity_recognition_battery_management", r"activity recognition|battery management"),
        ("stacked_chip_semiconductor", r"stacked chips|vias|semiconductor device"),
        ("delayed_memory_management", r"delayed memory management"),
        ("ai_accelerator_vector_processing", r"AI accelerator|vector processing|convolution"),
        ("scaleout_flash_storage", r"scaleout storage|storage nodes"),
        ("generic_memory_operation", r"memory device|memory system|storage device|semiconductor device|operation method"),
    ]
    for label, pat in patterns:
        if re.search(pat, text, flags=re.I):
            labels.append(label)
    return unique_keep_order(labels)


def infer_candidate_solution_labels(
    independent_claims: List[Dict[str, Any]],
    title_source: str = "",
    extra_terms: Optional[List[str]] = None,
) -> List[str]:
    text = " ".join([title_source] + (extra_terms or []) + [c.get("raw_text", "")[:1200] for c in independent_claims])
    labels = []
    claim_types = collect_secondary_claim_types(independent_claims)
    labels = infer_specific_solution_labels_from_text(text)
    return unique_keep_order(labels)


def infer_candidate_effect_labels(snippets: List[Dict[str, Any]], figures: List[Dict[str, Any]], extra_texts: List[str]) -> List[str]:
    text = " ".join(extra_texts + [s["text"] for s in snippets[:10]]) + " " + " ".join([f.get("caption_raw", "") for f in figures[:6]])
    labels = []
    if re.search(r"(降低功耗|减少功耗|low power|전력|power consumption)", text, flags=re.I):
        labels.append("power_reduction")
    if re.search(r"(增加.{0,12}带宽|提高.{0,12}带宽|带宽.{0,12}(增加|提高)|increase.{0,20}bandwidth|bandwidth.{0,20}increase|대역폭.{0,12}(증가|향상))", text, flags=re.I):
        labels.append("bandwidth_increase")
    if re.search(r"(并行数据传输|数据.{0,12}并行.{0,8}(读取|写入|传输)|并行.{0,12}数据.{0,8}(读取|写入|传输)|"
                 r"parallel.{0,20}(data transfer|transfer data|data read|data write)|병렬.{0,12}데이터.{0,8}(읽기|쓰기|전송))",
                 text, flags=re.I):
        labels.append("parallel_data_transfer")
    if re.search(r"(降低.{0,12}延迟|减少.{0,12}延迟|延迟.{0,12}(降低|减少)|reduce.{0,20}(latency|delay)|(latency|delay).{0,20}reduction|지연.{0,12}(감소|저감))", text, flags=re.I):
        labels.append("latency_reduction")
    if re.search(r"(提高|提升|改善|增强|增加|improv|increas|enhanc|높|향상).{0,24}(可靠性|稳定性|寿命|耐久|reliability|stability|lifetime|endurance|신뢰성|수명)|"
                 r"(可靠性|稳定性|寿命|耐久|reliability|stability|lifetime|endurance|신뢰성|수명).{0,24}(提高|提升|改善|增强|增加|improv|increas|enhanc|높|향상)",
                 text, flags=re.I):
        labels.append("reliability_improvement")
    if re.search(r"(提高|提升|改善|增强|增加|加快|优化|improv|increas|enhanc|accelerat|optimi|향상).{0,24}(性能|效率|速度|吞吐|读写|存取|performance|efficiency|speed|throughput|read|write|access|성능|효율)|"
                 r"(性能|效率|速度|吞吐|读写|存取|performance|efficiency|speed|throughput|read|write|access|성능|효율).{0,24}(提高|提升|改善|增强|增加|加快|优化|improv|increas|enhanc|accelerat|optimi|향상)",
                 text, flags=re.I):
        labels.append("performance_improvement")
    if re.search(r"(提高|提升|改善|增加|优化|improve|increase|optimi|향상).{0,24}(容量|利用率|存储效率|空间|capacity|utilization|storage efficiency|space|용량|활용률|저장 효율)|"
                 r"(容量|利用率|存储效率|空间|capacity|utilization|storage efficiency|space|용량|활용률|저장 효율).{0,24}(提高|提升|改善|增加|优化|improve|increase|optimi|향상)",
                 text, flags=re.I):
        labels.append("storage_efficiency_improvement")
    if re.search(r"(减少|降低|防止|避免|校正|抑制|reduce|prevent|avoid|correct|suppress|mitigate|감소|방지|억제).{0,24}(错误|误差|失效|故障|短路|干扰|纠错|error|failure|fault|short circuit|interference|disturb|오류|고장|단락|간섭)|"
                 r"(错误|误差|失效|故障|短路|干扰|纠错|error|failure|fault|short circuit|interference|disturb|오류|고장|단락|간섭).{0,24}(减少|降低|防止|避免|校正|抑制|reduce|prevent|avoid|correct|suppress|mitigate|감소|방지|억제)",
                 text, flags=re.I):
        labels.append("error_reduction")
    if re.search(r"丢失数据.{0,20}恢复|恢复.{0,20}丢失数据|数据恢复|自动重建|自动纠错|data recover|recover.{0,20}lost data|error correction|데이터 복구|오류 정정", text, flags=re.I):
        labels.append("data_recovery")
    if re.search(r"(降低|减少|节省|缩小|简化|reduce|save|shrink|simplify|감소|절감|축소|단순화).{0,24}(成本|面积|尺寸|资源|复杂度|cost|area|size|resource|complexity|비용|면적|크기|복잡도)|"
                 r"(成本|面积|尺寸|资源|复杂度|cost|area|size|resource|complexity|비용|면적|크기|복잡도).{0,24}(降低|减少|节省|缩小|简化|reduce|save|shrink|simplify|감소|절감|축소|단순화)",
                 text, flags=re.I):
        labels.append("resource_reduction")
    return unique_keep_order(labels)


def is_supported_solution_label(label: str, support_text: str) -> bool:
    high_risk_patterns = {
        "plane_dedicated_pad_group": r"平面专用焊盘组|plane dedicated pad|전용 패드",
        "multi_plane_architecture": r"多个存储器平面|memory plane|메모리 평면",
        "separate_data_path": r"数据路径|data path|데이터 경로",
        "page_buffer_circuit": r"页缓冲器|page buffer|페이지 버퍼",
        "no_plane_address_input": r"不接收平面地址|no plane address|평면 주소",
    }
    pat = high_risk_patterns.get(label)
    if not pat:
        return True
    return bool(re.search(pat, support_text, flags=re.I))


def derive_problem_labels_from_effects(effect_labels: List[str]) -> List[str]:
    derived = []
    for label in effect_labels:
        if label == "power_reduction":
            derived.append("power_consumption")
        elif label == "bandwidth_increase":
            derived.append("bandwidth_limit")
        elif label == "parallel_data_transfer":
            derived.append("serial_transfer_bottleneck")
        elif label == "latency_reduction":
            derived.append("data_transfer_latency")
        elif label == "reliability_improvement":
            derived.append("reliability_issue")
        elif label == "performance_improvement":
            derived.append("performance_limit")
        elif label == "storage_efficiency_improvement":
            derived.append("storage_efficiency_limit")
        elif label == "error_reduction":
            derived.append("error_risk")
        elif label == "data_recovery":
            derived.append("data_loss_risk")
        elif label == "resource_reduction":
            derived.append("resource_overhead")
    return unique_keep_order(derived)[:4]


def seed_evidence_ids(independent_claims: List[Dict[str, Any]], figures: List[Dict[str, Any]], snippets: List[Dict[str, Any]]) -> List[str]:
    ids = []
    for c in independent_claims[:4]:
        no = normalize_claim_no(c.get("claim_no", ""))
        if not no:
            continue
        ids.append(f"claim_{no}")
        page_no = normalize_ws(c.get("page_start") or c.get("page_end") or "")
        if page_no:
            ids.append(f"page_{page_no}")
    for f in figures[:4]:
        fig_no = normalize_ws(f.get("figure_no", ""))
        if fig_no:
            ids.append(f"fig_{fig_no}")
    for s in snippets[:4]:
        page_no = normalize_ws(s.get("page_no", ""))
        if page_no:
            ids.append(f"page_{page_no}")
    return unique_keep_order(ids)[:12]


# ---------------- LLM ----------------

def extract_json_candidate(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    first = t.find("{")
    last = t.rfind("}")
    return t[first:last + 1] if first != -1 and last != -1 and last > first else t


def minimal_llm_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "core_subject": {"type": "string", "maxLength": 120},
            "core_elements": {"type": "array", "items": {"type": "string", "maxLength": 60}, "maxItems": 8},
            "problem_labels": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 6},
            "solution_labels": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 8},
            "effect_labels": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 6},
            "evidence_ids": {"type": "array", "items": {"type": "string", "maxLength": 32}, "maxItems": 12},
            "confidence": {"type": "number"},
        },
        "required": [
            "core_subject",
            "core_elements",
            "problem_labels",
            "solution_labels",
            "effect_labels",
            "evidence_ids",
            "confidence",
        ],
    }


def build_prompt(
    meta: Dict[str, Any],
    independent_claims: List[Dict[str, Any]],
    claim_ref_counts: List[Dict[str, Any]],
    figures: List[Dict[str, Any]],
    snippets: List[Dict[str, Any]],
    claim_type_hint: str,
    candidate_problem_labels: List[str],
    candidate_solution_labels: List[str],
    candidate_effect_labels: List[str],
    candidate_evidence_ids: List[str],
    source_language: str,
    title_source: str,
    independent_claim_nos: List[str],
    core_terms: List[str],
    context_terms_not_core: List[str],
) -> str:
    schema_str = json.dumps(minimal_llm_schema(), ensure_ascii=False, indent=2)
    language_instruction = {
        "zh": "core_subject와 core_elements는 중국어 원문 용어로 작성하라. 번역하지 말라.",
        "en": "Write core_subject and core_elements in English source terms. Do not translate.",
        "ko": "core_subject와 core_elements는 한국어 원문 용어로 작성하라.",
    }.get(source_language, "Use the source-language technical terms. Do not translate unless the source is Korean.")

    lines = [
        "너는 특허를 빠르게 검색/비교/전략 연결하기 위한 최소 인덱스 JSON을 만드는 분석기다.",
        "짧고 비교 가능한 값만 출력하라.",
        "추정 금지. 아래 evidence에 직접 드러난 내용만 사용하라.",
        "긴 원문을 다시 복사하지 말라.",
        language_instruction,
        "core_elements에는 청구항의 구성요소 또는 직접적 기술 관계만 넣어라.",
        "required_core_terms는 가능한 한 원문 그대로 core_subject/core_elements에 반영하라.",
        "context_terms_not_core는 문제/비교/제거 대상 맥락이다. 그 표현 자체를 core_elements에 넣지 말라.",
        "problem_labels, solution_labels, effect_labels는 반드시 lower_snake_case 짧은 태그로 써라.",
        "가능하면 아래 candidate label 중에서만 선택하라.",
        "candidate_solution_labels가 비어 있으면 title_source와 core_elements에서 구체적인 solution_labels를 새로 만들어라.",
        "solution_labels에 claimed_method_process, claimed_system_architecture, claimed_device_structure, claimed_technical_subject 같은 generic 타입명은 쓰지 말라.",
        "evidence_ids는 아래 candidate_evidence_ids 안에서만 선택하라.",
        "출력은 반드시 JSON만 반환하라.",
        "",
        "schema:",
        schema_str,
        "",
        f"metadata: country={meta.get('country','')} source_language={source_language} title_source={title_source}",
        f"claim_type_hint: {claim_type_hint}",
        f"independent_claim_nos: {', '.join(independent_claim_nos)}",
        "",
        "[required_core_terms]",
        ", ".join(core_terms) if core_terms else "-",
        "",
        "[context_terms_not_core]",
        ", ".join(context_terms_not_core) if context_terms_not_core else "-",
        "",
        "[candidate_problem_labels]",
        ", ".join(candidate_problem_labels) if candidate_problem_labels else "-",
        "",
        "[candidate_solution_labels]",
        ", ".join(candidate_solution_labels) if candidate_solution_labels else "-",
        "",
        "[candidate_effect_labels]",
        ", ".join(candidate_effect_labels) if candidate_effect_labels else "-",
        "",
        "[candidate_evidence_ids]",
        ", ".join(candidate_evidence_ids) if candidate_evidence_ids else "-",
        "",
        "[independent_claims]",
    ]

    for c in independent_claims:
        lines.append(f"- claim {c.get('claim_no','')}: {c.get('raw_text','')[:1000]}")

    lines.append("")
    lines.append("[claim_ref_counts]")
    for r in claim_ref_counts[:20]:
        lines.append(f"- claim {r.get('claim_no','')} -> {r.get('ref_no_raw','')} (freq={r.get('freq',1)})")

    lines.append("")
    lines.append("[figures]")
    for f in figures[:8]:
        lines.append(f"- fig_{f.get('figure_no','')} page {f.get('page_no','')}: {normalize_ws(f.get('caption_raw',''))[:180]}")

    lines.append("")
    lines.append("[description_snippets]")
    for s in snippets[:6]:
        lines.append(f"- page_{s.get('page_no','')}: {s.get('text','')[:260]}")

    return "\n".join(lines).strip()


def call_ollama_minimal(prompt: str, patent_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "format": minimal_llm_schema(),
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }

    wall_start = time.time()
    r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    wall_seconds = time.time() - wall_start

    raw_text = data.get("response", "")
    candidate = extract_json_candidate(raw_text)

    parse_error = None
    try:
        parsed = json.loads(candidate)
    except Exception as e:
        parse_error = e
        repair_payload = dict(payload)
        repair_payload["prompt"] = (
            prompt
            + "\n\nPrevious response was invalid JSON. Regenerate one compact valid JSON object only. "
            + "Do not include markdown, comments, or trailing text."
        )
        repair_payload["options"] = dict(payload["options"])
        repair_payload["options"]["temperature"] = 0
        r2 = requests.post(OLLAMA_URL, json=repair_payload, timeout=TIMEOUT)
        r2.raise_for_status()
        data2 = r2.json()
        raw_text2 = data2.get("response", "")
        candidate2 = extract_json_candidate(raw_text2)
        try:
            parsed = json.loads(candidate2)
            data = data2
            raw_text = raw_text2
            candidate = candidate2
        except Exception as e2:
            parse_error = e2
            raw_text = raw_text + "\n\n[RETRY_RAW]\n" + raw_text2
            candidate = candidate2

    if parse_error is not None and "parsed" not in locals():
        raw_path = A4_RAW_INVALID / f"{safe_name(patent_id)}.minimal_v2.raw.txt"
        meta_path = A4_RAW_INVALID / f"{safe_name(patent_id)}.minimal_v2.meta.json"
        raw_path.write_text(raw_text, encoding="utf-8")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "done_reason": data.get("done_reason"),
                    "total_duration": data.get("total_duration"),
                    "prompt_eval_count": data.get("prompt_eval_count"),
                    "eval_count": data.get("eval_count"),
                    "error": str(parse_error),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        raise RuntimeError(f"minimal invalid JSON: {parse_error}")

    meta_info = {
        "wall_seconds": round(wall_seconds, 2),
        "ollama_total_seconds": round(float(data.get("total_duration", 0)) / 1e9, 2) if data.get("total_duration") else None,
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "done_reason": data.get("done_reason"),
    }
    return parsed, meta_info


def validate_llm_part(result: Dict[str, Any]) -> None:
    if not normalize_ws(result.get("core_subject", "")):
        raise RuntimeError("minimal empty core_subject")
    if not result.get("core_elements"):
        raise RuntimeError("minimal empty core_elements")


def fallback_core_elements_from_title(title_source: str, source_language: str) -> List[str]:
    title = normalize_ws(title_source)
    if not title:
        return []
    elements: List[str] = []
    title_upper = title.upper()
    title_lower = title.lower()
    if source_language == "en":
        keyword_pairs = [
            ("BMC", "BMC firmware"),
            ("BIOS", "BIOS firmware"),
            ("FLASH MEMORY", "flash memory"),
            ("ENCRYPT", "encryption processing"),
            ("HOST MANAGED BUFFER", "host managed buffer"),
            ("LOGICAL-TO PHYSICAL ADDRESS TABLE", "logical-to-physical address table"),
            ("SOLID STATE DRIVE", "solid state drive"),
            ("MICROELECTRONIC DEVICES", "microelectronic device"),
            ("CONDUCTIVE RAIL", "conductive rail"),
            ("PROMOTER MATERIAL", "promoter material"),
            ("CONDUCTIVE TIERS", "conductive tiers"),
            ("PHYSICALLY UNCLONABLE FUNCTION", "physically unclonable function"),
            ("FUNCTION BLOCK", "function block"),
        ]
        for needle, label in keyword_pairs:
            if needle in title_upper:
                elements.append(label)
    if source_language == "ko":
        for part in re.split(r"\s*(?:및|와|과|을 위한|를 위한)\s*", title):
            if 2 <= len(part) <= 40:
                elements.append(part)
    if source_language == "zh":
        for part in re.split(r"[、，,及和与]", title):
            if 2 <= len(part) <= 40:
                elements.append(part)
    if not elements:
        elements.append(title_lower if source_language == "en" else title)
    return unique_keep_order(elements)[:6]


def build_final_minimal(
    meta: Dict[str, Any],
    llm_part: Dict[str, Any],
    primary_claim_type: str,
    secondary_claim_types: List[str],
    candidate_evidence_ids: List[str],
    source_language: str,
    title_source: str,
    independent_claim_nos: List[str],
    core_terms: List[str],
    context_terms_not_core: List[str],
    candidate_problem_labels: List[str],
    candidate_solution_labels: List[str],
    candidate_effect_labels: List[str],
) -> Dict[str, Any]:
    valid_ids = set(candidate_evidence_ids)

    evidence_ids = []
    for x in llm_part.get("evidence_ids", []):
        s = normalize_ws(x)
        if s in valid_ids and s not in evidence_ids:
            evidence_ids.append(s)

    if not evidence_ids:
        evidence_ids = candidate_evidence_ids[:6]
    else:
        evidence_ids = unique_keep_order(evidence_ids + candidate_evidence_ids)[:12]

    core_elements = [
        x for x in unique_keep_order(llm_part.get("core_elements", []))
        if x not in set(context_terms_not_core)
    ][:8]
    if len(core_elements) < 3 and core_terms:
        core_elements = unique_keep_order(core_elements + core_terms)[:8]
    core_text_for_confirmation = " ".join([title_source, normalize_ws(llm_part.get("core_subject", ""))] + core_elements)
    confirmed_core_terms = [
        term for term in core_terms
        if term and (
            term in core_text_for_confirmation
            or any(el and (el in term or term in el) for el in core_elements)
        )
    ]
    llm_problem_labels = normalize_tags(llm_part.get("problem_labels", []), 6)
    llm_effect_labels = normalize_tags(llm_part.get("effect_labels", []), 6)
    problem_labels = unique_keep_order(candidate_problem_labels + [x for x in llm_problem_labels if x in candidate_problem_labels])[:6]
    core_subject = normalize_ws(llm_part.get("core_subject", ""))
    support_text_for_labels = " ".join([title_source, core_subject] + core_elements + confirmed_core_terms)
    llm_solution_labels = clean_solution_labels([
        x for x in normalize_tags(llm_part.get("solution_labels", []), 8)
        if x not in GENERIC_SOLUTION_LABELS and is_supported_solution_label(x, support_text_for_labels)
    ])
    core_based_solution_labels = infer_specific_solution_labels_from_text(
        support_text_for_labels
    )
    candidate_specific_solution_labels = clean_solution_labels([
        x for x in candidate_solution_labels if x not in GENERIC_SOLUTION_LABELS
    ])
    solution_labels = normalize_solution_labels_for_quality(
        llm_solution_labels + core_based_solution_labels,
        support_text_for_labels,
        secondary_claim_types,
    )[:8]
    if not solution_labels:
        solution_labels = normalize_solution_labels_for_quality(
            candidate_specific_solution_labels,
            support_text_for_labels,
            secondary_claim_types,
        )[:8]
    effect_labels = unique_keep_order(candidate_effect_labels + [x for x in llm_effect_labels if x in candidate_effect_labels])[:6]
    if not problem_labels and effect_labels:
        problem_labels = derive_problem_labels_from_effects(effect_labels)
    if not solution_labels:
        if "method" in secondary_claim_types or "process" in secondary_claim_types:
            solution_labels = ["core_process_feature_index"]
        elif "system" in secondary_claim_types:
            solution_labels = ["core_system_feature_index"]
        elif "device" in secondary_claim_types:
            solution_labels = ["core_device_feature_index"]
        else:
            solution_labels = ["core_technical_feature_index"]

    confidence = float(llm_part.get("confidence", 0.0) or 0.0)
    if is_generic_subject(core_subject) and title_source and not is_generic_subject(title_source):
        core_subject = title_source
        confidence = min(confidence, 0.76)
    if is_generic_subject(core_subject) and len(core_elements) >= 2:
        core_subject = normalize_ws(f"{core_subject}: {', '.join(core_elements[:3])}")
        confidence = min(confidence, 0.68)
    protected_hit = any(term and (term in normalize_ws(llm_part.get("core_subject", "")) or any(term in el for el in core_elements)) for term in confirmed_core_terms[:6])
    if core_terms and not protected_hit:
        confidence = min(confidence, 0.62)
    if not problem_labels:
        confidence = min(confidence, 0.78)
    if not effect_labels:
        confidence = min(confidence, 0.78)
    if len(core_elements) < 3:
        confidence = min(confidence, 0.62)
    if solution_labels and (solution_labels[0].startswith("claimed_") or solution_labels[0].startswith("core_")):
        confidence = min(confidence, 0.68)
    confidence = min(confidence, 0.86)

    final = {
        "patent_id": meta["patent_id"],
        "source_language": source_language,
        "summary_language": source_language,
        "title_source": title_source,
        "title_ko": title_source if source_language == "ko" else "",
        "primary_claim_type": primary_claim_type,
        "secondary_claim_types": secondary_claim_types,
        "independent_claim_nos": independent_claim_nos,
        "protected_terms": confirmed_core_terms,
        "context_terms_not_core": context_terms_not_core,
        "core_subject": core_subject,
        "core_elements": core_elements,
        "core_elements_ko": core_elements if source_language == "ko" else [],
        "problem_labels": problem_labels,
        "solution_labels": solution_labels,
        "effect_labels": effect_labels,
        "evidence_ids": evidence_ids[:12],
        "confidence": round(max(0.0, min(confidence, 1.0)), 2),
    }
    return final


def validate_final(result: Dict[str, Any], patent_id: str) -> None:
    if str(result.get("patent_id", "")) != patent_id:
        raise RuntimeError("final patent_id mismatch")
    if not normalize_ws(result.get("title_source", "")):
        raise RuntimeError("final empty title_source")
    if re.search(r"(摘要|ABSTRACT|Abstract|权利要求书|청구항|청구범위|요\s*약|\(\s*57\s*\)|\(\s*71\s*\)|\(\s*72\s*\)|Applicant:|Inventor:|Assignee:|U\.?\s*S\.?\s*Cl\.?|CPC|USPC|Date of Patent|Publication Date|Pub\.? Date|Filed:|H[I1][O0]B|et al\.)", result.get("title_source", ""), flags=re.I):
        raise RuntimeError("final contaminated title_source")
    if not normalize_ws(result.get("core_subject", "")):
        raise RuntimeError("final empty core_subject")
    if is_generic_subject(result.get("core_subject", "")):
        raise RuntimeError("final overly generic core_subject")
    if not result.get("core_elements"):
        raise RuntimeError("final empty core_elements")
    leaked_context_terms = set(result.get("context_terms_not_core", [])) & set(result.get("core_elements", []))
    if leaked_context_terms:
        raise RuntimeError(f"final context term leaked into core_elements: {sorted(leaked_context_terms)}")
    if any(x in DENIED_SOLUTION_LABELS for x in result.get("solution_labels", [])):
        raise RuntimeError("final denied solution_label")
    if not result.get("evidence_ids"):
        raise RuntimeError("final empty evidence_ids")
    if any(str(x).startswith("claim_0") for x in result.get("evidence_ids", [])):
        raise RuntimeError("final invalid claim_0 evidence_id")
    if not result.get("primary_claim_type"):
        raise RuntimeError("final empty primary_claim_type")
    if not result.get("independent_claim_nos"):
        raise RuntimeError("final empty independent_claim_nos")
    if any(not is_valid_claim_no(x) for x in result.get("independent_claim_nos", [])):
        raise RuntimeError("final invalid independent_claim_nos")


def process_one_patent(con: sqlite3.Connection, patent_id: str, overwrite: bool = False) -> Dict[str, Any]:
    out_path = MINIMAL_DIR / f"{safe_name(patent_id)}.minimal.json"
    if out_path.exists() and not overwrite:
        log(f"[캐시 사용] {out_path.name}")
        return {
            "patent_id": patent_id,
            "output_path": str(out_path),
            "elapsed": 0.0,
            "loaded_from_cache": True,
        }

    meta = get_patent_meta(con, patent_id)
    source_language = source_language_for_patent(patent_id, meta.get("country", ""))
    title_source = clean_title_source(meta)
    abstract_text = extract_abstract_text(meta)
    claims = get_claims(con, patent_id)
    independent_claims = [
        c for c in claims
        if c.get("claim_type") == "independent" and normalize_claim_no(c.get("claim_no", ""))
    ]
    recovered_claims = recover_front_claims_from_meta(meta, [str(c.get("claim_no", "")) for c in independent_claims])
    if recovered_claims:
        independent_claims = recovered_claims + independent_claims
        independent_claims.sort(key=lambda x: claim_sort_key(x.get("claim_no")))
        log(f"      · recovered_independent_claims={','.join([c['claim_no'] for c in recovered_claims])}")
    independent_claims = [c for c in independent_claims if normalize_claim_no(c.get("claim_no", ""))]
    non_dependent_claims = [c for c in independent_claims if not looks_dependent_claim(c.get("raw_text", ""))]
    if non_dependent_claims:
        independent_claims = non_dependent_claims
    if not independent_claims:
        raise RuntimeError("independent claims not found")

    claim_ref_counts = get_claim_ref_counts(con, patent_id)
    figures = get_figures(con, patent_id, limit=12)
    snippets = get_description_snippets(con, patent_id, claims, limit=10)
    front_matter_text = get_front_matter_text(con, patent_id, limit=2)

    primary_claim_type = choose_primary_claim_type(independent_claims)
    if primary_claim_type == "unknown":
        primary_claim_type = infer_claim_type_from_text(
            " ".join([title_source, abstract_text] + [c.get("raw_text", "") for c in independent_claims])
        )
    secondary_claim_types = collect_secondary_claim_types(independent_claims)
    if primary_claim_type != "unknown":
        secondary_claim_types = unique_keep_order(
            [primary_claim_type] + [x for x in secondary_claim_types if x != "unknown"]
        )
    core_terms, context_terms_not_core = technical_term_roles_for_language(
        source_language,
        [meta.get("title_raw", ""), title_source]
        + [c.get("raw_text", "") for c in independent_claims]
        + [s.get("text", "") for s in snippets],
    )
    label_context_texts = [
        title_source,
        abstract_text,
        front_matter_text,
    ] + [c.get("raw_text", "")[:1400] for c in independent_claims]
    candidate_problem_labels = infer_candidate_problem_labels(snippets, label_context_texts)
    candidate_solution_labels = infer_candidate_solution_labels(independent_claims, title_source, core_terms)
    candidate_effect_labels = infer_candidate_effect_labels(snippets, figures, label_context_texts)
    candidate_problem_labels = unique_keep_order(
        candidate_problem_labels + derive_problem_labels_from_effects(candidate_effect_labels)
    )[:6]
    candidate_evidence_ids = seed_evidence_ids(independent_claims, figures, snippets)
    independent_claim_nos = unique_keep_order([
        normalize_claim_no(c.get("claim_no", ""))
        for c in independent_claims
        if normalize_claim_no(c.get("claim_no", ""))
    ])

    log(
        f"      · independent_claims={len(independent_claims)}, claim_ref_counts={len(claim_ref_counts)}, "
        f"figures={len(figures)}, snippets={len(snippets)}, primary_claim_type={primary_claim_type}, "
        f"secondary_claim_types={secondary_claim_types}, source_language={source_language}, "
        f"independent_claim_nos={independent_claim_nos}, core_terms={core_terms}, "
        f"context_terms_not_core={context_terms_not_core}"
    )

    prompt = build_prompt(
        meta,
        independent_claims,
        claim_ref_counts,
        figures,
        snippets,
        primary_claim_type,
        candidate_problem_labels,
        candidate_solution_labels,
        candidate_effect_labels,
        candidate_evidence_ids,
        source_language,
        title_source,
        independent_claim_nos,
        core_terms,
        context_terms_not_core,
    )

    start = time.time()
    llm_part, meta_info = call_ollama_minimal(prompt, patent_id)
    if not normalize_ws(llm_part.get("core_subject", "")) and title_source:
        llm_part["core_subject"] = title_source
        meta_info["core_subject_fallback"] = "title_source"
    if not llm_part.get("core_elements"):
        llm_part["core_elements"] = unique_keep_order(
            core_terms + fallback_core_elements_from_title(title_source, source_language)
        )[:6]
        meta_info["core_elements_fallback"] = "title_source"
    validate_llm_part(llm_part)

    final = build_final_minimal(
        meta,
        llm_part,
        primary_claim_type,
        secondary_claim_types,
        candidate_evidence_ids,
        source_language,
        title_source,
        independent_claim_nos,
        core_terms,
        context_terms_not_core,
        candidate_problem_labels,
        candidate_solution_labels,
        candidate_effect_labels,
    )
    validate_final(final, patent_id)

    final["_build_meta"] = {
        "model": MODEL,
        "wall_seconds": meta_info["wall_seconds"],
        "ollama_total_seconds": meta_info["ollama_total_seconds"],
        "prompt_eval_count": meta_info["prompt_eval_count"],
        "eval_count": meta_info["eval_count"],
        "done_reason": meta_info["done_reason"],
        "core_subject_fallback": meta_info.get("core_subject_fallback"),
        "core_elements_fallback": meta_info.get("core_elements_fallback"),
    }

    save_json(out_path, final)

    return {
        "patent_id": patent_id,
        "output_path": str(out_path),
        "elapsed": time.time() - start,
        "loaded_from_cache": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate minimal indexing JSON from evidence DB.")
    parser.add_argument("--limit", type=int, default=1, help="How many patents to process")
    parser.add_argument("--patent-id", type=str, default=None, help="Specific patent_id to process")
    parser.add_argument("--patent-list-file", type=str, default=None, help="Text file with one patent_id per line")
    parser.add_argument("--skip", type=int, default=0, help="Skip this many targets from the ordered batch list")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if output exists")
    args = parser.parse_args()

    ensure_dirs()
    init_log_file()

    log("[로그 시작]")
    log(f"[설정] DB: {A4_DB}")
    log(f"[설정] output_dir: {MINIMAL_DIR}")
    log(f"[설정] model: {MODEL}")
    log(f"[설정] limit={args.limit}, patent_id={args.patent_id or '-'}, overwrite={args.overwrite}")

    con = open_db()
    counts = get_job_counts(con)
    log(f"[현황] jobs 상태: {counts}")

    processed = 0
    failed = 0
    total_start = time.time()

    if args.patent_list_file:
        with open(args.patent_list_file, "r", encoding="utf-8") as f:
            target_patent_ids = [
                normalize_ws(line)
                for line in f
                if normalize_ws(line) and not normalize_ws(line).startswith("#")
            ]
    else:
        target_patent_ids = fetch_patent_ids(con, args.patent_id, args.limit)
    if not args.patent_id and not args.patent_list_file and not args.overwrite:
        before_filter = len(target_patent_ids)
        target_patent_ids = [pid for pid in target_patent_ids if not minimal_output_exists(pid)]
        log(f"[대상] existing minimal 제외: {before_filter - len(target_patent_ids)}")
    if args.skip and not args.patent_id:
        target_patent_ids = target_patent_ids[args.skip:]
    if args.limit and not args.patent_id:
        target_patent_ids = target_patent_ids[:args.limit]
    log(f"[대상] patents={len(target_patent_ids)}, skip={args.skip}")
    if not target_patent_ids:
        log("[종료] 처리할 대상이 없습니다.")

    try:
        for patent_id in target_patent_ids:
            log("")
            log(f"[처리중] {processed + failed + 1}/{len(target_patent_ids)} patent_id={patent_id}")
            item_start = time.time()

            try:
                result = process_one_patent(con, patent_id, overwrite=args.overwrite)
                processed += 1
                log(f"    ✓ 완료: {result['patent_id']}")
                log(f"      · output_json: {result['output_path']}")
                log(f"      · 소요 시간: {human_seconds(result['elapsed'])}")
            except Exception as e:
                failed += 1
                log(f"    ✗ 실패: {patent_id}")
                log(f"      오류: {e}")
                log(f"      · 소요 시간: {human_seconds(time.time() - item_start)}")
    finally:
        con.close()

    total_elapsed = time.time() - total_start
    log("")
    log(f"[종료] 성공: {processed}, 실패: {failed}, 총 소요: {human_seconds(total_elapsed)}")
    log("[로그 종료]")


if __name__ == "__main__":
    main()
