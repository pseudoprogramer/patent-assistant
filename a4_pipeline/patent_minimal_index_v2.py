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
    return re.sub(r"\s+", " ", (s or "")).strip()


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
    con = sqlite3.connect(A4_DB, timeout=30)
    con.row_factory = sqlite3.Row
    return con


def claim_sort_key(v: Any) -> Tuple[int, str]:
    text = str(v or "")
    m = re.match(r"(\d+)", text)
    return (int(m.group(1)) if m else 10**9, text)


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


def normalize_tag(text: str) -> str:
    s = (text or "").strip().lower()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


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


def source_language_from_country(country: Any, patent_id: Any) -> str:
    code = normalize_ws(country).upper() or normalize_ws(patent_id)[:2].upper()
    return {"CN": "zh", "US": "en", "KR": "ko"}.get(code, "unknown")


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
        WHERE status='evidence_done'
        ORDER BY updated_at ASC, patent_id ASC
        LIMIT 1
        """
    ).fetchone()
    return row["patent_id"] if row else None


def mark_job_status(con: sqlite3.Connection, patent_id: str, status: str) -> None:
    con.execute(
        """
        UPDATE jobs
        SET status=?, updated_at=?
        WHERE patent_id=?
        """,
        (status, datetime.now().isoformat(timespec="seconds"), patent_id),
    )
    con.commit()


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


# ---------------- heuristics ----------------

def choose_primary_claim_type(independent_claims: List[Dict[str, Any]]) -> str:
    text = " ".join([c.get("raw_text", "")[:600] for c in independent_claims]).lower()
    if re.search(r"(方法|method|방법)", text, flags=re.I):
        return "method"
    if re.search(r"(device|apparatus|장치|memory device|半导体装置|存储器装置)", text, flags=re.I):
        return "device"
    if re.search(r"(system|系统|시스템)", text, flags=re.I):
        return "system"
    if re.search(r"(process|공정|制造方法|제조방법)", text, flags=re.I):
        return "process"
    return "unknown"


def collect_secondary_claim_types(independent_claims: List[Dict[str, Any]]) -> List[str]:
    out = []
    seen = set()
    for c in independent_claims:
        raw = c.get("raw_text", "")
        label = "unknown"
        if re.search(r"(方法|method|방법)", raw, flags=re.I):
            label = "method"
        elif re.search(r"(device|apparatus|장치|memory device|半导体装置|存储器装置)", raw, flags=re.I):
            label = "device"
        elif re.search(r"(system|系统|시스템)", raw, flags=re.I):
            label = "system"
        elif re.search(r"(process|공정|制造方法|제조방법)", raw, flags=re.I):
            label = "process"
        if label not in seen:
            seen.add(label)
            out.append(label)
    return out


def infer_candidate_problem_labels(snippets: List[Dict[str, Any]]) -> List[str]:
    text = " ".join([s["text"] for s in snippets[:8]])
    labels = []
    if re.search(r"(功耗|전력|power)", text, flags=re.I):
        labels.append("power_consumption")
    if re.search(r"(延迟|지연|latency)", text, flags=re.I):
        labels.append("data_transfer_latency")
    if re.search(r"(带宽|bandwidth|대역폭)", text, flags=re.I):
        labels.append("bandwidth_limit")
    if re.search(r"(多路复用|mux|multiplex)", text, flags=re.I):
        labels.append("mux_complexity")
    if re.search(r"(地址|address)", text, flags=re.I):
        labels.append("address_handling_complexity")
    return unique_keep_order(labels)


def infer_candidate_solution_labels(independent_claims: List[Dict[str, Any]]) -> List[str]:
    text = " ".join([c.get("raw_text", "")[:1200] for c in independent_claims])
    labels = []
    if re.search(r"(平面专用焊盘组|plane dedicated pad|전용 패드)", text, flags=re.I):
        labels.append("plane_dedicated_pad_group")
    if re.search(r"(多个存储器平面|memory plane|메모리 평면)", text, flags=re.I):
        labels.append("multi_plane_architecture")
    if re.search(r"(数据路径|data path)", text, flags=re.I):
        labels.append("separate_data_path")
    if re.search(r"(页缓冲器|page buffer)", text, flags=re.I):
        labels.append("page_buffer_per_plane")
    if re.search(r"(不接收平面地址|no plane address|평면 주소)", text, flags=re.I):
        labels.append("no_plane_address_input")
    return unique_keep_order(labels)


def infer_candidate_effect_labels(snippets: List[Dict[str, Any]], figures: List[Dict[str, Any]]) -> List[str]:
    text = " ".join([s["text"] for s in snippets[:10]]) + " " + " ".join([f.get("caption_raw", "") for f in figures[:6]])
    labels = []
    if re.search(r"(降低功耗|low power|전력)", text, flags=re.I):
        labels.append("power_reduction")
    if re.search(r"(带宽|bandwidth|대역폭)", text, flags=re.I):
        labels.append("bandwidth_increase")
    if re.search(r"(并行|parallel|병렬)", text, flags=re.I):
        labels.append("parallel_data_transfer")
    if re.search(r"(延迟|latency|지연)", text, flags=re.I):
        labels.append("latency_reduction")
    return unique_keep_order(labels)


def seed_evidence_ids(independent_claims: List[Dict[str, Any]], figures: List[Dict[str, Any]], snippets: List[Dict[str, Any]]) -> List[str]:
    ids = []
    for c in independent_claims[:4]:
        ids.append(f"claim_{c.get('claim_no','')}")
    for f in figures[:4]:
        ids.append(f"fig_{f.get('figure_no','')}")
    for s in snippets[:4]:
        ids.append(f"page_{s.get('page_no','')}")
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
            "core_subject": {"type": "string", "maxLength": 90},
            "core_elements_ko": {"type": "array", "items": {"type": "string", "maxLength": 48}, "maxItems": 8},
            "problem_labels": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 6},
            "solution_labels": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 8},
            "effect_labels": {"type": "array", "items": {"type": "string", "maxLength": 40}, "maxItems": 6},
            "evidence_ids": {"type": "array", "items": {"type": "string", "maxLength": 32}, "maxItems": 12},
            "confidence": {"type": "number"},
        },
        "required": [
            "core_subject",
            "core_elements_ko",
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
) -> str:
    schema_str = json.dumps(minimal_llm_schema(), ensure_ascii=False, indent=2)

    lines = [
        "너는 특허를 빠르게 검색/비교/전략 연결하기 위한 최소 인덱스 JSON을 만드는 분석기다.",
        "짧고 비교 가능한 값만 출력하라.",
        "추정 금지. 아래 evidence에 직접 드러난 내용만 사용하라.",
        "긴 원문을 다시 복사하지 말라.",
        "problem_labels, solution_labels, effect_labels는 반드시 lower_snake_case 짧은 태그로 써라.",
        "가능하면 아래 candidate label 중에서만 선택하라.",
        "evidence_ids는 아래 candidate_evidence_ids 안에서만 선택하라.",
        "출력은 반드시 JSON만 반환하라.",
        "",
        "schema:",
        schema_str,
        "",
        f"metadata: country={meta.get('country','')} title_raw={meta.get('title_raw','')[:180]}",
        f"claim_type_hint: {claim_type_hint}",
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

    try:
        parsed = json.loads(candidate)
    except Exception as e:
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
                    "error": str(e),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        raise RuntimeError(f"minimal invalid JSON: {e}")

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
    if not result.get("core_elements_ko"):
        raise RuntimeError("minimal empty core_elements_ko")
    if not result.get("solution_labels"):
        raise RuntimeError("minimal empty solution_labels")
    if not result.get("evidence_ids"):
        raise RuntimeError("minimal empty evidence_ids")


def build_final_minimal(
    meta: Dict[str, Any],
    llm_part: Dict[str, Any],
    primary_claim_type: str,
    secondary_claim_types: List[str],
    candidate_evidence_ids: List[str],
) -> Dict[str, Any]:
    valid_ids = set(candidate_evidence_ids)
    title = clean_title(meta.get("title_raw", ""))
    source_language = source_language_from_country(meta.get("country"), meta.get("patent_id"))

    evidence_ids = []
    for x in llm_part.get("evidence_ids", []):
        s = normalize_ws(x)
        if s in valid_ids and s not in evidence_ids:
            evidence_ids.append(s)

    if not evidence_ids:
        evidence_ids = candidate_evidence_ids[:6]

    final = {
        "patent_id": meta["patent_id"],
        "country": meta.get("country", ""),
        "source_language": source_language,
        "title_source": title,
        "title_ko": title,
        "primary_claim_type": primary_claim_type,
        "secondary_claim_types": secondary_claim_types,
        "core_subject": normalize_ws(llm_part.get("core_subject", "")),
        "core_elements_ko": unique_keep_order(llm_part.get("core_elements_ko", []))[:8],
        "problem_labels": normalize_tags(llm_part.get("problem_labels", []), 6),
        "solution_labels": normalize_tags(llm_part.get("solution_labels", []), 8),
        "effect_labels": normalize_tags(llm_part.get("effect_labels", []), 6),
        "evidence_ids": evidence_ids[:12],
        "confidence": llm_part.get("confidence", 0.0),
    }
    return final


def validate_final(result: Dict[str, Any], patent_id: str) -> None:
    if str(result.get("patent_id", "")) != patent_id:
        raise RuntimeError("final patent_id mismatch")
    if not normalize_ws(result.get("title_source", "")):
        raise RuntimeError("final empty title_source")
    if re.search(r"(权利要求|청구항|Claims?:|Abstract|摘要|발명의 설명)", str(result.get("title_source", "")), flags=re.I):
        raise RuntimeError("final contaminated title_source")
    if not normalize_ws(result.get("source_language", "")):
        raise RuntimeError("final empty source_language")
    if not normalize_ws(result.get("core_subject", "")):
        raise RuntimeError("final empty core_subject")
    if not result.get("core_elements_ko"):
        raise RuntimeError("final empty core_elements_ko")
    if not result.get("solution_labels"):
        raise RuntimeError("final empty solution_labels")
    if not result.get("evidence_ids"):
        raise RuntimeError("final empty evidence_ids")
    if not result.get("primary_claim_type"):
        raise RuntimeError("final empty primary_claim_type")


def process_one_patent(con: sqlite3.Connection, patent_id: str, overwrite: bool = False) -> Dict[str, Any]:
    out_path = MINIMAL_DIR / f"{safe_name(patent_id)}.minimal.json"
    if out_path.exists() and not overwrite:
        try:
            cached = json.loads(out_path.read_text(encoding="utf-8"))
            validate_final(cached, patent_id)
        except Exception as exc:
            log(f"[캐시 무효] {out_path.name}: {exc}; regenerate")
        else:
            log(f"[캐시 사용] {out_path.name}")
            mark_job_status(con, patent_id, "brief_done")
            return {
                "patent_id": patent_id,
                "output_path": str(out_path),
                "elapsed": 0.0,
                "loaded_from_cache": True,
            }

    meta = get_patent_meta(con, patent_id)
    claims = get_claims(con, patent_id)
    independent_claims = [c for c in claims if c.get("claim_type") == "independent"]
    if not independent_claims:
        raise RuntimeError("independent claims not found")

    claim_ref_counts = get_claim_ref_counts(con, patent_id)
    figures = get_figures(con, patent_id, limit=12)
    snippets = get_description_snippets(con, patent_id, claims, limit=10)

    primary_claim_type = choose_primary_claim_type(independent_claims)
    secondary_claim_types = collect_secondary_claim_types(independent_claims)
    candidate_problem_labels = infer_candidate_problem_labels(snippets)
    candidate_solution_labels = infer_candidate_solution_labels(independent_claims)
    candidate_effect_labels = infer_candidate_effect_labels(snippets, figures)
    candidate_evidence_ids = seed_evidence_ids(independent_claims, figures, snippets)

    log(
        f"      · independent_claims={len(independent_claims)}, claim_ref_counts={len(claim_ref_counts)}, "
        f"figures={len(figures)}, snippets={len(snippets)}, primary_claim_type={primary_claim_type}, "
        f"secondary_claim_types={secondary_claim_types}"
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
    )

    start = time.time()
    llm_part, meta_info = call_ollama_minimal(prompt, patent_id)
    validate_llm_part(llm_part)

    final = build_final_minimal(
        meta,
        llm_part,
        primary_claim_type,
        secondary_claim_types,
        candidate_evidence_ids,
    )
    validate_final(final, patent_id)

    final["_build_meta"] = {
        "model": MODEL,
        "wall_seconds": meta_info["wall_seconds"],
        "ollama_total_seconds": meta_info["ollama_total_seconds"],
        "prompt_eval_count": meta_info["prompt_eval_count"],
        "eval_count": meta_info["eval_count"],
        "done_reason": meta_info["done_reason"],
    }

    save_json(out_path, final)
    mark_job_status(con, patent_id, "brief_done")

    return {
        "patent_id": patent_id,
        "output_path": str(out_path),
        "elapsed": time.time() - start,
        "loaded_from_cache": False,
    }


def main() -> None:
    global A4_DB, A4_LOGS, A4_RAW_INVALID, MINIMAL_DIR, MODEL

    parser = argparse.ArgumentParser(description="Generate minimal indexing JSON from evidence DB.")
    parser.add_argument("--limit", type=int, default=1, help="How many patents to process")
    parser.add_argument("--patent-id", type=str, default=None, help="Specific patent_id to process")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate even if output exists")
    parser.add_argument("--db", default=str(A4_DB), help="Evidence SQLite DB path")
    parser.add_argument("--output-dir", default=str(MINIMAL_DIR), help="Minimal JSON output directory")
    parser.add_argument("--log-dir", default=str(A4_LOGS), help="Log directory")
    parser.add_argument("--raw-invalid-dir", default=str(A4_RAW_INVALID), help="Raw invalid output directory")
    parser.add_argument("--model", default=MODEL, help="Ollama model name")
    args = parser.parse_args()

    A4_DB = Path(args.db)
    MINIMAL_DIR = Path(args.output_dir)
    A4_LOGS = Path(args.log_dir)
    A4_RAW_INVALID = Path(args.raw_invalid_dir)
    MODEL = args.model

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

    try:
        while processed + failed < args.limit:
            patent_id = fetch_next_patent_id(con, args.patent_id if processed + failed == 0 else None)
            if not patent_id:
                log("[종료] 처리할 대상이 없습니다.")
                break

            log("")
            log(f"[처리중] patent_id={patent_id}")
            item_start = time.time()

            try:
                result = process_one_patent(con, patent_id, overwrite=args.overwrite)
                processed += 1
                log(f"    ✓ 완료: {result['patent_id']}")
                log(f"      · output_json: {result['output_path']}")
                log(f"      · 소요 시간: {human_seconds(result['elapsed'])}")
            except Exception as e:
                failed += 1
                try:
                    mark_job_status(con, patent_id, "minimal_failed")
                except Exception as mark_error:
                    log(f"      상태 업데이트 실패: {mark_error}")
                log(f"    ✗ 실패: {patent_id}")
                log(f"      오류: {e}")
                log(f"      · 소요 시간: {human_seconds(time.time() - item_start)}")

            if args.patent_id:
                break
    finally:
        con.close()

    total_elapsed = time.time() - total_start
    log("")
    log(f"[종료] 성공: {processed}, 실패: {failed}, 총 소요: {human_seconds(total_elapsed)}")
    log("[로그 종료]")


if __name__ == "__main__":
    main()
