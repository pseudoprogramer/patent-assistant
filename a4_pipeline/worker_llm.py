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

from db_schema import get_connection, upsert_job, increment_job_retry

try:
    from config import A4_DB, A4_BRIEFS, A4_LOGS, A4_RAW_INVALID
except Exception:
    BASE = Path("/Volumes/외장 2TB/cpu2026")
    HUB = BASE / "patent_hub"
    COMMON = BASE / "common"
    A4_DB = COMMON / "runtime" / "db" / "patent_A4.sqlite"
    A4_BRIEFS = HUB / "outputs" / "briefs" / "A4"
    A4_LOGS = COMMON / "runtime" / "logs" / "A4"
    A4_RAW_INVALID = COMMON / "runtime" / "raw_invalid" / "A4"

A4_DB = Path(A4_DB)
A4_BRIEFS = Path(A4_BRIEFS)
A4_LOGS = Path(A4_LOGS)
A4_RAW_INVALID = Path(A4_RAW_INVALID)

PASS_A_DIR = A4_BRIEFS / "pass_a"
PASS_B_DIR = A4_BRIEFS / "pass_b"
PASS_B_CHUNKS_DIR = A4_BRIEFS / "pass_b_chunks"

OLLAMA_URL = "http://localhost:11434/api/generate"
LLM_MODEL = "qwen3:14b"
KEEP_ALIVE = -1
TEMPERATURE = 0.03
SEED = 42
TIMEOUT = (20, 1200)

PASS_A_CHAR_BUDGET_FULL = 18000
PASS_A_CHAR_BUDGET_LITE = 12000
PASS_B_CHUNK_CHAR_BUDGET_FULL = 7000
PASS_B_CHUNK_CHAR_BUDGET_LITE = 5000
PASS_B_MERGE_CHAR_BUDGET_FULL = 15000
PASS_B_MERGE_CHAR_BUDGET_LITE = 10000

LOG_FILE_PATH = None


def ensure_dirs() -> None:
    for p in [A4_BRIEFS, PASS_A_DIR, PASS_B_DIR, PASS_B_CHUNKS_DIR, A4_LOGS, A4_RAW_INVALID]:
        Path(p).mkdir(parents=True, exist_ok=True)


def init_log_file() -> None:
    global LOG_FILE_PATH
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE_PATH = A4_LOGS / f"llm_worker_{ts}.log"


def log(msg: str) -> None:
    line = "[LLM] " + msg
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
    s = sec - m * 60
    return f"{m}분 {s:.1f}초" if m > 0 else f"{s:.1f}초"


def unique_keep_order(items: List[str]) -> List[str]:
    out = []
    seen = set()
    for x in items:
        s = normalize_ws(x)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def clip_text(value: Any, max_chars: int) -> str:
    return normalize_ws(str(value or ""))[:max_chars]


def compact_string_list(value: Any, max_items: int, max_chars: int) -> List[str]:
    if not isinstance(value, list):
        return []
    items = [clip_text(x, max_chars) for x in value]
    return unique_keep_order([x for x in items if x])[:max_items]


def compact_llm_result(result: Dict[str, Any], pass_name: str, patent_id: str) -> Dict[str, Any]:
    """Keep parsed model JSON short and deterministic before saving or merging."""
    if not isinstance(result, dict):
        return {}

    if pass_name == "pass_a":
        result["patent_id"] = patent_id
        for key, max_chars in {
            "source_language": 24,
            "title_source": 160,
            "title_ko": 120,
            "one_line_summary_ko": 180,
            "technical_field_ko": 90,
            "problem_to_solve_ko": 180,
        }.items():
            result[key] = clip_text(result.get(key, ""), max_chars)
        claims = []
        for item in result.get("representative_independent_claims_ko", [])[:4]:
            if isinstance(item, dict):
                claims.append({
                    "claim_no": clip_text(item.get("claim_no", ""), 16),
                    "gist_ko": clip_text(item.get("gist_ko", ""), 180),
                })
        result["representative_independent_claims_ko"] = claims
        result["claim_core_elements_ko"] = compact_string_list(result.get("claim_core_elements_ko"), 10, 80)
        result["explicit_effects_ko"] = compact_string_list(result.get("explicit_effects_ko"), 8, 90)
        result["notable_numbers_or_conditions_ko"] = compact_string_list(result.get("notable_numbers_or_conditions_ko"), 8, 80)
        result["evidence_anchors_ko"] = compact_string_list(result.get("evidence_anchors_ko"), 10, 80)
    elif pass_name.startswith("pass_b1_"):
        result["chunk_id"] = pass_name.removeprefix("pass_b1_")
        result["operation_steps_ko"] = compact_string_list(result.get("operation_steps_ko"), 6, 90)
        result["structural_elements_ko"] = compact_string_list(result.get("structural_elements_ko"), 8, 80)
        result["implementation_points_ko"] = compact_string_list(result.get("implementation_points_ko"), 6, 90)
        result["figure_signals_ko"] = compact_string_list(result.get("figure_signals_ko"), 6, 90)
        result["reference_signals_ko"] = compact_string_list(result.get("reference_signals_ko"), 6, 90)
        result["evidence_anchors_ko"] = compact_string_list(result.get("evidence_anchors_ko"), 8, 80)
    elif pass_name == "pass_b2_merge":
        result["patent_id"] = patent_id
        result["detailed_operation_ko"] = clip_text(result.get("detailed_operation_ko", ""), 500)
        result["operation_steps_ko"] = compact_string_list(result.get("operation_steps_ko"), 8, 90)
        result["structural_elements_ko"] = compact_string_list(result.get("structural_elements_ko"), 10, 80)
        result["implementation_points_ko"] = compact_string_list(result.get("implementation_points_ko"), 8, 90)
        result["figure_signals_ko"] = compact_string_list(result.get("figure_signals_ko"), 8, 90)
        result["reference_signals_ko"] = compact_string_list(result.get("reference_signals_ko"), 8, 90)
        result["evidence_anchors_ko"] = compact_string_list(result.get("evidence_anchors_ko"), 10, 80)
        result["evidence_gaps_ko"] = compact_string_list(result.get("evidence_gaps_ko"), 5, 90)

    return result


def append_with_budget(lines: List[str], text: str, used: int, budget: int) -> int:
    if used >= budget:
        return used
    remain = budget - used
    snippet = (text or "")[:remain]
    if snippet:
        lines.append(snippet)
        used += len(snippet)
    return used


def get_job_counts(con: sqlite3.Connection) -> Dict[str, int]:
    cur = con.cursor()
    rows = cur.execute("SELECT status, COUNT(*) AS c FROM jobs GROUP BY status").fetchall()
    out = {}
    for r in rows:
        out[r["status"]] = int(r["c"])
    return out


def claim_sort_key(v: Any) -> Tuple[int, str]:
    text = str(v or "")
    m = re.match(r"(\d+)", text)
    return (int(m.group(1)) if m else 10**9, text)


def fetch_next_patent_id(con: sqlite3.Connection, explicit_patent_id: Optional[str] = None) -> Optional[str]:
    cur = con.cursor()
    if explicit_patent_id:
        row = cur.execute("SELECT patent_id FROM jobs WHERE patent_id=?", (explicit_patent_id,)).fetchone()
        return row["patent_id"] if row else None
    row = cur.execute("""
        SELECT patent_id
        FROM jobs
        WHERE status='evidence_done'
        ORDER BY updated_at ASC, patent_id ASC
        LIMIT 1
    """).fetchone()
    return row["patent_id"] if row else None


def mark_status(con: sqlite3.Connection, patent_id: str, status: str) -> None:
    cur = con.cursor()
    row = cur.execute("SELECT pdf_path FROM jobs WHERE patent_id=?", (patent_id,)).fetchone()
    pdf_path = row["pdf_path"] if row else ""
    upsert_job(con, patent_id, pdf_path, status)


def mark_failed(con: sqlite3.Connection, patent_id: str, error_message: str) -> None:
    cur = con.cursor()
    row = cur.execute("SELECT pdf_path FROM jobs WHERE patent_id=?", (patent_id,)).fetchone()
    pdf_path = row["pdf_path"] if row else ""
    increment_job_retry(con, patent_id, pdf_path, error_message)


def get_patent_meta(con: sqlite3.Connection, patent_id: str) -> Dict[str, Any]:
    cur = con.cursor()
    row = cur.execute("""
        SELECT patent_id, country, title_raw, assignee_raw, application_no, publication_no, pdf_path, page_count, parser_version
        FROM patents
        WHERE patent_id=?
    """, (patent_id,)).fetchone()
    if not row:
        raise RuntimeError(f"patents row not found: {patent_id}")
    return dict(row)


def get_claims(con: sqlite3.Connection, patent_id: str) -> List[Dict[str, Any]]:
    cur = con.cursor()
    rows = cur.execute("""
        SELECT claim_no, parent_claim_no, claim_type, raw_text, norm_text, page_start, page_end
        FROM claims
        WHERE patent_id=?
        ORDER BY claim_no
    """, (patent_id,)).fetchall()
    out = [dict(r) for r in rows]
    out.sort(key=lambda x: claim_sort_key(x.get("claim_no")))
    return out


def get_refs(con: sqlite3.Connection, patent_id: str) -> List[Dict[str, Any]]:
    cur = con.cursor()
    rows = cur.execute("""
        SELECT ref_no_raw, COALESCE(label_raw, '') AS label_raw, source_section, page_no, COUNT(*) AS freq
        FROM ref_entities
        WHERE patent_id=?
        GROUP BY ref_no_raw, label_raw, source_section, page_no
        ORDER BY freq DESC, page_no ASC
    """, (patent_id,)).fetchall()
    return [dict(r) for r in rows]


def get_figures(con: sqlite3.Connection, patent_id: str) -> List[Dict[str, Any]]:
    cur = con.cursor()
    rows = cur.execute("""
        SELECT figure_no, caption_raw, caption_norm, page_no
        FROM figure_captions
        WHERE patent_id=?
        ORDER BY page_no ASC, figure_no ASC
    """, (patent_id,)).fetchall()
    return [dict(r) for r in rows]


def get_claim_ref_counts(con: sqlite3.Connection, patent_id: str) -> List[Dict[str, Any]]:
    cur = con.cursor()
    rows = cur.execute("""
        SELECT claim_no, ref_no_raw, COUNT(*) AS freq
        FROM claim_ref_map
        WHERE patent_id=?
        GROUP BY claim_no, ref_no_raw
        ORDER BY claim_no ASC, freq DESC
    """, (patent_id,)).fetchall()
    out = [dict(r) for r in rows]
    out.sort(key=lambda x: (claim_sort_key(x.get("claim_no")), -int(x.get("freq", 0))))
    return out


def get_description_snippets(con: sqlite3.Connection, patent_id: str, claims: List[Dict[str, Any]], limit: int = 9999) -> List[Dict[str, Any]]:
    cur = con.cursor()
    claim_end_pages = [c["page_end"] for c in claims if c.get("page_end")]
    min_claim_end = max(claim_end_pages) if claim_end_pages else 0
    rows = cur.execute("""
        SELECT page_no, raw_text
        FROM text_spans
        WHERE patent_id=?
          AND page_no > ?
          AND raw_text IS NOT NULL
          AND LENGTH(TRIM(raw_text)) BETWEEN 40 AND 1200
        ORDER BY page_no ASC, block_no ASC, line_no ASC, span_no ASC
        LIMIT ?
    """, (patent_id, min_claim_end, limit * 2)).fetchall()

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


def extract_numbers_or_conditions(texts: List[str], limit: int = 30) -> List[str]:
    pats = [
        r"\b\d+(?:\.\d+)?\s?(?:nm|um|μm|mm|cm|V|mV|A|mA|Ω|ohm|kΩ|MHz|GHz|kHz|ns|us|µs|ms|s|%)\b",
        r"\bW\d+\b",
        r"\bS\d+\b",
        r"\b\d+\s*(?:to|~|～|-)\s*\d+\b",
    ]
    out, seen = [], set()
    for txt in texts:
        for pat in pats:
            for m in re.findall(pat, txt, flags=re.I):
                s = normalize_ws(m)
                if s and s not in seen:
                    seen.add(s)
                    out.append(s)
                    if len(out) >= limit:
                        return out
    return out


def build_evidence_bundle(con: sqlite3.Connection, patent_id: str) -> Dict[str, Any]:
    meta = get_patent_meta(con, patent_id)
    claims = get_claims(con, patent_id)
    refs = get_refs(con, patent_id)
    figures = get_figures(con, patent_id)
    claim_ref_counts = get_claim_ref_counts(con, patent_id)
    snippets = get_description_snippets(con, patent_id, claims)
    independent_claims = [c for c in claims if c.get("claim_type") == "independent"]
    dependent_claims = [c for c in claims if c.get("claim_type") != "independent"]
    numbers = extract_numbers_or_conditions([c.get("raw_text", "") for c in claims] + [s.get("text", "") for s in snippets], 30)
    return {
        "meta": meta,
        "claims": claims,
        "independent_claims": independent_claims,
        "dependent_claims": dependent_claims,
        "refs": refs,
        "figures": figures,
        "claim_ref_counts": claim_ref_counts,
        "snippets": snippets,
        "numbers_or_conditions": numbers,
    }


def pass_a_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "source_language": {"type": "string", "maxLength": 24},
            "patent_id": {"type": "string"},
            "title_source": {"type": "string", "maxLength": 160},
            "title_ko": {"type": "string", "maxLength": 120},
            "one_line_summary_ko": {"type": "string", "maxLength": 180},
            "technical_field_ko": {"type": "string", "maxLength": 90},
            "problem_to_solve_ko": {"type": "string", "maxLength": 180},
            "representative_independent_claims_ko": {"type": "array", "items": {"type": "object", "properties": {"claim_no": {"type": "string", "maxLength": 16}, "gist_ko": {"type": "string", "maxLength": 180}}, "required": ["claim_no", "gist_ko"]}, "maxItems": 4},
            "claim_core_elements_ko": {"type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 10},
            "explicit_effects_ko": {"type": "array", "items": {"type": "string", "maxLength": 90}, "maxItems": 8},
            "notable_numbers_or_conditions_ko": {"type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 8},
            "evidence_anchors_ko": {"type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 10},
            "extraction_confidence": {"type": "number"},
        },
        "required": ["source_language", "patent_id", "title_source", "title_ko", "one_line_summary_ko", "technical_field_ko", "problem_to_solve_ko", "representative_independent_claims_ko", "claim_core_elements_ko", "explicit_effects_ko", "notable_numbers_or_conditions_ko", "evidence_anchors_ko", "extraction_confidence"],
    }


def pass_b_chunk_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "chunk_id": {"type": "string", "maxLength": 64},
            "operation_steps_ko": {"type": "array", "items": {"type": "string", "maxLength": 90}, "maxItems": 6},
            "structural_elements_ko": {"type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 8},
            "implementation_points_ko": {"type": "array", "items": {"type": "string", "maxLength": 90}, "maxItems": 6},
            "figure_signals_ko": {"type": "array", "items": {"type": "string", "maxLength": 90}, "maxItems": 6},
            "reference_signals_ko": {"type": "array", "items": {"type": "string", "maxLength": 90}, "maxItems": 6},
            "evidence_anchors_ko": {"type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 8},
            "chunk_confidence": {"type": "number"},
        },
        "required": ["chunk_id", "operation_steps_ko", "structural_elements_ko", "implementation_points_ko", "figure_signals_ko", "reference_signals_ko", "evidence_anchors_ko", "chunk_confidence"],
    }


def pass_b_merge_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "patent_id": {"type": "string"},
            "detailed_operation_ko": {"type": "string", "maxLength": 500},
            "operation_steps_ko": {"type": "array", "items": {"type": "string", "maxLength": 90}, "maxItems": 8},
            "structural_elements_ko": {"type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 10},
            "implementation_points_ko": {"type": "array", "items": {"type": "string", "maxLength": 90}, "maxItems": 8},
            "figure_signals_ko": {"type": "array", "items": {"type": "string", "maxLength": 90}, "maxItems": 8},
            "reference_signals_ko": {"type": "array", "items": {"type": "string", "maxLength": 90}, "maxItems": 8},
            "evidence_anchors_ko": {"type": "array", "items": {"type": "string", "maxLength": 80}, "maxItems": 10},
            "evidence_gaps_ko": {"type": "array", "items": {"type": "string", "maxLength": 90}, "maxItems": 5},
            "extraction_confidence": {"type": "number"},
        },
        "required": ["patent_id", "detailed_operation_ko", "operation_steps_ko", "structural_elements_ko", "implementation_points_ko", "figure_signals_ko", "reference_signals_ko", "evidence_anchors_ko", "evidence_gaps_ko", "extraction_confidence"],
    }


def extract_json_candidate(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    first = t.find("{")
    last = t.rfind("}")
    return t[first:last + 1] if first != -1 and last != -1 and last > first else t


def call_ollama_json(prompt: str, schema: Dict[str, Any], pass_name: str, patent_id: str, mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    num_ctx = 12288 if mode == "full" else 8192
    num_predict = 2200 if mode == "full" else 1400
    prompt_chars = len(prompt)
    log(
        f"      · LLM 호출 시작: pass={pass_name}, mode={mode}, "
        f"prompt_chars={prompt_chars}, num_ctx={num_ctx}, num_predict={num_predict}"
    )
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "format": schema,
        "options": {"temperature": TEMPERATURE, "seed": SEED, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    wall_start = time.time()
    r = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    data = r.json()
    wall_seconds = time.time() - wall_start
    raw_text = data.get("response", "")
    log(
        f"      · LLM 응답 완료: pass={pass_name}, mode={mode}, "
        f"wall={human_seconds(wall_seconds)}, done_reason={data.get('done_reason')}, "
        f"prompt_eval={data.get('prompt_eval_count')}, eval={data.get('eval_count')}, raw_chars={len(raw_text)}"
    )
    candidate = extract_json_candidate(raw_text)
    try:
        parsed = json.loads(candidate)
    except Exception as e:
        raw_path = A4_RAW_INVALID / f"{safe_name(patent_id)}.{pass_name}.{mode}.raw.txt"
        meta_path = A4_RAW_INVALID / f"{safe_name(patent_id)}.{pass_name}.{mode}.meta.json"
        raw_path.write_text(raw_text, encoding="utf-8")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "pass_name": pass_name,
                "mode": mode,
                "done_reason": data.get("done_reason"),
                "total_duration": data.get("total_duration"),
                "prompt_eval_count": data.get("prompt_eval_count"),
                "eval_count": data.get("eval_count"),
                "error": str(e),
            }, f, ensure_ascii=False, indent=2)
        raise RuntimeError(f"{pass_name} invalid JSON ({mode}): {e}")
    parsed = compact_llm_result(parsed, pass_name, patent_id)
    meta = {
        "pass_name": pass_name,
        "attempt_mode": mode,
        "wall_seconds": round(wall_seconds, 2),
        "ollama_total_seconds": round(float(data.get("total_duration", 0)) / 1e9, 2) if data.get("total_duration") else None,
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "done_reason": data.get("done_reason"),
    }
    return parsed, meta


def build_pass_a_prompt(bundle: Dict[str, Any], mode: str) -> str:
    meta = bundle["meta"]
    independent = bundle["independent_claims"]
    dependent = bundle["dependent_claims"]
    claim_ref_counts = bundle["claim_ref_counts"]
    figures = bundle["figures"]
    snippets = bundle["snippets"]
    nums = bundle["numbers_or_conditions"]
    budget = PASS_A_CHAR_BUDGET_FULL if mode == "full" else PASS_A_CHAR_BUDGET_LITE

    lines, used = [], 0
    header = [
        "당신은 특허 evidence DB를 읽고 claim/효과 중심 fact JSON을 만드는 분석기다.",
        "추정 금지. evidence에 직접 드러난 내용만 정리하라.",
        "모든 독립항은 가능한 한 반영하라.",
        "반복 금지. 같은 의미의 항목을 다시 쓰지 말라.",
        "각 배열은 최대 8개 이내로, 각 문자열은 짧은 한 문장으로 써라.",
        "출력은 반드시 JSON만 반환하라.",
        "",
        f"patent_id: {meta['patent_id']}",
        f"country: {meta.get('country','')}",
        f"title_raw: {meta.get('title_raw','')}",
        f"assignee_raw: {meta.get('assignee_raw','')}",
        f"application_no: {meta.get('application_no','')}",
        f"publication_no: {meta.get('publication_no','')}",
        "",
        "schema:",
        json.dumps(pass_a_schema(), ensure_ascii=False, indent=2),
        "",
        "[independent_claims]",
    ]
    for h in header:
        used = append_with_budget(lines, h + "\n", used, budget)
    for c in independent:
        used = append_with_budget(lines, f"- claim_no={c.get('claim_no','')}\n{c.get('raw_text','')[:2600]}\n", used, budget)
        if used >= budget:
            break

    if used < budget and dependent:
        used = append_with_budget(lines, "\n[representative_dependent_claims]\n", used, budget)
        seen_parents = set()
        for c in dependent:
            parent = str(c.get("parent_claim_no", "") or "")
            if parent and parent in seen_parents:
                continue
            if parent:
                seen_parents.add(parent)
            used = append_with_budget(lines, f"- claim_no={c.get('claim_no','')} | parent={parent}\n{c.get('raw_text','')[:1400]}\n", used, budget)
            if used >= budget:
                break

    if used < budget and claim_ref_counts:
        used = append_with_budget(lines, "\n[claim_ref_counts]\n", used, budget)
        for r in claim_ref_counts[:100]:
            used = append_with_budget(lines, f"- claim {r.get('claim_no','')} -> {r.get('ref_no_raw','')} (freq={r.get('freq',1)})\n", used, budget)
            if used >= budget:
                break

    if used < budget and figures:
        used = append_with_budget(lines, "\n[figure_captions]\n", used, budget)
        for f in figures[:20]:
            used = append_with_budget(lines, f"- figure={f.get('figure_no','')} | page={f.get('page_no','')} | caption={normalize_ws(f.get('caption_raw',''))[:300]}\n", used, budget)
            if used >= budget:
                break

    if used < budget and nums:
        used = append_with_budget(lines, "\n[numbers_or_conditions]\n", used, budget)
        for n in nums[:20]:
            used = append_with_budget(lines, f"- {n}\n", used, budget)
            if used >= budget:
                break

    if used < budget and snippets:
        used = append_with_budget(lines, "\n[supporting_snippets]\n", used, budget)
        for s in snippets[:15]:
            used = append_with_budget(lines, f"- page={s.get('page_no','')} | {s.get('text','')[:500]}\n", used, budget)
            if used >= budget:
                break

    return "".join(lines).strip()


def group_snippets_into_chunks(bundle: Dict[str, Any], mode: str) -> List[Dict[str, Any]]:
    snippets = bundle["snippets"]
    refs = bundle["refs"]
    figures = bundle["figures"]
    independent = bundle["independent_claims"]
    chunk_budget = PASS_B_CHUNK_CHAR_BUDGET_FULL if mode == "full" else PASS_B_CHUNK_CHAR_BUDGET_LITE

    chunks = []
    current = {"snippet_items": [], "pages": set(), "chars": 0}
    for s in snippets:
        t = s.get("text", "")
        if current["snippet_items"] and current["chars"] + len(t) > chunk_budget:
            chunks.append(current)
            current = {"snippet_items": [], "pages": set(), "chars": 0}
        current["snippet_items"].append(s)
        current["pages"].add(int(s.get("page_no", 0) or 0))
        current["chars"] += len(t)
    if current["snippet_items"]:
        chunks.append(current)
    if not chunks:
        chunks = [{"snippet_items": [], "pages": set(), "chars": 0}]

    out = []
    for idx, ch in enumerate(chunks, start=1):
        pages = set([p for p in ch["pages"] if p])
        rel_refs, seen_ref = [], set()
        for r in refs:
            page_no = int(r.get("page_no", 0) or 0)
            if pages and page_no not in pages:
                continue
            key = (str(r.get("ref_no_raw", "")), str(r.get("label_raw", "")))
            if key in seen_ref:
                continue
            seen_ref.add(key)
            rel_refs.append(r)

        rel_figs, seen_fig = [], set()
        for f in figures:
            page_no = int(f.get("page_no", 0) or 0)
            if pages and page_no not in pages:
                continue
            fig = str(f.get("figure_no", ""))
            if fig in seen_fig:
                continue
            seen_fig.add(fig)
            rel_figs.append(f)

        out.append({
            "chunk_id": f"chunk_{idx:03d}",
            "pages": sorted(list(pages)),
            "snippet_items": ch["snippet_items"],
            "refs": rel_refs,
            "figures": rel_figs,
            "independent_claims": independent,
        })
    return out


def build_pass_b_chunk_prompt(bundle: Dict[str, Any], chunk: Dict[str, Any], mode: str) -> str:
    meta = bundle["meta"]
    budget = PASS_B_CHUNK_CHAR_BUDGET_FULL if mode == "full" else PASS_B_CHUNK_CHAR_BUDGET_LITE
    lines, used = [], 0
    header = [
        "당신은 특허 evidence DB의 일부 chunk를 읽고 도면/실시형태/구현 중심 fact JSON을 만드는 분석기다.",
        "추정 금지. 이 chunk evidence에 직접 드러난 내용만 정리하라.",
        "반복 금지. 같은 의미의 단계/효과/구성요소를 다시 쓰지 말라.",
        "각 배열은 최대 6개 이내로, 각 문자열은 짧은 한 문장으로 써라.",
        f"chunk_id는 반드시 정확히 {chunk['chunk_id']} 로 써라.",
        "출력은 반드시 JSON만 반환하라.",
        "",
        f"patent_id: {meta['patent_id']}",
        f"country: {meta.get('country','')}",
        f"title_raw: {meta.get('title_raw','')}",
        f"chunk_id: {chunk['chunk_id']}",
        f"pages: {', '.join([str(x) for x in chunk['pages']]) if chunk['pages'] else '-'}",
        "",
        "schema:",
        json.dumps(pass_b_chunk_schema(), ensure_ascii=False, indent=2),
        "",
        "[independent_claim_gists]",
    ]
    for h in header:
        used = append_with_budget(lines, h + "\n", used, budget)
    for c in chunk["independent_claims"]:
        used = append_with_budget(lines, f"- claim_no={c.get('claim_no','')}\n{c.get('raw_text','')[:1000]}\n", used, budget)
        if used >= budget:
            break
    if used < budget:
        used = append_with_budget(lines, "\n[snippet_items]\n", used, budget)
        for s in chunk["snippet_items"]:
            used = append_with_budget(lines, f"- page={s.get('page_no','')} | {s.get('text','')[:900]}\n", used, budget)
            if used >= budget:
                break
    if used < budget and chunk["refs"]:
        used = append_with_budget(lines, "\n[reference_entities]\n", used, budget)
        for r in chunk["refs"]:
            used = append_with_budget(lines, f"- ref={r.get('ref_no_raw','')} | label={normalize_ws(r.get('label_raw',''))[:180]} | section={r.get('source_section','')} | page={r.get('page_no','')}\n", used, budget)
            if used >= budget:
                break
    if used < budget and chunk["figures"]:
        used = append_with_budget(lines, "\n[figures]\n", used, budget)
        for f in chunk["figures"]:
            used = append_with_budget(lines, f"- figure={f.get('figure_no','')} | page={f.get('page_no','')} | caption={normalize_ws(f.get('caption_raw',''))[:240]}\n", used, budget)
            if used >= budget:
                break
    return "".join(lines).strip()


def build_pass_b_merge_prompt(bundle: Dict[str, Any], chunk_results: List[Dict[str, Any]], mode: str) -> str:
    meta = bundle["meta"]
    budget = PASS_B_MERGE_CHAR_BUDGET_FULL if mode == "full" else PASS_B_MERGE_CHAR_BUDGET_LITE
    lines, used = [], 0
    header = [
        "당신은 여러 chunk fact JSON을 병합해 최종 Pass B JSON을 만드는 분석기다.",
        "원문 evidence를 직접 추정하지 말고, chunk 결과에 공통적으로 나타난 사실을 중심으로 병합하라.",
        "반복 금지. 같은 의미의 항목을 다시 쓰지 말라.",
        "각 배열은 최대 8개 이내로, detailed_operation_ko는 500자 이내로 써라.",
        "출력은 반드시 JSON만 반환하라.",
        "",
        f"patent_id: {meta['patent_id']}",
        f"country: {meta.get('country','')}",
        f"title_raw: {meta.get('title_raw','')}",
        "",
        "schema:",
        json.dumps(pass_b_merge_schema(), ensure_ascii=False, indent=2),
        "",
        "[chunk_results]",
    ]
    for h in header:
        used = append_with_budget(lines, h + "\n", used, budget)
    for ch in chunk_results:
        used = append_with_budget(lines, json.dumps(ch, ensure_ascii=False) + "\n", used, budget)
        if used >= budget:
            break
    return "".join(lines).strip()


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def merge_final_brief(bundle: Dict[str, Any], pass_a: Dict[str, Any], pass_b: Dict[str, Any], pass_a_meta: Dict[str, Any], pass_b_meta: Dict[str, Any]) -> Dict[str, Any]:
    meta = bundle["meta"]
    extraction_confidence = round((float(pass_a.get("extraction_confidence", 0.0)) + float(pass_b.get("extraction_confidence", 0.0))) / 2.0, 2)
    return {
        "source_language": pass_a.get("source_language", ""),
        "patent_id": meta["patent_id"],
        "title_source": pass_a.get("title_source", meta.get("title_raw", "")),
        "title_ko": pass_a.get("title_ko", ""),
        "one_line_summary_ko": pass_a.get("one_line_summary_ko", ""),
        "technical_field_ko": pass_a.get("technical_field_ko", ""),
        "problem_to_solve_ko": pass_a.get("problem_to_solve_ko", ""),
        "detailed_operation_ko": pass_b.get("detailed_operation_ko", ""),
        "representative_independent_claims_ko": pass_a.get("representative_independent_claims_ko", []),
        "claim_core_elements_ko": unique_keep_order(pass_a.get("claim_core_elements_ko", [])),
        "structural_elements_ko": unique_keep_order(pass_b.get("structural_elements_ko", [])),
        "operation_steps_ko": unique_keep_order(pass_b.get("operation_steps_ko", [])),
        "implementation_points_ko": unique_keep_order(pass_b.get("implementation_points_ko", [])),
        "explicit_effects_ko": unique_keep_order(pass_a.get("explicit_effects_ko", [])),
        "notable_numbers_or_conditions_ko": unique_keep_order(pass_a.get("notable_numbers_or_conditions_ko", [])),
        "figure_signals_ko": unique_keep_order(pass_b.get("figure_signals_ko", [])),
        "reference_signals_ko": unique_keep_order(pass_b.get("reference_signals_ko", [])),
        "evidence_anchors_ko": unique_keep_order(pass_a.get("evidence_anchors_ko", []) + pass_b.get("evidence_anchors_ko", [])),
        "evidence_gaps_ko": unique_keep_order(pass_b.get("evidence_gaps_ko", [])),
        "technical_specificity_score": extraction_confidence,
        "extraction_confidence": extraction_confidence,
        "build_meta": {
            "pipeline": "accuracy_first_two_pass_from_evidence_db_hierarchical_b",
            "model": LLM_MODEL,
            "keep_alive": KEEP_ALIVE,
            "temperature": TEMPERATURE,
            "seed": SEED,
            "pass_a_meta": pass_a_meta,
            "pass_b_meta": pass_b_meta,
        },
    }


def run_pass_a(bundle: Dict[str, Any], patent_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    try:
        return call_ollama_json(build_pass_a_prompt(bundle, "full"), pass_a_schema(), "pass_a", patent_id, "full")
    except Exception as e:
        log(f"      · pass_a 실패(mode=full): {e}")
        return call_ollama_json(build_pass_a_prompt(bundle, "lite"), pass_a_schema(), "pass_a", patent_id, "lite")


def run_pass_b_hierarchical(bundle: Dict[str, Any], patent_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    chunk_results, chunk_metas = [], []
    chunks = group_snippets_into_chunks(bundle, "full")
    log(f"      · PASS B-1 chunk 수: {len(chunks)}")
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        try:
            result, meta = call_ollama_json(build_pass_b_chunk_prompt(bundle, chunk, "full"), pass_b_chunk_schema(), f"pass_b1_{chunk_id}", patent_id, "full")
        except Exception as e:
            log(f"      · {chunk_id} 실패(mode=full): {e}")
            result, meta = call_ollama_json(build_pass_b_chunk_prompt(bundle, chunk, "lite"), pass_b_chunk_schema(), f"pass_b1_{chunk_id}", patent_id, "lite")
        chunk_results.append(result)
        chunk_metas.append(meta)

    try:
        pass_b, merge_meta = call_ollama_json(build_pass_b_merge_prompt(bundle, chunk_results, "full"), pass_b_merge_schema(), "pass_b2_merge", patent_id, "full")
    except Exception as e:
        log(f"      · pass_b2_merge 실패(mode=full): {e}")
        pass_b, merge_meta = call_ollama_json(build_pass_b_merge_prompt(bundle, chunk_results, "lite"), pass_b_merge_schema(), "pass_b2_merge", patent_id, "lite")

    return pass_b, {"pipeline": "hierarchical_b", "chunk_count": len(chunks), "chunk_metas": chunk_metas, "merge_meta": merge_meta}, chunk_results


def process_one_patent(con: sqlite3.Connection, patent_id: str) -> Dict[str, Any]:
    bundle = build_evidence_bundle(con, patent_id)
    mark_status(con, patent_id, "llm_running")
    item_start = time.time()

    pa_start = time.time()
    pass_a, pass_a_meta = run_pass_a(bundle, patent_id)
    pa_elapsed = time.time() - pa_start

    pb_start = time.time()
    pass_b, pass_b_meta, chunk_results = run_pass_b_hierarchical(bundle, patent_id)
    pb_elapsed = time.time() - pb_start

    final = merge_final_brief(bundle, pass_a, pass_b, pass_a_meta, pass_b_meta)

    pass_a_path = PASS_A_DIR / f"{safe_name(patent_id)}.pass_a.json"
    pass_b_path = PASS_B_DIR / f"{safe_name(patent_id)}.pass_b.json"
    final_path = A4_BRIEFS / f"{safe_name(patent_id)}.brief.json"

    save_json(pass_a_path, pass_a)
    save_json(pass_b_path, pass_b)
    save_json(final_path, final)

    chunk_dir = PASS_B_CHUNKS_DIR / safe_name(patent_id)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for idx, ch in enumerate(chunk_results, start=1):
        save_json(chunk_dir / f"chunk_{idx:03d}.json", ch)

    mark_status(con, patent_id, "brief_done")

    return {
        "patent_id": patent_id,
        "pass_a_json": str(pass_a_path),
        "pass_b_json": str(pass_b_path),
        "brief_json": str(final_path),
        "chunk_count": len(chunk_results),
        "pass_a_elapsed": pa_elapsed,
        "pass_b_elapsed": pb_elapsed,
        "overall_elapsed": time.time() - item_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run hierarchical local LLM fact extraction from evidence DB.")
    parser.add_argument("--limit", type=int, default=1, help="How many evidence_done patents to process in this run.")
    parser.add_argument("--patent-id", type=str, default=None, help="Process only the specified patent_id, regardless of current job status.")
    args = parser.parse_args()

    ensure_dirs()
    init_log_file()

    log("[로그 시작]")
    log(f"[설정] DB: {A4_DB}")
    log(f"[설정] briefs: {A4_BRIEFS}")
    log(f"[설정] model: {LLM_MODEL}")
    log("[설정] hierarchical Pass B enabled")
    log(f"[설정] limit={args.limit}, patent_id={args.patent_id or '-'}")

    con = get_connection()
    con.row_factory = sqlite3.Row
    counts = get_job_counts(con)
    log(f"[현황] jobs 상태: {counts}")

    processed, failed = 0, 0
    total_start = time.time()

    try:
        while processed + failed < args.limit:
            patent_id = fetch_next_patent_id(con, args.patent_id if processed + failed == 0 else None)
            if not patent_id:
                log("[종료] 처리할 evidence_done 건이 없습니다.")
                break

            log("")
            log(f"[처리중] patent_id={patent_id}")

            try:
                result = process_one_patent(con, patent_id)
                processed += 1
                log(f"    ✓ 완료: {result['patent_id']}")
                log(f"      · pass_a_json: {result['pass_a_json']}")
                log(f"      · pass_b_json: {result['pass_b_json']}")
                log(f"      · brief_json: {result['brief_json']}")
                log(f"      · PASS B chunk 수: {result['chunk_count']}")
                log(f"      · PASS A 소요: {human_seconds(result['pass_a_elapsed'])}")
                log(f"      · PASS B 소요: {human_seconds(result['pass_b_elapsed'])}")
                log(f"      · 전체 소요: {human_seconds(result['overall_elapsed'])}")
            except Exception as e:
                failed += 1
                mark_failed(con, patent_id, str(e))
                log(f"    ✗ 실패: {patent_id}")
                log(f"      오류: {e}")

            if args.patent_id:
                break
    finally:
        counts_end = get_job_counts(con)
        con.close()

    total_elapsed = time.time() - total_start
    log("")
    log(f"[종료] 성공: {processed}, 실패: {failed}, 총 소요: {human_seconds(total_elapsed)}")
    log(f"[종료] jobs 상태: {counts_end}")
    log("[로그 종료]")


if __name__ == "__main__":
    main()
