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

# ---------------- paths / settings ----------------

BASE = Path("/Volumes/외장 2TB/cpu2026")
HUB = BASE / "patent_hub"
COMMON = BASE / "common"

A4_DB = COMMON / "runtime" / "db" / "patent_A4.sqlite"
A4_LOGS = COMMON / "runtime" / "logs" / "A4"
A4_RAW_INVALID = COMMON / "runtime" / "raw_invalid" / "A4"

ANALYSIS_BASE = HUB / "outputs" / "analysis" / "A4"
CLAIM_SCOPE_DIR = ANALYSIS_BASE / "claim_scope"
CLAIM_ELEMENTS_DIR = ANALYSIS_BASE / "claim_elements"
GROUNDING_DIR = ANALYSIS_BASE / "grounding"
EFFECTS_DIR = ANALYSIS_BASE / "effects"
IMPLEMENTATION_DIR = ANALYSIS_BASE / "implementation"
PROFILES_DIR = ANALYSIS_BASE / "profiles"

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:14b"
KEEP_ALIVE = -1
TEMPERATURE = 0.03
SEED = 42
TIMEOUT = (20, 900)

M1_CTX = 4096
M2_CTX = 4096
M3_CTX = 4096
M4_CTX = 4096
M5_CTX = 4096

M1_PREDICT = 700
M2_PREDICT = 900
M3_PREDICT = 700
M4_PREDICT = 500
M5_PREDICT = 700

LOG_FILE_PATH = None

# ---------------- utils ----------------

def ensure_dirs() -> None:
    for p in [
        A4_LOGS, A4_RAW_INVALID, ANALYSIS_BASE,
        CLAIM_SCOPE_DIR, CLAIM_ELEMENTS_DIR, GROUNDING_DIR,
        EFFECTS_DIR, IMPLEMENTATION_DIR, PROFILES_DIR,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def init_log_file() -> None:
    global LOG_FILE_PATH
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_FILE_PATH = A4_LOGS / f"patent_analysis_{ts}.log"


def log(msg: str) -> None:
    line = "[ANALYSIS] " + msg
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
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def claim_sort_key(v: Any) -> Tuple[int, str]:
    text = str(v or "")
    m = re.match(r"(\d+)", text)
    return (int(m.group(1)) if m else 10**9, text)


def open_db() -> sqlite3.Connection:
    con = sqlite3.connect(A4_DB)
    con.row_factory = sqlite3.Row
    return con


# ---------------- DB helpers ----------------

def get_job_counts(con: sqlite3.Connection) -> Dict[str, int]:
    cur = con.cursor()
    rows = cur.execute("SELECT status, COUNT(*) AS c FROM jobs GROUP BY status").fetchall()
    return {r["status"]: int(r["c"]) for r in rows}


def fetch_next_patent_id(con: sqlite3.Connection, explicit_patent_id: Optional[str] = None) -> Optional[str]:
    cur = con.cursor()
    if explicit_patent_id:
        row = cur.execute(
            "SELECT patent_id FROM jobs WHERE patent_id=? AND status IN ('evidence_done','analysis_failed')",
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


def mark_job_status(con: sqlite3.Connection, patent_id: str, status: str, last_error: Optional[str] = None) -> None:
    cur = con.cursor()
    row = cur.execute("SELECT retry_count FROM jobs WHERE patent_id=?", (patent_id,)).fetchone()
    retry_count = int(row["retry_count"] or 0) if row else 0
    if status == "analysis_failed":
        retry_count += 1
    cur.execute(
        """
        UPDATE jobs
        SET status=?, retry_count=?, last_error=?, updated_at=CURRENT_TIMESTAMP
        WHERE patent_id=?
        """,
        (status, retry_count, last_error, patent_id),
    )
    con.commit()


def get_patent_meta(con: sqlite3.Connection, patent_id: str) -> Dict[str, Any]:
    cur = con.cursor()
    row = cur.execute(
        """
        SELECT patent_id, country, title_raw, assignee_raw, application_no, publication_no, pdf_path, page_count, parser_version
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
        """,
        (patent_id,),
    ).fetchall()
    out = [dict(r) for r in rows]
    out.sort(key=lambda x: (claim_sort_key(x.get("claim_no")), -int(x.get("freq", 0))))
    return out


def get_refs(con: sqlite3.Connection, patent_id: str) -> List[Dict[str, Any]]:
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT ref_no_raw, COALESCE(label_raw,'') AS label_raw, source_section, page_no, COUNT(*) AS freq
        FROM ref_entities
        WHERE patent_id=?
        GROUP BY ref_no_raw, label_raw, source_section, page_no
        ORDER BY freq DESC, page_no ASC
        """,
        (patent_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_figures(con: sqlite3.Connection, patent_id: str) -> List[Dict[str, Any]]:
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT figure_no, caption_raw, caption_norm, page_no
        FROM figure_captions
        WHERE patent_id=?
        ORDER BY page_no ASC, figure_no ASC
        """,
        (patent_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_description_snippets(con: sqlite3.Connection, patent_id: str, claims: List[Dict[str, Any]], limit: int = 200) -> List[Dict[str, Any]]:
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
          AND LENGTH(TRIM(raw_text)) BETWEEN 30 AND 1000
        ORDER BY page_no ASC, block_no ASC, line_no ASC, id ASC
        LIMIT ?
        """,
        (patent_id, min_claim_end, limit * 3),
    ).fetchall()

    snippets = []
    seen = set()
    for r in rows:
        txt = normalize_ws(r["raw_text"])
        if not txt:
            continue
        key = txt[:200]
        if key in seen:
            continue
        seen.add(key)
        snippets.append({"page_no": r["page_no"], "text": txt})
        if len(snippets) >= limit:
            break
    return snippets


def get_effect_snippets(snippets: List[Dict[str, Any]], limit: int = 20) -> List[Dict[str, Any]]:
    cue = re.compile(r"(향상|감소|증가|개선|최소화|절감|reduce|improv|decreas|increas|降低|提高|减少|改善)", re.I)
    out = []
    for s in snippets:
        if cue.search(s["text"]):
            out.append(s)
            if len(out) >= limit:
                break
    return out


def get_method_claims(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for c in claims:
        raw = c.get("raw_text", "")
        if re.search(r"(방법|method|方法)", raw, flags=re.I):
            out.append(c)
    return out


# ---------------- LLM helpers ----------------

def extract_json_candidate(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    first = t.find("{")
    last = t.rfind("}")
    return t[first:last + 1] if first != -1 and last != -1 and last > first else t


def call_ollama_json(
    patent_id: str,
    module_name: str,
    prompt: str,
    schema: Dict[str, Any],
    num_ctx: int,
    num_predict: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "keep_alive": KEEP_ALIVE,
        "format": schema,
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
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
        raw_path = A4_RAW_INVALID / f"{safe_name(patent_id)}.{module_name}.raw.txt"
        meta_path = A4_RAW_INVALID / f"{safe_name(patent_id)}.{module_name}.meta.json"
        raw_path.write_text(raw_text, encoding="utf-8")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "module_name": module_name,
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
        raise RuntimeError(f"{module_name} invalid JSON: {e}")

    meta = {
        "module_name": module_name,
        "wall_seconds": round(wall_seconds, 2),
        "ollama_total_seconds": round(float(data.get("total_duration", 0)) / 1e9, 2) if data.get("total_duration") else None,
        "prompt_eval_count": data.get("prompt_eval_count"),
        "eval_count": data.get("eval_count"),
        "done_reason": data.get("done_reason"),
    }
    return parsed, meta


# ---------------- schemas ----------------

def claim_scope_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "patent_id": {"type": "string"},
            "module": {"type": "string"},
            "claim_no": {"type": "string"},
            "claim_type": {"type": "string"},
            "scope_category": {"type": "string"},
            "claim_purpose_ko": {"type": "string"},
            "core_scope_ko": {"type": "string"},
            "essential_relations_ko": {"type": "array", "items": {"type": "string"}},
            "evidence_anchors": {"type": "array", "items": {"type": "string"}},
            "extraction_confidence": {"type": "number"},
        },
        "required": ["patent_id","module","claim_no","claim_type","scope_category","claim_purpose_ko","core_scope_ko","essential_relations_ko","evidence_anchors","extraction_confidence"],
    }


def claim_elements_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "patent_id": {"type": "string"},
            "module": {"type": "string"},
            "claim_no": {"type": "string"},
            "claim_type": {"type": "string"},
            "elements": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "element_id": {"type": "string"},
                        "element_text_ko": {"type": "string"},
                        "element_role_ko": {"type": "string"},
                        "required": {"type": "boolean"},
                        "parent_element_id": {"type": "string"},
                    },
                    "required": ["element_id","element_text_ko","element_role_ko","required","parent_element_id"],
                },
            },
            "evidence_anchors": {"type": "array", "items": {"type": "string"}},
            "extraction_confidence": {"type": "number"},
        },
        "required": ["patent_id","module","claim_no","claim_type","elements","evidence_anchors","extraction_confidence"],
    }


def grounding_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "patent_id": {"type": "string"},
            "module": {"type": "string"},
            "claim_no": {"type": "string"},
            "element_id": {"type": "string"},
            "element_text_ko": {"type": "string"},
            "supporting_description_ko": {"type": "string"},
            "supporting_pages": {"type": "array", "items": {"type": "integer"}},
            "supporting_figures": {"type": "array", "items": {"type": "string"}},
            "supporting_ref_entities": {"type": "array", "items": {"type": "string"}},
            "supporting_snippets_ko": {"type": "array", "items": {"type": "string"}},
            "grounding_strength": {"type": "string"},
            "evidence_anchors": {"type": "array", "items": {"type": "string"}},
            "extraction_confidence": {"type": "number"},
        },
        "required": ["patent_id","module","claim_no","element_id","element_text_ko","supporting_description_ko","supporting_pages","supporting_figures","supporting_ref_entities","supporting_snippets_ko","grounding_strength","evidence_anchors","extraction_confidence"],
    }


def effect_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "patent_id": {"type": "string"},
            "module": {"type": "string"},
            "claim_no": {"type": "string"},
            "element_id": {"type": "string"},
            "effect_ko": {"type": "string"},
            "effect_type": {"type": "string"},
            "why_ko": {"type": "string"},
            "related_grounding_ids": {"type": "array", "items": {"type": "string"}},
            "evidence_anchors": {"type": "array", "items": {"type": "string"}},
            "extraction_confidence": {"type": "number"},
        },
        "required": ["patent_id","module","claim_no","element_id","effect_ko","effect_type","why_ko","related_grounding_ids","evidence_anchors","extraction_confidence"],
    }


def implementation_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "patent_id": {"type": "string"},
            "module": {"type": "string"},
            "source_type": {"type": "string"},
            "source_id": {"type": "string"},
            "operation_steps_ko": {"type": "array", "items": {"type": "string"}},
            "implementation_points_ko": {"type": "array", "items": {"type": "string"}},
            "key_materials_or_conditions_ko": {"type": "array", "items": {"type": "string"}},
            "supporting_pages": {"type": "array", "items": {"type": "integer"}},
            "supporting_figures": {"type": "array", "items": {"type": "string"}},
            "evidence_anchors": {"type": "array", "items": {"type": "string"}},
            "extraction_confidence": {"type": "number"},
        },
        "required": ["patent_id","module","source_type","source_id","operation_steps_ko","implementation_points_ko","key_materials_or_conditions_ko","supporting_pages","supporting_figures","evidence_anchors","extraction_confidence"],
    }


# ---------------- prompts ----------------

def build_claim_scope_prompt(meta: Dict[str, Any], claim: Dict[str, Any]) -> str:
    schema_str = json.dumps(claim_scope_schema(), ensure_ascii=False, indent=2)
    return f"""
너는 특허 독립항 범위를 해석하는 분석기다.
추정하거나 선행기술 비교를 하지 말고, 아래 독립항 텍스트만 보고 권리범위 핵심을 정리하라.
출력은 반드시 JSON만 반환하라.
claim_no는 입력 claim_no와 동일해야 한다.
evidence_anchors에는 반드시 "claim {claim['claim_no']}"를 포함하라.

schema:
{schema_str}

metadata:
patent_id: {meta['patent_id']}
country: {meta.get('country','')}
title_raw: {meta.get('title_raw','')}

claim_no: {claim['claim_no']}
claim_type: {claim.get('claim_type','')}
claim_text:
{claim.get('raw_text','')}
""".strip()


def build_claim_elements_prompt(meta: Dict[str, Any], claim: Dict[str, Any], claim_scope: Dict[str, Any]) -> str:
    schema_str = json.dumps(claim_elements_schema(), ensure_ascii=False, indent=2)
    return f"""
너는 특허 독립항을 claim chart용 요소로 분해하는 분석기다.
오직 아래 독립항 텍스트만 기준으로 필수 구성요소를 분해하라.
새로운 요소를 발명하지 말라.
element_id는 "{claim['claim_no']}-E1", "{claim['claim_no']}-E2" 형식으로 만들어라.
출력은 반드시 JSON만 반환하라.
claim_no는 입력과 동일해야 한다.

schema:
{schema_str}

metadata:
patent_id: {meta['patent_id']}
country: {meta.get('country','')}
title_raw: {meta.get('title_raw','')}

claim_scope_summary:
{claim_scope.get('core_scope_ko','')}

claim_no: {claim['claim_no']}
claim_text:
{claim.get('raw_text','')}
""".strip()


def build_grounding_prompt(
    meta: Dict[str, Any],
    claim: Dict[str, Any],
    element: Dict[str, Any],
    candidate_snippets: List[Dict[str, Any]],
    candidate_refs: List[Dict[str, Any]],
    candidate_figures: List[Dict[str, Any]],
) -> str:
    schema_str = json.dumps(grounding_schema(), ensure_ascii=False, indent=2)
    lines = [
        "너는 특허 청구항 요소의 grounding을 찾는 분석기다.",
        "추정하지 말고, 아래 후보 evidence 안에서만 근거를 골라라.",
        "근거가 약하면 grounding_strength를 weak로 두고, 없는 값은 빈 배열로 둬라.",
        "출력은 반드시 JSON만 반환하라.",
        "",
        "schema:",
        schema_str,
        "",
        f"metadata: patent_id={meta['patent_id']} country={meta.get('country','')}",
        f"claim_no: {claim['claim_no']}",
        f"claim_text: {claim.get('raw_text','')[:1600]}",
        f"element_id: {element['element_id']}",
        f"element_text_ko: {element['element_text_ko']}",
        "",
        "[candidate_snippets]",
    ]
    for s in candidate_snippets[:8]:
        lines.append(f"- page={s['page_no']} | {s['text'][:450]}")
    lines.append("")
    lines.append("[candidate_refs]")
    for r in candidate_refs[:12]:
        lines.append(f"- ref={r.get('ref_no_raw','')} | label={normalize_ws(r.get('label_raw',''))[:160]} | section={r.get('source_section','')} | page={r.get('page_no','')}")
    lines.append("")
    lines.append("[candidate_figures]")
    for f in candidate_figures[:8]:
        lines.append(f"- figure={f.get('figure_no','')} | page={f.get('page_no','')} | caption={normalize_ws(f.get('caption_raw',''))[:220]}")
    return "\n".join(lines).strip()


def build_effect_prompt(
    meta: Dict[str, Any],
    claim: Dict[str, Any],
    element: Dict[str, Any],
    grounding: Dict[str, Any],
    effect_snippets: List[Dict[str, Any]],
) -> str:
    schema_str = json.dumps(effect_schema(), ensure_ascii=False, indent=2)
    lines = [
        "너는 특허 요소별 명시적 효과를 추출하는 분석기다.",
        "추정하지 말고, 아래 grounding과 설명문 후보에 직접 드러난 효과만 써라.",
        "효과가 분명하지 않으면 effect_ko는 빈 문자열로 두고 confidence를 낮춰라.",
        "출력은 반드시 JSON만 반환하라.",
        "",
        "schema:",
        schema_str,
        "",
        f"metadata: patent_id={meta['patent_id']} country={meta.get('country','')}",
        f"claim_no: {claim['claim_no']}",
        f"element_id: {element['element_id']}",
        f"element_text_ko: {element['element_text_ko']}",
        "",
        "[grounding]",
        json.dumps(grounding, ensure_ascii=False),
        "",
        "[effect_candidate_snippets]",
    ]
    for s in effect_snippets[:6]:
        lines.append(f"- page={s['page_no']} | {s['text'][:450]}")
    return "\n".join(lines).strip()


def build_implementation_prompt(
    meta: Dict[str, Any],
    source_type: str,
    source_id: str,
    source_text: str,
    figures: List[Dict[str, Any]],
) -> str:
    schema_str = json.dumps(implementation_schema(), ensure_ascii=False, indent=2)
    lines = [
        "너는 특허의 방법/구현/공정 포인트를 추출하는 분석기다.",
        "추정하지 말고, 아래 source_text와 figure 정보에서 직접 드러난 구현 단계만 정리하라.",
        "출력은 반드시 JSON만 반환하라.",
        "",
        "schema:",
        schema_str,
        "",
        f"metadata: patent_id={meta['patent_id']} country={meta.get('country','')}",
        f"source_type: {source_type}",
        f"source_id: {source_id}",
        "",
        "[source_text]",
        source_text[:2200],
        "",
        "[candidate_figures]",
    ]
    for f in figures[:8]:
        lines.append(f"- figure={f.get('figure_no','')} | page={f.get('page_no','')} | caption={normalize_ws(f.get('caption_raw',''))[:220]}")
    return "\n".join(lines).strip()


# ---------------- module runners ----------------

def validate_claim_scope(result: Dict[str, Any], patent_id: str, claim_no: str) -> None:
    if str(result.get("patent_id", "")) != patent_id:
        raise RuntimeError("claim_scope patent_id mismatch")
    if str(result.get("claim_no", "")) != str(claim_no):
        raise RuntimeError("claim_scope claim_no mismatch")
    if not normalize_ws(result.get("core_scope_ko", "")):
        raise RuntimeError("claim_scope empty core_scope_ko")


def validate_claim_elements(result: Dict[str, Any], patent_id: str, claim_no: str) -> None:
    if str(result.get("patent_id", "")) != patent_id:
        raise RuntimeError("claim_elements patent_id mismatch")
    if str(result.get("claim_no", "")) != str(claim_no):
        raise RuntimeError("claim_elements claim_no mismatch")
    elems = result.get("elements", [])
    if not elems:
        raise RuntimeError("claim_elements empty")
    seen = set()
    for e in elems:
        eid = e.get("element_id", "")
        if not eid or eid in seen:
            raise RuntimeError("claim_elements invalid/duplicate element_id")
        seen.add(eid)


def run_claim_scope(meta: Dict[str, Any], claim: Dict[str, Any], overwrite: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    out_path = CLAIM_SCOPE_DIR / f"{safe_name(meta['patent_id'])}.claim_scope.claim_{safe_name(claim['claim_no'])}.json"
    if out_path.exists() and not overwrite:
        return load_json(out_path), {"module_name": "claim_scope", "loaded_from_cache": True}, out_path
    prompt = build_claim_scope_prompt(meta, claim)
    result, meta_info = call_ollama_json(meta["patent_id"], f"claim_scope.claim_{claim['claim_no']}", prompt, claim_scope_schema(), M1_CTX, M1_PREDICT)
    validate_claim_scope(result, meta["patent_id"], claim["claim_no"])
    save_json(out_path, result)
    return result, meta_info, out_path


def run_claim_elements(meta: Dict[str, Any], claim: Dict[str, Any], claim_scope: Dict[str, Any], overwrite: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    out_path = CLAIM_ELEMENTS_DIR / f"{safe_name(meta['patent_id'])}.claim_elements.claim_{safe_name(claim['claim_no'])}.json"
    if out_path.exists() and not overwrite:
        return load_json(out_path), {"module_name": "claim_elements", "loaded_from_cache": True}, out_path
    prompt = build_claim_elements_prompt(meta, claim, claim_scope)
    result, meta_info = call_ollama_json(meta["patent_id"], f"claim_elements.claim_{claim['claim_no']}", prompt, claim_elements_schema(), M2_CTX, M2_PREDICT)
    validate_claim_elements(result, meta["patent_id"], claim["claim_no"])
    save_json(out_path, result)
    return result, meta_info, out_path


def fallback_grounding(meta: Dict[str, Any], claim_no: str, element: Dict[str, Any], error_note: str) -> Dict[str, Any]:
    return {
        "patent_id": meta["patent_id"],
        "module": "element_grounding",
        "claim_no": str(claim_no),
        "element_id": element["element_id"],
        "element_text_ko": element["element_text_ko"],
        "supporting_description_ko": "",
        "supporting_pages": [],
        "supporting_figures": [],
        "supporting_ref_entities": [],
        "supporting_snippets_ko": [],
        "grounding_strength": "weak",
        "evidence_anchors": [],
        "extraction_confidence": 0.0,
        "error_note": error_note,
    }


def run_grounding(meta: Dict[str, Any], claim: Dict[str, Any], element: Dict[str, Any], candidate_snippets: List[Dict[str, Any]], candidate_refs: List[Dict[str, Any]], candidate_figures: List[Dict[str, Any]], overwrite: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    out_path = GROUNDING_DIR / f"{safe_name(meta['patent_id'])}.grounding.{safe_name(element['element_id'])}.json"
    if out_path.exists() and not overwrite:
        return load_json(out_path), {"module_name": "grounding", "loaded_from_cache": True}, out_path
    prompt = build_grounding_prompt(meta, claim, element, candidate_snippets, candidate_refs, candidate_figures)
    try:
        result, meta_info = call_ollama_json(meta["patent_id"], f"grounding.{element['element_id']}", prompt, grounding_schema(), M3_CTX, M3_PREDICT)
    except Exception as e:
        result = fallback_grounding(meta, claim["claim_no"], element, str(e))
        meta_info = {"module_name": "grounding", "fallback": True, "error": str(e)}
    save_json(out_path, result)
    return result, meta_info, out_path


def fallback_effect(meta: Dict[str, Any], claim_no: str, element_id: str, error_note: str) -> Dict[str, Any]:
    return {
        "patent_id": meta["patent_id"],
        "module": "element_effect",
        "claim_no": str(claim_no),
        "element_id": element_id,
        "effect_ko": "",
        "effect_type": "other",
        "why_ko": "",
        "related_grounding_ids": [element_id],
        "evidence_anchors": [],
        "extraction_confidence": 0.0,
        "error_note": error_note,
    }


def run_effect(meta: Dict[str, Any], claim: Dict[str, Any], element: Dict[str, Any], grounding: Dict[str, Any], effect_snippets: List[Dict[str, Any]], overwrite: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    out_path = EFFECTS_DIR / f"{safe_name(meta['patent_id'])}.effects.{safe_name(element['element_id'])}.json"
    if out_path.exists() and not overwrite:
        return load_json(out_path), {"module_name": "effects", "loaded_from_cache": True}, out_path
    prompt = build_effect_prompt(meta, claim, element, grounding, effect_snippets)
    try:
        result, meta_info = call_ollama_json(meta["patent_id"], f"effects.{element['element_id']}", prompt, effect_schema(), M4_CTX, M4_PREDICT)
    except Exception as e:
        result = fallback_effect(meta, claim["claim_no"], element["element_id"], str(e))
        meta_info = {"module_name": "effects", "fallback": True, "error": str(e)}
    save_json(out_path, result)
    return result, meta_info, out_path


def fallback_implementation(meta: Dict[str, Any], source_type: str, source_id: str, error_note: str) -> Dict[str, Any]:
    return {
        "patent_id": meta["patent_id"],
        "module": "implementation",
        "source_type": source_type,
        "source_id": source_id,
        "operation_steps_ko": [],
        "implementation_points_ko": [],
        "key_materials_or_conditions_ko": [],
        "supporting_pages": [],
        "supporting_figures": [],
        "evidence_anchors": [],
        "extraction_confidence": 0.0,
        "error_note": error_note,
    }


def run_implementation(meta: Dict[str, Any], source_type: str, source_id: str, source_text: str, figures: List[Dict[str, Any]], overwrite: bool = False) -> Tuple[Dict[str, Any], Dict[str, Any], Path]:
    out_path = IMPLEMENTATION_DIR / f"{safe_name(meta['patent_id'])}.implementation.{safe_name(source_id)}.json"
    if out_path.exists() and not overwrite:
        return load_json(out_path), {"module_name": "implementation", "loaded_from_cache": True}, out_path
    prompt = build_implementation_prompt(meta, source_type, source_id, source_text, figures)
    try:
        result, meta_info = call_ollama_json(meta["patent_id"], f"implementation.{source_id}", prompt, implementation_schema(), M5_CTX, M5_PREDICT)
    except Exception as e:
        result = fallback_implementation(meta, source_type, source_id, str(e))
        meta_info = {"module_name": "implementation", "fallback": True, "error": str(e)}
    save_json(out_path, result)
    return result, meta_info, out_path


# ---------------- candidate selection ----------------

def candidate_snippets_for_element(snippets: List[Dict[str, Any]], element_text_ko: str, limit: int = 8) -> List[Dict[str, Any]]:
    tokens = [t for t in re.split(r"[,\s/;:()\[\]{}\-]+", element_text_ko) if len(t.strip()) >= 2]
    scored = []
    for s in snippets:
        txt = s["text"]
        score = 0
        for tok in tokens:
            if tok and tok in txt:
                score += 1
        scored.append((score, s))
    scored.sort(key=lambda x: (-x[0], x[1]["page_no"]))
    selected = [x[1] for x in scored[:limit]]
    return selected if selected else snippets[:limit]


def candidate_refs_for_claim(claim_ref_counts: List[Dict[str, Any]], refs: List[Dict[str, Any]], claim_no: str, limit: int = 12) -> List[Dict[str, Any]]:
    wanted = [r["ref_no_raw"] for r in claim_ref_counts if str(r["claim_no"]) == str(claim_no)]
    out = []
    seen = set()
    for ref_no in wanted:
        for r in refs:
            if str(r.get("ref_no_raw", "")) == str(ref_no):
                key = (r.get("ref_no_raw", ""), r.get("page_no", ""))
                if key not in seen:
                    seen.add(key)
                    out.append(r)
                    if len(out) >= limit:
                        return out
    return out[:limit] if out else refs[:limit]


# ---------------- profile merge ----------------

def build_profile(meta: Dict[str, Any], claim_scope_results: List[Dict[str, Any]], claim_elements_results: List[Dict[str, Any]], grounding_results: List[Dict[str, Any]], effect_results: List[Dict[str, Any]], implementation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    title_source = meta.get("title_raw", "")
    title_ko = title_source

    independent_claim_scopes = []
    scope_summaries = []
    for r in claim_scope_results:
        independent_claim_scopes.append({"claim_no": r.get("claim_no", ""), "core_scope_ko": r.get("core_scope_ko", "")})
        if normalize_ws(r.get("core_scope_ko", "")):
            scope_summaries.append(r.get("core_scope_ko", ""))

    core_elements = []
    for ce in claim_elements_results:
        for e in ce.get("elements", []):
            if e.get("required", True):
                core_elements.append(e.get("element_text_ko", ""))

    grounded_figures = []
    grounded_refs = []
    grounding_anchors = []
    for g in grounding_results:
        grounded_figures.extend(g.get("supporting_figures", []))
        grounded_refs.extend(g.get("supporting_ref_entities", []))
        grounding_anchors.extend(g.get("evidence_anchors", []))

    effects = []
    effect_anchors = []
    for e in effect_results:
        if normalize_ws(e.get("effect_ko", "")):
            effects.append(e.get("effect_ko", ""))
            effect_anchors.extend(e.get("evidence_anchors", []))

    implementation_points = []
    important_conditions = []
    impl_anchors = []
    for im in implementation_results:
        implementation_points.extend(im.get("implementation_points_ko", []))
        important_conditions.extend(im.get("key_materials_or_conditions_ko", []))
        impl_anchors.extend(im.get("evidence_anchors", []))

    anchors = unique_keep_order(
        [f"claim {x.get('claim_no','')}" for x in claim_scope_results]
        + grounding_anchors + effect_anchors + impl_anchors
    )

    conf_values = []
    for coll in [claim_scope_results, claim_elements_results, grounding_results, effect_results, implementation_results]:
        for x in coll:
            try:
                conf_values.append(float(x.get("extraction_confidence", 0.0)))
            except Exception:
                pass
    profile_conf = round(sum(conf_values) / len(conf_values), 2) if conf_values else 0.0

    return {
        "patent_id": meta["patent_id"],
        "module": "single_patent_profile",
        "title_source": title_source,
        "title_ko": title_ko,
        "scope_summary_ko": " / ".join(unique_keep_order(scope_summaries[:6])),
        "independent_claim_scopes": independent_claim_scopes,
        "core_elements_ko": unique_keep_order(core_elements),
        "grounded_figures": unique_keep_order(grounded_figures),
        "grounded_ref_entities": unique_keep_order(grounded_refs),
        "explicit_effects_ko": unique_keep_order(effects),
        "implementation_points_ko": unique_keep_order(implementation_points),
        "important_conditions_ko": unique_keep_order(important_conditions),
        "evidence_anchors": anchors,
        "profile_confidence": profile_conf,
    }


# ---------------- pipeline ----------------

def process_one_patent(con: sqlite3.Connection, patent_id: str, overwrite: bool = False) -> Dict[str, Any]:
    meta = get_patent_meta(con, patent_id)
    claims = get_claims(con, patent_id)
    if not claims:
        raise RuntimeError("claims not found")

    independent_claims = [c for c in claims if c.get("claim_type") == "independent"]
    if not independent_claims:
        raise RuntimeError("independent claims not found")

    claim_ref_counts = get_claim_ref_counts(con, patent_id)
    refs = get_refs(con, patent_id)
    figures = get_figures(con, patent_id)
    snippets = get_description_snippets(con, patent_id, claims, limit=120)
    effect_snips = get_effect_snippets(snippets, limit=20)
    method_claims = get_method_claims(claims)

    log(f"      · independent_claims={len(independent_claims)}, refs={len(refs)}, figures={len(figures)}, snippets={len(snippets)}, method_claims={len(method_claims)}")

    claim_scope_results = []
    claim_elements_results = []
    grounding_results = []
    effect_results = []
    implementation_results = []

    item_start = time.time()

    # M1 + M2 + M3 + M4
    for idx, claim in enumerate(independent_claims, start=1):
        log(f"      · M1 claim_scope 시작 {idx}/{len(independent_claims)} claim_no={claim['claim_no']}")
        cs, cs_meta, cs_path = run_claim_scope(meta, claim, overwrite=overwrite)
        claim_scope_results.append(cs)
        log(f"        ↳ 완료 {human_seconds(cs_meta.get('wall_seconds', 0.0)) if 'wall_seconds' in cs_meta else 'cache'} | {cs_path.name}")

        log(f"      · M2 claim_elements 시작 {idx}/{len(independent_claims)} claim_no={claim['claim_no']}")
        ce, ce_meta, ce_path = run_claim_elements(meta, claim, cs, overwrite=overwrite)
        claim_elements_results.append(ce)
        log(f"        ↳ 완료 {human_seconds(cs_meta.get('wall_seconds', 0.0)) if 'wall_seconds' in ce_meta else 'cache'} | elements={len(ce.get('elements', []))} | {ce_path.name}")

        for eidx, element in enumerate(ce.get("elements", []), start=1):
            cand_snips = candidate_snippets_for_element(snippets, element.get("element_text_ko", ""), limit=8)
            cand_refs = candidate_refs_for_claim(claim_ref_counts, refs, claim["claim_no"], limit=12)
            cand_figs = figures[:8]

            log(f"      · M3 grounding 시작 claim={claim['claim_no']} element={element['element_id']} ({eidx}/{len(ce.get('elements', []))})")
            gr, gr_meta, gr_path = run_grounding(meta, claim, element, cand_snips, cand_refs, cand_figs, overwrite=overwrite)
            grounding_results.append(gr)
            if "wall_seconds" in gr_meta:
                log(f"        ↳ 완료 {human_seconds(gr_meta.get('wall_seconds', 0.0))} | {gr_path.name}")
            else:
                log(f"        ↳ fallback | {gr_path.name}")

            log(f"      · M4 effect 시작 claim={claim['claim_no']} element={element['element_id']} ({eidx}/{len(ce.get('elements', []))})")
            ef, ef_meta, ef_path = run_effect(meta, claim, element, gr, effect_snips, overwrite=overwrite)
            effect_results.append(ef)
            if "wall_seconds" in ef_meta:
                log(f"        ↳ 완료 {human_seconds(ef_meta.get('wall_seconds', 0.0))} | {ef_path.name}")
            else:
                log(f"        ↳ fallback | {ef_path.name}")

    # M5 implementation
    if method_claims:
        for midx, mclaim in enumerate(method_claims, start=1):
            source_id = f"claim_{mclaim['claim_no']}"
            log(f"      · M5 implementation 시작 method_claim {midx}/{len(method_claims)} claim_no={mclaim['claim_no']}")
            im, im_meta, im_path = run_implementation(meta, "method_claim", source_id, mclaim.get("raw_text", ""), figures[:8], overwrite=overwrite)
            implementation_results.append(im)
            if "wall_seconds" in im_meta:
                log(f"        ↳ 완료 {human_seconds(im_meta.get('wall_seconds', 0.0))} | {im_path.name}")
            else:
                log(f"        ↳ fallback | {im_path.name}")
    else:
        desc_chunks = []
        chunk = []
        for s in snippets[:18]:
            chunk.append(s)
            if len(chunk) >= 6:
                desc_chunks.append(chunk)
                chunk = []
        if chunk:
            desc_chunks.append(chunk)

        for didx, chunk_items in enumerate(desc_chunks[:3], start=1):
            source_id = f"desc_chunk_{didx:03d}"
            source_text = "\n".join([f"page={x['page_no']} | {x['text']}" for x in chunk_items])
            log(f"      · M5 implementation 시작 description_chunk {didx}/{min(len(desc_chunks),3)}")
            im, im_meta, im_path = run_implementation(meta, "description_chunk", source_id, source_text, figures[:8], overwrite=overwrite)
            implementation_results.append(im)
            if "wall_seconds" in im_meta:
                log(f"        ↳ 완료 {human_seconds(im_meta.get('wall_seconds', 0.0))} | {im_path.name}")
            else:
                log(f"        ↳ fallback | {im_path.name}")

    # M6 profile
    log("      · M6 profile merge 시작")
    profile = build_profile(meta, claim_scope_results, claim_elements_results, grounding_results, effect_results, implementation_results)
    profile_path = PROFILES_DIR / f"{safe_name(meta['patent_id'])}.profile.json"
    save_json(profile_path, profile)
    elapsed = time.time() - item_start
    log(f"        ↳ 완료 {human_seconds(elapsed)} | {profile_path.name}")

    return {
        "patent_id": patent_id,
        "profile_path": str(profile_path),
        "independent_claim_count": len(independent_claims),
        "element_count": sum(len(x.get("elements", [])) for x in claim_elements_results),
        "grounding_count": len(grounding_results),
        "effect_count": len(effect_results),
        "implementation_count": len(implementation_results),
        "elapsed": elapsed,
    }


# ---------------- main ----------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential single-patent analysis pipeline from evidence DB.")
    parser.add_argument("--limit", type=int, default=1, help="How many evidence_done patents to process")
    parser.add_argument("--patent-id", type=str, default=None, help="Process only one patent_id")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs even if files exist")
    args = parser.parse_args()

    ensure_dirs()
    init_log_file()

    log("[로그 시작]")
    log(f"[설정] DB: {A4_DB}")
    log(f"[설정] analysis_base: {ANALYSIS_BASE}")
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
                log("[종료] 처리할 evidence_done/analysis_failed 건이 없습니다.")
                break

            log("")
            log(f"[처리중] patent_id={patent_id}")
            item_start = time.time()

            try:
                mark_job_status(con, patent_id, "analysis_running", None)
                result = process_one_patent(con, patent_id, overwrite=args.overwrite)
                mark_job_status(con, patent_id, "profile_done", None)
                processed += 1

                log(f"    ✓ 완료: {result['patent_id']}")
                log(f"      · profile_json: {result['profile_path']}")
                log(f"      · independent_claims={result['independent_claim_count']}, elements={result['element_count']}, groundings={result['grounding_count']}, effects={result['effect_count']}, implementations={result['implementation_count']}")
                log(f"      · 전체 소요: {human_seconds(result['elapsed'])}")

            except Exception as e:
                failed += 1
                mark_job_status(con, patent_id, "analysis_failed", str(e))
                log(f"    ✗ 실패: {patent_id}")
                log(f"      오류: {e}")
                log(f"      · 소요 시간: {human_seconds(time.time() - item_start)}")

            if args.patent_id:
                break

    finally:
        end_counts = get_job_counts(con)
        con.close()

    total_elapsed = time.time() - total_start
    log("")
    log(f"[종료] 성공: {processed}, 실패: {failed}, 총 소요: {human_seconds(total_elapsed)}")
    log(f"[종료] jobs 상태: {end_counts}")
    log("[로그 종료]")


if __name__ == "__main__":
    main()
