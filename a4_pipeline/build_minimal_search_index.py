from __future__ import annotations

import argparse
import json
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List


BASE = Path("/Volumes/외장 2TB/cpu2026")
HUB = BASE / "patent_hub"

MINIMAL_DIR = HUB / "outputs" / "minimal_analysis" / "A4"
INDEX_DIR = HUB / "outputs" / "indexes" / "A4"
INDEX_SQLITE = INDEX_DIR / "patent_minimal_index.sqlite"
INDEX_JSONL = INDEX_DIR / "patent_minimal_index.jsonl"
QC_REPORT = INDEX_DIR / "patent_minimal_qc_report.json"


REQUIRED_FIELDS = [
    "patent_id",
    "source_language",
    "title_source",
    "primary_claim_type",
    "secondary_claim_types",
    "core_subject",
    "core_elements",
    "solution_labels",
    "evidence_ids",
    "confidence",
]

TITLE_CONTAMINATION_RE = re.compile(
    r"(摘要|ABSTRACT|Abstract|权利要求书|청구항|청구범위|요\s*약|"
    r"\(\s*(?:30|57|60|71|72|86)\s*\)|Applicant[:!]?|Applicants[:!]?|"
    r"Inventor[:!]?|Inventors[:!]?|Assignee[:!]?|Appl\.?\s*No\.?|Filed:|"
    r"Date of Patent|Publication Date|Pub\.? Date|Related U\.?S\.? Application|"
    r"Foreign Application Priority Data|provisional application|"
    r"U\.?\s*S\.?\s*Cl\.?|CPC|USPC|H[I1][O0]B|HOB\s*\d|GO6F|GUC\s*\d|G1I1C|"
    r"See application file|References Cited|et al\.|wees|ceeee|o\.\.|frorn|vaive)",
    re.I,
)

GENERIC_SUBJECT_RE = re.compile(
    r"^(memory device|memory system|semiconductor device|storage device|method|system|device|"
    r"non-volatile memory device|operation method|메모리 장치|메모리 시스템|반도체 장치|방법|시스템|장치|"
    r"存储器装置|存储器系统|半导体装置|方法|系统|装置)$",
    re.I,
)

SUSPICIOUS_LABELS = {
    "ect_data_labeling",
    "ect_data_labeling_platform",
    "composite_electrode_preparation",
}

LOW_VALUE_LABELS = {
    "generic_memory_operation",
    "general_data_processing",
}


def unique_keep_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        value = normalize_ws(item)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def infer_replacement_solution_labels_from_context(text: str, claim_types: List[str]) -> List[str]:
    text = normalize_ws(text)
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
    for label, pattern in patterns:
        if re.search(pattern, text, flags=re.I):
            labels.append(label)
    if not labels:
        if "method" in claim_types or "process" in claim_types:
            labels.append("method_feature_extraction")
        elif "system" in claim_types:
            labels.append("system_feature_extraction")
        elif "device" in claim_types:
            labels.append("device_feature_extraction")
    return unique_keep_order(labels)[:6]


def normalize_solution_labels_for_quality(obj: Dict[str, Any]) -> List[str]:
    labels = unique_keep_order([str(x) for x in (obj.get("solution_labels") or [])])
    labels = [x for x in labels if x not in SUSPICIOUS_LABELS]
    specific = [
        x for x in labels
        if x not in LOW_VALUE_LABELS and not x.startswith(("core_", "claimed_", "safe_"))
    ]
    if specific:
        return specific
    context_text = " ".join(
        [
            normalize_ws(obj.get("title_source")),
            normalize_ws(obj.get("core_subject")),
            " ".join([str(x) for x in (obj.get("core_elements") or [])]),
            " ".join([str(x) for x in (obj.get("protected_terms") or [])]),
        ]
    )
    claim_types = unique_keep_order(
        [normalize_ws(obj.get("primary_claim_type"))]
        + [str(x) for x in (obj.get("secondary_claim_types") or [])]
    )
    replacements = infer_replacement_solution_labels_from_context(context_text, claim_types)
    return replacements or labels


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_minimal(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def expected_language_for_patent_id(patent_id: str) -> str:
    prefix = patent_id[:2].lower()
    return {"cn": "zh", "us": "en", "kr": "ko"}.get(prefix, "")


def qc_flags(obj: Dict[str, Any], path: Path) -> List[str]:
    flags: List[str] = []
    for field in REQUIRED_FIELDS:
        if obj.get(field) in (None, "", []):
            flags.append(f"missing_{field}")

    patent_id = normalize_ws(obj.get("patent_id"))
    if patent_id and patent_id != path.stem.replace(".minimal", ""):
        flags.append("patent_id_filename_mismatch")

    title = normalize_ws(obj.get("title_source"))
    core_subject = normalize_ws(obj.get("core_subject"))
    if len(title) > 180:
        flags.append("long_title_gt160")
    if TITLE_CONTAMINATION_RE.search(title):
        flags.append("title_contamination")
    if TITLE_CONTAMINATION_RE.search(core_subject):
        flags.append("core_subject_contamination")
    if GENERIC_SUBJECT_RE.search(core_subject):
        flags.append("generic_core_subject")

    if obj.get("primary_claim_type") == "unknown":
        flags.append("unknown_primary_claim_type")
    if len(obj.get("evidence_ids") or []) < 2:
        flags.append("low_evidence_lt2")

    labels = normalize_solution_labels_for_quality(obj)
    if any(label in SUSPICIOUS_LABELS for label in labels):
        flags.append("suspicious_solution_label")
    if any(label in LOW_VALUE_LABELS for label in labels):
        flags.append("low_value_solution_label")
    if any(label.startswith(("core_", "claimed_", "safe_")) for label in labels):
        flags.append("fallback_like_solution_label")

    expected_language = expected_language_for_patent_id(patent_id)
    if expected_language and obj.get("source_language") != expected_language:
        flags.append("source_language_mismatch")

    return flags


def iter_rows() -> Iterable[Dict[str, Any]]:
    for path in sorted(MINIMAL_DIR.glob("*.minimal.json")):
        obj = load_minimal(path)
        flags = qc_flags(obj, path)
        row = {
            "patent_id": obj.get("patent_id"),
            "source_language": obj.get("source_language"),
            "summary_language": obj.get("summary_language"),
            "title_source": obj.get("title_source"),
            "title_ko": obj.get("title_ko", ""),
            "primary_claim_type": obj.get("primary_claim_type"),
            "secondary_claim_types": obj.get("secondary_claim_types") or [],
            "independent_claim_nos": obj.get("independent_claim_nos") or [],
            "protected_terms": obj.get("protected_terms") or [],
            "core_subject": obj.get("core_subject"),
            "core_elements": obj.get("core_elements") or [],
            "problem_labels": obj.get("problem_labels") or [],
            "solution_labels": normalize_solution_labels_for_quality(obj),
            "effect_labels": obj.get("effect_labels") or [],
            "evidence_ids": obj.get("evidence_ids") or [],
            "confidence": obj.get("confidence"),
            "json_path": str(path),
            "qc_flags": flags,
        }
        row["search_text"] = normalize_ws(
            " ".join(
                [
                    row["patent_id"] or "",
                    row["title_source"] or "",
                    row["core_subject"] or "",
                    " ".join(row["core_elements"]),
                    " ".join(row["problem_labels"]),
                    " ".join(row["solution_labels"]),
                    " ".join(row["effect_labels"]),
                    " ".join(row["protected_terms"]),
                ]
            )
        )
        yield row


def build_sqlite(rows: List[Dict[str, Any]], db_path: Path) -> None:
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        """
        CREATE TABLE minimal_index (
            patent_id TEXT PRIMARY KEY,
            source_language TEXT,
            summary_language TEXT,
            title_source TEXT,
            title_ko TEXT,
            primary_claim_type TEXT,
            secondary_claim_types_json TEXT,
            independent_claim_nos_json TEXT,
            protected_terms_json TEXT,
            core_subject TEXT,
            core_elements_json TEXT,
            problem_labels_json TEXT,
            solution_labels_json TEXT,
            effect_labels_json TEXT,
            evidence_ids_json TEXT,
            confidence REAL,
            json_path TEXT,
            qc_flags_json TEXT,
            search_text TEXT
        )
        """
    )
    cur.execute("CREATE INDEX idx_minimal_language ON minimal_index(source_language)")
    cur.execute("CREATE INDEX idx_minimal_primary_claim_type ON minimal_index(primary_claim_type)")
    cur.execute("CREATE INDEX idx_minimal_confidence ON minimal_index(confidence)")

    cur.execute(
        """
        CREATE TABLE minimal_labels (
            patent_id TEXT,
            label_type TEXT,
            label TEXT,
            FOREIGN KEY(patent_id) REFERENCES minimal_index(patent_id)
        )
        """
    )
    cur.execute("CREATE INDEX idx_minimal_labels_label ON minimal_labels(label)")
    cur.execute("CREATE INDEX idx_minimal_labels_type ON minimal_labels(label_type)")

    cur.execute(
        """
        CREATE TABLE minimal_evidence (
            patent_id TEXT,
            evidence_id TEXT,
            FOREIGN KEY(patent_id) REFERENCES minimal_index(patent_id)
        )
        """
    )
    cur.execute("CREATE INDEX idx_minimal_evidence_id ON minimal_evidence(evidence_id)")

    cur.executemany(
        """
        INSERT INTO minimal_index VALUES (
            :patent_id,
            :source_language,
            :summary_language,
            :title_source,
            :title_ko,
            :primary_claim_type,
            :secondary_claim_types_json,
            :independent_claim_nos_json,
            :protected_terms_json,
            :core_subject,
            :core_elements_json,
            :problem_labels_json,
            :solution_labels_json,
            :effect_labels_json,
            :evidence_ids_json,
            :confidence,
            :json_path,
            :qc_flags_json,
            :search_text
        )
        """,
        [
            {
                **row,
                "secondary_claim_types_json": json_dumps(row["secondary_claim_types"]),
                "independent_claim_nos_json": json_dumps(row["independent_claim_nos"]),
                "protected_terms_json": json_dumps(row["protected_terms"]),
                "core_elements_json": json_dumps(row["core_elements"]),
                "problem_labels_json": json_dumps(row["problem_labels"]),
                "solution_labels_json": json_dumps(row["solution_labels"]),
                "effect_labels_json": json_dumps(row["effect_labels"]),
                "evidence_ids_json": json_dumps(row["evidence_ids"]),
                "qc_flags_json": json_dumps(row["qc_flags"]),
            }
            for row in rows
        ],
    )

    label_rows = []
    for row in rows:
        for label_type in ["problem_labels", "solution_labels", "effect_labels"]:
            for label in row[label_type]:
                label_rows.append((row["patent_id"], label_type, label))
    cur.executemany("INSERT INTO minimal_labels VALUES (?, ?, ?)", label_rows)

    evidence_rows = [
        (row["patent_id"], evidence_id)
        for row in rows
        for evidence_id in row["evidence_ids"]
    ]
    cur.executemany("INSERT INTO minimal_evidence VALUES (?, ?)", evidence_rows)

    try:
        cur.execute(
            "CREATE VIRTUAL TABLE minimal_index_fts USING fts5("
            "patent_id UNINDEXED, title_source, core_subject, search_text)"
        )
        cur.executemany(
            "INSERT INTO minimal_index_fts VALUES (?, ?, ?, ?)",
            [
                (
                    row["patent_id"],
                    row["title_source"],
                    row["core_subject"],
                    row["search_text"],
                )
                for row in rows
            ],
        )
    except sqlite3.OperationalError:
        pass

    con.commit()
    con.close()


def build_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json_dumps(row) + "\n")


def build_qc_report(rows: List[Dict[str, Any]], path: Path) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    examples: Dict[str, List[str]] = {}
    for row in rows:
        for flag in row["qc_flags"]:
            counts[flag] = counts.get(flag, 0) + 1
            examples.setdefault(flag, [])
            if len(examples[flag]) < 12:
                examples[flag].append(row["patent_id"])
    report = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "minimal_dir": str(MINIMAL_DIR),
        "total": len(rows),
        "qc_counts": dict(sorted(counts.items())),
        "qc_examples": {k: examples[k] for k in sorted(examples)},
        "outputs": {
            "sqlite": str(INDEX_SQLITE),
            "jsonl": str(INDEX_JSONL),
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    global MINIMAL_DIR, INDEX_DIR, INDEX_SQLITE, INDEX_JSONL, QC_REPORT

    parser = argparse.ArgumentParser()
    parser.add_argument("--minimal-dir", type=Path, default=MINIMAL_DIR)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    args = parser.parse_args()

    MINIMAL_DIR = args.minimal_dir
    INDEX_DIR = args.index_dir
    INDEX_SQLITE = INDEX_DIR / "patent_minimal_index.sqlite"
    INDEX_JSONL = INDEX_DIR / "patent_minimal_index.jsonl"
    QC_REPORT = INDEX_DIR / "patent_minimal_qc_report.json"

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(iter_rows())
    build_sqlite(rows, INDEX_SQLITE)
    build_jsonl(rows, INDEX_JSONL)
    report = build_qc_report(rows, QC_REPORT)

    print(f"[INDEX] total={report['total']}")
    print(f"[INDEX] sqlite={INDEX_SQLITE}")
    print(f"[INDEX] jsonl={INDEX_JSONL}")
    print(f"[INDEX] qc_report={QC_REPORT}")
    print(f"[INDEX] qc_counts={json_dumps(report['qc_counts'])}")


if __name__ == "__main__":
    main()
