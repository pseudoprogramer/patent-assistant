from __future__ import annotations

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BASE = Path("/Volumes/외장 2TB/cpu2026")
DEFAULT_EVIDENCE_DB = BASE / "common" / "runtime" / "db" / "patent_A4.sqlite"
DEFAULT_PACK_DB = BASE / "patent_hub" / "outputs" / "indexes" / "A4" / "patent_evidence_pack_index.sqlite"
DEFAULT_OUT_DIR = BASE / "common" / "runtime" / "reports" / "A4"

FIG_FRONTMATTER_RE = re.compile(
    r"(U\.S\.\s+Patent|Sheet\s+\d+\s+of\s+\d+|FIG\.?\s*\d+|cited by examiner|"
    r"Foreign Application Priority Data|References Cited|Primary Examiner)",
    re.I,
)
FLOWCHART_STEP_RE = re.compile(r"\bS\d{2,4}\b|\$[0-9]{2,4}|[-—]{2,}\s*\$?[0-9]{2,4}", re.I)
CLAIM_START_RE = re.compile(
    r"^\s*(?:\d+\.?\s*)?(?:A|An|The|That|In|Method|Device|System|Apparatus|Circuit|Memory|Non-transitory|One or more|"
    r"What is claimed is|一种|一種|根据|如权利要求|基于|제\s*\d+\s*항|청구항|"
    r"장치|방법|시스템|메모리|반도체)",
    re.I,
)
CLAIM_KEYWORD_RE = re.compile(
    r"\b(comprising|wherein|configured|including|method|device|system|apparatus|circuit|memory|claim)\b|"
    r"(包括|包含|其中|配置|方法|装置|系统|电路|存储器|权利要求|청구항|제\s*\d+\s*항|상기|있어서|포함|구비|방법|장치|시스템|회로|메모리)",
    re.I,
)
OCR_GIBBERISH_RE = re.compile(r"\b(?:Memary|Mernory|EONTRELLER|CENERATOR|Whas|Unesiacted|desver|Aysueg)\b", re.I)


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def json_loads(value: Any) -> Any:
    try:
        return json.loads(value or "[]")
    except Exception:
        return []


def char_stats(text: str) -> Dict[str, float]:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return {"length": 0, "letter_ratio": 0.0, "symbol_ratio": 0.0, "digit_ratio": 0.0}
    letters = sum(ch.isalpha() or "\u4e00" <= ch <= "\u9fff" or "\uac00" <= ch <= "\ud7a3" for ch in compact)
    digits = sum(ch.isdigit() for ch in compact)
    symbols = len(compact) - letters - digits
    return {
        "length": len(compact),
        "letter_ratio": round(letters / len(compact), 3),
        "symbol_ratio": round(symbols / len(compact), 3),
        "digit_ratio": round(digits / len(compact), 3),
    }


def repeated_short_token_score(text: str) -> int:
    tokens = re.findall(r"\b[A-Za-z]{1,3}\b", text)
    counts = Counter(token.lower() for token in tokens)
    return sum(count for token, count in counts.items() if count >= 8)


def looks_like_substantive_claim(text: str) -> bool:
    return bool(CLAIM_START_RE.search(text) and CLAIM_KEYWORD_RE.search(text))


def claim_quality_flags(text: str, claim_type: str = "") -> Tuple[List[str], int]:
    text = normalize_ws(text)
    stats = char_stats(text)
    flags: List[str] = []
    severity = 0
    substantive_claim = looks_like_substantive_claim(text)

    if not text:
        flags.append("empty_text")
        return flags, 100
    if len(text) < 80:
        flags.append("very_short_claim_text")
        severity += 25
    if FIG_FRONTMATTER_RE.search(text):
        flags.append("figure_or_frontmatter_text")
        severity += 10 if substantive_claim else 45
    if FLOWCHART_STEP_RE.search(text):
        flags.append("flowchart_step_text")
        severity += 5 if substantive_claim else 20
    if OCR_GIBBERISH_RE.search(text):
        flags.append("ocr_gibberish_terms")
        severity += 35
    if stats["symbol_ratio"] > 0.24:
        flags.append("high_symbol_ratio")
        severity += 8 if substantive_claim else 30
    if stats["letter_ratio"] < 0.48 and stats["length"] >= 120:
        flags.append("low_letter_ratio")
        severity += 8 if substantive_claim else 30
    if text.count("|") >= 4 or text.count("—") + text.count("-") >= 14:
        flags.append("diagram_line_art")
        severity += 8 if substantive_claim else 25
    if repeated_short_token_score(text) >= 18:
        flags.append("repeated_short_ocr_tokens")
        severity += 5 if substantive_claim else 20
    if claim_type == "independent" and len(text) >= 120 and not CLAIM_KEYWORD_RE.search(text):
        flags.append("independent_without_claim_keywords")
        severity += 20
    if len(text) >= 120 and not CLAIM_START_RE.search(text) and FIG_FRONTMATTER_RE.search(text):
        flags.append("does_not_start_like_claim")
        severity += 20

    return flags, severity


def severity_bucket(score: int) -> str:
    if score >= 70:
        return "severe"
    if score >= 45:
        return "high"
    if score >= 25:
        return "medium"
    if score > 0:
        return "low"
    return "ok"


def audit_claims(evidence_db: Path, sample_limit: int) -> Dict[str, Any]:
    con = sqlite3.connect(evidence_db)
    rows = con.execute(
        """
        SELECT patent_id, claim_no, parent_claim_no, claim_type, raw_text, norm_text, page_start, page_end
        FROM claims
        ORDER BY patent_id, CAST(claim_no AS INT), claim_no
        """
    ).fetchall()
    con.close()

    total = len(rows)
    by_bucket: Counter[str] = Counter()
    by_flag: Counter[str] = Counter()
    patents_by_bucket: Dict[str, set] = defaultdict(set)
    bad_patents: Counter[str] = Counter()
    samples: List[Dict[str, Any]] = []

    for row in rows:
        patent_id, claim_no, parent_claim_no, claim_type, raw_text, norm_text, page_start, page_end = row
        text = normalize_ws(raw_text or norm_text)
        flags, score = claim_quality_flags(text, claim_type=claim_type or "")
        bucket = severity_bucket(score)
        by_bucket[bucket] += 1
        patents_by_bucket[bucket].add(patent_id)
        for flag in flags:
            by_flag[flag] += 1
        if bucket in {"severe", "high"}:
            bad_patents[patent_id] += 1
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "patent_id": patent_id,
                        "claim_no": str(claim_no),
                        "claim_type": claim_type or "",
                        "page_start": page_start,
                        "page_end": page_end,
                        "bucket": bucket,
                        "score": score,
                        "flags": flags,
                        "text_preview": text[:700],
                    }
                )

    return {
        "total_claims": total,
        "bucket_counts": dict(by_bucket),
        "patent_counts_by_bucket": {bucket: len(values) for bucket, values in patents_by_bucket.items()},
        "flag_counts": dict(by_flag.most_common()),
        "high_or_severe_patents": len(bad_patents),
        "top_bad_patents": bad_patents.most_common(30),
        "samples": samples,
    }


def audit_packs(pack_db: Path, sample_limit: int) -> Dict[str, Any]:
    con = sqlite3.connect(pack_db)
    rows = con.execute(
        """
        SELECT patent_id, strong_claims_json, support_claims_json, quality_flags_json
        FROM evidence_pack_index
        ORDER BY patent_id
        """
    ).fetchall()
    con.close()

    pack_count = len(rows)
    affected_packs: Counter[str] = Counter()
    by_flag: Counter[str] = Counter()
    by_bucket: Counter[str] = Counter()
    samples: List[Dict[str, Any]] = []
    missing_claim_text = 0

    for patent_id, strong_json, support_json, quality_json in rows:
        claims = []
        for source, values in (("strong", json_loads(strong_json)), ("support", json_loads(support_json))):
            for claim in values:
                item = dict(claim)
                item["_source"] = source
                claims.append(item)
        if not claims:
            missing_claim_text += 1
        pack_max_score = 0
        pack_flags: List[str] = []
        for claim in claims:
            flags, score = claim_quality_flags(claim.get("text", ""), claim_type=claim.get("claim_type", ""))
            bucket = severity_bucket(score)
            if bucket != "ok":
                by_bucket[bucket] += 1
            for flag in flags:
                by_flag[flag] += 1
            if score > pack_max_score:
                pack_max_score = score
                pack_flags = flags
        if pack_max_score >= 45:
            affected_packs[patent_id] += 1
            if len(samples) < sample_limit:
                samples.append(
                    {
                        "patent_id": patent_id,
                        "bucket": severity_bucket(pack_max_score),
                        "score": pack_max_score,
                        "flags": pack_flags,
                        "existing_quality_flags": json_loads(quality_json),
                        "claim_count_in_pack": len(claims),
                        "claim_preview": normalize_ws((claims[0] if claims else {}).get("text", ""))[:700],
                    }
                )

    return {
        "total_packs": pack_count,
        "packs_without_claim_text": missing_claim_text,
        "affected_packs_high_or_severe": len(affected_packs),
        "claim_bucket_counts_inside_packs": dict(by_bucket),
        "flag_counts_inside_packs": dict(by_flag.most_common()),
        "samples": samples,
    }


def write_markdown(report: Dict[str, Any], md_path: Path) -> None:
    claims = report["claims"]
    packs = report["packs"]
    lines = [
        "# A4 Evidence Quality Audit",
        "",
        "## Summary",
        f"- total claims: {claims['total_claims']}",
        f"- claim bucket counts: `{claims['bucket_counts']}`",
        f"- high/severe claim patents: {claims['high_or_severe_patents']}",
        f"- total evidence packs: {packs['total_packs']}",
        f"- packs without claim text: {packs['packs_without_claim_text']}",
        f"- packs with high/severe contaminated claim text: {packs['affected_packs_high_or_severe']}",
        "",
        "## Top Claim Flags",
    ]
    for flag, count in list(claims["flag_counts"].items())[:16]:
        lines.append(f"- {flag}: {count}")
    lines.extend(["", "## Top Bad Patents"])
    for patent_id, count in claims["top_bad_patents"][:20]:
        lines.append(f"- {patent_id}: {count}")
    lines.extend(["", "## Claim Samples"])
    for sample in claims["samples"][:20]:
        lines.extend(
            [
                f"### {sample['patent_id']} claim_{sample['claim_no']} ({sample['bucket']}, score={sample['score']})",
                f"- flags: {', '.join(sample['flags'])}",
                "```text",
                sample["text_preview"],
                "```",
            ]
        )
    lines.extend(["", "## Pack Samples"])
    for sample in packs["samples"][:20]:
        lines.extend(
            [
                f"### {sample['patent_id']} ({sample['bucket']}, score={sample['score']})",
                f"- flags: {', '.join(sample['flags'])}",
                f"- existing quality flags: {', '.join(sample['existing_quality_flags'])}",
                "```text",
                sample["claim_preview"],
                "```",
            ]
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit claim/evidence pack contamination without modifying DBs.")
    parser.add_argument("--evidence-db", default=str(DEFAULT_EVIDENCE_DB))
    parser.add_argument("--pack-db", default=str(DEFAULT_PACK_DB))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--sample-limit", type=int, default=40)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "evidence_db": str(args.evidence_db),
        "pack_db": str(args.pack_db),
        "claims": audit_claims(Path(args.evidence_db), args.sample_limit),
        "packs": audit_packs(Path(args.pack_db), args.sample_limit),
    }
    json_path = out_dir / "evidence_quality_audit.json"
    md_path = out_dir / "evidence_quality_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        summary = {
            "json_path": str(json_path),
            "md_path": str(md_path),
            "claim_bucket_counts": report["claims"]["bucket_counts"],
            "high_or_severe_claim_patents": report["claims"]["high_or_severe_patents"],
            "affected_packs_high_or_severe": report["packs"]["affected_packs_high_or_severe"],
            "packs_without_claim_text": report["packs"]["packs_without_claim_text"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
