from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import patent_minimal_index as minimal
from llm_clients import LLMClient, json_from_text


BASE = Path("/Volumes/외장 2TB/cpu2026")
DEFAULT_DB = BASE / "common" / "runtime" / "db" / "patent_A4_v2_compare200.sqlite"
DEFAULT_QWEN_DIR = BASE / "patent_hub" / "outputs" / "minimal_analysis" / "A4" / "v2_compare200_qwen3_14b"
DEFAULT_GEMINI_DIR = BASE / "patent_hub" / "outputs" / "minimal_analysis" / "A4" / "v2_compare200_gemini"
DEFAULT_REPORT_DIR = BASE / "common" / "runtime" / "reports" / "A4" / "minimal_labeler_compare"


def select_patents(db_path: Path, explicit: List[str], limit: int) -> List[str]:
    if explicit:
        return explicit[:limit] if limit else explicit
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("SELECT patent_id FROM patents ORDER BY patent_id LIMIT ?", (limit,)).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        con.close()


def make_gemini_caller(client: LLMClient):
    def call_gemini_minimal(prompt: str, patent_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        started = time.monotonic()
        instructions = (
            "You are a strict patent indexing JSON generator. "
            "Return exactly one valid JSON object matching the schema in the prompt. "
            "Do not include markdown, comments, prose, or extra keys."
        )
        text = client.generate(prompt, instructions=instructions, max_tokens=1200, temperature=0.0)
        parsed = json_from_text(text)
        return parsed, {
            "wall_seconds": round(time.monotonic() - started, 2),
            "ollama_total_seconds": None,
            "prompt_eval_count": None,
            "eval_count": None,
            "done_reason": "gemini",
            "provider": client.provider,
            "model": client.model,
        }

    return call_gemini_minimal


def run_gemini_labels(
    db_path: Path,
    output_dir: Path,
    patent_ids: List[str],
    provider: str,
    model: str,
) -> List[Dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    minimal.A4_DB = db_path
    minimal.MINIMAL_DIR = output_dir
    minimal.ensure_dirs()
    minimal.init_log_file()

    client = LLMClient(provider=provider, model=model or None, timeout=300)
    minimal.MODEL = client.model
    minimal.call_ollama_minimal = make_gemini_caller(client)  # type: ignore[assignment]

    con = minimal.open_db()
    results: List[Dict[str, Any]] = []
    try:
        for index, patent_id in enumerate(patent_ids, 1):
            print(f"[gemini-label] {index}/{len(patent_ids)} {patent_id}", flush=True)
            try:
                result = minimal.process_one_patent(con, patent_id, overwrite=True)
                result["status"] = "success"
            except Exception as exc:
                result = {"patent_id": patent_id, "status": "failed", "error": repr(exc)}
            results.append(result)
    finally:
        con.close()
    return results


def load_minimal(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def label_score(card: Dict[str, Any]) -> Dict[str, Any]:
    if not card:
        return {"score": 0, "missing": ["file_missing"]}
    missing = []
    for key in ["core_subject", "core_elements", "solution_labels", "evidence_ids"]:
        if not card.get(key):
            missing.append(key)
    for key in ["problem_labels", "effect_labels"]:
        if not card.get(key):
            missing.append(key)
    score = 100
    score -= 18 * len([x for x in missing if x in {"core_subject", "core_elements", "solution_labels", "evidence_ids"}])
    score -= 8 * len([x for x in missing if x in {"problem_labels", "effect_labels"}])
    if float(card.get("confidence") or 0) < 0.75:
        score -= 8
    generic = {"memory_program_operation", "memory_control_operation", "claimed_device_structure", "claimed_method_process"}
    generic_hits = sorted(set(card.get("solution_labels") or []) & generic)
    score -= 5 * len(generic_hits)
    return {
        "score": max(0, score),
        "missing": missing,
        "generic_solution_labels": generic_hits,
        "confidence": card.get("confidence"),
    }


def compare_outputs(patent_ids: List[str], qwen_dir: Path, gemini_dir: Path) -> Dict[str, Any]:
    items = []
    for patent_id in patent_ids:
        qwen = load_minimal(qwen_dir / f"{minimal.safe_name(patent_id)}.minimal.json")
        gemini = load_minimal(gemini_dir / f"{minimal.safe_name(patent_id)}.minimal.json")
        item = {
            "patent_id": patent_id,
            "qwen_score": label_score(qwen),
            "gemini_score": label_score(gemini),
            "qwen": {
                "core_subject": qwen.get("core_subject"),
                "core_elements": qwen.get("core_elements"),
                "problem_labels": qwen.get("problem_labels"),
                "solution_labels": qwen.get("solution_labels"),
                "effect_labels": qwen.get("effect_labels"),
                "evidence_ids": qwen.get("evidence_ids"),
                "confidence": qwen.get("confidence"),
            },
            "gemini": {
                "core_subject": gemini.get("core_subject"),
                "core_elements": gemini.get("core_elements"),
                "problem_labels": gemini.get("problem_labels"),
                "solution_labels": gemini.get("solution_labels"),
                "effect_labels": gemini.get("effect_labels"),
                "evidence_ids": gemini.get("evidence_ids"),
                "confidence": gemini.get("confidence"),
            },
        }
        items.append(item)
    summary = {
        "patents": len(items),
        "qwen_avg_score": round(sum(x["qwen_score"]["score"] for x in items) / max(1, len(items)), 1),
        "gemini_avg_score": round(sum(x["gemini_score"]["score"] for x in items) / max(1, len(items)), 1),
        "qwen_missing_problem": sum(1 for x in items if "problem_labels" in x["qwen_score"]["missing"]),
        "gemini_missing_problem": sum(1 for x in items if "problem_labels" in x["gemini_score"]["missing"]),
        "qwen_missing_effect": sum(1 for x in items if "effect_labels" in x["qwen_score"]["missing"]),
        "gemini_missing_effect": sum(1 for x in items if "effect_labels" in x["gemini_score"]["missing"]),
    }
    return {"summary": summary, "items": items}


def write_markdown(report: Dict[str, Any], path: Path) -> None:
    lines = ["# Minimal Labeler Comparison", ""]
    for key, value in report["summary"].items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "| patent_id | qwen score | gemini score | qwen missing | gemini missing |",
            "|---|---:|---:|---|---|",
        ]
    )
    for item in report["items"]:
        lines.append(
            f"| {item['patent_id']} | {item['qwen_score']['score']} | {item['gemini_score']['score']} | "
            f"{', '.join(item['qwen_score']['missing']) or '-'} | "
            f"{', '.join(item['gemini_score']['missing']) or '-'} |"
        )
    lines.append("")
    for item in report["items"]:
        lines.append(f"## {item['patent_id']}")
        lines.append("")
        lines.append(f"- qwen core_subject: `{item['qwen']['core_subject']}`")
        lines.append(f"- gemini core_subject: `{item['gemini']['core_subject']}`")
        lines.append(f"- qwen problem/effect: `{item['qwen']['problem_labels']}` / `{item['qwen']['effect_labels']}`")
        lines.append(f"- gemini problem/effect: `{item['gemini']['problem_labels']}` / `{item['gemini']['effect_labels']}`")
        lines.append(f"- qwen solution: `{item['qwen']['solution_labels']}`")
        lines.append(f"- gemini solution: `{item['gemini']['solution_labels']}`")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare local Qwen minimal labels with Gemini labels.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--patent-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--provider", default="gemini")
    parser.add_argument("--model", default="")
    parser.add_argument("--qwen-dir", default=str(DEFAULT_QWEN_DIR))
    parser.add_argument("--gemini-dir", default=str(DEFAULT_GEMINI_DIR))
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR))
    parser.add_argument("--skip-gemini-run", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db)
    qwen_dir = Path(args.qwen_dir)
    gemini_dir = Path(args.gemini_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    patent_ids = select_patents(db_path, args.patent_id, args.limit)
    if not args.skip_gemini_run:
        run_results = run_gemini_labels(
            db_path=db_path,
            output_dir=gemini_dir,
            patent_ids=patent_ids,
            provider=args.provider,
            model=args.model,
        )
    else:
        run_results = []

    report = compare_outputs(patent_ids, qwen_dir, gemini_dir)
    report["gemini_run_results"] = run_results
    json_path = report_dir / "minimal_labeler_compare.json"
    md_path = report_dir / "minimal_labeler_compare.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(report, md_path)
    print(json.dumps({"summary": report["summary"], "json_path": str(json_path), "md_path": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
