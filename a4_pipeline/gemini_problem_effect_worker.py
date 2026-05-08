from __future__ import annotations

import argparse
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import probe_problem_effect_evidence as probe
from repair_problem_effect_with_pro import repair_one


BASE = Path("/Volumes/외장 2TB/cpu2026")
DEFAULT_REBUILD = BASE / "rebuilds" / "A4_evidence_v2_full_20260508"
DEFAULT_DB = DEFAULT_REBUILD / "db" / "patent_A4_evidence_v2_full.sqlite"
DEFAULT_MINIMAL_DIR = DEFAULT_REBUILD / "minimal_analysis"
DEFAULT_WORK_DIR = DEFAULT_REBUILD / "gemini_problem_effect_repair"


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_seen_keys(paths: Iterable[Path]) -> Set[Tuple[str, str]]:
    seen: Set[Tuple[str, str]] = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                patent_id = str(item.get("patent_id") or "")
                for reason in item.get("weak_reasons") or []:
                    if patent_id and reason:
                        seen.add((patent_id, str(reason)))
    return seen


def load_recent_failed_keys(path: Path, retry_after_sec: float) -> Set[Tuple[str, str]]:
    if not path.exists() or retry_after_sec <= 0:
        return set()
    cutoff = datetime.now() - timedelta(seconds=retry_after_sec)
    seen: Set[Tuple[str, str]] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                at = item.get("at")
                if not at or datetime.fromisoformat(str(at)) < cutoff:
                    continue
            except Exception:
                continue
            patent_id = str(item.get("patent_id") or "")
            for reason in item.get("weak_reasons") or []:
                if patent_id and reason:
                    seen.add((patent_id, str(reason)))
    return seen


def weak_reasons(card: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    if not card.get("problem_labels"):
        reasons.append("empty_problem_labels")
    if not card.get("effect_labels"):
        reasons.append("empty_effect_labels")
    if not card.get("solution_labels"):
        reasons.append("empty_solution_labels")
    if card.get("fallback_reason"):
        reasons.append("fallback_output")
    return reasons


def iter_weak_minimal_cards(minimal_dir: Path, seen_keys: Set[Tuple[str, str]], max_items: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in sorted(minimal_dir.glob("*.minimal.json")):
        try:
            card = load_json(path)
        except Exception:
            continue
        patent_id = str(card.get("patent_id") or path.name.split(".")[0])
        if not patent_id:
            continue
        reasons = weak_reasons(card)
        if not reasons:
            continue
        reasons = [reason for reason in reasons if (patent_id, reason) not in seen_keys]
        if not reasons:
            continue
        items.append({"patent_id": patent_id, "path": str(path), "weak_reasons": reasons, "card": card})
        if len(items) >= max_items:
            break
    return items


def get_minimal_failed_ids(db_path: Path) -> List[str]:
    if not db_path.exists():
        return []
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT patent_id
            FROM jobs
            WHERE status='minimal_failed'
            ORDER BY updated_at ASC, patent_id ASC
            """
        ).fetchall()
        return [str(row[0]) for row in rows]
    finally:
        con.close()


def load_independent_claims(con: sqlite3.Connection, patent_id: str) -> List[Dict[str, Any]]:
    rows = con.execute(
        """
        SELECT claim_no, raw_text, norm_text
        FROM claims
        WHERE patent_id=? AND claim_type='independent'
        ORDER BY claim_no
        LIMIT 4
        """,
        (patent_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def mark_brief_done(db_path: Path, patent_id: str) -> None:
    con = sqlite3.connect(db_path)
    try:
        con.execute(
            """
            UPDATE jobs
            SET status='brief_done', updated_at=?
            WHERE patent_id=?
            """,
            (now(), patent_id),
        )
        con.commit()
    finally:
        con.close()


def minimal_card_valid(card: Dict[str, Any]) -> bool:
    required = ["patent_id", "title_source", "source_language", "core_subject", "core_elements_ko", "solution_labels", "evidence_ids"]
    return all(card.get(key) for key in required)


def enforce_requested_repairs(result: Dict[str, Any], weak_reasons: List[str]) -> None:
    repair = result.get("repair") or {}
    quality_flags = list(result.get("quality_flags") or [])
    required_pairs = [
        ("empty_problem_labels", "problem_labels", "missing_repaired_problem_labels"),
        ("empty_effect_labels", "effect_labels", "missing_repaired_effect_labels"),
        ("empty_solution_labels", "solution_labels", "missing_repaired_solution_labels"),
    ]
    for reason, field, flag in required_pairs:
        if reason in weak_reasons and not repair.get(field) and flag not in quality_flags:
            quality_flags.append(flag)
    result["quality_flags"] = quality_flags
    result["quality_pass"] = result.get("status") == "success" and not quality_flags


def apply_quality_repair(minimal_path: Path, result: Dict[str, Any]) -> bool:
    if not result.get("quality_pass"):
        return False
    repair = result.get("repair") or {}
    card = load_json(minimal_path)
    changed = False
    if repair.get("problem_labels") and not card.get("problem_labels"):
        card["problem_labels"] = repair["problem_labels"]
        changed = True
    if repair.get("effect_labels") and not card.get("effect_labels"):
        card["effect_labels"] = repair["effect_labels"]
        changed = True
    if repair.get("solution_labels") and not card.get("solution_labels"):
        card["solution_labels"] = repair["solution_labels"]
        changed = True
    card["_gemini_problem_effect_repair"] = {
        "repaired_at": now(),
        "source": "gemini-cli",
        "quality_pass": bool(result.get("quality_pass")),
        "supporting_snippet_ids": repair.get("supporting_snippet_ids") or [],
        "supporting_claim_ids": repair.get("supporting_claim_ids") or [],
        "why_needed": repair.get("why_needed", ""),
        "expected_effect": repair.get("expected_effect", ""),
        "confidence": repair.get("confidence", 0),
    }
    save_json(minimal_path, card)
    return changed


def write_queue_snapshots(work_dir: Path, candidates: List[Dict[str, Any]], failed_ids: List[str]) -> None:
    (work_dir / "weak_candidates_latest.txt").write_text(
        "\n".join(item["patent_id"] for item in candidates) + ("\n" if candidates else ""),
        encoding="utf-8",
    )
    (work_dir / "qwen_minimal_failed_latest.txt").write_text(
        "\n".join(failed_ids) + ("\n" if failed_ids else ""),
        encoding="utf-8",
    )


def run_loop(args: argparse.Namespace) -> None:
    db_path = Path(args.db)
    minimal_dir = Path(args.minimal_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    results_path = work_dir / "gemini_repair_results.jsonl"
    failed_path = work_dir / "gemini_repair_failed.jsonl"
    skipped_path = work_dir / "gemini_repair_skipped.jsonl"

    attempts = 0
    while True:
        # Do not let old successful JSONL rows suppress repair forever.
        # The source of truth is the current minimal card: if the card still has
        # weak fields, retry it. Only explicit skips and recent failures back off.
        seen_keys = load_seen_keys([skipped_path])
        seen_keys.update(load_recent_failed_keys(failed_path, args.retry_failed_after_sec))
        candidates = iter_weak_minimal_cards(minimal_dir, seen_keys, max_items=args.batch_size)
        failed_minimal_ids = get_minimal_failed_ids(db_path)
        write_queue_snapshots(work_dir, candidates, failed_minimal_ids)

        if not candidates:
            print(f"[gemini-worker] {now()} no weak minimal cards; sleep={args.poll_sec}s", flush=True)
            if args.once:
                break
            time.sleep(args.poll_sec)
            continue

        con = probe.connect(db_path)
        try:
            for item in candidates:
                patent_id = item["patent_id"]
                minimal_path = Path(item["path"])
                print(
                    f"[gemini-worker] repair patent_id={patent_id} reasons={','.join(item['weak_reasons'])}",
                    flush=True,
                )
                try:
                    probe_item = probe.run_one(con, patent_id, max_pages=None)
                    probe_item["independent_claims"] = load_independent_claims(con, patent_id)
                    result = repair_one(
                        None,
                        probe_item,
                        provider="gemini-cli",
                        model=args.model,
                        timeout_sec=args.timeout_sec,
                    )
                    result["weak_reasons"] = item["weak_reasons"]
                    result["minimal_path"] = str(minimal_path)
                    result["at"] = now()
                    enforce_requested_repairs(result, item["weak_reasons"])
                    result["applied"] = False
                    if args.apply and result.get("quality_pass"):
                        result["applied"] = apply_quality_repair(minimal_path, result)
                        try:
                            repaired_card = load_json(minimal_path)
                            if minimal_card_valid(repaired_card):
                                mark_brief_done(db_path, patent_id)
                        except Exception:
                            pass
                    target = results_path if result.get("quality_pass") else failed_path
                    append_jsonl(target, result)
                    print(
                        f"[gemini-worker]   status={result['status']} quality_pass={result['quality_pass']} "
                        f"applied={result['applied']} elapsed={result['elapsed']}s",
                        flush=True,
                    )
                except Exception as exc:
                    append_jsonl(
                        failed_path,
                        {
                            "patent_id": patent_id,
                            "minimal_path": str(minimal_path),
                            "weak_reasons": item["weak_reasons"],
                            "status": "failed",
                            "quality_pass": False,
                            "error": repr(exc),
                            "elapsed": 0,
                            "at": now(),
                        },
                    )
                    print(f"[gemini-worker]   failed error={exc}", flush=True)
                attempts += 1
                if args.limit and attempts >= args.limit:
                    return
                if args.delay_sec > 0:
                    print(f"[gemini-worker]   sleep={args.delay_sec}s", flush=True)
                    time.sleep(args.delay_sec)
        finally:
            con.close()

        if args.once:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously repair weak minimal problem/effect fields with Gemini CLI.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--minimal-dir", default=str(DEFAULT_MINIMAL_DIR))
    parser.add_argument("--work-dir", default=str(DEFAULT_WORK_DIR))
    parser.add_argument("--model", default="gemini-2.5-flash-lite")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--delay-sec", type=float, default=60)
    parser.add_argument("--poll-sec", type=float, default=300)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--retry-failed-after-sec", type=float, default=21600)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    run_loop(args)


if __name__ == "__main__":
    main()
