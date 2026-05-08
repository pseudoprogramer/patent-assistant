from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

from llm_clients import load_env_file


BASE = Path("/Volumes/외장 2TB/cpu2026")
REBUILD = BASE / "rebuilds" / "A4_evidence_v2_full_20260508"
CODE = BASE / "common" / "code"
DB = REBUILD / "db" / "patent_A4_evidence_v2_full.sqlite"
MINIMAL_DIR = REBUILD / "minimal_analysis"
LOG_DIR = REBUILD / "logs"
APPROVAL_DIR = REBUILD / "telegram_approvals"
QUARANTINE_DIR = APPROVAL_DIR / "quarantined_minimal"
PENDING_PATH = APPROVAL_DIR / "pending_actions.json"
EVENT_LOG = APPROVAL_DIR / "approval_events.jsonl"
CHAT_LOG = BASE / "common" / "runtime" / "logs" / "A4" / "patent_telegram_chat.jsonl"

EVIDENCE_SCREEN = "a4_evidence_v2_full_20260508"
MINIMAL_SCREEN = "a4_minimal_v2_full_20260508"
GEMINI_SCREEN = "a4_gemini_pe_worker_20260508"

EVIDENCE_RUN = REBUILD / "scripts" / "run_evidence_full.sh"
MINIMAL_RUN = REBUILD / "scripts" / "run_minimal_full.sh"
GEMINI_RUN = REBUILD / "scripts" / "run_gemini_problem_effect_worker.sh"

TITLE_CONTAMINATION_RE = re.compile(
    r"(\(\s*57\s*\)|摘要|ABSTRACT|权利要求|Claims|청구범위|발명의 설명)",
    re.I,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_ws(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def append_event(event: Dict[str, Any]) -> None:
    APPROVAL_DIR.mkdir(parents=True, exist_ok=True)
    with EVENT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": now(), **event}, ensure_ascii=False) + "\n")


def load_pending() -> Dict[str, Dict[str, Any]]:
    if not PENDING_PATH.exists():
        return {}
    try:
        data = json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {str(item["id"]): item for item in data.get("pending", []) if item.get("id")}


def save_pending(pending: Dict[str, Dict[str, Any]]) -> None:
    APPROVAL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"updated_at": now(), "pending": list(pending.values())}
    tmp = PENDING_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(PENDING_PATH)


def next_action_id(action_type: str) -> str:
    stamp = datetime.now().strftime("%m%d%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", action_type.lower()).strip("_")[:18]
    return f"{slug}_{stamp}"


class TelegramClient:
    def __init__(self, token: str, timeout: int = 30) -> None:
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.timeout = timeout

    def send_message(self, chat_id: int, text: str) -> None:
        requests.post(
            f"{self.base_url}/sendMessage",
            json={"chat_id": chat_id, "text": text[:3900], "disable_web_page_preview": True},
            timeout=self.timeout,
        ).raise_for_status()


def telegram_client() -> Optional[TelegramClient]:
    load_env_file()
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        return None
    return TelegramClient(token)


def target_chat_ids() -> List[int]:
    load_env_file()
    ids: List[int] = []
    for part in os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", "").split(","):
        part = part.strip()
        if part:
            ids.append(int(part))
    if ids:
        return ids
    if CHAT_LOG.exists():
        seen: List[int] = []
        try:
            with CHAT_LOG.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        chat_id = int((json.loads(line).get("chat_id")))
                    except Exception:
                        continue
                    if chat_id not in seen:
                        seen.append(chat_id)
            ids = seen[-3:]
        except Exception as exc:
            append_event({"event": "chat_log_read_error", "error": repr(exc)})
    return ids


def send_telegram(text: str) -> None:
    client = telegram_client()
    chats = target_chat_ids()
    if not client or not chats:
        append_event({"event": "telegram_not_configured", "text": text[:500]})
        return
    for chat_id in chats:
        client.send_message(chat_id, text)


def run_cmd(args: List[str], timeout: int = 20) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout)


def screen_names() -> List[str]:
    proc = run_cmd(["screen", "-ls"], timeout=10)
    text = proc.stdout + "\n" + proc.stderr
    return re.findall(r"\d+\.([A-Za-z0-9_.-]+)\s+\(", text)


def mtime_age(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    return time.time() - path.stat().st_mtime


def job_counts() -> Dict[str, int]:
    if not DB.exists():
        return {}
    con = sqlite3.connect(str(DB))
    try:
        return {str(status): int(count) for status, count in con.execute("SELECT status, COUNT(*) FROM jobs GROUP BY status")}
    finally:
        con.close()


def patent_count() -> int:
    if not DB.exists():
        return 0
    con = sqlite3.connect(str(DB))
    try:
        return int(con.execute("SELECT COUNT(*) FROM patents").fetchone()[0])
    finally:
        con.close()


def inspect_minimal_cards() -> Dict[str, Any]:
    files = sorted(MINIMAL_DIR.glob("*.minimal.json"))
    invalid: List[Dict[str, Any]] = []
    weak: List[str] = []
    repaired = 0
    for path in files:
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            invalid.append({"patent_id": path.name.split(".")[0], "path": str(path), "reason": f"bad_json:{exc}"})
            continue
        reasons: List[str] = []
        title = str(card.get("title_source") or "")
        if TITLE_CONTAMINATION_RE.search(title):
            reasons.append("contaminated_title")
        if not card.get("country"):
            reasons.append("missing_country")
        if not card.get("source_language"):
            reasons.append("missing_source_language")
        if not card.get("solution_labels"):
            reasons.append("empty_solution_labels")
        if not card.get("evidence_ids"):
            reasons.append("empty_evidence_ids")
        if reasons:
            invalid.append(
                {
                    "patent_id": card.get("patent_id") or path.name.split(".")[0],
                    "path": str(path),
                    "reason": ",".join(reasons),
                }
            )
        if not card.get("problem_labels") or not card.get("effect_labels"):
            weak.append(str(card.get("patent_id") or path.name.split(".")[0]))
        if card.get("_gemini_problem_effect_repair"):
            repaired += 1
    return {"files": len(files), "invalid": invalid, "weak": weak, "repaired": repaired}


def create_pending_action(action_type: str, reason: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    pending = load_pending()
    mergeable_types = {"requeue_invalid_minimal", "requeue_minimal_failed"}
    if action_type in mergeable_types and payload.get("patent_ids"):
        new_ids = [str(x) for x in payload.get("patent_ids", []) if str(x).strip()]
        for item in pending.values():
            if item.get("type") != action_type or item.get("status") != "pending":
                continue
            item_payload = item.setdefault("payload", {})
            old_ids = [str(x) for x in item_payload.get("patent_ids", []) if str(x).strip()]
            merged_ids = sorted(set(old_ids) | set(new_ids))
            if merged_ids == old_ids:
                return item
            item_payload["patent_ids"] = merged_ids
            item_payload["count"] = len(merged_ids)
            item["reason"] = f"{action_type} accumulated: {len(merged_ids)} patents"
            item["updated_at"] = now()
            item["dedupe_key"] = f"{action_type}:accumulated"
            pending[item["id"]] = item
            save_pending(pending)
            append_event({"event": "pending_merged", "action": item, "added_count": len(set(new_ids) - set(old_ids))})
            return item
    key = f"{action_type}:{json.dumps(payload, sort_keys=True, ensure_ascii=False)}"
    for item in pending.values():
        if item.get("dedupe_key") == key and item.get("status") == "pending":
            return item
    action = {
        "id": next_action_id(action_type),
        "type": action_type,
        "status": "pending",
        "created_at": now(),
        "reason": reason,
        "payload": payload,
        "dedupe_key": key,
    }
    pending[action["id"]] = action
    save_pending(pending)
    append_event({"event": "pending_created", "action": action})
    return action


def approval_message(action: Dict[str, Any], status: Dict[str, Any]) -> str:
    payload = action.get("payload") or {}
    lines = [
        "A4 파이프라인 승인 요청",
        f"- id: {action['id']}",
        f"- action: {action['type']}",
        f"- reason: {action['reason']}",
        f"- evidence: {status.get('patents', 0)} / 21345",
        f"- minimal_json: {status.get('minimal_files', 0)}",
        f"- jobs: {status.get('jobs')}",
    ]
    if payload.get("patent_ids"):
        lines.append(f"- pending_patents: {len(payload['patent_ids'])}")
        lines.append("- patent_ids: " + ", ".join(payload["patent_ids"][:12]))
    lines.extend(
        [
            "",
            "대기 목록을 보려면: /pending",
            "대기 요청이 1건뿐이면: 승인 또는 거절",
            "승인 시점까지 누적된 같은 종류의 요청을 함께 처리함",
            f"실행하려면: 승인 {action['id']}",
            f"보류하려면: 거절 {action['id']}",
        ]
    )
    return "\n".join(lines)


def monitor_once(notify: bool = True) -> Dict[str, Any]:
    names = set(screen_names())
    jobs = job_counts()
    cards = inspect_minimal_cards()
    status = {
        "at": now(),
        "screens": sorted(names),
        "jobs": jobs,
        "patents": patent_count(),
        "minimal_files": cards["files"],
        "invalid_count": len(cards["invalid"]),
        "weak_count": len(cards["weak"]),
        "gemini_repaired": cards["repaired"],
    }
    actions: List[Dict[str, Any]] = []

    missing = []
    for name, run_path, action_type in [
        (EVIDENCE_SCREEN, EVIDENCE_RUN, "restart_evidence"),
        (MINIMAL_SCREEN, MINIMAL_RUN, "restart_minimal"),
        (GEMINI_SCREEN, GEMINI_RUN, "restart_gemini_worker"),
    ]:
        if name not in names:
            missing.append(name)
            actions.append(create_pending_action(action_type, f"screen session missing: {name}", {"screen": name, "script": str(run_path)}))

    if cards["invalid"]:
        patent_ids = [item["patent_id"] for item in cards["invalid"][:100]]
        actions.append(
            create_pending_action(
                "requeue_invalid_minimal",
                f"invalid minimal cards detected: {len(cards['invalid'])}",
                {"patent_ids": patent_ids, "count": len(cards["invalid"])},
            )
        )

    if jobs.get("minimal_failed", 0) >= 10:
        rows = failed_patent_ids(limit=100)
        actions.append(
            create_pending_action(
                "requeue_minimal_failed",
                f"minimal_failed accumulated: {jobs.get('minimal_failed', 0)}",
                {"patent_ids": rows, "count": jobs.get("minimal_failed", 0)},
            )
        )

    evidence_age = mtime_age(LOG_DIR / "evidence_full.out")
    minimal_age = mtime_age(LOG_DIR / "minimal_full.out")
    if EVIDENCE_SCREEN in names and evidence_age is not None and evidence_age > 1800:
        actions.append(create_pending_action("restart_evidence", "evidence log has not advanced for 30+ minutes", {"screen": EVIDENCE_SCREEN, "script": str(EVIDENCE_RUN)}))
    if MINIMAL_SCREEN in names and minimal_age is not None and minimal_age > 3600:
        actions.append(create_pending_action("restart_minimal", "minimal log has not advanced for 60+ minutes", {"screen": MINIMAL_SCREEN, "script": str(MINIMAL_RUN)}))

    if notify:
        for action in actions:
            if not action.get("notified_at"):
                action["notified_at"] = now()
                pending = load_pending()
                pending[action["id"]] = action
                save_pending(pending)
                send_telegram(approval_message(action, status))

    append_event({"event": "monitor", "status": status, "actions": [a["id"] for a in actions]})
    return status


def failed_patent_ids(limit: int = 100) -> List[str]:
    if not DB.exists():
        return []
    con = sqlite3.connect(str(DB))
    try:
        return [str(row[0]) for row in con.execute("SELECT patent_id FROM jobs WHERE status='minimal_failed' ORDER BY updated_at ASC LIMIT ?", (limit,))]
    finally:
        con.close()


def requeue_patents(patent_ids: Iterable[str]) -> int:
    ids = [normalize_ws(x) for x in patent_ids if normalize_ws(x)]
    if not ids:
        return 0
    con = sqlite3.connect(str(DB))
    try:
        con.executemany(
            "UPDATE jobs SET status='evidence_done', updated_at=? WHERE patent_id=?",
            [(datetime.now().isoformat(timespec="seconds"), patent_id) for patent_id in ids],
        )
        con.commit()
        return con.total_changes
    finally:
        con.close()


def quarantine_minimal_outputs(patent_ids: Iterable[str]) -> int:
    ids = [normalize_ws(x) for x in patent_ids if normalize_ws(x)]
    if not ids:
        return 0
    QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    moved = 0
    for patent_id in ids:
        src = MINIMAL_DIR / f"{patent_id}.minimal.json"
        if not src.exists():
            continue
        dst = QUARANTINE_DIR / f"{patent_id}.{stamp}.minimal.json"
        shutil.move(str(src), str(dst))
        moved += 1
    return moved


def restart_screen(screen_name: str, script_path: str) -> str:
    run_cmd(["screen", "-S", screen_name, "-X", "quit"], timeout=10)
    proc = run_cmd(["screen", "-dmS", screen_name, "bash", script_path], timeout=20)
    if proc.returncode != 0:
        raise RuntimeError((proc.stdout + proc.stderr).strip() or f"screen restart failed: {screen_name}")
    return f"restarted {screen_name}"


def execute_action(action_id: str, approved_by: str = "telegram") -> str:
    pending = load_pending()
    action = pending.get(action_id)
    if not action:
        return f"pending action not found: {action_id}"
    if action.get("status") != "pending":
        return f"action already {action.get('status')}: {action_id}"
    action_type = action.get("type")
    payload = action.get("payload") or {}
    if action_type in {"requeue_invalid_minimal", "requeue_minimal_failed"}:
        moved = quarantine_minimal_outputs(payload.get("patent_ids") or [])
        changed = requeue_patents(payload.get("patent_ids") or [])
        result = f"quarantined {moved} minimal files; requeued {changed} job rows to evidence_done"
    elif action_type == "restart_evidence":
        result = restart_screen(EVIDENCE_SCREEN, str(EVIDENCE_RUN))
    elif action_type == "restart_minimal":
        result = restart_screen(MINIMAL_SCREEN, str(MINIMAL_RUN))
    elif action_type == "restart_gemini_worker":
        result = restart_screen(GEMINI_SCREEN, str(GEMINI_RUN))
    else:
        raise RuntimeError(f"unsupported action type: {action_type}")
    action["status"] = "approved_executed"
    action["approved_by"] = approved_by
    action["executed_at"] = now()
    action["result"] = result
    pending[action_id] = action
    save_pending(pending)
    append_event({"event": "action_executed", "action": action})
    return result


def reject_action(action_id: str, rejected_by: str = "telegram") -> str:
    pending = load_pending()
    action = pending.get(action_id)
    if not action:
        return f"pending action not found: {action_id}"
    action["status"] = "rejected"
    action["rejected_by"] = rejected_by
    action["rejected_at"] = now()
    pending[action_id] = action
    save_pending(pending)
    append_event({"event": "action_rejected", "action": action})
    return f"rejected {action_id}"


def format_pending() -> str:
    pending = [item for item in load_pending().values() if item.get("status") == "pending"]
    if not pending:
        return "대기 중인 승인 요청이 없어."
    lines = ["대기 중인 승인 요청"]
    for item in pending[:20]:
        payload = item.get("payload") or {}
        count = payload.get("count")
        count_text = f" | {count}건" if count else ""
        lines.append(f"- {item['id']} | {item['type']}{count_text} | {item['reason']}")
    return "\n".join(lines)


def run_loop(interval_sec: float) -> None:
    while True:
        try:
            monitor_once(notify=True)
        except Exception as exc:
            append_event({"event": "monitor_error", "error": repr(exc)})
        time.sleep(interval_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram approval monitor for A4 rebuild pipeline.")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval-sec", type=float, default=3600)
    parser.add_argument("--execute", default="")
    parser.add_argument("--reject", default="")
    parser.add_argument("--pending", action="store_true")
    args = parser.parse_args()

    if args.execute:
        print(execute_action(args.execute, approved_by="cli"))
        return
    if args.reject:
        print(reject_action(args.reject, rejected_by="cli"))
        return
    if args.pending:
        print(format_pending())
        return
    if args.once or not args.loop:
        print(json.dumps(monitor_once(notify=True), ensure_ascii=False, indent=2))
        return
    run_loop(args.interval_sec)


if __name__ == "__main__":
    main()
