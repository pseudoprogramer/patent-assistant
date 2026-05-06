from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import requests


CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from patent_dictionary_ask import (  # noqa: E402
    DEFAULT_MODEL,
    ask_llm,
    build_prompt,
    compact_card,
    infer_search_query,
)
from patent_dictionary_search import DEFAULT_DB, lookup, search  # noqa: E402
from patent_judge import judge_question  # noqa: E402


DEFAULT_LOG_PATH = Path("/Volumes/외장 2TB/cpu2026/common/runtime/logs/A4/patent_telegram_chat.jsonl")
TELEGRAM_MAX_MESSAGE = 3900
MAX_SEARCH_LIMIT = 30
PATENT_ID_RE = re.compile(r"\b(?:us|cn|kr)[a-z0-9]{6,}p\b", re.IGNORECASE)
PATENT_NUMBER_RE = re.compile(r"\d{7,}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_allowed_chat_ids(value: str) -> Set[int]:
    ids: Set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        ids.add(int(part))
    return ids


def split_message(text: str, limit: int = TELEGRAM_MAX_MESSAGE) -> Iterable[str]:
    text = text.strip() or "(empty)"
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = text.rfind(" ", 0, limit)
        if cut < limit // 2:
            cut = limit
        yield text[:cut].strip()
        text = text[cut:].strip()
    if text:
        yield text


def clamp_limit(value: int) -> int:
    return max(1, min(MAX_SEARCH_LIMIT, value))


def extract_patent_id(text: str) -> str:
    match = PATENT_ID_RE.search(text or "")
    if match:
        return match.group(0).lower()
    first = (text or "").strip().split(maxsplit=1)[0] if (text or "").strip() else ""
    cleaned = first.strip("[](){}.,:;`'\"<>").lower()
    return cleaned if PATENT_ID_RE.fullmatch(cleaned) else ""


def parse_limited_query(text: str, default_limit: int) -> tuple[str, int]:
    parts = text.strip().split()
    limit = clamp_limit(default_limit)
    query_parts: List[str] = []
    i = 0
    while i < len(parts):
        part = parts[i]
        if part.isdigit() and 1 <= int(part) <= MAX_SEARCH_LIMIT:
            limit = clamp_limit(int(part))
            i += 1
            continue
        if part in {"--limit", "-n"} and i + 1 < len(parts) and parts[i + 1].isdigit():
            limit = clamp_limit(int(parts[i + 1]))
            i += 2
            continue
        if part.startswith("--limit=") and part.split("=", 1)[1].isdigit():
            limit = clamp_limit(int(part.split("=", 1)[1]))
            i += 1
            continue
        query_parts.append(part)
        i += 1
    return " ".join(query_parts).strip(), limit


class TelegramClient:
    def __init__(self, token: str, timeout: int = 60) -> None:
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.timeout = timeout

    def request(self, method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        r = requests.post(f"{self.base_url}/{method}", json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"Telegram API failed: {data}")
        return data

    def get_updates(self, offset: Optional[int], timeout: int) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {"timeout": timeout, "allowed_updates": ["message"]}
        if offset is not None:
            payload["offset"] = offset
        return self.request("getUpdates", payload).get("result", [])

    def send_message(self, chat_id: int, text: str) -> None:
        for chunk in split_message(text):
            self.request(
                "sendMessage",
                {
                    "chat_id": chat_id,
                    "text": chunk,
                    "disable_web_page_preview": True,
                },
            )

    def set_commands(self) -> None:
        self.request(
            "setMyCommands",
            {
                "commands": [
                    {"command": "ask", "description": "특허 사전에 질문하고 로컬 LLM 답변 받기"},
                    {"command": "ask_pro", "description": "GPT/Gemini가 로컬 근거를 보고 판단"},
                    {"command": "verify", "description": "특정 특허/요약을 근거 기반 검증"},
                    {"command": "search", "description": "관련 특허 후보 카드 검색"},
                    {"command": "patent", "description": "patent_id로 특정 특허 조회"},
                    {"command": "status", "description": "인덱스 상태 확인"},
                    {"command": "help", "description": "사용법 보기"},
                ]
            },
        )


class PatentTelegramBot:
    def __init__(
        self,
        token: str,
        db_path: Path,
        log_path: Path,
        allowed_chat_ids: Set[int],
        model: str,
        limit: int,
        timeout: int,
        num_predict: int,
        ask_workers: int,
    ) -> None:
        self.telegram = TelegramClient(token)
        self.db_path = db_path
        self.log_path = log_path
        self.allowed_chat_ids = allowed_chat_ids
        self.model = model
        self.limit = limit
        self.timeout = timeout
        self.num_predict = num_predict
        self.ask_executor = ThreadPoolExecutor(max_workers=max(1, ask_workers))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: Dict[str, Any]) -> None:
        event = {"ts": utc_now(), **event}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def fuzzy_patent_cards(self, con: sqlite3.Connection, text: str, limit: int = 5) -> List[Dict[str, Any]]:
        patent_id = extract_patent_id(text)
        if patent_id:
            card = lookup(con, patent_id)
            if card:
                return [card]

        fragments = PATENT_NUMBER_RE.findall(text or "")
        if not fragments:
            return []

        lower = (text or "").lower()
        prefix = ""
        if "us" in lower or "미국" in lower:
            prefix = "us"
        elif "cn" in lower or "중국" in lower:
            prefix = "cn"
        elif "kr" in lower or "한국" in lower:
            prefix = "kr"

        seen: Set[str] = set()
        cards: List[Dict[str, Any]] = []
        for fragment in fragments:
            stripped = fragment.lstrip("0") or fragment
            patterns = list(dict.fromkeys([f"%{fragment.lower()}%", f"%{stripped.lower()}%"]))
            for pattern in patterns:
                params: List[Any] = [pattern]
                where = "patent_id LIKE ?"
                if prefix:
                    where += " AND patent_id LIKE ?"
                    params.append(f"{prefix}%")
                rows = con.execute(
                    f"""
                    SELECT patent_id
                    FROM minimal_index
                    WHERE {where}
                    ORDER BY length(patent_id), patent_id
                    LIMIT ?
                    """,
                    [*params, limit],
                ).fetchall()
                for (candidate_id,) in rows:
                    if candidate_id in seen:
                        continue
                    card = lookup(con, candidate_id)
                    if card:
                        seen.add(candidate_id)
                        cards.append(card)
                    if len(cards) >= limit:
                        return cards
        return cards

    def retrieve(self, question: str, limit: Optional[int] = None) -> tuple[str, List[Dict[str, Any]]]:
        con = self.connect()
        retrieval_query = question
        cards = self.fuzzy_patent_cards(con, question, limit=limit or self.limit)
        if cards:
            retrieval_query = cards[0]["patent_id"] if len(cards) == 1 else "patent_id_fuzzy_match"
        else:
            cards = search(con, question, limit or self.limit)
        cards = [card for card in cards if card]
        if not cards:
            fallback_query = infer_search_query(question)
            if fallback_query and fallback_query != question:
                retrieval_query = fallback_query
                cards = search(con, fallback_query, limit or self.limit)
                cards = [card for card in cards if card]
        con.close()
        return retrieval_query, cards

    def format_cards(self, cards: List[Dict[str, Any]]) -> str:
        if not cards:
            return "검색 결과가 없어."
        lines = [f"검색 결과: {len(cards)}건"]
        for i, card in enumerate(cards, 1):
            labels = ", ".join(card["solution_labels"][:5])
            evidence = ", ".join(card["evidence_ids"][:6])
            lines.extend(
                [
                    "",
                    f"[{i}] {card['patent_id']} ({card['language']}, {card['primary_claim_type']}, conf={card['confidence']})",
                    f"Title: {card['title']}",
                    f"Core: {card['core_subject']}",
                ]
            )
            if labels:
                lines.append(f"Labels: {labels}")
            if evidence:
                lines.append(f"Evidence: {evidence}")
            if card["qc_flags"]:
                lines.append("QC: " + ", ".join(card["qc_flags"]))
        return "\n".join(lines)

    def format_patent_detail(self, card: Dict[str, Any]) -> str:
        lines = [
            f"{card['patent_id']} ({card['language']}, {card['primary_claim_type']}, conf={card['confidence']})",
            f"Title: {card['title']}",
            f"Core: {card['core_subject']}",
            f"Independent claims: {', '.join(card['independent_claim_nos']) or '-'}",
        ]
        if card["secondary_claim_types"]:
            lines.append("Secondary claim types: " + ", ".join(card["secondary_claim_types"]))
        if card["core_elements"]:
            lines.append("Core elements:\n- " + "\n- ".join(card["core_elements"][:12]))
        if card["problem_labels"]:
            lines.append("Problem labels: " + ", ".join(card["problem_labels"][:10]))
        if card["solution_labels"]:
            lines.append("Solution labels: " + ", ".join(card["solution_labels"][:12]))
        if card["effect_labels"]:
            lines.append("Effect labels: " + ", ".join(card["effect_labels"][:10]))
        if card["evidence_ids"]:
            lines.append("Evidence ids: " + ", ".join(card["evidence_ids"][:16]))
        if card["qc_flags"]:
            lines.append("QC flags: " + ", ".join(card["qc_flags"]))
        lines.append(f"JSON: {card['json_path']}")
        return "\n".join(lines)

    def ask(self, question: str) -> str:
        retrieval_query, cards = self.retrieve(question)
        if not cards:
            return "관련 특허를 찾지 못했어. 핵심 키워드를 조금 더 짧게 넣어줘."
        prompt = build_prompt(question, cards)
        answer = ask_llm(prompt, self.model, self.timeout, self.num_predict)
        prefix = ""
        if retrieval_query != question:
            prefix = f"Retrieval query: {retrieval_query}\n\n"
        return prefix + answer

    def status(self) -> str:
        con = self.connect()
        total = con.execute("SELECT COUNT(*) FROM minimal_index").fetchone()[0]
        qc_rows = con.execute(
            "SELECT COUNT(*) FROM minimal_index WHERE qc_flags_json IS NOT NULL AND qc_flags_json != '[]'"
        ).fetchone()[0]
        langs = con.execute(
            "SELECT source_language, COUNT(*) FROM minimal_index GROUP BY source_language ORDER BY COUNT(*) DESC"
        ).fetchall()
        con.close()
        lang_text = ", ".join(f"{lang or 'unknown'}={count}" for lang, count in langs)
        return (
            "특허 사전 상태\n"
            f"- index: {self.db_path}\n"
            f"- indexed patents: {total}\n"
            f"- qc flagged rows: {qc_rows}\n"
            f"- languages: {lang_text}\n"
            f"- chat log: {self.log_path}"
        )

    def help_text(self) -> str:
        return (
            "특허 사전 봇 명령어\n"
            "/ask 질문 - 관련 특허를 찾아 로컬 LLM으로 답변\n"
            "/ask_pro 질문 - GPT/Gemini가 검색 계획을 세우고 근거 기반 답변\n"
            "/verify 질문 - 특정 특허나 요약의 정확성 검증\n"
            "/search 키워드 - 후보 카드만 빠르게 검색, 기본 10건\n"
            "/search 20 키워드 - 후보 개수 지정, 최대 30건\n"
            "/patent patent_id - 특정 특허 카드 조회\n"
            "/status - 인덱스 상태 확인\n"
            "/help - 도움말\n\n"
            "예: /ask page buffer와 bit line 제어 관련 특허 후보 비교해줘"
        )

    def strip_bot_suffix(self, text: str) -> str:
        if not text.startswith("/"):
            return text
        first, *rest = text.split(maxsplit=1)
        if "@" in first:
            first = first.split("@", 1)[0]
        return " ".join([first] + rest).strip()

    def run_ask_job(self, chat_id: int, original_text: str, question: str) -> None:
        started = time.monotonic()
        try:
            answer = self.ask(question)
            elapsed = time.monotonic() - started
            answer = f"생성 시간: {elapsed:.1f}초\n\n{answer}"
            self.telegram.send_message(chat_id, answer)
            self.log_event({"event": "answer", "chat_id": chat_id, "text": original_text, "answer": answer})
        except Exception as exc:
            elapsed = time.monotonic() - started
            answer = f"처리 중 오류가 났어: {exc}"
            self.telegram.send_message(chat_id, answer)
            self.log_event(
                {
                    "event": "error",
                    "chat_id": chat_id,
                    "text": original_text,
                    "elapsed_sec": round(elapsed, 1),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    def run_pro_job(self, chat_id: int, original_text: str, question: str, mode: str) -> None:
        started = time.monotonic()
        try:
            result = judge_question(question, provider="auto", planner_provider="auto", limit=self.limit, timeout=self.timeout)
            answer = (
                f"판단 모드: {mode}\n"
                f"Provider: {result['provider']} / {result['model']}\n"
                f"Planner: {result['planner_provider']} / {result['planner_model']}\n"
                f"생성 시간: {result['elapsed_sec']}초\n\n"
                f"{result['answer']}"
            )
            self.telegram.send_message(chat_id, answer)
            self.log_event(
                {
                    "event": "pro_answer",
                    "chat_id": chat_id,
                    "text": original_text,
                    "mode": mode,
                    "elapsed_sec": round(time.monotonic() - started, 1),
                    "answer": answer,
                    "query_plan": result["evidence_pack"].get("query_plan"),
                    "retrieved_patents": [c["patent_id"] for c in result["evidence_pack"].get("retrieved_cards", [])],
                }
            )
        except Exception as exc:
            answer = f"프로 판단 처리 중 오류가 났어: {exc}"
            self.telegram.send_message(chat_id, answer)
            self.log_event(
                {
                    "event": "pro_error",
                    "chat_id": chat_id,
                    "text": original_text,
                    "mode": mode,
                    "elapsed_sec": round(time.monotonic() - started, 1),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )

    def enqueue_ask(self, chat_id: int, original_text: str, question: str) -> None:
        self.telegram.send_message(chat_id, "질문 받았어. 관련 특허를 찾고 로컬 LLM으로 답변 생성 중이야.")
        self.log_event({"event": "ask_queued", "chat_id": chat_id, "text": original_text, "question": question})
        self.ask_executor.submit(self.run_ask_job, chat_id, original_text, question)

    def enqueue_pro(self, chat_id: int, original_text: str, question: str, mode: str) -> None:
        self.telegram.send_message(chat_id, "프로 판단 질문 받았어. GPT/Gemini가 검색 계획을 만들고 로컬 근거를 검토하는 중이야.")
        self.log_event({"event": "pro_queued", "chat_id": chat_id, "text": original_text, "question": question, "mode": mode})
        self.ask_executor.submit(self.run_pro_job, chat_id, original_text, question, mode)

    def handle_text(self, chat_id: int, text: str) -> Optional[str]:
        text = self.strip_bot_suffix(text.strip())
        if text in {"/start", "/help"}:
            return self.help_text()
        if text == "/status":
            return self.status()
        if text.startswith("/search"):
            query, search_limit = parse_limited_query(text.removeprefix("/search").strip(), self.limit)
            if not query:
                return "검색어를 같이 보내줘. 예: /search page buffer bit line 또는 /search 20 page buffer bit line"
            _, cards = self.retrieve(query, limit=search_limit)
            return self.format_cards(cards)
        if text.startswith("/patent"):
            con = self.connect()
            cards = self.fuzzy_patent_cards(con, text.removeprefix("/patent").strip(), limit=5)
            con.close()
            if not cards:
                return "해당 특허를 찾지 못했어. 예: /patent us20250191658a1p 또는 /patent 20250191658"
            if len(cards) > 1:
                return "후보가 여러 개 잡혔어. 더 정확한 번호로 다시 물어봐.\n\n" + self.format_cards(cards)
            return self.format_patent_detail(cards[0])
        if text.startswith("/ask_pro"):
            question = text.removeprefix("/ask_pro").strip()
            if not question:
                return "질문을 같이 보내줘. 예: /ask_pro page buffer와 bit line 관련 특허 비교해줘"
            self.enqueue_pro(chat_id, text, question, "ask_pro")
            return None
        if text.startswith("/verify"):
            question = text.removeprefix("/verify").strip()
            if not question:
                return "검증할 특허나 요약을 같이 보내줘. 예: /verify 0012062403 이 요약이 맞는지 봐줘"
            self.enqueue_pro(chat_id, text, question, "verify")
            return None
        if text.startswith("/ask"):
            question = text.removeprefix("/ask").strip()
            if not question:
                return "질문을 같이 보내줘. 예: /ask SSD garbage collection 관련 특허 묶어줘"
            self.enqueue_ask(chat_id, text, question)
            return None
        self.enqueue_ask(chat_id, text, text)
        return None

    def is_allowed(self, chat_id: int) -> bool:
        return not self.allowed_chat_ids or chat_id in self.allowed_chat_ids

    def run(self, poll_timeout: int = 30) -> None:
        offset: Optional[int] = None
        try:
            self.telegram.set_commands()
            self.log_event({"event": "commands_registered"})
        except Exception as exc:
            self.log_event({"event": "commands_register_error", "error": repr(exc)})
        print(f"[telegram] started, db={self.db_path}")
        while True:
            try:
                updates = self.telegram.get_updates(offset, poll_timeout)
                for update in updates:
                    offset = update["update_id"] + 1
                    message = update.get("message") or {}
                    chat = message.get("chat") or {}
                    chat_id = int(chat.get("id"))
                    text = str(message.get("text") or "").strip()
                    if not text:
                        continue
                    if not self.is_allowed(chat_id):
                        self.log_event({"event": "rejected_chat", "chat_id": chat_id, "text": text})
                        self.telegram.send_message(chat_id, "이 봇은 허용된 채팅방에서만 사용할 수 있어.")
                        continue
                    self.log_event({"event": "question", "chat_id": chat_id, "text": text})
                    try:
                        answer = self.handle_text(chat_id, text)
                    except Exception as exc:
                        answer = f"처리 중 오류가 났어: {exc}"
                        self.log_event(
                            {
                                "event": "error",
                                "chat_id": chat_id,
                                "text": text,
                                "error": repr(exc),
                                "traceback": traceback.format_exc(),
                            }
                        )
                    if answer is not None:
                        self.telegram.send_message(chat_id, answer)
                        self.log_event({"event": "answer", "chat_id": chat_id, "text": text, "answer": answer})
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                self.log_event({"event": "poll_error", "error": repr(exc), "traceback": traceback.format_exc()})
                time.sleep(5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Telegram chat interface for the local patent dictionary.")
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--log", default=str(DEFAULT_LOG_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--num-predict", type=int, default=1600)
    parser.add_argument("--ask-workers", type=int, default=1)
    parser.add_argument("--poll-timeout", type=int, default=30)
    args = parser.parse_args()

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN first.")
    allowed_chat_ids = parse_allowed_chat_ids(os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", ""))
    bot = PatentTelegramBot(
        token=token,
        db_path=Path(args.db),
        log_path=Path(args.log),
        allowed_chat_ids=allowed_chat_ids,
        model=args.model,
        limit=args.limit,
        timeout=args.timeout,
        num_predict=args.num_predict,
        ask_workers=args.ask_workers,
    )
    bot.run(args.poll_timeout)


if __name__ == "__main__":
    main()
