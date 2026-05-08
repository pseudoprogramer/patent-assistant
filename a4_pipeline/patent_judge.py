from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from evidence_pack import build_evidence_pack  # noqa: E402
from llm_clients import LLMClient  # noqa: E402


JUDGE_INSTRUCTIONS = """You are a patent analysis judge.
Answer in Korean, but preserve patent titles and technical terms in their source language.
Use only the provided evidence pack. Do not invent patents, claims, figures, or facts.
Every important assertion must cite patent_id and evidence id or claim_no/figure_no.
If evidence is weak, noisy, OCR-contaminated, or insufficient, say so explicitly.
Treat retrieval/ranker weaknesses as retrieval-quality notes, not as patent facts.
If a claim is marked dependent or dependent_inferred, do not describe it as an independent claim.
Prefer concise, concrete, comparison-oriented answers."""


def build_judge_prompt(question: str, pack: Dict[str, Any]) -> str:
    return f"""User question:
{question}

Evidence pack JSON:
{json.dumps(pack, ensure_ascii=False, indent=2)}

Write under these headings:
1. 결론
2. 근거가 강한 부분
3. 관련 특허별 판단
4. 불확실하거나 추가 확인할 부분
5. 다음 검색/검증 제안
"""


def judge_question(
    question: str,
    provider: str = "auto",
    model: str = "",
    planner_provider: str = "auto",
    planner_model: str = "",
    limit: int = 8,
    max_tokens: int = 2200,
    timeout: int = 240,
) -> Dict[str, Any]:
    started = time.monotonic()
    judge_client = LLMClient(provider, model=model or None, timeout=timeout)
    planner_client = None if planner_provider == "none" else LLMClient(planner_provider, model=planner_model or model or None, timeout=timeout)
    pack = build_evidence_pack(question, planner_client=planner_client, limit=limit)
    prompt = build_judge_prompt(question, pack)
    answer = judge_client.generate(prompt, instructions=JUDGE_INSTRUCTIONS, max_tokens=max_tokens, temperature=0.1)
    elapsed = time.monotonic() - started
    return {
        "provider": judge_client.provider,
        "model": judge_client.model,
        "planner_provider": planner_client.provider if planner_client else "none",
        "planner_model": planner_client.model if planner_client else "fallback",
        "elapsed_sec": round(elapsed, 1),
        "answer": answer,
        "evidence_pack": pack,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask GPT/Gemini/Ollama to judge a question using local patent evidence.")
    parser.add_argument("question")
    parser.add_argument("--provider", default="auto", choices=["auto", "openai", "gemini", "ollama"])
    parser.add_argument("--model", default="")
    parser.add_argument("--planner-provider", default="auto", choices=["none", "auto", "openai", "gemini", "ollama"])
    parser.add_argument("--planner-model", default="")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=2200)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--show-pack", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = judge_question(
        args.question,
        provider=args.provider,
        model=args.model,
        planner_provider=args.planner_provider,
        planner_model=args.planner_model,
        limit=args.limit,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"Provider: {result['provider']} / {result['model']}")
    print(f"Planner: {result['planner_provider']} / {result['planner_model']}")
    print(f"Elapsed: {result['elapsed_sec']} sec")
    print("\n--- Answer ---")
    print(result["answer"])
    if args.show_pack:
        print("\n--- Evidence pack ---")
        print(json.dumps(result["evidence_pack"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
