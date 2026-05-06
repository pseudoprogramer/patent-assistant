# A4 Patent Evidence Pipeline

Local A4 patent pipeline code for:

- building the evidence SQLite database
- generating minimal patent index JSON
- rebuilding the searchable minimal index
- searching the local patent dictionary
- asking a local Ollama model questions over retrieved patent cards
- building GPT/Gemini evidence packs for higher-accuracy judgment
- serving the same dictionary through a Telegram bot

Data files, logs, model files, raw patents, generated JSON, and SQLite indexes are intentionally not committed.

## Typical Flow

```bash
cd "/Volumes/외장 2TB/cpu2026/common/code"

python build_evidence_db.py --image-folders --no-quarantine
python patent_minimal_index.py --limit 100
python build_minimal_search_index.py
python patent_dictionary_search.py "page buffer bit line" --limit 10
python patent_dictionary_ask.py "page buffer와 bit line 제어 관련 특허 후보 비교해줘"
python evidence_pack.py "0012062403 특허에 대해서 알려줘" --provider none --limit 3
python patent_judge.py "0012062403 특허의 핵심을 근거 중심으로 알려줘" --provider ollama --planner-provider none
```

## Pro Judgment Mode

The pro path separates retrieval from judgment:

1. GPT/Gemini/Ollama converts the user question into a compact search plan.
2. Local SQLite search retrieves patent cards and claim/figure evidence.
3. GPT/Gemini/Ollama judges only the retrieved evidence pack.

Create a local `.env` from `.env.example` and set keys as needed:

```bash
cp .env.example .env
# edit .env locally; never commit real keys
```

Useful variables:

- `PATENT_PRO_PROVIDER=openai` or `gemini`
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-5.1`
- `GEMINI_API_KEY=...`
- `GEMINI_MODEL=gemini-2.5-pro`

## Telegram Bot

```bash
export TELEGRAM_BOT_TOKEN="..."
export TELEGRAM_ALLOWED_CHAT_IDS="7479642309"
python patent_telegram_bot.py
```

Supported commands:

- `/status`
- `/search page buffer bit line`
- `/search 20 page buffer bit line`
- `/patent us20250191658a1p`
- `/patent 0012062403`
- `/ask page buffer와 bit line 제어 관련 특허 후보 비교해줘`
- `/ask_pro page buffer와 bit line 제어 관련 특허 후보 비교해줘`
- `/verify 0012062403 이 요약이 맞는지 검증해줘`

The bot can resolve partial patent numbers such as `0012062403` to matching local patent IDs when the match is unambiguous.
