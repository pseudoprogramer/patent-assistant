# A4 Patent Evidence Pipeline

Local A4 patent pipeline code for:

- building the evidence SQLite database
- generating minimal patent index JSON
- rebuilding the searchable minimal index
- searching the local patent dictionary
- asking a local Ollama model questions over retrieved patent cards
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
```

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

The bot can resolve partial patent numbers such as `0012062403` to matching local patent IDs when the match is unambiguous.
