# A4 Evidence v2 Full Rebuild Workspace

This workspace is for rebuilding the A4 patent evidence database from raw PDFs without touching the current production DB or outputs.

## Main Paths

- Raw inbox: `/Volumes/외장 2TB/cpu2026/patent_hub/raw_patents/inbox/A4`
- Code dir: `/Volumes/외장 2TB/cpu2026/common/code`
- Rebuild root: `/Volumes/외장 2TB/cpu2026/rebuilds/A4_evidence_v2_full_20260508`
- Rebuild evidence DB: `/Volumes/외장 2TB/cpu2026/rebuilds/A4_evidence_v2_full_20260508/db/patent_A4_evidence_v2_full.sqlite`
- Rebuild parsed JSON: `/Volumes/외장 2TB/cpu2026/rebuilds/A4_evidence_v2_full_20260508/parsed_json`
- Rebuild evidence pack DB: `/Volumes/외장 2TB/cpu2026/rebuilds/A4_evidence_v2_full_20260508/indexes/patent_evidence_pack_index.sqlite`
- Rebuild minimal JSON: `/Volumes/외장 2TB/cpu2026/rebuilds/A4_evidence_v2_full_20260508/minimal_analysis`
- Rebuild reports: `/Volumes/외장 2TB/cpu2026/rebuilds/A4_evidence_v2_full_20260508/reports`
- Rebuild logs: `/Volumes/외장 2TB/cpu2026/rebuilds/A4_evidence_v2_full_20260508/logs`

## Intended Flow

1. Put all raw PDF folders under the raw inbox.
2. Run `scripts/run_evidence_smoke20.sh`.
3. Inspect the generated audit report.
4. If smoke test is clean, run `scripts/run_evidence_full.sh`.
5. Run `scripts/run_minimal_smoke20.sh`.
6. If labels look good, run the full Qwen minimal labeling batch.
7. Send weak problem/effect candidates to the Gemini CLI repair queue.
8. Rebuild the search/index DB used by the Telegram bot.

## Safety Rules

- Do not overwrite `/Volumes/외장 2TB/cpu2026/common/runtime/db/patent_A4.sqlite` during rebuild.
- Do not write new minimal JSON into `/Volumes/외장 2TB/cpu2026/patent_hub/outputs/minimal_analysis/A4` until rebuild QA passes.
- Only promote rebuilt DB/outputs after a smoke test, audit, and spot-check pass.
- Keep raw PDFs in inbox unless explicitly moving processed files is desired.

## Smoke Test

```bash
cd "/Volumes/외장 2TB/cpu2026/rebuilds/A4_evidence_v2_full_20260508"
bash scripts/run_evidence_smoke20.sh
```

## Full Evidence Rebuild

Run this only after the user gives the signal.

```bash
cd "/Volumes/외장 2TB/cpu2026/rebuilds/A4_evidence_v2_full_20260508"
bash scripts/run_evidence_full.sh
```

## Minimal Labeling Smoke Test

```bash
cd "/Volumes/외장 2TB/cpu2026/rebuilds/A4_evidence_v2_full_20260508"
bash scripts/run_minimal_smoke20.sh
```

