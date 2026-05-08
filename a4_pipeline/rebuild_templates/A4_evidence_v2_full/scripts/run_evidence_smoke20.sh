#!/usr/bin/env bash
set -euo pipefail

ROOT="/Volumes/외장 2TB/cpu2026/rebuilds/A4_evidence_v2_full_20260508"
CODE="/Volumes/외장 2TB/cpu2026/common/code"
PY="/Volumes/외장 2TB/cpu2026/common/.venv/bin/python"

cd "$CODE"
env PYTHONPYCACHEPREFIX=/tmp/codex_pycache "$PY" build_evidence_db_v2.py \
  --image-folders \
  --recursive \
  --limit 20 \
  --db "$ROOT/db/patent_A4_evidence_v2_smoke20.sqlite" \
  --parsed-json-dir "$ROOT/parsed_json/smoke20" \
  --pack-db "$ROOT/indexes/patent_evidence_pack_index_smoke20.sqlite" \
  --minimal-db "$ROOT/indexes/patent_minimal_index.sqlite" \
  --report-dir "$ROOT/reports/evidence_smoke20" \
  > "$ROOT/logs/evidence_smoke20.out" 2>&1

echo "Smoke evidence rebuild complete:"
echo "$ROOT/reports/evidence_smoke20/evidence_db_v2_run.json"

