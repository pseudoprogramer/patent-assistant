#!/usr/bin/env bash
set -euo pipefail

ROOT="/Volumes/외장 2TB/cpu2026/rebuilds/A4_evidence_v2_full_20260508"
CODE="/Volumes/외장 2TB/cpu2026/common/code"
PY="/Volumes/외장 2TB/cpu2026/common/.venv/bin/python"

cd "$CODE"
env PYTHONPYCACHEPREFIX=/tmp/codex_pycache "$PY" build_evidence_db_v2.py \
  --image-folders \
  --recursive \
  --db "$ROOT/db/patent_A4_evidence_v2_full.sqlite" \
  --parsed-json-dir "$ROOT/parsed_json/full" \
  --pack-db "$ROOT/indexes/patent_evidence_pack_index.sqlite" \
  --minimal-db "$ROOT/indexes/patent_minimal_index.sqlite" \
  --report-dir "$ROOT/reports/evidence_full" \
  > "$ROOT/logs/evidence_full.out" 2>&1

echo "Full evidence rebuild complete:"
echo "$ROOT/reports/evidence_full/evidence_db_v2_run.json"

