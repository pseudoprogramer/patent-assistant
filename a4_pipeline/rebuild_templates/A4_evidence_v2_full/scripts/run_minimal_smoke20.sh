#!/usr/bin/env bash
set -euo pipefail

ROOT="/Volumes/외장 2TB/cpu2026/rebuilds/A4_evidence_v2_full_20260508"
CODE="/Volumes/외장 2TB/cpu2026/common/code"
PY="/Volumes/외장 2TB/cpu2026/common/.venv/bin/python"

cd "$CODE"
env PYTHONPYCACHEPREFIX=/tmp/codex_pycache "$PY" patent_minimal_index_v2.py \
  --limit 20 \
  --db "$ROOT/db/patent_A4_evidence_v2_full.sqlite" \
  --output-dir "$ROOT/minimal_analysis" \
  --log-dir "$ROOT/logs" \
  --raw-invalid-dir "$ROOT/raw_invalid/minimal" \
  --model qwen3:14b \
  > "$ROOT/logs/minimal_smoke20.out" 2>&1

echo "Minimal smoke labeling complete:"
echo "$ROOT/minimal_analysis"

