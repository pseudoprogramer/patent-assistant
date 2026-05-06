from pathlib import Path

BASE = Path("/Volumes/외장 2TB/cpu2026")

HUB = BASE / "patent_hub"
COMMON = BASE / "common"

# A4 기준
A4_INBOX = HUB / "raw_patents" / "inbox" / "A4"
A4_PROCESSED = HUB / "raw_patents" / "processed" / "A4"
A4_PARSED_JSON = HUB / "outputs" / "parsed_json" / "A4"
A4_BRIEFS = HUB / "outputs" / "briefs" / "A4"
A4_REPORTS = HUB / "outputs" / "reports" / "A4"
A4_DB_SNAPSHOTS = HUB / "outputs" / "db_snapshots" / "A4"

A4_DB = COMMON / "runtime" / "db" / "patent_A4.sqlite"
A4_LOGS = COMMON / "runtime" / "logs" / "A4"
A4_QUARANTINE = COMMON / "runtime" / "quarantine" / "A4"
A4_TMP = COMMON / "runtime" / "tmp" / "A4"
A4_CACHE = COMMON / "runtime" / "cache" / "A4"
A4_RAW_INVALID = COMMON / "runtime" / "raw_invalid" / "A4"

A4_POLICY = COMMON / "policy" / "A4_problem_policy.json"

def ensure_runtime_dirs():
    dirs = [
        A4_LOGS,
        A4_QUARANTINE,
        A4_TMP,
        A4_CACHE,
        A4_RAW_INVALID,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
