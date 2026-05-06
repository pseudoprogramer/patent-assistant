from __future__ import annotations
import sqlite3
from pathlib import Path

try:
    from config import A4_DB
except Exception:
    BASE = Path("/Volumes/외장 2TB/cpu2026")
    A4_DB = BASE / "common" / "runtime" / "db" / "patent_A4.sqlite"

DB_PATH = Path(A4_DB)


def get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON;")
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con


def ensure_db() -> Path:
    con = get_connection()
    cur = con.cursor()

    cur.executescript(
        """
        CREATE TABLE IF NOT EXISTS patents (
            patent_id TEXT PRIMARY KEY,
            country TEXT,
            title_raw TEXT,
            assignee_raw TEXT,
            application_no TEXT,
            publication_no TEXT,
            pdf_path TEXT,
            page_count INTEGER,
            parser_version TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS pages (
            patent_id TEXT NOT NULL,
            page_no INTEGER NOT NULL,
            width REAL,
            height REAL,
            PRIMARY KEY (patent_id, page_no),
            FOREIGN KEY (patent_id) REFERENCES patents(patent_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS text_spans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patent_id TEXT NOT NULL,
            page_no INTEGER NOT NULL,
            span_id TEXT NOT NULL,
            block_no INTEGER,
            line_no INTEGER,
            span_no INTEGER,
            raw_text TEXT,
            norm_text TEXT,
            x0 REAL,
            y0 REAL,
            x1 REAL,
            y1 REAL,
            UNIQUE (patent_id, page_no, span_id),
            FOREIGN KEY (patent_id) REFERENCES patents(patent_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS claims (
            patent_id TEXT NOT NULL,
            claim_no TEXT NOT NULL,
            parent_claim_no TEXT,
            claim_type TEXT,
            raw_text TEXT,
            norm_text TEXT,
            page_start INTEGER,
            page_end INTEGER,
            PRIMARY KEY (patent_id, claim_no),
            FOREIGN KEY (patent_id) REFERENCES patents(patent_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ref_entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patent_id TEXT NOT NULL,
            ref_no_raw TEXT NOT NULL,
            ref_no_norm TEXT,
            label_raw TEXT,
            label_norm TEXT,
            source_section TEXT,
            page_no INTEGER,
            x0 REAL,
            y0 REAL,
            x1 REAL,
            y1 REAL,
            UNIQUE (
                patent_id, ref_no_raw, label_raw, source_section, page_no, x0, y0, x1, y1
            ),
            FOREIGN KEY (patent_id) REFERENCES patents(patent_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS claim_ref_map (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patent_id TEXT NOT NULL,
            claim_no TEXT NOT NULL,
            ref_no_raw TEXT NOT NULL,
            mention_text TEXT,
            page_no INTEGER,
            x0 REAL,
            y0 REAL,
            x1 REAL,
            y1 REAL,
            UNIQUE (
                patent_id, claim_no, ref_no_raw, page_no, x0, y0, x1, y1
            ),
            FOREIGN KEY (patent_id) REFERENCES patents(patent_id) ON DELETE CASCADE,
            FOREIGN KEY (patent_id, claim_no) REFERENCES claims(patent_id, claim_no) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS figure_captions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patent_id TEXT NOT NULL,
            figure_no TEXT NOT NULL,
            caption_raw TEXT,
            caption_norm TEXT,
            page_no INTEGER,
            UNIQUE (patent_id, figure_no, page_no),
            FOREIGN KEY (patent_id) REFERENCES patents(patent_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS drawing_ref_map (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patent_id TEXT NOT NULL,
            figure_no TEXT NOT NULL,
            ref_no_raw TEXT NOT NULL,
            page_no INTEGER,
            x0 REAL,
            y0 REAL,
            x1 REAL,
            y1 REAL,
            UNIQUE (
                patent_id, figure_no, ref_no_raw, page_no, x0, y0, x1, y1
            ),
            FOREIGN KEY (patent_id) REFERENCES patents(patent_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS jobs (
            patent_id TEXT PRIMARY KEY,
            pdf_path TEXT NOT NULL,
            status TEXT NOT NULL,
            retry_count INTEGER DEFAULT 0,
            last_error TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_text_spans_patent_page ON text_spans (patent_id, page_no);
        CREATE INDEX IF NOT EXISTS idx_claims_patent ON claims (patent_id);
        CREATE INDEX IF NOT EXISTS idx_ref_entities_patent ON ref_entities (patent_id);
        CREATE INDEX IF NOT EXISTS idx_figure_captions_patent ON figure_captions (patent_id);
        CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs (status);
        """
    )

    con.commit()
    con.close()
    return DB_PATH


# ---------- helper write functions ----------

def reset_patent_artifacts(con: sqlite3.Connection, patent_id: str) -> None:
    cur = con.cursor()
    cur.execute("DELETE FROM drawing_ref_map WHERE patent_id=?", (patent_id,))
    cur.execute("DELETE FROM figure_captions WHERE patent_id=?", (patent_id,))
    cur.execute("DELETE FROM claim_ref_map WHERE patent_id=?", (patent_id,))
    cur.execute("DELETE FROM ref_entities WHERE patent_id=?", (patent_id,))
    cur.execute("DELETE FROM claims WHERE patent_id=?", (patent_id,))
    cur.execute("DELETE FROM text_spans WHERE patent_id=?", (patent_id,))
    cur.execute("DELETE FROM pages WHERE patent_id=?", (patent_id,))
    cur.execute("DELETE FROM patents WHERE patent_id=?", (patent_id,))
    con.commit()


def upsert_job(
    con: sqlite3.Connection,
    patent_id: str,
    pdf_path: str,
    status: str,
    retry_count: int | None = None,
    last_error: str | None = None,
) -> None:
    cur = con.cursor()
    existing = cur.execute(
        "SELECT retry_count FROM jobs WHERE patent_id=?", (patent_id,)
    ).fetchone()
    current_retry = int(existing[0]) if existing else 0
    new_retry = current_retry if retry_count is None else retry_count

    cur.execute(
        """
        INSERT INTO jobs (patent_id, pdf_path, status, retry_count, last_error)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(patent_id) DO UPDATE SET
            pdf_path=excluded.pdf_path,
            status=excluded.status,
            retry_count=excluded.retry_count,
            last_error=excluded.last_error,
            updated_at=CURRENT_TIMESTAMP
        """,
        (patent_id, pdf_path, status, new_retry, last_error),
    )
    con.commit()


def increment_job_retry(con: sqlite3.Connection, patent_id: str, pdf_path: str, last_error: str) -> None:
    cur = con.cursor()
    existing = cur.execute(
        "SELECT retry_count FROM jobs WHERE patent_id=?", (patent_id,)
    ).fetchone()
    current_retry = int(existing[0]) if existing else 0
    upsert_job(con, patent_id, pdf_path, "failed", retry_count=current_retry + 1, last_error=last_error)


if __name__ == "__main__":
    path = ensure_db()
    print(f"[db_schema] initialized: {path}")
