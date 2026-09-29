"""Lightweight SQLite persistence for:

- the document registry (the doc's "Simple Data Model" Documents table:
  id, name, upload_date, source -- plus indexing status),
- the query audit log (the doc's Queries table: question, response,
  timestamp), and
- per-run timings for summary/obligation analysis,

which together back the live KPI view (Response Time, Citation
Coverage, etc. from the doc's "MVP Success Criteria").

This is intentionally a single local SQLite file (stdlib sqlite3, no new
dependency) -- enough for a single-instance MVP. A multi-instance
deployment would swap this for Postgres; the schema and queries below
would carry over almost unchanged.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from app.config import DB_PATH


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT UNIQUE NOT NULL,
                source TEXT,
                upload_date TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'uploaded',
                chunks_count INTEGER DEFAULT 0,
                indexed_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                intent TEXT,
                grounded INTEGER,
                citation_count INTEGER,
                response_time_ms INTEGER,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                kind TEXT NOT NULL,
                response_time_ms INTEGER,
                timestamp TEXT NOT NULL
            )
            """
        )


# --------------------------------------------------------------------------
# Document registry
# --------------------------------------------------------------------------


def record_upload(filename: str, source: str = "manual_upload") -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO documents (filename, source, upload_date, status)
            VALUES (?, ?, ?, 'uploaded')
            ON CONFLICT(filename) DO UPDATE SET
                upload_date = excluded.upload_date,
                source = excluded.source
            """,
            (filename, source, _now()),
        )


def record_indexed(filename: str, chunks_count: int, source: str = "manual_upload") -> None:
    with _connect() as conn:
        # Insert the row if it doesn't exist yet (e.g. a file that was
        # dropped straight into the documents folder rather than
        # uploaded via the API), otherwise just update it.
        conn.execute(
            """
            INSERT INTO documents (filename, source, upload_date, status, chunks_count, indexed_at)
            VALUES (?, ?, ?, 'indexed', ?, ?)
            ON CONFLICT(filename) DO UPDATE SET
                status = 'indexed',
                chunks_count = excluded.chunks_count,
                indexed_at = excluded.indexed_at
            """,
            (filename, source, _now(), chunks_count, _now()),
        )


def remove_document_record(filename: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM documents WHERE filename = ?", (filename,))


def list_documents() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY upload_date DESC"
        ).fetchall()
        return [dict(row) for row in rows]


# --------------------------------------------------------------------------
# Query audit log
# --------------------------------------------------------------------------


def log_query(
    session_id: str | None,
    question: str,
    answer: str,
    intent: str | None,
    grounded: bool,
    citation_count: int,
    response_time_ms: int,
) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO queries
                (session_id, question, answer, intent, grounded,
                 citation_count, response_time_ms, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                question,
                answer,
                intent,
                1 if grounded else 0,
                citation_count,
                response_time_ms,
                _now(),
            ),
        )


def recent_queries(limit: int = 20) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM queries ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]


# --------------------------------------------------------------------------
# Analysis run timings (summary / obligations)
# --------------------------------------------------------------------------


def log_analysis_run(filename: str, kind: str, response_time_ms: int) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO analysis_runs (filename, kind, response_time_ms, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (filename, kind, response_time_ms, _now()),
        )


# --------------------------------------------------------------------------
# KPI aggregation (MVP Success Criteria)
# --------------------------------------------------------------------------


def get_kpis() -> dict:
    with _connect() as conn:
        query_stats = conn.execute(
            """
            SELECT
                COUNT(*) AS total_queries,
                AVG(response_time_ms) AS avg_response_time_ms,
                AVG(grounded) AS grounded_rate,
                SUM(CASE WHEN citation_count > 0 THEN 1 ELSE 0 END) * 1.0
                    / NULLIF(
                        SUM(CASE WHEN intent = 'domain_query' THEN 1 ELSE 0 END), 0
                      ) AS citation_coverage
            FROM queries
            """
        ).fetchone()

        summary_stats = conn.execute(
            """
            SELECT AVG(response_time_ms) AS avg_ms
            FROM analysis_runs WHERE kind = 'summary'
            """
        ).fetchone()

        obligation_stats = conn.execute(
            """
            SELECT AVG(response_time_ms) AS avg_ms
            FROM analysis_runs WHERE kind = 'obligations'
            """
        ).fetchone()

        doc_stats = conn.execute(
            """
            SELECT
                COUNT(*) AS total_documents,
                SUM(CASE WHEN status = 'indexed' THEN 1 ELSE 0 END) AS indexed_documents
            FROM documents
            """
        ).fetchone()

        return {
            "total_queries": query_stats["total_queries"] or 0,
            "avg_response_time_ms": round(query_stats["avg_response_time_ms"] or 0, 1),
            "grounded_rate_pct": round((query_stats["grounded_rate"] or 0) * 100, 1),
            "citation_coverage_pct": round(
                (query_stats["citation_coverage"] or 0) * 100, 1
            ),
            "avg_summary_time_ms": round(summary_stats["avg_ms"] or 0, 1),
            "avg_obligation_time_ms": round(obligation_stats["avg_ms"] or 0, 1),
            "total_documents": doc_stats["total_documents"] or 0,
            "indexed_documents": doc_stats["indexed_documents"] or 0,
            # From the doc's MVP Success Criteria table, for the UI to
            # compare live numbers against at a glance.
            "targets": {
                "response_time_ms": 5000,
                "summary_time_ms": 15000,
                "citation_coverage_pct": 100,
            },
        }
