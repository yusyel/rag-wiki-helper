import os
import logging
from typing import Optional

import streamlit as st

try:
    import psycopg2
except ImportError:  # pragma: no cover
    psycopg2 = None

logger = logging.getLogger(__name__)


def _connection_kwargs() -> Optional[dict]:
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return {"dsn": database_url}

    required = {
        "host": os.getenv("PGHOST"),
        "port": os.getenv("PGPORT", "5432"),
        "dbname": os.getenv("PGDATABASE"),
        "user": os.getenv("PGUSER"),
        "password": os.getenv("PGPASSWORD"),
    }
    if not all(required.values()):
        return None
    return required


@st.cache_resource(show_spinner=False)
def init_db() -> None:
    if psycopg2 is None:
        logger.warning("psycopg2 is not installed; database init is skipped.")
        return

    kwargs = _connection_kwargs()
    if not kwargs:
        return

    conn = psycopg2.connect(**kwargs)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS qa_history (
                        id BIGSERIAL PRIMARY KEY,
                        query TEXT NOT NULL,
                        answer TEXT NOT NULL,
                        feedback TEXT,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                    """
                )
                cur.execute(
                    """
                    ALTER TABLE qa_history
                    ADD COLUMN IF NOT EXISTS feedback TEXT
                    """
                )
    finally:
        conn.close()


def save_qa_exchange(query: str, answer: str) -> Optional[int]:
    if psycopg2 is None:
        logger.warning("psycopg2 is not installed; qa exchange is not saved.")
        return None

    kwargs = _connection_kwargs()
    if not kwargs:
        return None

    try:
        conn = psycopg2.connect(**kwargs)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO qa_history (query, answer) VALUES (%s, %s) RETURNING id",
                        (query, answer),
                    )
                    row = cur.fetchone()
                    return int(row[0]) if row else None
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed to save qa exchange.")
        return None


def save_feedback(exchange_id: int, feedback: str) -> bool:
    if psycopg2 is None:
        logger.warning("psycopg2 is not installed; feedback is not saved.")
        return False
    if feedback not in {"up", "down"}:
        return False

    kwargs = _connection_kwargs()
    if not kwargs:
        return False

    try:
        conn = psycopg2.connect(**kwargs)
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE qa_history SET feedback = %s WHERE id = %s",
                        (feedback, exchange_id),
                    )
                    return cur.rowcount == 1
        finally:
            conn.close()
    except Exception:
        logger.exception("Failed to save feedback.")
        return False
