"""SQLite-based session and cost memory for the image privacy agent."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent.parent / "data" / "privacy_agent.db"


class PrivacyMemoryManager:
    """Thread-safe SQLite memory. Stores sessions, threat reports, costs, paper hashes."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id      TEXT PRIMARY KEY,
                    created_at      TEXT NOT NULL,
                    api_provider    TEXT,
                    image_hash      TEXT,
                    epsilon         REAL,
                    ssim            REAL,
                    clip_cosine     REAL,
                    psnr_db         REAL,
                    protection_mode TEXT,
                    reversal_applied INTEGER DEFAULT 0,
                    cost_usd        REAL DEFAULT 0.0
                );

                CREATE TABLE IF NOT EXISTS threat_reports (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id      TEXT,
                    created_at      TEXT NOT NULL,
                    pii_types       TEXT,
                    risk_level      TEXT,
                    recommended_epsilon REAL,
                    recommendations TEXT,
                    attack_scenarios TEXT,
                    confidence      REAL,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS llm_cost_log (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at      TEXT NOT NULL,
                    provider        TEXT,
                    model           TEXT,
                    prompt_tokens   INTEGER,
                    completion_tokens INTEGER,
                    cost_usd        REAL,
                    task            TEXT
                );

                CREATE TABLE IF NOT EXISTS knowledge_hashes (
                    paper_hash      TEXT PRIMARY KEY,
                    added_at        TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_sessions_created ON sessions(created_at);
                CREATE INDEX IF NOT EXISTS idx_costs_created ON llm_cost_log(created_at);
            """)

    def save_session(
        self,
        session_id: str,
        api_provider: str,
        image_hash: str,
        epsilon: float,
        ssim: float,
        clip_cosine: float,
        psnr_db: float,
        protection_mode: str = "pixel",
        cost_usd: float = 0.0,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO sessions
                   (session_id, created_at, api_provider, image_hash, epsilon,
                    ssim, clip_cosine, psnr_db, protection_mode, cost_usd)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    datetime.now(timezone.utc).isoformat(),
                    api_provider,
                    image_hash,
                    epsilon,
                    ssim,
                    clip_cosine,
                    psnr_db,
                    protection_mode,
                    cost_usd,
                ),
            )

    def mark_reversal(self, session_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET reversal_applied=1 WHERE session_id=?",
                (session_id,),
            )

    def save_threat_report(
        self,
        session_id: Optional[str],
        pii_types: list[str],
        risk_level: str,
        recommended_epsilon: float,
        recommendations: list[str],
        attack_scenarios: list[str],
        confidence: float,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO threat_reports
                   (session_id, created_at, pii_types, risk_level,
                    recommended_epsilon, recommendations, attack_scenarios, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session_id,
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(pii_types),
                    risk_level,
                    recommended_epsilon,
                    json.dumps(recommendations),
                    json.dumps(attack_scenarios),
                    confidence,
                ),
            )

    def get_session_stats(self) -> dict:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """SELECT COUNT(*) as total,
                          AVG(ssim) as avg_ssim,
                          AVG(clip_cosine) as avg_clip,
                          AVG(psnr_db) as avg_psnr,
                          SUM(reversal_applied) as reversals
                   FROM sessions"""
            ).fetchone()
            return {
                "total_sessions": row["total"] or 0,
                "avg_ssim": round(row["avg_ssim"] or 0, 4),
                "avg_clip_cosine": round(row["avg_clip"] or 0, 4),
                "avg_psnr_db": round(row["avg_psnr"] or 0, 2),
                "total_reversals": row["reversals"] or 0,
            }

    def log_llm_cost(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        task: str = "",
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """INSERT INTO llm_cost_log
                   (created_at, provider, model, prompt_tokens, completion_tokens, cost_usd, task)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    provider,
                    model,
                    prompt_tokens,
                    completion_tokens,
                    cost_usd,
                    task,
                ),
            )

    def get_cost_summary(self, days: int = 30) -> dict:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """SELECT provider, SUM(cost_usd) as total_cost, COUNT(*) as calls
                   FROM llm_cost_log
                   WHERE created_at > datetime('now', ?)
                   GROUP BY provider""",
                (f"-{days} days",),
            ).fetchall()
            return {r["provider"]: {"total_cost": round(r["total_cost"], 6), "calls": r["calls"]} for r in rows}

    def is_known_paper(self, paper_hash: str) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM knowledge_hashes WHERE paper_hash=?", (paper_hash,)
            ).fetchone()
            return row is not None

    def mark_paper_known(self, paper_hash: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO knowledge_hashes (paper_hash, added_at) VALUES (?, ?)",
                (paper_hash, datetime.now(timezone.utc).isoformat()),
            )

    def get_known_paper_hashes(self) -> list[str]:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT paper_hash FROM knowledge_hashes").fetchall()
            return [r["paper_hash"] for r in rows]

    def get_stats(self) -> dict:
        sessions = self.get_session_stats()
        costs = self.get_cost_summary()
        with self._lock, self._connect() as conn:
            papers = conn.execute("SELECT COUNT(*) as n FROM knowledge_hashes").fetchone()["n"]
        return {
            "sessions": sessions,
            "costs_30d": costs,
            "known_papers": papers,
        }
