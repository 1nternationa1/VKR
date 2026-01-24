import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "history.db"


def init_db() -> None:
    """Ensure the SQLite database and table exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                mode TEXT,
                input_json TEXT NOT NULL,
                output_json TEXT,
                raw_response TEXT,
                recommendation TEXT,
                risk_score REAL
            )
            """
        )
        conn.commit()


def log_history(
    property_data: Dict[str, Any],
    report_data: Optional[Any],
    raw_response: Optional[str],
    mode: str,
) -> int:
    created_at = datetime.utcnow().isoformat()
    recommendation = None
    risk_score = None
    serialized_report: Optional[str] = None

    if isinstance(report_data, dict):
        recommendation = report_data.get("recommendation")
        risk_score = report_data.get("risk_score")
        serialized_report = json.dumps(report_data, ensure_ascii=False)
    elif report_data:
        serialized_report = str(report_data)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.execute(
            """
            INSERT INTO history (created_at, mode, input_json, output_json, raw_response, recommendation, risk_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                mode,
                json.dumps(property_data, ensure_ascii=False),
                serialized_report,
                raw_response,
                recommendation,
                risk_score,
            ),
        )
        conn.commit()
        return cursor.lastrowid


def fetch_history(limit: int = 20) -> List[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, recommendation, risk_score, mode
            FROM history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


def fetch_history_item(record_id: int) -> Optional[Dict[str, Any]]:
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT id, created_at, recommendation, risk_score, mode, input_json, output_json, raw_response
            FROM history
            WHERE id = ?
            """,
            (record_id,),
        ).fetchone()
        if not row:
            return None

        record = dict(row)
        record["input_json"] = json.loads(record["input_json"])
        if record["output_json"]:
            try:
                record["output_json"] = json.loads(record["output_json"])
            except json.JSONDecodeError:
                pass
        return record
