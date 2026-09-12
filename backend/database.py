import json
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List

DB_PATH = "audit_logs.db"


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes SQLite database for audit trail storage."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scan_id TEXT NOT NULL UNIQUE,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            compliance_status TEXT NOT NULL,
            confidence REAL NOT NULL,
            violations TEXT NOT NULL,
            warnings TEXT NOT NULL,
            audit_trail TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def save_audit_log(
    scan_id: str,
    filename: str,
    compliance_status: str,
    confidence: float,
    violations: List[str],
    warnings: List[str],
    audit_trail: List[Dict[str, Any]],
    db_path: str = DB_PATH,
) -> None:
    """Persists an immutable audit log entry."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    timestamp = datetime.now(timezone.utc).isoformat()

    cursor.execute(
        """
        INSERT INTO audit_logs (
            scan_id, timestamp, filename, compliance_status,
            confidence, violations, warnings, audit_trail
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            scan_id,
            timestamp,
            filename,
            compliance_status,
            float(confidence),
            json.dumps(violations),
            json.dumps(warnings),
            json.dumps(audit_trail),
        ),
    )
    conn.commit()
    conn.close()


def fetch_all_logs(db_path: str = DB_PATH) -> List[Dict[str, Any]]:
    """Retrieves all past scan records sorted by timestamp descending."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT scan_id, timestamp, filename, compliance_status,
               confidence, violations, warnings, audit_trail
        FROM audit_logs ORDER BY id DESC
        """
    )
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "scan_id": row[0],
            "timestamp": row[1],
            "filename": row[2],
            "compliance_status": row[3],
            "confidence": row[4],
            "violations": json.loads(row[5]),
            "warnings": json.loads(row[6]),
            "audit_trail": json.loads(row[7]),
        }
        for row in rows
    ]