"""SQLite storage backend for calyxos."""

import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteStorage:
    """SQLite-based storage backend for persisting calyxos object state.

    Keys are user-supplied strings that remain stable across process restarts.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calyxos_stored (
                    key TEXT PRIMARY KEY,
                    stored_values TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def save(self, key: str, stored_values: dict[str, Any]) -> None:
        json_data = json.dumps(stored_values)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO calyxos_stored (key, stored_values, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    stored_values = excluded.stored_values,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, json_data),
            )
            conn.commit()

    def load(self, key: str) -> dict[str, Any] | None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT stored_values FROM calyxos_stored WHERE key = ?",
                (key,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        data: dict[str, Any] = json.loads(row[0])
        return data

    def delete(self, key: str) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM calyxos_stored WHERE key = ?", (key,))
            conn.commit()

    def exists(self, key: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM calyxos_stored WHERE key = ?", (key,)
            )
            return cursor.fetchone() is not None

    def clear_all(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM calyxos_stored")
            conn.commit()

    def close(self) -> None:
        pass
