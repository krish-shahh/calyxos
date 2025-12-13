"""SQLite storage backend for Talos."""

import json
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteStorage:
    """SQLite-based storage backend for persisting Talos object state."""

    def __init__(self, db_path: str | Path) -> None:
        """
        Initialize SQLite storage.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calyxos_stored_values (
                    object_id INTEGER PRIMARY KEY,
                    stored_values TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def save(self, object_id: int, stored_values: dict[str, Any]) -> None:
        """Save stored values for an object."""
        json_data = json.dumps(stored_values)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO calyxos_stored_values (object_id, stored_values, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(object_id) DO UPDATE SET
                    stored_values = excluded.stored_values,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (object_id, json_data),
            )
            conn.commit()

    def load(self, object_id: int) -> dict[str, Any] | None:
        """Load stored values for an object."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT stored_values FROM calyxos_stored_values WHERE object_id = ?",
                (object_id,),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        data: dict[str, Any] = json.loads(row[0])
        return data

    def delete(self, object_id: int) -> None:
        """Delete stored values for an object."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM calyxos_stored_values WHERE object_id = ?", (object_id,))
            conn.commit()

    def exists(self, object_id: int) -> bool:
        """Check if stored values exist for an object."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT 1 FROM calyxos_stored_values WHERE object_id = ?", (object_id,)
            )
            return cursor.fetchone() is not None

    def clear_all(self) -> None:
        """Clear all stored values (for testing)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM calyxos_stored_values")
            conn.commit()

    def close(self) -> None:
        """Close the database connection if needed."""
        pass
