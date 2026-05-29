"""Simple JSON file storage backend for calyxos."""

import json
from pathlib import Path
from typing import Any


class JSONStorage:
    """Simple JSON file-based storage backend for calyxos.

    Files are named ``<key>.json`` inside the storage directory.
    """

    def __init__(self, dir_path: str | Path) -> None:
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, key: str) -> Path:
        # Sanitise key for filesystem safety
        safe = key.replace("/", "_").replace("\\", "_")
        return self.dir_path / f"{safe}.json"

    def save(self, key: str, stored_values: dict[str, Any]) -> None:
        file_path = self._get_file_path(key)
        with open(file_path, "w") as f:
            json.dump(stored_values, f, indent=2)

    def load(self, key: str) -> dict[str, Any] | None:
        file_path = self._get_file_path(key)
        if not file_path.exists():
            return None
        with open(file_path) as f:
            data: dict[str, Any] = json.load(f)
            return data

    def delete(self, key: str) -> None:
        file_path = self._get_file_path(key)
        if file_path.exists():
            file_path.unlink()

    def exists(self, key: str) -> bool:
        return self._get_file_path(key).exists()

    def clear_all(self) -> None:
        """Clear all stored values (for testing)."""
        for file_path in self.dir_path.glob("*.json"):
            file_path.unlink()
