"""Simple JSON file storage backend for Talos."""

import json
from pathlib import Path
from typing import Any


class JSONStorage:
    """Simple JSON file-based storage backend for Talos."""

    def __init__(self, dir_path: str | Path) -> None:
        """
        Initialize JSON storage.

        Args:
            dir_path: Directory to store JSON files
        """
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, object_id: int) -> Path:
        """Get the file path for an object."""
        return self.dir_path / f"object_{object_id}.json"

    def save(self, object_id: int, stored_values: dict[str, Any]) -> None:
        """Save stored values for an object."""
        file_path = self._get_file_path(object_id)
        with open(file_path, "w") as f:
            json.dump(stored_values, f, indent=2)

    def load(self, object_id: int) -> dict[str, Any] | None:
        """Load stored values for an object."""
        file_path = self._get_file_path(object_id)

        if not file_path.exists():
            return None

        with open(file_path) as f:
            data: dict[str, Any] = json.load(f)
            return data

    def delete(self, object_id: int) -> None:
        """Delete stored values for an object."""
        file_path = self._get_file_path(object_id)
        if file_path.exists():
            file_path.unlink()

    def exists(self, object_id: int) -> bool:
        """Check if stored values exist for an object."""
        file_path = self._get_file_path(object_id)
        return file_path.exists()

    def clear_all(self) -> None:
        """Clear all stored values (for testing)."""
        for file_path in self.dir_path.glob("object_*.json"):
            file_path.unlink()
