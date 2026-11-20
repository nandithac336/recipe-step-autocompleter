"""Small JSON key-value store retained from the starter package."""
import json
import os
from pathlib import Path

class Store:
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.environ.get("DATA_PATH", "data/app.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}", encoding="utf-8")
    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8") or "{}")
    def put(self, key: str, value) -> None:
        data = self._read()
        data[key] = value
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    def get(self, key: str, default=None):
        return self._read().get(key, default)
    def all(self) -> dict:
        return dict(sorted(self._read().items()))
