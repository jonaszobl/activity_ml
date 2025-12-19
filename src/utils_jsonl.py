import json
from typing import Dict, Iterable

def read_jsonl(path: str) -> Iterable[Dict]:
    """Liest eine JSONL-Datei. Überspringt meta/meta_end-Zeilen."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if obj.get("type") in ("meta", "meta_end"):
                continue
            yield obj
