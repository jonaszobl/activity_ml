# src/utils_jsonl.py
from typing import Iterator, TextIO
import json

def _iter_jsonl_lines(io: TextIO) -> Iterator[dict]:
    for line in io:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            # ungültige Zeile überspringen
            continue

def read_jsonl_from_io(io: TextIO) -> Iterator[dict]:
    # öffentliche API, die du in app.py verwendest
    yield from _iter_jsonl_lines(io)

def read_jsonl(path: str) -> Iterator[dict]:
    # Kompatibilitäts-Wrapper für predict_workout.py (erwartet read_jsonl)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        yield from _iter_jsonl_lines(f)
