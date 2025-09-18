# src/utils_jsonl.py
from typing import Iterator, Union, TextIO
import json

def read_jsonl_from_io(io: TextIO) -> Iterator[dict]:
    for line in io:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except Exception:
            # invalid line -> skip
            continue
