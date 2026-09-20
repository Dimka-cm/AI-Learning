"""Память агента: короткая (история диалога) + долгая (файл знаний).

Намеренно просто и прозрачно: JSON-файлы, никаких скрытых состояний.
Апгрейд-путь: заменить долгую память на векторный store (chroma/faiss),
когда появится эмбеддер — интерфейс от этого не меняется.
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional


class Memory:
    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)
        self.history_path = os.path.join(directory, "history.jsonl")
        self.knowledge_path = os.path.join(directory, "knowledge.json")
        self._history: List[dict] = []
        self._knowledge: List[dict] = self._load_knowledge()

    # ------------------------------------------------------------- история
    def _append_jsonl(self, path: str, record: dict):
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add(self, role: str, text: str, meta: Optional[dict] = None):
        rec = {"ts": time.time(), "role": role, "text": text, "meta": meta or {}}
        self._history.append(rec)
        self._append_jsonl(self.history_path, rec)

    def recent(self, n: int = 6) -> List[dict]:
        return self._history[-n:]

    def context(self, n: int = 6) -> str:
        return "\n".join(f"{r['role']}: {r['text']}" for r in self.recent(n))

    # ------------------------------------------------------------- знания
    def _load_knowledge(self) -> List[dict]:
        if not os.path.exists(self.knowledge_path):
            return []
        try:
            with open(self.knowledge_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def remember_knowledge(self, key: str, value: str):
        self._knowledge = [k for k in self._knowledge if k["key"] != key]
        self._knowledge.append({"key": key, "value": value})
        with open(self.knowledge_path, "w", encoding="utf-8") as f:
            json.dump(self._knowledge, f, ensure_ascii=False, indent=2)

    def search_knowledge(self, query: str, limit: int = 5) -> List[dict]:
        q = query.lower()
        hits = [k for k in self._knowledge if q in k["key"].lower() or q in k["value"].lower()]
        return hits[:limit]
