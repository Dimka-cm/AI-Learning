"""Роутер-переключатель: решает, какой эксперт/инструмент отвечает.

Это и есть твой «переключатель, если надо»: детерминированный диспетчер
(никакой магии) поверх экспертов. Легко заменить на LLM-классификатор,
когда текстовая модель дообучится — интерфейс route() не меняется.
"""
from __future__ import annotations

import re

# Ключи, по которым мы узнаём тему запроса.
_MATH_WORDS = [r"\d+\s*[+\-*/%^]\s*\d+", r"реши", r"посчита", r"сколько будет",
               r"вычисли", r"пример", r"уравнение", r"матем", r"арифмети", r"="]

_CODE_WORDS = [r"\b(c\+\+|c#|c\b|python|java|типы|функци[яи]|класс|компил|объект|переменная|цикл|массив)",
               r"напиши (код|программу|скрипт)", r"с#", r"си\+\+|чистый си\b"]

_VISION_WORDS = [r"глаз", r"распозна", r"что на (картинк|фото|скрин|изображ)",
                 r"посмотри на (картинк|фото|скрин)", r"какая среда", r"узнай по (картинк|фото)",
                 r"\.(png|jpg|jpeg|bmp|webp)\b"]

_TOOL_WORDS = [r"(открой|покажи|прочитай|прочти)\s+(файл|папку|содержимое|директорию)",
               r"(создай|запиши|напиши в файл|сохрани)\s+(файл|папку|файл с)",
               r"удали\s+(файл|папку)", r"запусти\s+(команду|скрипт|программу)",
               r"список файлов", r"содержимое папки"]

# Слэш-команды инструментов (быстрый ручной доступ к «воркспейсу»).
TOOL_COMMANDS = {
    "/list": "list_dir", "/ls": "list_dir",
    "/read": "read_file", "/cat": "read_file",
    "/write": "write_file", "/mkdir": "mkdir",
    "/delete": "delete", "/rm": "delete",
    "/run": "run_command", "/see": "vision",
}


def route(text: str) -> str:
    """Возвращает имя эксперта/обработчика: code | math | vision | tool | chat."""
    t = text.strip().lower()

    if any(k in t for k in TOOL_COMMANDS) and t.startswith("/"):
        return "tool"
    if any(re.search(p, t) for p in _TOOL_WORDS):
        return "tool"
    if any(re.search(p, t) for p in _VISION_WORDS):
        return "vision"
    if any(re.search(p, t) for p in _MATH_WORDS):
        return "math"
    if any(re.search(p, t) for p in _CODE_WORDS):
        return "code"
    return "chat"


def intent_of(text: str) -> str:
    """Грубое извлечение намерения для tool-режима (fallback без LLM)."""
    t = text.lower()
    if "удали" in t or "удалить" in t:
        return "delete"
    if "создай" in t or "запиши" in t or "напиши файл" in t or "сохрани" in t:
        return "write"
    if "запусти" in t:
        return "run"
    if "создай папку" in t or "mkdir" in t:
        return "mkdir"
    if "список" in t or "содержимое" in t:
        return "list"
    return "read"
