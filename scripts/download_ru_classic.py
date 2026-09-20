#!/usr/bin/env python3
"""Скачивание русского корпуса Imperius/ru-classic (866 МБ чистой классики).

Раскладывает в data/raw/text/ кусками по ~50 МБ (не грузит всё в RAM —
стриминг). После этого: python3 steps/step_00_prepare.py

Источники (оба встроены):
  * datasets (по умолчанию, streaming=True) — колонка "text";
  * mlcroissant (код, который ты прислал) — --source croissant.

Объёмы:
  python3 scripts/download_ru_classic.py --limit-mb 20   # тест: 20 МБ, пара минут
  python3 scripts/download_ru_classic.py                # весь корпус (~866 МБ)

Запускать на ноуте при интернете. Потом данные офлайн и обучение офлайн.
"""
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

OUT_DIR = os.path.join(ROOT, "data", "raw", "text")
CHUNK_MB = 50
DEFAULT_LIMIT_MB = 0  # 0 = без ограничения (весь корпус)


def _extract_text(record) -> str:
    """Достаёт поле 'text' из записи mlcroissant/datasets устойчиво.

    mlcroissant отдаёт dict, но ключ зависит от версии схемы:
      * v0:            record["text"]   (имя поля)
      * новая схема:   record["default/text"]  или UUID вида record[".../text"]
    Плюс поле может быть вложено в dict ({"/text": {...}}).
    """
    if not isinstance(record, dict):
        # на случай, если отдают не dict, а объект строки — пробуем атрибуты
        return str(getattr(record, "text", "") or "").strip()

    for key in ("text", "/text", "default/text"):
        v = record.get(key)
        if isinstance(v, str):
            return v.strip()
        if isinstance(v, dict):
            # вложенный словарь: ищем любое строковое значение внутри
            for inner in v.values():
                if isinstance(inner, str) and inner.strip():
                    return inner.strip()

    # fallback: любой ключ, оканчивающийся на 'text'
    for key, val in record.items():
        if str(key).endswith("text") and isinstance(val, str):
            return val.strip()
        if str(key).endswith("text") and isinstance(val, dict):
            for inner in val.values():
                if isinstance(inner, str) and inner.strip():
                    return inner.strip()
    return ""


def iter_texts_datasets(limit_mb: int):
    from datasets import load_dataset
    ds = load_dataset("Imperius/ru-classic", split="train", streaming=True)
    for row in ds:
        text = _extract_text(row)
        if text:
            yield text


def iter_texts_croissant(limit_mb: int):
    try:
        from mlcroissant import Dataset
    except ImportError:
        sys.exit(
            "Нет mlcroissant. Поставь: pip install mlcroissant\n"
            "или используй --source datasets (по умолчанию)."
        )
    ds = Dataset(jsonld="https://huggingface.co/api/datasets/Imperius/ru-classic/croissant")
    for record in ds.records("default"):
        text = _extract_text(record)
        if text:
            yield text


def main() -> int:
    ap = argparse.ArgumentParser(description="Скачать Imperius/ru-classic в data/raw/text/")
    ap.add_argument("--source", choices=["datasets", "croissant"], default="datasets")
    ap.add_argument("--limit-mb", type=int, default=DEFAULT_LIMIT_MB,
                    help="Ограничить объём (МБ). 20 — быстрый тест, 0 — всё.")
    ap.add_argument("--chunk-mb", type=int, default=CHUNK_MB,
                    help="Размер выходного куска в МБ")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    limit_bytes = args.limit_mb * 1024 * 1024

    if args.source == "datasets":
        stream = iter_texts_datasets(args.limit_mb)
    else:
        stream = iter_texts_croissant(args.limit_mb)

    def open_next(part):
        p = os.path.join(OUT_DIR, f"ru_classic_{part:03d}.txt")
        print(f"[write] {p}")
        return open(p, "w", encoding="utf-8")

    part = 0
    written = 0
    total = 0
    chunk_roof = args.chunk_mb * 1024 * 1024
    records = 0
    f = open_next(part)
    try:
        for text in stream:
            f.write(text)
            f.write("\n")
            n = len(text) + 1
            written += n
            total += n
            records += 1
            # честная обрезка по объёму: --limit-mb 20 остановит на ~20 МБ
            if limit_bytes and total >= limit_bytes:
                break
            if written >= chunk_roof:
                f.close()
                part += 1
                written = 0
                f = open_next(part)
    except KeyboardInterrupt:
        print("\n[!] Прервано — то, что скачано, уже на диске.")
        return 130
    except Exception as exc:
        f.close()
        print(f"\n[!] Ошибка: {exc.__class__.__name__}: {exc}")
        print("    Проверь интернет (нужен доступ к huggingface.co) и версию библиотек.")
        return 1
    finally:
        try:
            f.close()
        except Exception:
            pass

    print(f"\n[ok] строк: {records:,}, байт: {total / 1e6:.1f} МБ -> {OUT_DIR}")
    print("Дальше:  python3 steps/step_00_prepare.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
