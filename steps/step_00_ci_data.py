#!/usr/bin/env python3
"""Подготовка данных: сырые .txt (Drive / HF / демо) -> токенизация в 10 порций.

Работает и в CI, и на ноуте. Порядок по умолчанию (DATA_SOURCE=auto):

  1) если в data/raw уже есть .txt (рекурсивно) — ничего не качаем, сразу токенизируем;
  2) иначе DRIVE_FOLDER_URL — качаем .txt из папки Drive, но НЕ БОЛЬШЕ LIMIT_MB;
  3) иначе HF_DATASET — стримим датасет (тоже с потолком LIMIT_MB);
  4) иначе — демо-корпус (офлайн, только stdlib).

Почему с потолком: папка Drive владельца — библиотека книг, она может весить
гигабайты. Раньше `gdown --folder` тянул её целиком, шаг «данные» съедал все
300 минут джобы и падал по таймауту. Теперь объём ограничен, а если сеть/Drive
недоступны — пайплайн всё равно доезжает до конца на демо-корпусе.

Переменные окружения:
    DATA_SOURCE        auto | drive | hf | demo   (по умолчанию auto)
    DRIVE_FOLDER_URL   ссылка на папку Google Drive с .txt
    HF_DATASET         имя датасета на HuggingFace (напр. Imperius/ru-classic)
    LIMIT_MB           потолок объёма в МБ (по умолчанию 25; 0 = без потолка)
    DRIVE_BUDGET_S     сколько секунд максимум отдать Drive (по умолчанию 900)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW = os.path.join(ROOT, "data", "raw")


def has_txt(root: str) -> bool:
    """Есть ли .txt где-нибудь внутри — gdown/HF кладут их в подпапки."""
    p = Path(root)
    return p.is_dir() and any(f.suffix.lower() == ".txt" for f in p.rglob("*.txt"))


def count_txt(root: str) -> tuple[int, int]:
    files = [f for f in Path(root).rglob("*.txt")] if os.path.isdir(root) else []
    total = sum(f.stat().st_size for f in files)
    return len(files), total


def report(msg: str) -> None:
    """Пишем в лог шага и, если есть, в Job Summary."""
    print(msg, flush=True)
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(msg + "\n\n")
        except OSError:
            pass


# ------------------------------------------------------------------ Drive

def _rel_name(f) -> str:
    """Путь файла внутри папки Drive (без выхода за пределы каталога)."""
    raw = getattr(f, "path", None) or getattr(f, "name", "") or ""
    parts = [x for x in str(raw).replace("\\", "/").split("/") if x not in ("", ".", "..")]
    return "/".join(parts[-2:]) or f"drive_{getattr(f, 'id', 'file')}.txt"


def fetch_drive(url: str, out_dir: str, limit_mb: int, budget_s: int) -> int:
    """Качает .txt из папки Drive, не превышая limit_mb и budget_s. -> байт."""
    limit = limit_mb * 1024 * 1024 if limit_mb else 0
    started = time.time()

    try:
        import gdown
    except ImportError:
        report("[data] gdown не установлен — папку Drive пропускаю")
        return 0

    try:
        files = gdown.download_folder(url, quiet=True, skip_download=True)
    except Exception as exc:  # сеть, доступ, смена API Google — не повод падать
        report(f"[data] не смог получить список файлов Drive ({type(exc).__name__}) — пропускаю")
        return 0

    txts = [f for f in (files or []) if str(getattr(f, "name", "")).lower().endswith(".txt")]
    others = [f for f in (files or []) if f not in txts]
    report(f"[data] в папке Drive: {len(files or [])} объектов, из них .txt — {len(txts)}")
    if others:
        report(f"[data] пропускаю {len(others)} файлов не-.txt (токенизатор читает только .txt)")
    if not txts:
        return 0

    got = 0
    taken = 0
    for f in txts:
        if limit and got >= limit:
            report(f"[data] потолок {limit_mb} МБ достигнут — остальные файлы не качаю")
            break
        if time.time() - started > budget_s:
            report(f"[data] вышло время на Drive ({budget_s} с) — дальше без него")
            break
        dst = os.path.join(out_dir, _rel_name(f))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            gdown.download(id=getattr(f, "id"), output=dst, quiet=True)
        except Exception as exc:
            report(f"[data] не скачался {getattr(f, 'name', '?')}: {type(exc).__name__}")
            continue
        if os.path.exists(dst):
            got += os.path.getsize(dst)
            taken += 1

    report(f"[data] с Drive скачано: {taken} файлов, {got / 1e6:.1f} МБ")
    return got


# ------------------------------------------------------------------ HF

def fetch_hf(dataset: str, limit_mb: int) -> int:
    """Стриминг датасета через готовый скрипт (он сам пишет куски по 50 МБ)."""
    cmd = [sys.executable, "scripts/download_ru_classic.py",
           "--source", "datasets", "--limit-mb", str(limit_mb)]
    if dataset and dataset != "Imperius/ru-classic":
        report(f"[data] HF_DATASET={dataset}: скачиватель пока умеет только Imperius/ru-classic")
    rc = subprocess.call(cmd, cwd=ROOT)
    return 0 if rc == 0 else 0


# ------------------------------------------------------------------ main

def main() -> int:
    os.environ["CI"] = "1"
    os.environ.setdefault("HF_HOME", os.path.join(ROOT, "hf-cache"))

    source = (os.environ.get("DATA_SOURCE") or "auto").strip().lower()
    drive_url = (os.environ.get("DRIVE_FOLDER_URL") or "").strip()
    hf_ds = (os.environ.get("HF_DATASET") or "").strip()
    limit_mb = int(os.environ.get("LIMIT_MB") or "25")
    budget_s = int(os.environ.get("DRIVE_BUDGET_S") or "900")

    out_text = os.path.join(RAW, "text")
    os.makedirs(out_text, exist_ok=True)

    if has_txt(RAW) and source == "auto":
        n, size = count_txt(RAW)
        report(f"[data] свои .txt уже на месте: {n} файлов, {size / 1e6:.1f} МБ — качать нечего")
    elif source == "demo":
        report("[data] выбрано демо-корпус (сеть не трогаем)")
    else:
        # Каскад: Drive -> HF. Что-то да сработает, а если нет — ниже демо.
        if source in ("drive", "auto") and drive_url:
            report(f"[data] папка Drive: потолок {limit_mb} МБ, лимит времени {budget_s} с")
            fetch_drive(drive_url, out_text, limit_mb, budget_s)
        if not has_txt(RAW) and source in ("hf", "auto") and (hf_ds or source == "hf"):
            name = hf_ds or "Imperius/ru-classic"
            report(f"[data] стримлю датасет {name}, потолок {limit_mb} МБ")
            fetch_hf(name, limit_mb)

    if not has_txt(RAW):
        report("[data] сырых .txt нет — генерирую демо-корпус (офлайн)")
        from scripts.make_data import main as make_demo
        make_demo()

    n, size = count_txt(RAW)
    report(f"[data] итого сырья: {n} файлов, {size / 1e6:.1f} МБ")
    if n == 0:
        report("[data] ПУСТО: ни Drive, ни HF, ни демо не дали текста — токенизировать нечего")
        return 1

    report("[data] токенизация в 10 порций…")
    from steps.step_00_prepare import main as prep
    return prep()


if __name__ == "__main__":
    raise SystemExit(main())
