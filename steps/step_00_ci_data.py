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
import re
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



def _parse_size_bytes(value) -> int | None:
    """Пытается прочитать размер файла из объекта gdown/строки.

    gdown в разных версиях может отдавать размер как int, float или строку
    вроде ``"460 MB"``/``"1.2GB"``. Если размер неизвестен — None.
    """
    if value in (None, ""):
        return None
    if isinstance(value, (int, float)):
        return int(value) if value >= 0 else None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    if text.isdigit():
        return int(text)
    m = re.match(r"^([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b?|байт|кб|мб|гб)\s*$", text, re.IGNORECASE)
    if not m:
        return None
    n = float(m.group(1))
    unit = m.group(2).lower()
    mul = 1
    if unit in ("k", "kb", "kib", "кб"):
        mul = 1024
    elif unit in ("m", "mb", "mib", "мб"):
        mul = 1024 ** 2
    elif unit in ("g", "gb", "gib", "гб"):
        mul = 1024 ** 3
    elif unit in ("t", "tb", "tib"):
        mul = 1024 ** 4
    return int(n * mul)


def _drive_file_size(f) -> int | None:
    """Размер файла Drive, если gdown смог его вернуть."""
    for attr in ("size", "size_bytes", "bytes", "file_size", "filesize"):
        n = _parse_size_bytes(getattr(f, attr, None))
        if n is not None:
            return n
    return None



def _natural_key(text: str) -> list[object]:
    """Сортировка part_2 перед part_10."""
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", text)]


def _filter_drive_txts(files, include_regex: str = "", max_files: int = 0):
    """Фильтр .txt из Drive по имени/пути.

    Нужен для корпуса владельца: 7 частей по ~0.5-1+ ГБ, удобнее запускать
    по одной части, например DRIVE_INCLUDE_REGEX=part_07 и DRIVE_MAX_FILES=1.
    """
    txts = [f for f in (files or []) if str(getattr(f, "name", "")).lower().endswith(".txt")]
    txts.sort(key=lambda f: _natural_key(_rel_name(f)))
    if include_regex:
        rx = re.compile(include_regex, re.IGNORECASE)
        txts = [
            f for f in txts
            if rx.search(str(getattr(f, "name", ""))) or rx.search(_rel_name(f))
        ]
    if max_files and max_files > 0:
        txts = txts[:max_files]
    return txts

def _rel_name(f) -> str:
    """Путь файла внутри папки Drive (без выхода за пределы каталога)."""
    raw = getattr(f, "path", None) or getattr(f, "name", "") or ""
    parts = [x for x in str(raw).replace("\\", "/").split("/") if x not in ("", ".", "..")]
    return "/".join(parts[-2:]) or f"drive_{getattr(f, 'id', 'file')}.txt"


def fetch_drive(url: str, out_dir: str, limit_mb: int, budget_s: int,
                include_regex: str = "", max_files: int = 0) -> int:
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

    all_txts = [f for f in (files or []) if str(getattr(f, "name", "")).lower().endswith(".txt")]
    others = [f for f in (files or []) if f not in all_txts]
    txts = _filter_drive_txts(files, include_regex, max_files)
    report(f"[data] в папке Drive: {len(files or [])} объектов, из них .txt — {len(all_txts)}")
    if include_regex:
        report(f"[data] фильтр Drive: /{include_regex}/ -> {len(txts)} .txt")
    if max_files:
        report(f"[data] максимум .txt после фильтра: {max_files}")
    if others:
        report(f"[data] пропускаю {len(others)} файлов не-.txt (токенизатор читает только .txt)")
    if not txts:
        report("[data] после фильтра не осталось .txt")
        return 0

    got = 0
    taken = 0
    skipped_big = 0
    unknown_size = 0
    for f in txts:
        if limit and got >= limit:
            report(f"[data] потолок {limit_mb} МБ достигнут — остальные файлы не качаю")
            break
        if time.time() - started > budget_s:
            report(f"[data] вышло время на Drive ({budget_s} с) — дальше без него")
            break

        name = getattr(f, "name", "?")
        size = _drive_file_size(f)
        remaining = limit - got if limit else 0
        if limit and size is not None and size > remaining:
            skipped_big += 1
            report(
                f"[data] пропускаю {name}: {size / 1e6:.1f} МБ больше "
                f"остатка лимита {remaining / 1e6:.1f} МБ"
            )
            continue
        if limit and size is None:
            unknown_size += 1
            report(f"[data] размер {name} неизвестен — скачаю, но лимит проверю после файла")

        dst = os.path.join(out_dir, _rel_name(f))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            gdown.download(id=getattr(f, "id"), output=dst, quiet=True)
        except Exception as exc:
            report(f"[data] не скачался {name}: {type(exc).__name__}")
            continue
        if os.path.exists(dst):
            got += os.path.getsize(dst)
            taken += 1

    report(f"[data] с Drive скачано: {taken} файлов, {got / 1e6:.1f} МБ")
    if skipped_big:
        report(
            f"[data] пропущено больших .txt: {skipped_big}. "
            f"Если все части корпуса по сотни МБ/ГБ, для smoke сделай отдельный "
            f"маленький .txt или подними data_limit_mb выше размера нужной части."
        )
    if unknown_size:
        report(f"[data] файлов с неизвестным размером: {unknown_size}")
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
    drive_include_regex = (os.environ.get("DRIVE_INCLUDE_REGEX") or "").strip()
    drive_max_files = int(os.environ.get("DRIVE_MAX_FILES") or "0")

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
            fetch_drive(drive_url, out_text, limit_mb, budget_s, drive_include_regex, drive_max_files)
        if not has_txt(RAW) and source in ("hf", "auto") and (hf_ds or source == "hf"):
            name = hf_ds or "Imperius/ru-classic"
            report(f"[data] стримлю датасет {name}, потолок {limit_mb} МБ")
            fetch_hf(name, limit_mb)

    if not has_txt(RAW):
        # Масштаб важен: при scale=1 демо-корпус даёт ~230 токенов на порцию,
        # а модели для одного окна нужно block_size+2. Отсюда была ошибка
        # «Слишком мало токенов в shard_01: 232 < block_size+2».
        scale = os.environ.get("DATA_SCALE", "20")
        report(f"[data] сырых .txt нет — генерирую демо-корпус (офлайн, масштаб x{scale})")
        rc = subprocess.call([sys.executable, "scripts/make_data.py", "--scale", str(scale)], cwd=ROOT)
        if rc != 0:
            report(f"[data] make_data.py упал с кодом {rc}")
            return 1

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
