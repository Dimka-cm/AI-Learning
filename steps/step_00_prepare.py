"""Шаг 00: токенизация сырых .txt в 10 порций data/shards/shard_XX.bin.

Раскладка по доменам (папки в data/raw/):
    text/      -> shards 01..04   (русский текст, диалоги — база и разговор)
    math/      -> shards 05..06   (математика)
    code/      -> shards 07..08   (C / C++ / C#)
    behavior/  -> shards 09..10   (поведение: без обзывательств, вежливость)
Если папки нет — её порции останутся пустыми; если есть data/raw/*.txt в корне
(старый формат) — они делятся round-robin по всем 10 порциям.

Запуск:  python3 steps/step_00_prepare.py
Токенизатор качается один раз (кэш HF). Это CPU-задача, GPU не обязателен.
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_cfg
from src.data_utils import token_id_dtype

# домен -> диапазон порций (включительно)
DOMAIN_SHARDS = {
    "text": (1, 4),
    "math": (5, 6),
    "code": (7, 8),
    "behavior": (9, 10),
}


# Распознаём Q/A-диалоги, чтобы не разрывать пару «вопрос -> ответ» между
# разными shard. Это важно для корпуса владельца: там много вручную написанных
# русских вопрос-ответов. Если вопрос и ответ попадут в разные порции, модель
# хуже выучит именно формат диалога.
QUESTION_PREFIX_RE = re.compile(
    r"^\s*(пользователь|user|human|человек|вопрос|q)\s*[:：-]",
    re.IGNORECASE,
)
ANSWER_PREFIX_RE = re.compile(
    r"^\s*(агент|assistant|ассистент|бот|модель|ответ|a)\s*[:：-]",
    re.IGNORECASE,
)
INLINE_ANSWER_RE = re.compile(
    r"(агент|assistant|ассистент|бот|модель|ответ|a)\s*[:：-]",
    re.IGNORECASE,
)
SPEAKER_PREFIX_RE = re.compile(
    r"^\s*(пользователь|user|human|человек|вопрос|q|агент|assistant|ассистент|бот|модель|ответ|a)\s*[:：-]",
    re.IGNORECASE,
)


def iter_training_units(fh):
    """Поток учебных блоков из txt.

    Обычная проза остаётся построчной. Диалоги вида:

        Пользователь: вопрос
        Агент: ответ
        Пользователь: следующий вопрос
        Агент: следующий ответ

    группируются парами/блоками, чтобы round-robin раскладка по shards не
    отправила вопрос в shard_01, а ответ — в shard_02. Пустая строка тоже
    завершает текущий блок, поэтому формат с blank-line-separated Q/A работает.

    Yield: (text, raw_line_count).
    """
    block: list[str] = []
    raw_count = 0
    has_answer = False

    def flush():
        nonlocal block, raw_count, has_answer
        if not block:
            return None
        item = ("\n".join(block), raw_count)
        block = []
        raw_count = 0
        has_answer = False
        return item

    for raw_line in fh:
        line = raw_line.strip()
        if not line:
            item = flush()
            if item is not None:
                yield item
            continue

        is_question = bool(QUESTION_PREFIX_RE.match(line))
        is_answer = bool(ANSWER_PREFIX_RE.match(line))
        has_inline_answer = bool(INLINE_ANSWER_RE.search(line))
        has_speaker = bool(SPEAKER_PREFIX_RE.match(line))

        if has_speaker:
            # Новый вопрос после уже увиденного ответа = новая Q/A-пара.
            if is_question and block and has_answer:
                item = flush()
                if item is not None:
                    yield item
            block.append(line)
            raw_count += 1
            has_answer = has_answer or is_answer or has_inline_answer
            continue

        if block:
            # Продолжение многострочного вопроса/ответа. Защита от слишком
            # огромного блока: если в txt нет пустых строк, режем по 40 строкам.
            block.append(line)
            raw_count += 1
            if raw_count >= 40:
                item = flush()
                if item is not None:
                    yield item
            continue

        # Обычная строка прозы/кода без Q/A-маркеров.
        yield line, 1

    item = flush()
    if item is not None:
        yield item


def main() -> int:
    cfg = get_cfg()
    raw_dir = cfg.p(cfg.text_dir)
    npdtype = token_id_dtype(cfg)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token_id or 0

    out_dir = cfg.p(cfg.tokenized_dir)
    os.makedirs(out_dir, exist_ok=True)
    # сбрасываем старые порции, чтобы не дублировать при повторных прогонах
    for i in range(cfg.shards):
        p = os.path.join(out_dir, f"shard_{i + 1:02d}.bin")
        if os.path.exists(p):
            os.remove(p)

    targets = {s: 0 for s in range(1, cfg.shards + 1)}  # счётчик токенов на порцию
    files: list[tuple[str, int, int]] = []               # (path, lo, hi) диапазон домена

    # 1) доменные папки (рекурсивно: gdown --folder кладёт файлы в подпапку)
    for domain, (lo, hi) in DOMAIN_SHARDS.items():
        d = os.path.join(raw_dir, domain)
        if not os.path.isdir(d):
            continue
        txts = sorted(str(p) for p in Path(d).rglob("*.txt"))
        txts = sorted(set(txts))  # без дублей
        if not txts:
            continue
        for path in txts:
            files.append((path, lo, hi))
        print(f"[domain] {domain}: {len(txts)} файлов -> shards {lo}..{hi}")

    # 2) старый формат: *.txt в корне -> round-robin по всем 10
    legacy = sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
        if f.endswith(".txt") and os.path.isfile(os.path.join(raw_dir, f))
    ) if os.path.isdir(raw_dir) else []
    if legacy:
        print(f"[legacy] {len(legacy)} файлов в корне data/raw -> round-robin")
        for p in legacy:
            files.append((p, 1, cfg.shards))

    # читаем потоком, не держим весь корпус в памяти
    buffer: dict[int, list[int]] = {s: [] for s in range(1, cfg.shards + 1)}

    def flush(s: int):
        if buffer[s]:
            arr = np.asarray(buffer[s], dtype=npdtype)
            with open(os.path.join(out_dir, f"shard_{s:02d}.bin"), "ab") as fw:
                arr.tofile(fw)
            targets[s] += len(buffer[s])
            buffer[s].clear()

    # печатаем прогресс: на большом корпусе (гигабайты) иначе непонятно,
    # работает шаг или завис
    lines_done = 0
    started = time.time()

    for path, lo, hi in files:
        try:
            fh = open(path, "r", encoding="utf-8", errors="replace")
        except OSError as ex:
            print(f"[warn] пропуск {path}: {ex}")
            continue
        span = hi - lo + 1
        file_lines = 0
        file_units = 0
        with fh:
            for i, (unit, raw_lines) in enumerate(iter_training_units(fh)):
                # каждый учебный блок идёт в свой шард диапазона домена
                # (равномерно), но Q/A-пара остаётся внутри одного блока.
                s = lo + (i % span)
                ids = tok(unit)["input_ids"] + [eos]
                buffer[s].extend(ids)
                if len(buffer[s]) > 500_000:
                    flush(s)
                file_units += 1
                file_lines += raw_lines
                lines_done += raw_lines
                if lines_done % 200_000 == 0:
                    mins = (time.time() - started) / 60
                    print(f"[token] строк {lines_done:,} за {mins:.1f} мин "
                          f"({lines_done / max(mins, 1e-6) / 1000:.0f} тыс. строк/мин)", flush=True)
        print(f"[token] {os.path.basename(path)}: {file_lines:,} строк, "
              f"{file_units:,} учебных блоков", flush=True)

    for s in range(1, cfg.shards + 1):
        flush(s)

    need = cfg.block_size + 2  # ровно столько нужно для одного окна обучения
    print(f"\n[токенов в порции] (для обучения нужно минимум {need:,})")
    small = []
    for s in range(1, cfg.shards + 1):
        mb = targets[s] * np.dtype(npdtype).itemsize / 1e6
        mark = ""
        if targets[s] == 0:
            mark = "  <- пусто: нет .txt для этого домена"
            small.append(s)
        elif targets[s] < need:
            mark = f"  <- мало (меньше {need:,}): данные будут повторены"
            small.append(s)
        print(f"  shard_{s:02d}: {targets[s]:,} токенов ({mb:.1f} MB){mark}")

    if small:
        print("\n[!] Не все порции готовы к обучению. Домены по порциям:")
        for domain, (lo, hi) in DOMAIN_SHARDS.items():
            print(f"    {domain}: порции {lo}..{hi}")
        print("    Добавь .txt в нужную папку data/raw/<домен>/ и запусти шаг заново.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
