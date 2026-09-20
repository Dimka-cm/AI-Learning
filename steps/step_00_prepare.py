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
import sys

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

    # 1) доменные папки
    for domain, (lo, hi) in DOMAIN_SHARDS.items():
        d = os.path.join(raw_dir, domain)
        if not os.path.isdir(d):
            continue
        txts = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".txt"))
        if not txts:
            continue
        for path in txts:
            files.append((path, lo, hi))
        print(f"[domain] {domain}: {len(txts)} файлов -> shards {lo}..{hi}")

    # 2) старый формат: *.txt в корне -> round-robin по всем 10
    legacy = sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir)
        if f.endswith(".txt") and os.path.isfile(os.path.join(raw_dir, f))
    )
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

    for path, lo, hi in files:
        try:
            fh = open(path, "r", encoding="utf-8", errors="replace")
        except OSError as ex:
            print(f"[warn] пропуск {path}: {ex}")
            continue
        span = hi - lo + 1
        with fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                # каждая строка идёт в свой шард диапазона домена (равномерно)
                s = lo + (i % span)
                ids = tok(line)["input_ids"] + [eos]
                buffer[s].extend(ids)
                if len(buffer[s]) > 500_000:
                    flush(s)

    for s in range(1, cfg.shards + 1):
        flush(s)

    print("\n[токенов в порции]")
    for s in range(1, cfg.shards + 1):
        mb = targets[s] * np.dtype(npdtype).itemsize / 1e6
        print(f"  shard_{s:02d}: {targets[s]:,} токенов ({mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
