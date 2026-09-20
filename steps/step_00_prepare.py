"""Шаг 00: токенизация сырых .txt из data/raw в 10 порций data/shards/shard_XX.bin.

Запуск:  python3 steps/step_00_prepare.py
Токенизатор качается один раз (кэш HF). Порции пишутся как uint16 (или uint32),
по ним потом идут training-шаги. Это CPU-задача — GPU не обязателен.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import get_cfg
from src.data_utils import list_text_files, token_id_dtype


def main() -> int:
    cfg = get_cfg()
    text_files = list_text_files(cfg)
    if not text_files:
        print(f"[!] Нет .txt в {cfg.p(cfg.text_dir)} — положи туда русский корпус.")
        return 1

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    out_dir = cfg.p(cfg.tokenized_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Сбрасываем старые порции, чтобы не дублировать данные при повторных прогонах
    for i in range(cfg.shards):
        p = os.path.join(out_dir, f"shard_{i + 1:02d}.bin")
        if os.path.exists(p):
            os.remove(p)

    npdtype = token_id_dtype(cfg)
    eos = tok.eos_token_id or 0
    buffers = [[] for _ in range(cfg.shards)]
    counts = [0] * cfg.shards
    # чередование: чтобы разные домены (текст/код/математика) попадали во все порции
    round_robin = 0

    def flush(i: int):
        if buffers[i]:
            arr = np.asarray(buffers[i], dtype=npdtype)
            with open(os.path.join(out_dir, f"shard_{i + 1:02d}.bin"), "ab") as fw:
                arr.tofile(fw)
            counts[i] += len(buffers[i])
            buffers[i].clear()

    for path in text_files:
        print(f"[read] {path}")
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ids = tok(line)["input_ids"] + [eos]
                i = round_robin % cfg.shards
                round_robin += 1
                buffers[i].extend(ids)
                if len(buffers[i]) > 500_000:
                    flush(i)

    for i in range(cfg.shards):
        flush(i)

    print("\n[токенов в порции]")
    for i in range(cfg.shards):
        mb = counts[i] * np.dtype(npdtype).itemsize / 1e6
        print(f"  shard_{i + 1:02d}: {counts[i]:,} токенов ({mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
