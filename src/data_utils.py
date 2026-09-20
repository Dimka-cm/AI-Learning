"""Утилиты: чтение порции данных, поиск чекпоинта для резюме, размер модели."""
from __future__ import annotations

import os
import re
from typing import List, Optional

import numpy as np

from .config import CFG


def token_id_dtype(cfg: CFG):
    return np.uint32 if cfg.dtype == "uint32" else np.uint16


def shard_path(cfg: CFG, step: int) -> str:
    return cfg.p(cfg.tokenized_dir, f"shard_{step:02d}.bin")


def shard_token_count(cfg: CFG, step: int) -> int:
    """Сколько токенов в порции. 0 — файла нет или он пустой."""
    p = shard_path(cfg, step)
    if not os.path.exists(p):
        return 0
    return int(os.path.getsize(p) // np.dtype(token_id_dtype(cfg)).itemsize)


def load_shard(cfg: CFG, step: int) -> np.ndarray:
    """Читает токенизированную порцию шага (data/shards/shard_XX.bin) в память.

    Если порция меньше, чем нужно для одного окна обучения (block_size + 2),
    данные повторяются до нужного размера, с предупреждением в лог. Так демо-корпус
    и маленькие домены не валят прогон: «Слишком мало токенов: 232 < 258» больше
    не роняет шаг — модель просто несколько раз увидит один и тот же кусок.
    """
    p = shard_path(cfg, step)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Нет {p}. Сначала запусти steps/step_00_prepare.py "
            f"(положи .txt в {cfg.text_dir}/)"
        )
    arr = np.fromfile(p, dtype=token_id_dtype(cfg))
    if arr.ndim != 1:
        arr = arr.reshape(-1)

    need = cfg.block_size + 2
    if arr.size == 0:
        raise ValueError(
            f"Порция shard_{step:02d} пустая: в ней нет ни одного токена.\n"
            f"  Причина: для этого домена не нашлось .txt в {cfg.text_dir}/.\n"
            f"  Что делать: положи текст в нужную папку домена и запусти "
            f"python steps/step_00_prepare.py заново."
        )
    if arr.size < need:
        reps = int(np.ceil(need / arr.size))
        print(f"[data] в shard_{step:02d} всего {arr.size} токенов — меньше окна "
              f"({need}). Повторяю данные x{reps}, чтобы обучение прошло. "
              f"Для качества добавь текста (см. docs/DATA.md).")
        arr = np.tile(arr, reps)
    return arr


def find_resume_checkpoint(cfg: CFG, step: int) -> Optional[str]:
    """Последний чекпоинт: сначала в папке текущего шага, затем предыдущего и т.д."""
    for s in range(step, 0, -1):
        d = cfg.p(cfg.checkpoints_dir, f"step_{s:02d}")
        if os.path.isdir(d):
            candidates = []
            for f in os.listdir(d):
                m = re.match(r"last(?:-(\d+))?\.pt$", f)
                if m:
                    candidates.append((int(m.group(1) or 0), f))
            if candidates:
                candidates.sort()
                return os.path.join(d, candidates[-1][1])
    return None


def base_model_config(cfg: CFG):
    return {"local_files_only": bool(cfg.extra.get("local_files_only", False))}


def param_count(model) -> str:
    n = sum(p.numel() for p in model.parameters())
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    return f"{n/1e6:.1f}M"


def list_text_files(cfg: CFG) -> List[str]:
    d = cfg.p(cfg.text_dir)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".txt"))
