"""Приведение переменных окружения с потоками к нормальному виду.

История ошибки: в workflow стояло `OMP_NUM_THREADS: "auto"`, и libgomp падал с
«Недопустимое значение переменной окружения OMP_NUM_THREADS». Значение `auto`
понимает наш конфиг, но не OpenMP/BLAS — им нужно целое число.

Вызывать ДО импорта torch/numpy: эти библиотеки читают переменные при загрузке.
"""
from __future__ import annotations

import os

# Библиотеки, которые читают число потоков из окружения.
THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def default_threads() -> int:
    """Сколько потоков дать: все ядра минус одно (одно оставляем системе)."""
    return max(1, (os.cpu_count() or 2) - 1)


def _as_int(value: str) -> int | None:
    """'4' -> 4, 'auto'/'' -> None (значит «реши сам»), 'abc' -> None."""
    v = (value or "").strip()
    if not v or v.lower() in ("auto", "none", "default"):
        return None
    try:
        n = int(float(v))
    except ValueError:
        return None
    return n if n > 0 else None


def fix_thread_env(verbose: bool = False) -> None:
    """Чинит потоки в окружении на месте.

    * `auto` / пусто / мусор -> число ядер минус одно;
    * 0 или отрицательное -> то же самое (OpenMP такого не любит);
    * корректное число -> не трогаем.
    """
    fallback = default_threads()
    for name in THREAD_VARS:
        if name not in os.environ:
            continue  # переменной нет — пусть каждая библиотека решает сама
        raw = os.environ.get(name, "")
        n = _as_int(raw)
        if n is None:
            os.environ[name] = str(fallback)
            if verbose:
                print(f"[env] {name}={raw!r} -> {fallback}")
