#!/usr/bin/env python3
"""Оркестратор пайплайна: прогоняет подготовку, диапазон шагов и оценку.

Работает одинаково локально (Windows/WSL/Linux) и в GitHub Actions.
Примеры:
  python orchestrator.py                        # шаги 1..10, GPU 1
  python orchestrator.py --start 3 --end 5      # только шаги 3..5
  python orchestrator.py --epochs 5 --device cpu  # демо на CPU
  python orchestrator.py --prepare              # заново токенизировать
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--end", type=int, default=10)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--gpu", type=str, default="auto", help="какую NVIDIA взять: auto | индекс-CUDA | подстрока имени (напр. 3060)")
    p.add_argument("--prepare", action="store_true", help="запустить токенизацию перед шагами")
    p.add_argument("--evaluate", action="store_true", help="сводный отчёт в конце")
    return p.parse_args()


def run(cmd: list[str]) -> int:
    print("\n$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def step_skip_reason(step: int) -> str | None:
    """Почему шаг пропускается, или None, если данные на месте."""
    try:
        from src.config import get_cfg, get_cfg_ci
        from src.data_utils import shard_token_count

        cfg = get_cfg_ci() if os.environ.get("CI") == "1" else get_cfg()
        shard = f"data/shards/shard_{step:02d}.bin"
        if not os.path.exists(os.path.join(ROOT, shard)):
            return f"нет {shard} — для этой порции не нашлось .txt"
        n = shard_token_count(cfg, step)
        if n == 0:
            return f"{shard} пустой"
        if n < 8:
            return f"в {shard} всего {n} токенов — учить нечего"
        return None
    except Exception as exc:  # проверка не должна ломать сам прогон
        print(f"[warn] не смог проверить данные шага {step:02d}: {exc}")
        return None


def main() -> int:
    args = parse_args()

    if args.device == "cpu":
        print("[gpu] режим CPU — CUDA не используется")

    # OMP_NUM_THREADS: на ноуте под загрузку данных отдаём 5 потоков, но если
    # переменная уже задана мусором ("auto" из workflow) — чиним на число.
    from src.threads import fix_thread_env
    os.environ.setdefault("OMP_NUM_THREADS", "5")
    fix_thread_env()

    if args.prepare:
        rc = run([sys.executable, "steps/step_00_prepare.py"])
        if rc:
            return rc

    total = 0
    skipped: list[tuple[int, str]] = []
    for step in range(args.start, args.end + 1):
        # Нет порции — не падаем, а пропускаем шаг с понятной причиной:
        # у владельца сначала будут только книги (порции 1..4), и прогон
        # «1..10» не должен умирать на математике или коде.
        reason = step_skip_reason(step)
        if reason:
            print(f"[skip] шаг {step:02d}: {reason}", flush=True)
            skipped.append((step, reason))
            continue
        cmd = [sys.executable, f"steps/step_{step:02d}.py", "--device", args.device,
               "--gpu", args.gpu]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        rc = run(cmd)
        if rc:
            print(f"[!] шаг {step:02d} упал с кодом {rc}")
            return rc
        total += 1

    if args.evaluate and total > 0:
        rc = run([sys.executable, "steps/step_11_evaluate.py"])
        if rc:
            return rc

    if skipped:
        print(f"\n[ok] обучено шагов: {total}, пропущено: {len(skipped)} (нет данных)")
        for step, reason in skipped:
            print(f"     шаг {step:02d}: {reason}")
        print("     Чтобы включить эти шаги — положи .txt в нужную папку домена "
              "и запусти: python steps/step_00_prepare.py")
        return 0

    print(f"[ok] пройдено шагов: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
