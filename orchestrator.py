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
    print("\\n$ " + " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    args = parse_args()

    if args.device == "cpu":
        print("[gpu] режим CPU — CUDA не используется")

    # меньше всего CPU — под загрузку данных (на твоём ноуте свободно 5)
    os.environ.setdefault("OMP_NUM_THREADS", "5")

    if args.prepare:
        rc = run([sys.executable, "steps/step_00_prepare.py"])
        if rc:
            return rc

    total = 0
    for step in range(args.start, args.end + 1):
        cmd = [sys.executable, f"steps/step_{step:02d}.py", "--device", args.device,
               "--gpu", args.gpu]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        rc = run(cmd)
        if rc:
            print(f"[!] шаг {step:02d} упал с кодом {rc}")
            return rc
        total += 1

    if args.evaluate:
        return run([sys.executable, "steps/step_11_evaluate.py"])

    print(f"[ok] пройдено шагов: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
