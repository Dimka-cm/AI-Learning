#!/usr/bin/env python3
"""Сборка «обучающего мешка» в свободном формате: любые .txt → один data/bag.txt.

Смысл: сделать по-настоящему просто — берёшь любые .txt (книги, диалоги,
выгрузки), любые пути, складываешь в data/bag/, запускаешь — и получаешь
единый data/raw/text/bag.txt, готовый к токенизации. Один шаг, без форматов.

Запуск:  python3 scripts/make_bag.py [--src data/mycustom] 
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/bag", help="папка с любыми .txt")
    ap.add_argument("--out", default="data/raw/text/bag.txt")
    args = ap.parse_args()

    src = os.path.join(ROOT, args.src)
    out = os.path.join(ROOT, args.out)

    if not Path(src).is_dir():
        print(f"[!] Папка {src} не найдена — создай её и положи туда .txt")
        return 1

    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_files = n_bad = 0
    with open(out, "w", encoding="utf-8") as out_f:
        for p in sorted(Path(src).rglob("*.txt")):
            n_files += 1
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                n_bad += 1
                continue
            out_f.write(text)
            if not text.endswith("\n"):
                out_f.write("\n")
    print(f"[ok] {n_files} файлов -> {out} (пропущено битых: {n_bad})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
