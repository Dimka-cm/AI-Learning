#!/usr/bin/env python3
"""Подготовка данных ДЛЯ CI: без HF-сети и без torchvision.

Скачивает корпус (если задан DRIVE/URL/датасет) или генерирует демо-корпус,
затем токенизирует. Токенизатор качаем с HF — раннер имеет интернет.
Порции кладём так же, как пайплайн: text->1..4, math->5..6, code->7..8, behavior->9..10.

Запускается только внутри CI (env CI=1); локально — обычный step_00_prepare.py.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def gen_demo():
    from scripts.make_data import main as m
    m()


def main() -> int:
    os.environ["CI"] = "1"
    os.environ["HF_HOME"] = os.environ.get("HF_HOME", os.path.join(ROOT, "hf-cache"))

    # --- 1. Получить сырые тексты ---
    drive_url = os.environ.get("DRIVE_FOLDER_URL", "").strip()
    hf_ds = os.environ.get("HF_DATASET", "").strip()
    out_text = os.path.join(ROOT, "data", "raw", "text")
    os.makedirs(out_text, exist_ok=True)

    if drive_url:
        print(f"[data] скачиваю папку с Drive: {drive_url}")
        rc = os.system(f"gdown --folder \"{drive_url}\" -O {out_text}/")
        if rc != 0:
            print("[warn] gdown не смог; пробуем curl-driven обход…")
            rc = os.system(f"gdown \"{drive_url.replace('/folders/', '/uc?id=')}\" -O {out_text}/")
    elif hf_ds:
        # стримингом из HF-датасета (напр. Imperius/ru-classic)
        print(f"[data] стримлю датасет: {hf_ds}")
        rc = os.system(
            f"python scripts/download_ru_classic.py --source datasets --limit-mb {os.environ.get('LIMIT_MB', '25')}"
        )

    # если ничего не скачалось — генерим демо (офлайн, надёжно)
    has_txt = any(
        os.path.exists(os.path.join(out_text, f)) and f.endswith(".txt")
        for f in os.listdir(out_text)
    ) if os.path.isdir(out_text) else False
    if not has_txt:
        print("[data] сырых .txt нет/не скачались — генерирую демо-корпус")
        gen_demo()

    # --- 2. Токенизация в порции ---
    print("[data] токенизация…")
    from steps.step_00_prepare import main as prep_main
    return prep_main()


if __name__ == "__main__":
    raise SystemExit(main())
