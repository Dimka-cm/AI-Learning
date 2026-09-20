#!/usr/bin/env python3
"""
Качает НАСТОЯЩИЕ иконки интерфейса Unreal Engine 5 (набор Starship).

Почему не из установленного движка: UE5 весит десятки гигабайт, а в CI
и в песочнице его нет. Официальный EpicGames/UnrealEngine закрыт — нужна
привязка аккаунта GitHub к Epic Games. Поэтому берём из публичного зеркала,
где лежит нетронутая папка Engine/Content/Slate.

Ассеты Epic НЕ кладутся в репозиторий: их скачивает эта команда.

    python3 tools/fetch_ue5_assets.py
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import urllib.request
from pathlib import Path

REPO = "cgt507/Zombie_Release"
RAW = f"https://raw.githubusercontent.com/{REPO}/HEAD/"
TREE = f"https://api.github.com/repos/{REPO}/git/trees/HEAD?recursive=1"
SLATE = "Engine/Content/Slate/"


def grab(item: tuple[str, Path]) -> bool:
    url, dst = item
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url, timeout=60) as r:
            dst.write_bytes(r.read())
        return True
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Скачать ассеты интерфейса UE5")
    ap.add_argument("--out", default="assets/ue5", help="куда складывать")
    ap.add_argument("--token", default="", help="токен GitHub (для лимита API)")
    args = ap.parse_args()

    out = Path(args.out)
    req = urllib.request.Request(TREE)
    if args.token:
        req.add_header("Authorization", f"Bearer {args.token}")
    print(f"читаю дерево {REPO} ...")
    tree = json.load(urllib.request.urlopen(req, timeout=90))["tree"]

    want = [t["path"] for t in tree
            if t["path"].startswith(SLATE)
            and t["path"].lower().endswith((".svg", ".png"))]
    print(f"найдено файлов интерфейса: {len(want)}")

    jobs = [(RAW + p, out / p[len(SLATE):]) for p in want]
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for good in ex.map(grab, jobs):
            ok += bool(good)
    print(f"скачано: {ok} из {len(jobs)} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
