#!/usr/bin/env python3
"""CLI-вход агента: python -m agent.cli [опции]

Примеры:
  python -m agent.cli                        # интерактивный режим
  python -m agent.cli --say "2 + 2" --trace  # один запрос
  python -m agent.cli --say "/list"          # список файлов в воркспейсе
  python -m agent.cli --say "посчитай (17*3)%5"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import yaml

from agent.core import Agent


def load_cfg() -> dict:
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "agent_config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    p = argparse.ArgumentParser(description="ПК-агент: эксперты + инструменты + зрение + память")
    p.add_argument("--say", type=str, default=None, help="один запрос без интерактива")
    p.add_argument("--trace", action="store_true", help="показывать решения роутера")
    p.add_argument("--dry-run", action="store_true", help="инструменты только проговаривают действия")
    p.add_argument("--allow-dangerous", action="store_true",
                   help="разрешить delete/run без подтверждения")
    args = p.parse_args()

    cfg = load_cfg()
    if args.trace:
        cfg["trace"] = True
    if args.dry_run:
        cfg["tools"]["dry_run"] = True
    if args.allow_dangerous:
        cfg["tools"]["allow_dangerous"] = True

    agent = Agent(cfg)
    if args.say:
        print(agent.reply(args.say))
        return 0
    agent.run_interactive()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
