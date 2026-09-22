#!/usr/bin/env python3
"""Генерация пробных ответов модели после дообучения.

Скрипт не ставит автоматическую оценку «хорошо/плохо»: он прогоняет выбранные
промпты из eval/model_quality_prompts.yaml и пишет markdown-отчёт. В CI это
помогает сразу увидеть: модель отвечает по-русски, не выдумывает ли грубо, и
появился ли чекпоинт.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def latest_checkpoint() -> str | None:
    candidates: list[tuple[int, str]] = []
    for p in glob.glob(str(ROOT / "checkpoints" / "step_*" / "last.pt")):
        step_name = Path(p).parent.name
        try:
            step = int(step_name.split("_")[1])
        except Exception:
            step = 0
        candidates.append((step, p))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def load_prompts(path: str, limit: int) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    prompts = data.get("prompts", [])
    if not isinstance(prompts, list):
        raise ValueError("eval-файл должен содержать список prompts")
    return prompts[:limit] if limit > 0 else prompts


def main() -> int:
    ap = argparse.ArgumentParser(description="Пробные ответы дообученной модели")
    ap.add_argument("--prompts", default=str(ROOT / "eval" / "model_quality_prompts.yaml"))
    ap.add_argument("--checkpoint", default="", help="Путь к checkpoints/step_XX/last.pt; пусто = найти последний")
    ap.add_argument("--limit", type=int, default=12, help="Сколько промптов прогнать (0 = все)")
    ap.add_argument("--out", default=str(ROOT / "logs" / "eval_answers.md"))
    ap.add_argument("--max-new-tokens", type=int, default=96)
    args = ap.parse_args()

    os.environ.setdefault("CI", os.environ.get("CI", "0"))
    from src.config import get_cfg, get_cfg_ci

    cfg = get_cfg_ci() if os.environ.get("CI") == "1" else get_cfg()
    ckpt = args.checkpoint or latest_checkpoint()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)

    loaded = "базовая модель, чекпоинт не найден"
    if ckpt and Path(ckpt).exists():
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state["model"])
        loaded = ckpt
    model.eval()

    prompts = load_prompts(args.prompts, args.limit)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("# Пробные ответы модели\n\n")
        f.write(f"- model: `{cfg.model_name}`\n")
        f.write(f"- checkpoint: `{loaded}`\n")
        f.write(f"- prompts: `{args.prompts}`\n\n")
        for item in prompts:
            prompt = str(item.get("prompt", "")).strip()
            if not prompt:
                continue
            full = f"Пользователь: {prompt}\nАгент:"
            ids = tok(full, return_tensors="pt")
            with torch.no_grad():
                gen = model.generate(
                    ids["input_ids"],
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tok.eos_token_id,
                )
            text = tok.decode(gen[0], skip_special_tokens=True)
            answer = text[len(full):].strip()
            f.write(f"## {item.get('id', 'prompt')} · {item.get('category', '-')}\n\n")
            f.write(f"**Вопрос:** {prompt}\n\n")
            f.write("**Ответ модели:**\n\n")
            f.write(answer or "(пустой ответ)")
            f.write("\n\n")

    print(f"[eval] отчёт: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
