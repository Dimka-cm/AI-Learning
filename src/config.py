"""Загрузка конфига train.yaml и выбор устройства (GPU 1 > любой GPU > CPU)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

from .threads import default_threads, fix_thread_env

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def path(*parts: str) -> str:
    """Абсолютный путь от корня репозитория."""
    return os.path.join(ROOT, *parts)


@dataclass
class CFG:
    model_name: str = "ai-forever/rugpt3small_based_on_gpt2"
    epochs_per_step: int = 3
    batch_size: int = 4
    grad_accum: int = 8
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: int = 0
    checkpoint_every_steps: int = 500
    log_every_steps: int = 20
    fp16: bool = True
    gradient_checkpointing: bool = True
    num_workers: int = 5
    seed: int = 42
    block_size: int = 1024
    dtype: str = "uint16"
    shards: int = 10
    text_dir: str = "data/raw"
    tokenized_dir: str = "data/shards"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, p: str) -> "CFG":
        with open(p, "r", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        model = y.get("model", {})
        tr = y.get("training", {})
        data = y.get("data", {})
        paths = y.get("paths", {})
        return cls(
            model_name=model.get("name", cls.model_name),
            epochs_per_step=int(tr.get("epochs_per_step", 3)),
            batch_size=int(tr.get("batch_size", 4)),
            grad_accum=int(tr.get("grad_accum", 8)),
            lr=float(tr.get("lr", 3e-4)),
            weight_decay=float(tr.get("weight_decay", 0.01)),
            warmup_steps=int(tr.get("warmup_steps", 200)),
            max_steps=int(tr.get("max_steps", 0)),
            checkpoint_every_steps=int(tr.get("checkpoint_every_steps", 500)),
            log_every_steps=int(tr.get("log_every_steps", 20)),
            fp16=bool(tr.get("fp16", True)),
            gradient_checkpointing=bool(tr.get("gradient_checkpointing", True)),
            num_workers=int(tr.get("num_workers", 4)),
            seed=int(tr.get("seed", 42)),
            block_size=int(data.get("block_size", 1024)),
            dtype=str(data.get("dtype", "uint16")),
            shards=int(data.get("shards", 10)),
            text_dir=str(data.get("text_dir", "data/raw")),
            tokenized_dir=str(data.get("tokenized_dir", "data/shards")),
            checkpoints_dir=str(paths.get("checkpoints", "checkpoints")),
            logs_dir=str(paths.get("logs", "logs")),
            extra=y.get("extra", {}),
        )

    def apply_env_overrides(self) -> "CFG":
        """Переопределение из переменных окружения (использует GitHub Actions)."""
        fix_thread_env()  # до torch/numpy: OMP_NUM_THREADS обязан быть числом
        overrides = {
            "MAX_STEPS": "max_steps",
            "EPOCHS_PER_STEP": "epochs_per_step",
            "BATCH_SIZE": "batch_size",
            "BLOCK_SIZE": "block_size",
            "MODEL_NAME": "model_name",
            "FP16": "fp16",
            "LOG_EVERY_STEPS": "log_every_steps",
            "NUM_WORKERS": "num_workers",
        }
        for env, attr in overrides.items():
            v = os.environ.get(env)
            if v in (None, ""):
                continue
            v = v.strip()
            if attr == "model_name":
                setattr(self, attr, v)
            elif attr == "fp16":
                setattr(self, attr, v.lower() in ("1", "true", "yes"))
            elif attr == "num_workers":
                # "auto" (так ставят workflows) — это не число: берём ядра минус одно.
                if v.lower() in ("auto", "none", "default"):
                    setattr(self, attr, default_threads())
                else:
                    try:
                        setattr(self, attr, max(1, int(v)))
                    except ValueError:
                        pass
            else:
                try:
                    setattr(self, attr, int(v))
                except ValueError:
                    pass
        return self

    def p(self, *parts: str) -> str:
        return os.path.join(ROOT, *parts)


def get_cfg() -> CFG:
    return CFG.load(os.path.join(ROOT, "config", "train.yaml")).apply_env_overrides()


def get_cfg_ci() -> CFG:
    """Конфиг для GitHub Actions (CPU-раннер): компактный прогон дообучения."""
    return CFG.load(os.path.join(ROOT, "config", "train_ci.yaml")).apply_env_overrides()


def choose_device(preference: str = "auto") -> str:
    """Возвращает "cuda" / "cpu".

    Вся логика выбора (включая путаницу Windows-нумерации GPU на Gigabyte G5 KD)
    живёт в src/gpu.py — здесь просто прокси, чтобы интерфейс не менялся.
    """
    from . import gpu  # локальный импорт: не тянем subprocess при старте
    return gpu.choose_device(preference)
