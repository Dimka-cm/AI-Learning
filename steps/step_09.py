"""Шаг NN (01..10): обучение порции shard_NN с резюме с последнего чекпоинта.

Запуск:  python3 steps/step_01.py [--epochs N] [--device auto|cuda|cpu] [--gpu auto|индекс|имя]
Выбор GPU — src/gpu.py: находит NVIDIA-карту, переживает путаницу Windows-нумерации
(в Windows «GPU 1» = RTX 3060, у CUDA она = 0; Intel UHD в CUDA не видна).
Резюме: checkpoints/step_{наибольший s<NN}/last*.pt -> checkpoints/step_NN/.
Журнал прогресса: logs/step_NN.jsonl.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.config import get_cfg, get_cfg_ci
from src.trainer import Trainer
from src.data_utils import find_resume_checkpoint, load_shard, param_count


def _cfg():
    import os
    return get_cfg_ci() if os.environ.get("CI") == "1" else get_cfg()

STEP = int(os.path.basename(__file__).split("_")[1].split(".")[0])  # 1..10


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=None, help="эпох на шаг (по умолчанию из конфига)")
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--gpu", type=str, default="auto",
                   help="какую NVIDIA взять: auto | CUDA-индекс | подстрока имени (напр. 3060)")
    return p.parse_args()


def make_batch_fn(cfg, data, device):
    """Генератор батчей через DataLoader (параллельная сборка на num_workers ядер)."""
    import torch
    from torch.utils.data import DataLoader, IterableDataset

    class ShardIter(IterableDataset):
        def __iter__(self):
            g = torch.Generator()
            B = cfg.batch_size
            n = len(data) - cfg.block_size
            while True:
                idx = torch.randint(0, n, (B,), generator=g).tolist()
                xs = torch.stack([torch.from_numpy(data[i:i + cfg.block_size].astype("int64"))
                                  for i in idx])
                ys = torch.stack([torch.from_numpy(data[i + 1:i + 1 + cfg.block_size].astype("int64"))
                                  for i in idx])
                yield {"input_ids": xs, "labels": ys}

    kwargs = {}
    if device == "cpu" and getattr(cfg, "num_workers", 0) > 0:
        kwargs = {"num_workers": cfg.num_workers, "persistent_workers": False}
    dl = DataLoader(ShardIter(), batch_size=None, **kwargs)

    def gen():
        for batch in dl:
            yield {"input_ids": batch["input_ids"].to(device),
                   "labels": batch["labels"].to(device)}

    return gen


def main() -> int:
    args = parse_args()
    cfg = _cfg()
    if args.epochs:
        cfg.epochs_per_step = args.epochs

    # Правильный выбор GPU: на Gigabyte G5 KD «GPU 1» из Windows = RTX 3060,
    # а её CUDA-индекс = 0 (Intel UHD в CUDA не видна). Модуль сам разруливает.
    from src import gpu
    device = gpu.choose_device(args.device if args.device != "cpu" else "cpu")
    if args.gpu not in ("", "auto") and device == "cuda":
        # точечный выбор конкретной карты, если пользователю не хватило auto
        device = gpu.choose_device(args.gpu)
    print(f"[step {STEP:02d}] device={device} epochs={cfg.epochs_per_step}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    data = load_shard(cfg, STEP)
    print(f"[step {STEP:02d}] shard tokens={data.size:,}")

    resume = find_resume_checkpoint(cfg, STEP)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name)
    if resume:
        print(f"[resume] {resume}")
        sd = torch.load(resume, map_location="cpu")["model"]
        model.load_state_dict(sd)
    else:
        print("[resume] нет — стартуем с базовых весов")

    model.config.use_cache = False
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.to(device)

    print(f"[model] {cfg.model_name} ~{param_count(model)} параметров")

    trainer = Trainer(cfg, model, tok, device, STEP, cfg.shards)

    block = cfg.block_size
    n = len(data) - block
    batches_per_epoch = max(1, n // (block * cfg.batch_size))
    steps_per_epoch = min(batches_per_epoch, 2000)  # ограничение длины эпохи
    print(f"[step {STEP:02d}] step/epoch={steps_per_epoch} "
          f"total={steps_per_epoch * cfg.epochs_per_step}")

    gs = trainer.fit(make_batch_fn(cfg, data, device), steps_per_epoch, cfg.epochs_per_step)
    print(f"[step {STEP:02d}] done: {gs} итераций")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
