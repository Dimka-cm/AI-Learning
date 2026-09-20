"""Тренер: цикл обучения, чекпоинты, JSONL-журнал шагов (прогресс)."""
from __future__ import annotations

import json
import math
import os
import time

import torch
from tqdm import tqdm

from .config import CFG


class Trainer:
    def __init__(self, cfg: CFG, model, tokenizer, device: str, step: int, total_steps: int):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.step = step
        self.total_steps = total_steps
        self.log_path = cfg.p(cfg.logs_dir, f"step_{step:02d}.jsonl")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.start_ts = time.time()

    def log(self, record: dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def fit(self, train_batches, steps_per_epoch: int, epochs: int):
        """train_batches — генератор, отдающий dict(input_ids=, labels=) на device."""
        cfg = self.cfg
        model = self.model
        device = self.device

        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        total_epoch_steps = steps_per_epoch * epochs
        warmup = min(cfg.warmup_steps, max(1, total_epoch_steps // 10))

        def lr_at(s):
            if s < warmup:
                return cfg.lr * (s + 1) / max(1, warmup)
            prog = (s - warmup) / max(1, total_epoch_steps - warmup)
            return cfg.lr * 0.5 * (1 + math.cos(math.pi * prog))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
        scaler = None
        if cfg.fp16 and device == "cuda":
            scaler = torch.amp.GradScaler("cuda")

        max_steps = cfg.max_steps if cfg.max_steps and cfg.max_steps > 0 else total_epoch_steps

        global_step = 0
        epoch = 0
        running = 0.0
        tokens_cum = 0

        self.log({"event": "start", "step": self.step, "epochs": epochs,
                  "steps_per_epoch": steps_per_epoch, "max_steps": max_steps,
                  "device": device})

        bar = tqdm(total=max_steps, desc=f"step {self.step:02d}", unit="it")
        it = train_batches()
        for epoch in range(epochs):
            for _ in range(steps_per_epoch):
                if global_step >= max_steps:
                    break
                batch = next(it)
                opt.zero_grad(set_to_none=True)
                x = batch["input_ids"]
                y = batch["labels"]
                t0 = time.time()

                if scaler is not None:
                    with torch.autocast("cuda", dtype=torch.float16):
                        out = model(input_ids=x, labels=y)
                        loss = out.loss / cfg.grad_accum
                    scaler.scale(loss).backward()
                else:
                    out = model(input_ids=x, labels=y)
                    loss = out.loss / cfg.grad_accum
                    loss.backward()

                if (global_step + 1) % cfg.grad_accum == 0:
                    if scaler is not None:
                        scaler.step(opt)
                        scaler.update()
                    else:
                        opt.step()
                    sched.step()

                dt = time.time() - t0
                running += loss.detach().float().item() * cfg.grad_accum
                tokens_cum += x.numel()

                global_step += 1
                bar.update(1)

                if global_step % cfg.log_every_steps == 0:
                    avg = running / cfg.log_every_steps
                    running = 0.0
                    elapsed = time.time() - self.start_ts
                    self.log({
                        "event": "train", "global_step": global_step,
                        "epoch": epoch + 1, "loss": round(avg, 4),
                        "lr": round(float(sched.get_last_lr()[0]), 7),
                        "tokens_per_sec": round(tokens_cum / max(elapsed, 1e-6), 1),
                        "elapsed_sec": round(elapsed, 1),
                    })
                    tqdm.write(
                        f"[step {self.step:02d}] epoch {epoch+1}/{epochs} "
                        f"it {global_step}/{max_steps} loss={avg:.4f} "
                        f"{tokens_cum / max(elapsed, 1e-6):.0f} tok/s"
                    )

                if global_step % cfg.checkpoint_every_steps == 0:
                    self.save(global_step, opt)

            self.save(0, opt)  # 'last.pt' в конце каждой эпохи
            if global_step >= max_steps:
                break

        bar.close()
        elapsed = time.time() - self.start_ts
        res = {
            "event": "done", "global_step": global_step,
            "elapsed_sec": round(elapsed, 1),
            "avg_tokens_per_sec": round(tokens_cum / max(elapsed, 1e-6), 1),
        }
        self.log(res)
        return global_step

    def save(self, global_step: int, opt):
        cfg = self.cfg
        out = cfg.p(cfg.checkpoints_dir, f"step_{self.step:02d}")
        os.makedirs(out, exist_ok=True)
        fname = os.path.join(out, f"last-{global_step}.pt" if global_step else "last.pt")
        ckpt = {
            "model": self.model.state_dict(),
            "opt": opt.state_dict(),
            "tokenizer": self.tokenizer.name_or_path,
            "step": self.step,
            "global_step_in_step": global_step,
        }
        torch.save(ckpt, fname)
        tqdm.write(f"[ckpt] {fname}")
