"""Выбор GPU с защитой от путаницы в нумерации (Windows vs CUDA).

Ключевой момент на ноуте Gigabyte G5 KD:
  - Windows «Диспетчер задач»:  GPU 0 = Intel UHD (iGPU), GPU 1 = RTX 3060.
  - CUDA / nvidia-smi / PyTorch видят ТОЛЬКО NVIDIA-карты: Intel UHD там нет,
    поэтому RTX 3060 имеет CUDA-индекс 0 (а не 1).

Поэтому «не трогать GPU 0, учить на GPU 1» переводится в CUDA-индекс 0.
Если пользователь всё же передаст «1» (по привычке из Windows), мы это
поймём, поправим на реальную дискретную карту и громко об этом сообщим.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

_INTEL_MARKERS = ("INTEL", "UHD", "IRIS")


def _is_nvidia(name: str) -> bool:
    n = name.upper()
    if any(m in n for m in _INTEL_MARKERS):
        return False
    return any(m in n for m in ("NVIDIA", "RTX", "GEFORCE", "TESLA", "QUADRO", "TITAN"))


def nvidia_gpus() -> list[dict]:
    """Список ДИСКРЕТНЫХ NVIDIA-карт из nvidia-smi (то, что реально видит CUDA)."""
    if shutil.which("nvidia-smi") is None:
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.free,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=15,
        )
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        m = re.match(r"\s*(\d+)\s*,\s*(.*?)\s*,\s*(\d+)\s*,\s*(\d+)", line)
        if m:
            gpus.append({
                "index": int(m.group(1)),
                "name": m.group(2).strip(),
                "free_mb": int(m.group(3)),
                "total_mb": int(m.group(4)),
            })
    return gpus


def pick_device_index(pref: str, gpus: list[dict] | None = None):
    """Возвращает dict-запись выбранной карты или None.

    pref:
      "auto" / "cuda" / "" — самая свободная дискретная NVIDIA;
      "0", "1", ...        — CUDA-индекс; если его нет, а пользователь,
                             похоже, называл «GPU 1» из Windows — берём
                             дискретную карту и предупреждаем;
      "3060", "NVIDIA" ... — по подстроке имени.
    """
    if gpus is None:
        gpus = nvidia_gpus()
    if not gpus:
        return None

    def by_free():
        return max(gpus, key=lambda g: g["free_mb"])

    if pref in ("auto", "cuda", "", None):
        return by_free()

    def by_name():
        lo = pref.lower()
        return next((g for g in gpus if lo in g["name"].lower()), None)

    if pref.isdigit():
        want = int(pref)
        for g in gpus:
            if g["index"] == want:
                return g
        # числа вроде "3060" — это модель карты, а не CUDA-индекс
        named = by_name()
        if named is not None:
            print(f"[gpu] ℹ \"{pref}\" совпало с именем '{named['name']}' — беру её.")
            return named
        print(f"[gpu] ⚠ запрошен CUDA-индекс {want}, но в CUDA только: "
              f"{[(g['index'], g['name']) for g in gpus]}.")
        print("[gpu] ℹ Похоже, индекс взят из Windows (GPU 0 = Intel iGPU, "
              f"GPU {want} = RTX 3060). Беру дискретную NVIDIA-карту.")
        return by_free()

    named = by_name()
    if named is not None:
        return named
    return None


def brief() -> str:
    """Короткий человекочитаемый отчёт о видимых CUDA-устройствах."""
    gpus = nvidia_gpus()
    if not gpus:
        return "CUDA-карт не видно (нет nvidia-smi или драйвера NVIDIA)"
    return "; ".join(
        f"nvml-индекс {g['index']}: {g['name']} (свободно {g['free_mb']:.0f} МБ)"
        for g in gpus
    )


def choose_device(pref: str = "auto") -> str:
    """Возвращает "cuda"/"cpu". Если можно — выставляет CUDA_VISIBLE_DEVICES
    на корректную NVIDIA-карту и печатает, что реально будет использовано.

    Вызывать ДО первого обращения к torch.cuda (до инициализации CUDA).
    """
    if pref == "cpu":
        return "cpu"

    # 1) Если окружение уже отфильтровано (orchestrator/workflow выставили
    #    CUDA_VISIBLE_DEVICES) — просто проверяем торчем, что карта есть.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
                print(f"[gpu] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} "
                      f"-> cuda:0 {torch.cuda.get_device_name(0)} "
                      f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
                return "cuda"
        except Exception:
            pass
        return "cpu"

    # 2) Нужно выбрать карту до инициализации CUDA -> смотрим nvidia-smi.
    pick = pick_device_index(pref)
    if pick is None:
        print(f"[gpu] ⚠ NVIDIA-карта не найдена ({brief()}). Идём на CPU.")
        return "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pick["index"])
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() >= 1:
            print(f"[gpu] выбрана {pick['name']} (nvml-индекс {pick['index']}) "
                  f"-> torch видит cuda:0 {torch.cuda.get_device_name(0)} "
                  f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
            return "cuda"
    except Exception:
        pass
    return "cpu"
