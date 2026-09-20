"""Модель зрения для интерфейса UE5.

Общий свёрточный энкодер и три головы:
  1. координаты панелей — где вьюпорт, Outliner, Details, Content Browser;
  2. состояние редактора — режим (edit/play/simulate/paused) и есть ли
     выделенный объект;
  3. иконки — какие пиктограммы Starship присутствуют на панели.

Энкодер тот же, что обкатан в Mine-AI: там он на 18 000 кадров дал 91%
по стороне света. Менять проверенную часть без причины смысла нет.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .render import MODES, PANELS
from . import icons

# Панели, координаты которых предсказываем (четыре числа на каждую).
LOCATE = ["viewport", "outliner", "details", "content", "toolbar", "statusbar"]
N_BOX = len(LOCATE) * 4
N_MODE = len(MODES)
N_ICON = len(icons.TOOLBAR)


class UIVision(nn.Module):
    """Глаза для интерфейса: кадр -> панели, состояние, иконки."""

    def __init__(self, height: int = 360, width: int = 640):
        super().__init__()
        self.h, self.w = height, width
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, 4), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1), nn.SiLU(),
            nn.Conv2d(64, 64, 3, 2), nn.SiLU(),
            nn.Conv2d(64, 32, 1), nn.SiLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n = self.conv(torch.zeros(1, 3, height, width)).shape[1]
        self.flat = n
        self.trunk = nn.Sequential(nn.Linear(n, 512), nn.SiLU())
        # Координаты в долях кадра, поэтому сигмоида на выходе.
        self.head_box = nn.Sequential(nn.Linear(512, 256), nn.SiLU(),
                                      nn.Linear(256, N_BOX), nn.Sigmoid())
        self.head_mode = nn.Linear(512, N_MODE)
        self.head_sel = nn.Linear(512, 1)
        self.head_icon = nn.Linear(512, N_ICON)

    def forward(self, x):
        z = self.trunk(self.conv(x))
        return (self.head_box(z), self.head_mode(z),
                self.head_sel(z).squeeze(-1), self.head_icon(z))


def load_into(model: UIVision, path: str, verbose: bool = True) -> int:
    """Переносит обученные свёртки в другую модель (для дообучения)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["conv"] if "conv" in ck else ck
    model.conv.load_state_dict(sd)
    if verbose:
        print(f"перенесено тензоров: {len(sd)}")
    return len(sd)
