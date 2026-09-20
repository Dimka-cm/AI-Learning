"""Палитра интерфейса UE5 — подлинные цвета Epic.

Взяты из Engine/Source/Runtime/SlateCore/Private/Styling/StyleColors.cpp,
строки вида SetDefaultColor(EStyleColor::Panel, COLOR("#242424FF")).
Подбирать их на глаз нельзя: агент должен видеть тот же редактор, что и
человек, иначе он выучит несуществующий интерфейс.
"""
from __future__ import annotations

import json
from pathlib import Path

_PATH = Path(__file__).resolve().parent.parent / "assets" / "ue5_theme.json"

# Запасная палитра на случай, если ассеты ещё не скачаны: те же значения
# из StyleColors.cpp, чтобы модуль импортировался без сети.
_FALLBACK = {
    "Background": [21, 21, 21], "Title": [21, 21, 21], "Panel": [36, 36, 36],
    "Header": [47, 47, 47], "Input": [15, 15, 15], "InputOutline": [56, 56, 56],
    "Recessed": [26, 26, 26], "Dropdown": [56, 56, 56], "Hover": [87, 87, 87],
    "Foreground": [192, 192, 192], "ForegroundHover": [255, 255, 255],
    "Primary": [0, 112, 224], "PrimaryHover": [14, 134, 255],
    "PrimaryPress": [0, 80, 160], "Select": [70, 75, 80],
    "SelectHover": [66, 66, 66], "White": [255, 255, 255],
    "Black": [0, 0, 0], "Highlight": [0, 112, 224],
    "AccentBlue": [38, 187, 255], "AccentGreen": [139, 194, 74],
    "AccentOrange": [254, 155, 7], "AccentRed": [255, 64, 64],
    "AccentYellow": [255, 220, 26], "AccentPurple": [161, 57, 191],
    "Error": [239, 53, 53], "Warning": [255, 184, 0], "Success": [31, 228, 75],
}


def load() -> dict[str, list[int]]:
    """Цвета темы. Из файла, если он есть, иначе встроенная копия."""
    if _PATH.exists():
        pal = json.loads(_PATH.read_text())
        if pal:
            return {**_FALLBACK, **pal}
    return dict(_FALLBACK)


COLORS = load()


def rgb(name: str, fallback: str = "Panel") -> tuple[int, int, int]:
    """Цвет по имени из темы UE5."""
    c = COLORS.get(name) or COLORS.get(fallback) or [36, 36, 36]
    return (int(c[0]), int(c[1]), int(c[2]))
