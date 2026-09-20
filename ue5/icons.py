"""Загрузка настоящих иконок интерфейса UE5 (набор Starship).

Иконки — это SVG из Engine/Content/Slate/Starship. Растеризуем их через
cairosvg и держим в памяти как RGBA-массивы. Рисовать «похожие» значки
самому нельзя: агент должен узнавать те самые пиктограммы, которые видит
пользователь в редакторе.
"""
from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "assets" / "ue5"

# Иконки, которые реально видны в главном окне редактора. Порядок задаёт
# номер класса для головы классификации, поэтому его нельзя менять.
TOOLBAR = [
    # Имена сверены с файлами в Engine/Content/Slate/Starship/Common,
    # а не написаны по памяти: половина «очевидных» путей там не существует.
    "Starship/Common/save.svg",
    "Starship/Common/play.svg",
    "Starship/Common/stop.svg",
    "Starship/Common/settings.svg",
    "Starship/Common/blueprint.svg",
    "Starship/Common/search.svg",
    "Starship/Common/filter.svg",
    "Starship/Common/folder-closed.svg",
    "Starship/Common/folder-open.svg",
    "Starship/Common/file.svg",
    "Starship/Common/Copy.svg",
    "Starship/Common/Paste.svg",
    "Starship/Common/Cut.svg",
    "Starship/Common/Delete.svg",
    "Starship/Common/Undo.svg",
    "Starship/Common/Redo.svg",
    "Starship/Common/Console.svg",
    "Starship/Common/OutputLog.svg",
    "Starship/Common/visible.svg",
    "Starship/Common/hidden.svg",
    "Starship/Common/lock.svg",
    "Starship/Common/lock-unlocked.svg",
    "Starship/Common/plus.svg",
    "Starship/Common/minus.svg",
    "Starship/Common/close.svg",
    "Starship/Common/check.svg",
    "Starship/Common/edit.svg",
    "Starship/Common/import.svg",
    "Starship/Common/export.svg",
    "Starship/Common/world.svg",
    "Starship/Common/sphere.svg",
    "Starship/Common/cylinder.svg",
    "Starship/Common/box-perspective.svg",
    "Starship/Common/UELogo.svg",
    "Starship/Common/help.svg",
    "Starship/Common/Info.svg",
    "Starship/Common/alert-triangle.svg",
    "Starship/Common/chevron-down.svg",
    "Starship/Common/chevron-right.svg",
    "Starship/Common/caret-down.svg",
]


def available() -> bool:
    """Скачаны ли ассеты Epic."""
    return ROOT.exists() and any(ROOT.rglob("*.svg"))


@lru_cache(maxsize=512)
def load_svg(rel: str, size: int = 20) -> np.ndarray | None:
    """Растеризует SVG в RGBA (size, size, 4). None, если файла нет."""
    path = ROOT / rel
    if not path.exists():
        return None
    try:
        import cairosvg
    except ImportError:
        return None
    try:
        png = cairosvg.svg2png(url=str(path), output_width=size,
                               output_height=size)
    except Exception:
        return None
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def catalog(size: int = 20) -> list[tuple[str, np.ndarray]]:
    """Все иконки панели инструментов, что удалось прочитать."""
    out = []
    for rel in TOOLBAR:
        img = load_svg(rel, size)
        if img is not None:
            out.append((Path(rel).stem, img))
    return out


def any_icons(limit: int = 64, size: int = 20) -> list[tuple[str, np.ndarray]]:
    """Запасной набор: первые найденные SVG, если списка TOOLBAR не хватило."""
    out = []
    for p in sorted(ROOT.rglob("*.svg")):
        img = load_svg(str(p.relative_to(ROOT)), size)
        if img is not None:
            out.append((p.stem, img))
        if len(out) >= limit:
            break
    return out
