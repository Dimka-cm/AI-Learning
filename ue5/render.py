"""Рендер главного окна Unreal Editor 5 с точной разметкой.

Кадры собираются кодом, но из подлинных материалов Epic: цвета берутся из
StyleColors.cpp, иконки — SVG из Engine/Content/Slate/Starship. Благодаря
этому разметка бесплатна и точна до пикселя: мы сами ставим каждую панель
и знаем её координаты, тогда как на живых скриншотах их пришлось бы
размечать руками.

Раскладка повторяет стандартную: меню и панель инструментов сверху,
вьюпорт в центре, Outliner справа сверху, Details под ним, Content Browser
внизу, статусная строка в самом низу.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import icons
from .theme import rgb

# Классы панелей. Номер — это метка для головы «что за элемент», порядок
# менять нельзя, иначе обученные веса перестанут совпадать с разметкой.
PANELS = [
    "menubar", "toolbar", "viewport", "outliner",
    "details", "content", "statusbar", "tabbar",
]
PANEL_ID = {n: i for i, n in enumerate(PANELS)}

# Режимы редактора, которые модель должна различать.
MODES = ["edit", "play", "simulate", "paused"]
MODE_ID = {n: i for i, n in enumerate(MODES)}


@dataclass
class Box:
    """Прямоугольник элемента с меткой."""
    x0: int
    y0: int
    x1: int
    y1: int
    kind: str
    name: str = ""

    def norm(self, w: int, h: int) -> list[float]:
        """Координаты в долях кадра — в таком виде их учит сеть."""
        return [self.x0 / w, self.y0 / h, self.x1 / w, self.y1 / h]


@dataclass
class Frame:
    """Кадр редактора и всё, что про него известно."""
    img: np.ndarray
    boxes: list[Box] = field(default_factory=list)
    mode: str = "edit"
    selected: bool = False
    icon_ids: list[int] = field(default_factory=list)

    def box(self, kind: str) -> Box | None:
        for b in self.boxes:
            if b.kind == kind:
                return b
        return None


def _fill(img, x0, y0, x1, y1, color) -> None:
    """Заливка прямоугольника с обрезкой по краям кадра."""
    h, w = img.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 > x0 and y1 > y0:
        img[y0:y1, x0:x1] = color


def _outline(img, x0, y0, x1, y1, color, t: int = 1) -> None:
    """Рамка толщиной t."""
    _fill(img, x0, y0, x1, y0 + t, color)
    _fill(img, x0, y1 - t, x1, y1, color)
    _fill(img, x0, y0, x0 + t, y1, color)
    _fill(img, x1 - t, y0, x1, y1, color)


def _blit(img, icon: np.ndarray, x: int, y: int, tint=None) -> None:
    """Накладывает RGBA-иконку с учётом прозрачности."""
    ih, iw = icon.shape[:2]
    h, w = img.shape[:2]
    if x < 0 or y < 0 or x + iw > w or y + ih > h:
        return
    a = icon[..., 3:4].astype(np.float32) / 255.0
    src = icon[..., :3].astype(np.float32)
    if tint is not None:
        # В движке белые пиктограммы перекрашиваются стилем, повторяем это.
        lum = src.mean(axis=2, keepdims=True) / 255.0
        src = lum * np.asarray(tint, np.float32)
    dst = img[y:y + ih, x:x + iw].astype(np.float32)
    img[y:y + ih, x:x + iw] = (src * a + dst * (1 - a)).astype(np.uint8)


def _text_run(img, x, y, length, color, size: int = 3) -> int:
    """Имитация строки текста: столбики разной ширины, как у мелкого шрифта.

    На кадре 640x360 надписи редактора занимают 3-5 пикселей по высоте, и
    буквы на таком масштабе неразличимы. Нам нужен не текст, а его вид:
    ритм светлых штрихов на тёмном фоне. Иконки при этом настоящие.
    """
    cx = x
    rng = np.random.default_rng(length * 7919 + x * 31 + y)
    while cx < x + length:
        w = int(rng.integers(1, 4))
        if rng.random() < 0.78:
            _fill(img, cx, y, cx + w, y + size, color)
        cx += w + 1
    return cx


def _viewport(img, x0, y0, x1, y1, rng, mode: str) -> None:
    """Содержимое 3D-вьюпорта: небо, земля, сетка, пара объектов."""
    h = y1 - y0
    horizon = y0 + int(h * rng.uniform(0.38, 0.55))
    # Небо градиентом — в редакторе это фон уровня.
    sky_top = np.array(rng.choice([[46, 52, 64], [38, 44, 56], [55, 60, 72]]),
                       np.float32)
    sky_bot = sky_top * 1.45
    for y in range(y0, horizon):
        t = (y - y0) / max(1, horizon - y0)
        _fill(img, x0, y, x1, y + 1, (sky_top * (1 - t) + sky_bot * t))
    ground = np.array(rng.choice([[34, 36, 34], [40, 38, 35], [30, 33, 36]]),
                      np.float32)
    _fill(img, x0, horizon, x1, y1, ground)

    # Сетка пола с перспективой: линии сходятся к горизонту.
    grid = tuple(int(c) for c in ground * 1.7)
    for i in range(1, 14):
        t = i / 14.0
        yy = horizon + int((y1 - horizon) * t * t)
        if yy < y1:
            _fill(img, x0, yy, x1, yy + 1, grid)
    cx = (x0 + x1) // 2
    for i in range(-7, 8):
        xx = cx + int(i * (x1 - x0) * 0.09)
        for yy in range(horizon, y1):
            t = (yy - horizon) / max(1, y1 - horizon)
            px = int(cx + (xx - cx) * (0.25 + t * 1.75))
            if x0 <= px < x1:
                img[yy, px] = grid

    # Несколько объектов сцены — кубы и сферы разного тона.
    for _ in range(int(rng.integers(1, 5))):
        bw = int(rng.integers(18, 60))
        bh = int(rng.integers(18, 55))
        bx = int(rng.integers(x0 + 5, max(x0 + 6, x1 - bw - 5)))
        by = int(rng.integers(horizon - bh, max(horizon - bh + 1, y1 - bh)))
        tone = float(rng.uniform(0.5, 1.0))
        base = np.array([120, 118, 112], np.float32) * tone
        _fill(img, bx, by, bx + bw, by + bh, base)
        _fill(img, bx, by, bx + bw, by + max(2, bh // 6), base * 1.3)
        _fill(img, bx, by + bh - max(2, bh // 6), bx + bw, by + bh, base * 0.6)

    if mode == "play":
        # В режиме игры вьюпорт обводится оранжевой рамкой.
        _outline(img, x0, y0, x1, y1, rgb("AccentOrange"), 2)


def _tree(img, x0, y0, x1, y1, rng, rows: int, sel_row: int = -1) -> None:
    """Список с отступами — Outliner и дерево папок Content Browser."""
    fg, sel = rgb("Foreground"), rgb("Select")
    y = y0 + 4
    n = 0
    while y + 12 < y1 and n < rows:
        depth = int(rng.integers(0, 3))
        if n == sel_row:
            _fill(img, x0 + 1, y - 2, x1 - 1, y + 10, sel)
        x = x0 + 6 + depth * 10
        if rng.random() < 0.35:                      # треугольник раскрытия
            _fill(img, x - 5, y + 2, x - 2, y + 5, fg)
        _fill(img, x, y + 1, x + 7, y + 8, tuple(int(c * 0.8) for c in fg))
        _text_run(img, x + 11, y + 2, int(rng.integers(24, 78)), fg, 4)
        y += 13
        n += 1


def _props(img, x0, y0, x1, y1, rng) -> None:
    """Панель Details: строки «подпись — поле ввода»."""
    fg, inp, hdr = rgb("Foreground"), rgb("Input"), rgb("Header")
    y = y0 + 4
    while y + 14 < y1:
        if rng.random() < 0.22:                      # заголовок группы
            _fill(img, x0 + 1, y, x1 - 1, y + 11, hdr)
            _text_run(img, x0 + 12, y + 3, int(rng.integers(30, 60)),
                      rgb("ForegroundHover"), 4)
            y += 14
            continue
        half = x0 + int((x1 - x0) * 0.42)
        _text_run(img, x0 + 6, y + 3, int(rng.integers(20, 50)), fg, 4)
        nf = int(rng.choice([1, 1, 3]))              # одно поле или XYZ
        fw = (x1 - half - 8) // nf
        for k in range(nf):
            fx = half + k * fw
            _fill(img, fx, y, fx + fw - 3, y + 11, inp)
            _text_run(img, fx + 3, y + 3, min(22, fw - 8), fg, 4)
        y += 14


def _tiles(img, x0, y0, x1, y1, rng) -> None:
    """Плитки ассетов в Content Browser."""
    fg = rgb("Foreground")
    tw, th = 46, 54
    y = y0 + 6
    while y + th < y1:
        x = x0 + 8
        while x + tw < x1:
            tone = float(rng.uniform(0.35, 0.95))
            col = np.array(rng.choice([[90, 110, 140], [110, 95, 80],
                                       [80, 120, 90], [130, 110, 130]]),
                           np.float32) * tone
            _fill(img, x, y, x + tw, y + th - 12, col)
            _fill(img, x, y + th - 14, x + tw, y + th - 12,
                  rgb(rng.choice(["AccentBlue", "AccentGreen",
                                  "AccentOrange", "AccentPurple"])))
            _text_run(img, x + 4, y + th - 9, tw - 10, fg, 4)
            x += tw + 8
        y += th + 6


def render(width: int = 640, height: int = 360, rng=None,
           icon_size: int = 16) -> Frame:
    """Собирает один кадр редактора и возвращает его вместе с разметкой."""
    if rng is None:
        rng = np.random.default_rng()
    w, h = width, height
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = rgb("Background")
    boxes: list[Box] = []
    used_icons: list[int] = []

    mode = str(rng.choice(MODES, p=[0.55, 0.25, 0.10, 0.10]))
    fg, panel, header = rgb("Foreground"), rgb("Panel"), rgb("Header")

    # -- меню --------------------------------------------------------------
    mh = max(12, int(h * 0.045))
    _fill(img, 0, 0, w, mh, rgb("Title"))
    boxes.append(Box(0, 0, w, mh, "menubar"))
    logo = icons.load_svg("Starship/Common/UELogo.svg", mh - 4)
    if logo is not None:
        _blit(img, logo, 3, 2, rgb("ForegroundHover"))
    x = mh + 6
    for _ in range(int(rng.integers(5, 8))):
        x = _text_run(img, x, mh // 2 - 2, int(rng.integers(16, 34)), fg, 4) + 9

    # -- панель инструментов ----------------------------------------------
    th = max(20, int(h * 0.075))
    _fill(img, 0, mh, w, mh + th, panel)
    _fill(img, 0, mh + th - 1, w, mh + th, rgb("Black"))
    boxes.append(Box(0, mh, w, mh + th, "toolbar"))

    cat = icons.catalog(icon_size)
    if cat:
        x = 8
        k = int(rng.integers(6, 11))
        for i in range(k):
            name, ic = cat[int(rng.integers(0, len(cat)))]
            used_icons.append(icons.TOOLBAR.index(
                next(p for p in icons.TOOLBAR if p.endswith(name + ".svg"))))
            hovered = rng.random() < 0.12
            if hovered:
                _fill(img, x - 3, mh + 3, x + icon_size + 3,
                      mh + th - 4, rgb("Hover"))
            _blit(img, ic, x, mh + (th - icon_size) // 2,
                  rgb("ForegroundHover" if hovered else "Foreground"))
            boxes.append(Box(x, mh + 2, x + icon_size, mh + th - 2,
                             "toolbar", name))
            x += icon_size + 12
        # Кнопка Play выделена цветом режима.
        play = icons.load_svg("Starship/Common/play.svg", icon_size)
        stop = icons.load_svg("Starship/Common/stop.svg", icon_size)
        bx = w // 2 - 30
        _fill(img, bx - 6, mh + 3, bx + 58, mh + th - 4,
              rgb("PrimaryPress") if mode == "play" else rgb("Dropdown"))
        ic = stop if mode in ("play", "paused") else play
        if ic is not None:
            _blit(img, ic, bx, mh + (th - icon_size) // 2,
                  rgb("AccentGreen") if mode == "edit" else rgb("AccentRed"))
        _text_run(img, bx + icon_size + 6, mh + th // 2 - 2, 30, fg, 4)

    # -- раскладка ---------------------------------------------------------
    sb = max(10, int(h * 0.04))                      # статусная строка
    cb_h = int(h * rng.uniform(0.22, 0.32))          # Content Browser
    right_w = int(w * rng.uniform(0.20, 0.27))       # правая колонка
    top = mh + th
    bottom = h - sb
    vp_bottom = bottom - cb_h

    # вкладка над вьюпортом
    tab_h = 13
    _fill(img, 0, top, w - right_w, top + tab_h, rgb("Recessed"))
    _fill(img, 4, top + 2, 74, top + tab_h, panel)
    _text_run(img, 10, top + 5, 46, rgb("ForegroundHover"), 4)
    boxes.append(Box(0, top, w - right_w, top + tab_h, "tabbar"))

    # вьюпорт
    vx0, vy0, vx1, vy1 = 0, top + tab_h, w - right_w, vp_bottom
    _viewport(img, vx0, vy0, vx1, vy1, rng, mode)
    boxes.append(Box(vx0, vy0, vx1, vy1, "viewport"))

    # правая колонка: Outliner сверху, Details под ним
    ox0 = w - right_w
    out_h = int((vp_bottom - top) * rng.uniform(0.35, 0.5))
    _fill(img, ox0, top, w, top + out_h, panel)
    _fill(img, ox0, top, w, top + 12, header)
    _text_run(img, ox0 + 6, top + 4, 44, rgb("ForegroundHover"), 4)
    selected = bool(rng.random() < 0.55)
    _tree(img, ox0, top + 12, w, top + out_h, rng, 14,
          int(rng.integers(0, 8)) if selected else -1)
    boxes.append(Box(ox0, top, w, top + out_h, "outliner"))

    _fill(img, ox0, top + out_h, w, vp_bottom, panel)
    _fill(img, ox0, top + out_h, w, top + out_h + 12, header)
    _text_run(img, ox0 + 6, top + out_h + 4, 38, rgb("ForegroundHover"), 4)
    if selected:
        _props(img, ox0, top + out_h + 12, w, vp_bottom, rng)
    boxes.append(Box(ox0, top + out_h, w, vp_bottom, "details"))

    # Content Browser внизу
    _fill(img, 0, vp_bottom, w, bottom, panel)
    _fill(img, 0, vp_bottom, w, vp_bottom + 14, header)
    _text_run(img, 22, vp_bottom + 5, 50, rgb("ForegroundHover"), 4)
    src = icons.load_svg("Starship/Common/folder-open.svg", 11)
    if src is not None:
        _blit(img, src, 6, vp_bottom + 2, rgb("Foreground"))
    split = int(w * 0.18)
    _fill(img, 0, vp_bottom + 14, split, bottom, rgb("Recessed"))
    _tree(img, 0, vp_bottom + 14, split, bottom, rng, 8, -1)
    _tiles(img, split, vp_bottom + 14, w, bottom, rng)
    boxes.append(Box(0, vp_bottom, w, bottom, "content"))

    # статусная строка
    _fill(img, 0, bottom, w, h, rgb("Title"))
    _text_run(img, 6, bottom + sb // 2 - 2, 70, fg, 4)
    _text_run(img, w - 120, bottom + sb // 2 - 2, 60, fg, 4)
    boxes.append(Box(0, bottom, w, h, "statusbar"))

    # вертикальные границы панелей — в редакторе они чёрные
    _fill(img, ox0 - 1, top, ox0, vp_bottom, rgb("Black"))
    _fill(img, 0, vp_bottom - 1, w, vp_bottom, rgb("Black"))

    _finish(img, rng)
    return Frame(img=img, boxes=boxes, mode=mode, selected=selected,
                 icon_ids=used_icons)


def _finish(img, rng) -> None:
    """Доводка кадра до вида настоящего скриншота.

    Голая отрисовка даёт 261 цвет: плоские заливки и ступенчатые границы.
    Реальный снимок экрана так не выглядит — там субпиксельное сглаживание
    шрифтов, лёгкая неравномерность подсветки и слабый шум сжатия. Без
    этого сеть выучит идеальные прямоугольники и развалится на живом
    скриншоте, как это уже случилось с текстурами Minecraft.
    """
    h, w = img.shape[:2]
    f = img.astype(np.float32)

    # Неравномерность подсветки панели: по краям экрана чуть темнее.
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    vign = 1.0 - 0.045 * (xx * xx + yy * yy)
    f *= vign[..., None]

    # Слабый цветной шум — след сжатия и матрицы монитора.
    f += rng.normal(0.0, 1.9, f.shape).astype(np.float32)

    # Сглаживание границ: настоящий скриншот не имеет ступенек в один пиксель.
    blur = f.copy()
    blur[1:-1] = (f[:-2] + 2.0 * f[1:-1] + f[2:]) / 4.0
    blur[:, 1:-1] = (blur[:, :-2] + 2.0 * blur[:, 1:-1] + blur[:, 2:]) / 4.0
    f = f * 0.72 + blur * 0.28

    np.clip(f, 0, 255, out=f)
    img[:] = f.astype(np.uint8)
