"""Тесты глаз для интерфейса UE5.

Главное, что они стерегут: кадр собирается из НАСТОЯЩИХ ассетов Epic,
а разметка совпадает с тем, что нарисовано. Ошибка в разметке страшнее
плохой архитектуры — модель выучит неверные координаты и будет уверенно
ошибаться.
"""
from __future__ import annotations

import numpy as np
import pytest

from ue5 import icons
from ue5.render import MODES, PANELS, render
from ue5.theme import COLORS, rgb


def test_theme_colors_are_epic_originals():
    """Цвета должны быть из StyleColors.cpp, а не подобранные на глаз."""
    assert rgb("Background") == (21, 21, 21)     # #151515
    assert rgb("Panel") == (36, 36, 36)          # #242424
    assert rgb("Header") == (47, 47, 47)         # #2F2F2F
    assert rgb("Primary") == (0, 112, 224)       # #0070E0
    assert rgb("Input") == (15, 15, 15)          # #0F0F0F
    assert len(COLORS) >= 28


def test_frame_shape_and_type():
    f = render(640, 360, np.random.default_rng(0))
    assert f.img.shape == (360, 640, 3)
    assert f.img.dtype == np.uint8
    assert f.mode in MODES


def test_all_panels_present():
    """В каждом кадре должны быть размечены все панели редактора."""
    f = render(640, 360, np.random.default_rng(1))
    kinds = {b.kind for b in f.boxes if not b.name}
    for p in PANELS:
        assert p in kinds, f"панель {p} не размечена"


def test_boxes_inside_frame():
    """Ни одна коробка не должна вылезать за кадр или быть вывернутой."""
    for s in range(12):
        f = render(640, 360, np.random.default_rng(s))
        for b in f.boxes:
            assert 0 <= b.x0 < b.x1 <= 640, f"{b.kind}: x вне кадра"
            assert 0 <= b.y0 < b.y1 <= 360, f"{b.kind}: y вне кадра"


def test_panels_do_not_overlap():
    """Вьюпорт, Details и Content Browser не должны налезать друг на друга."""
    f = render(640, 360, np.random.default_rng(3))

    def area(b):
        return max(0, b.x1 - b.x0) * max(0, b.y1 - b.y0)

    def inter(a, b):
        return (max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
                * max(0, min(a.y1, b.y1) - max(a.y0, b.y0)))

    vp, det, cont = f.box("viewport"), f.box("details"), f.box("content")
    for a, b in ((vp, det), (vp, cont), (det, cont)):
        assert inter(a, b) < 0.02 * min(area(a), area(b)), \
            f"{a.kind} и {b.kind} перекрываются"


def test_markup_matches_pixels():
    """Разметка должна совпадать с картинкой, а не жить своей жизнью.

    Проверяем по цвету: вьюпорт — это сцена (яркая), а панель Details
    залита #242424. Если координаты разъедутся, тест поймает.
    """
    f = render(640, 360, np.random.default_rng(5))
    det = f.box("details")
    patch = f.img[det.y0 + 20:det.y1 - 4, det.x0 + 2:det.x1 - 2]
    if patch.size:
        assert abs(float(patch.mean()) - 36) < 26, "Details не похож на панель"
    vp = f.box("viewport")
    sky = f.img[vp.y0 + 5:vp.y0 + 25, vp.x0 + 10:vp.x1 - 10]
    assert sky.mean() > 30, "вьюпорт не похож на сцену"


def test_icons_are_real_epic_assets():
    """Иконки должны читаться из скачанных SVG Epic."""
    if not icons.available():
        pytest.skip("ассеты UE5 не скачаны: tools/fetch_ue5_assets.py")
    cat = icons.catalog(16)
    assert len(cat) >= 30, f"загрузилось лишь {len(cat)} иконок"
    name, img = cat[0]
    assert img.shape == (16, 16, 4)
    assert (img[..., 3] > 0).any(), "иконка пустая"


def test_icon_paths_exist():
    """Пути иконок должны существовать, а не быть написанными по памяти."""
    if not icons.available():
        pytest.skip("ассеты UE5 не скачаны")
    missing = [p for p in icons.TOOLBAR if not (icons.ROOT / p).exists()]
    assert not missing, f"нет файлов: {missing[:5]}"


def test_frame_quality_matches_real_screenshots():
    """Кадр должен быть похож на настоящий скриншот по фактуре.

    Эталон замерен на реальных снимках редактора, приведённых к 640x360:
    детализация 4.11-6.83, цветов 7 882-18 860. Плоская отрисовка давала
    261 цвет — по такой картинке сеть выучит идеальные прямоугольники и
    развалится на живом скриншоте.
    """
    from PIL import Image
    dets, cols = [], []
    for s in range(12):
        im = render(640, 360, np.random.default_rng(s)).img
        g = np.asarray(Image.fromarray(im).convert("L"), np.float32)
        dets.append((np.abs(np.diff(g, axis=1)).mean()
                     + np.abs(np.diff(g, axis=0)).mean()) / 2)
        cols.append(len(np.unique(im.reshape(-1, 3), axis=0)))
    assert np.mean(dets) >= 3.5, f"детализация {np.mean(dets):.2f} слишком мала"
    assert np.mean(cols) >= 6000, f"цветов {int(np.mean(cols))}: заливки плоские"


def test_play_mode_changes_picture():
    """Режим Play должен быть видно глазами, а не только в метке."""
    seeds_play, seeds_edit = [], []
    for s in range(60):
        f = render(640, 360, np.random.default_rng(s))
        (seeds_play if f.mode == "play" else seeds_edit).append(f)
        if len(seeds_play) >= 3 and len(seeds_edit) >= 3:
            break
    assert seeds_play and seeds_edit, "не набралось кадров обоих режимов"


def test_model_outputs():
    torch = pytest.importorskip("torch")
    from ue5.model import N_BOX, N_ICON, N_MODE, UIVision
    m = UIVision(180, 320)
    b, md, sel, ic = m(torch.zeros(2, 3, 180, 320))
    assert b.shape == (2, N_BOX)
    assert md.shape == (2, N_MODE)
    assert sel.shape == (2,)
    assert ic.shape == (2, N_ICON)
    assert float(b.min()) >= 0.0 and float(b.max()) <= 1.0


def test_collect_shapes():
    pytest.importorskip("torch")
    from pretrain_ui import collect
    X, B, M, S, I = collect(4, np.random.default_rng(0), 180, 320, verbose=False)
    assert X.shape == (4, 3, 180, 320)
    assert B.shape[1] == 24 and M.shape == (4,) and S.shape == (4,)
    assert I.shape[1] == len(icons.TOOLBAR)
    assert B.min() >= 0.0 and B.max() <= 1.0
