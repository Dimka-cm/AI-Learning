"""Зрение агента: 2D-классификатор сред; 3D — заглушка с интерфейсом.

Твой «глаз» уже на обучении (2D). PytorchClassifier — это то место, куда
подключается его результат: либо он сам сериализован как state_dict, либо
мы дообучим заглушку здесь. Интерфейсы ядра от весов не зависят.

  2D: вход — путь к изображению -> предсказание класса (Unreal Engine и т.п.)
  3D: вход — облако точек/глубина -> заглушка, интерфейс готов
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import numpy as np


class PytorchClassifier:
    """Классификатор изображений: произвольная torch-модель + классы."""

    def __init__(self, state_dict_path: str, class_names: list[str],
                 arch: str = "mobilenet_v3_small", input_size: int = 224):
        self.state_dict_path = state_dict_path
        self.class_names = class_names
        self.arch = arch
        self.input_size = input_size
        self._model = None
        self._transform = None
        self._backend = "torch"

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch
            from torchvision import models, transforms
        except Exception as exc:  # нет torch/torchvision
            raise RuntimeError(f"Не установлены torch/torchvision: {exc}")

        if self.arch == "mobilenet_v3_small":
            net = models.mobilenet_v3_small(num_classes=len(self.class_names))
        else:
            raise ValueError(f"Неизвестная архитектура {self.arch}")

        p = Path(self.state_dict_path).expanduser()
        if p.exists():
            sd = torch.load(str(p), map_location="cpu")
            if isinstance(sd, dict) and any("conv" in k or "classifier" in k for k in sd.keys()):
                net.load_state_dict(sd)
            else:  # сохранён весь чекпоинт модели
                state = sd.get("state_dict", sd.get("model", sd))
                net.load_state_dict(state, strict=False)
        net.eval()

        self._transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self._model = net

    def predict(self, image_path: str) -> str:
        """Возвращает человекочитаемый результат классификации."""
        if not Path(image_path).expanduser().exists():
            return f"[2D] Файл не найден: {image_path}"
        try:
            self._ensure_loaded()
            import torch
            from PIL import Image
            img = Image.open(image_path).convert("RGB")
            x = self._transform(img).unsqueeze(0)
            with torch.no_grad():
                logits = self._model(x)
            probs = torch.softmax(logits, dim=1)[0]
            top = int(probs.argmax())
            return f"[2D] {self.class_names[top]} (уверенность {float(probs[top]):.2f})"
        except RuntimeError as exc:
            return f"[2D] Веса ещё не готовы — {exc}"
        except Exception as exc:
            return f"[2D] Ошибка: {exc}"


class DepthDummy:
    """3D-зрение: интерфейс готов, модель/сенсор подключается позже."""

    def predict(self, data) -> str:
        return "[3D] Модуль 3D-зрения ещё не обучен — интерфейс готов, нужен датасет глубин/облаков точек"


def make_vision(cfg: dict):
    """Собирает vision-эксперта по конфигу. Поддерживает переход на numpy-веса,
    если «глаз» — не torch-формат (см. load_array_weights ниже)."""
    v = cfg.get("vision", {})
    if v.get("kind") == "torch":
        classes = v.get("classes", ["not_game", "unreal_engine", "unity", "other_3d_engine"])
        return PytorchClassifier(
            v.get("state_dict_path", "models/eye_2d.pt"),
            classes,
            arch=v.get("arch", "mobilenet_v3_small"),
        )
    return DepthDummy() if v.get("kind") == "depth" else PytorchClassifier("models/eye_2d.pt", ["class0", "class1"])


def load_array_weights(path: str) -> Optional[dict]:
    """Если «глаз» пришёл не torch, а numpy/текстом (список чисел) — сюда."""
    p = Path(path).expanduser()
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    try:
        val = ast.literal_eval(text)
        if isinstance(val, dict):
            arr = np.array(list(val.values()))
        else:
            arr = np.array(val)
        return {"array": arr, "shape": list(arr.shape)}
    except Exception:
        return None
