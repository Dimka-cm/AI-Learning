"""Эксперты агента: несколько «обученных ИИ» и заглушки до их появления.

Каждый эксперт — независимая способность:
  * chat_gpt2 / chat_rugpt — языковая модель (генерация);
  * code   — дообученный на C/C++/C# вариант модели;
  * math   — детерминированный вычислитель + провайдер мировой модели при наличии;
  * vision — вынесен в vision.py (2D есть, 3D заглушка).

Единый интерфейс: respond(text, context) -> str.
Это и есть «скрещение нескольких ИИ», где каждый — отдельный блок.
"""
from __future__ import annotations

import re
from typing import Optional


# =========================================================== LLM-обёртка
class HFModelExpert:
    """Обёртка над моделью HF (AutoModelForCausalLM). Лениво грузит веса."""

    def __init__(self, model_id: str, tokenizer=None, model=None, gen_kwargs: Optional[dict] = None):
        self.model_id = model_id
        self._tok = tokenizer
        self._model = model
        self._reachable = None  # кэш проверки доступа к HF
        self.gen_kwargs = gen_kwargs or {
            "max_new_tokens": 96, "do_sample": True, "temperature": 0.8, "top_p": 0.9,
        }
        self.on_generate = None  # перехватчик, чтобы агент логировал/сохранял прогоны

    def _hf_reachable(self) -> bool:
        import os
        if os.path.exists(self.model_id):  # локальная папка с весами
            return True
        if self._reachable is not None:
            return self._reachable
        try:
            import urllib.request
            urllib.request.urlopen("https://huggingface.co", timeout=3)
            self._reachable = True
        except Exception:
            self._reachable = False
        return self._reachable

    def _ensure_model(self):
        if self._model is not None:
            return
        if not self._hf_reachable():
            raise RuntimeError(
                "Hugging Face недоступен (нет сети или веса не в кэше). На ноуте "
                "запусти при интернете, либо укажи локальный путь к весам в "
                "config/agent_config.yaml (experts.chat.model)."
            )
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self._model.eval()

    def respond(self, text: str, context: str = "") -> str:
        try:
            self._ensure_model()
        except RuntimeError as exc:
            return str(exc)
        prompt = (context + "\n" if context else "") + "Пользователь: " + text + "\nАгент:"
        ids = self._tok(prompt, return_tensors="pt")
        out = self._model.generate(ids["input_ids"], **self.gen_kwargs)
        reply = self._tok.decode(out[0], skip_special_tokens=True)
        reply = reply[len(prompt):].strip()
        if self.on_generate:
            self.on_generate(self.move_id, text, reply)
        return reply


# =========================================================== математика
class MathExpert:
    """Надёжная арифметика/выражения через AST (не даём нейронке путаться в 2+2*2)."""

    # захватываем выражение из любого текста («сколько будет (17*3)%5» -> "(17*3)%5")
    _CAND = re.compile(r"[0-9+\-*/%^().\s]+")

    def respond(self, text: str, context: str = "") -> str:
        # вытаскиваем выражение из любой фразы: «сколько будет (17*3)%5» -> "(17*3)%5"
        t = text.replace(",", ".") if text else ""
        for chunk in re.findall(r"[0-9+\-*/%^().]+", t):
            chunk = chunk.strip()
            if any(ch.isdigit() for ch in chunk) and any(ch in chunk for ch in "+-*/%^"):
                safe = chunk.replace("^", "**").replace(" ", "")
                # Безопасно: строка из цифр/операторов/точек/скобок (без букв и
                # кавычек), eval получает пустые builtins => вызов функций невозможен.
                try:
                    val = eval(safe, {"__builtins__": {}}, {})  # noqa: S307
                    return f"{chunk} = {val}"
                except Exception:
                    continue
        return explain_math()


def explain_math() -> str:
    return (
        "Математику умею, уточни выражение: например, «сколько будет 2+2*2», "
        "«посчитай (17*3)%5», «2**10». Пока беру только числовые/алгебраические выражения."
    )


# =========================================================== заглушки
class StubExpert:
    """Заглушка для «обученной ИИ», которой пока нет (код/чат-вариант и т.п.)."""

    def respond(self, text: str, context: str = "") -> str:
        return (
            f"(эксперт '{self.move_id}' ещё не подключён: "
            f"нужна готовая модель/чекпоинт — см. config/agent_config.yaml)"
        )


def make_experts(cfg: dict):
    """Собирает экспертов по конфигу. Каждому проставляется move_id."""
    exps = {}
    for name, spec in cfg.get("experts", {}).items():
        kind = spec.get("kind", "stub")
        if kind == "hf":
            e = HFModelExpert(
                spec["model"],
                gen_kwargs=spec.get("generate", None),
            )
        elif kind == "math":
            e = MathExpert()
        elif kind == "stub":
            e = StubExpert()
        else:
            raise ValueError(f"Неизвестный kind эксперта '{kind}' для '{name}'")
        e.move_id = name
        exps[name] = e
    return exps
