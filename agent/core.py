"""Ядро агента: цикл «понимание → переключатель → эксперт/инструмент → память».

Архитектура повторяет то, как устроен я сам:
    запрос -> роутер (переключатель) -> эксперт или инструмент -> ответ
    + память (короткая история + долгие знания)
    + песочница (твой ПК вместо моего воркспейса)
    + зрение 2D (глаз) и 3D-заглушка
"""
from __future__ import annotations

import re
import sys
from typing import Optional

from . import memory, router, vision
from .experts import make_experts
from .tools import ToolResult, make_pc_from_config


class Agent:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.mem = memory.Memory(cfg.get("memory_dir", "agent/memory"))
        self.experts = make_experts(cfg)
        self.pc = make_pc_from_config(cfg.get("tools", {}))
        self.vision = vision.make_vision(cfg)
        self.name = cfg.get("name", "agent")
        self.trace = cfg.get("trace", False)
        # перехват: каждый прогон модели попадает в журнал агента
        for e in self.experts.values():
            if hasattr(e, "on_generate"):
                e.on_generate = self._log_generation

    # --------------------------------------------------------------- лог
    def _log_generation(self, expert: str, prompt: str, reply: str):
        self.mem.add("model:" + expert, prompt + " → " + reply[:200])

    def _say(self, who: str, text: str):
        if self.trace:
            print(f"\033[90m[{who}]\033[0m {text}")

    # ------------------------------------------------------------- эксперты
    def _call_expert(self, name: str, text: str, context: str = "") -> str:
        if name in self.experts:
            out = self.experts[name].respond(text, context)
        elif name == "chat":
            out = self._stub_or_chat(text)
        else:
            out = f"(эксперт '{name}' не найден)"
        return (out or "").strip() or "(пустой ответ)"

    def _stub_or_chat(self, text: str) -> str:
        # чат-модели пока нет -> честная заглушка с подсказками
        return (
            "Я каркас агента: текстовая чат-модель ещё не подключена — "
            "дообучи её (см. steps/ и docs/SELF_HOSTED_RUNNER.md), а пока "
            "я умею инструменты (/list /read /write /run), математику и подключаю зрение."
        )

    # ------------------------------------------------------------- инструменты
    def _call_tool(self, text: str) -> str:
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in router.TOOL_COMMANDS:
            op = router.TOOL_COMMANDS[cmd]
            if op == "vision":
                return self._see(arg)
            return self._exec_pc(op, arg)
        return self._tool_fallback(text)

    def _exec_pc(self, op: str, arg: str) -> str:
        pc = self.pc
        arg = arg.strip().strip("\"'")
        if not arg:
            if op in ("list_dir",):
                arg = "."  # список корня воркспейса по умолчанию
            else:
                return f"Укажи цель для {op}"

        fn = getattr(pc, op, None)
        if fn is None:
            return f"Операция {op} неизвестна"
        if op in ("write_file",):
            # формат: /write путь##содержимое
            if "##" in arg:
                p, content = arg.split("##", 1)
                res = fn(p.strip(), content)
            else:
                return "Формат: /write путь##содержимое"
        else:
            res = fn(arg)
        return self._fmt(res)

    def _fmt(self, r: ToolResult) -> str:
        head = "✅" if r.ok else "⛔"
        suffix = " [dry-run]" if r.dry_run else ""
        return f"{head} {r.action}({r.target}){suffix}\n{r.output}"

    def _tool_fallback(self, text: str) -> str:
        """Без LLM: грубое намерение -> инструмент. Пути не выдумываем —
        если аргумента нет, просим пользователя."""
        intent = router.intent_of(text)
        self._say("tool", f"намерение={intent}")
        if intent == "list":
            return self._fmt(self.pc.list_dir("."))
        if intent == "read" or intent == "write":
            return ("Уточни, пожалуйста: какой файл? "
                    "Пример: /read notes.txt или /write notes.txt##привет")
        if intent == "delete":
            return ("Что удалить? Удалительная операция требует полного пути "
                    "и подтверждения (--allow-dangerous).")
        if intent == "run":
            return "Какую команду запустить? Пример: /run python --version"
        if intent == "mkdir":
            return "Какую папку создать? Пример: /mkdir projects/new"
        return ("Не понял, что сделать с файлами. Доступны: "
                "/list /read /write /mkdir /delete /run /see")

    # ------------------------------------------------------------- зрение
    def _see(self, arg: str) -> str:
        path = (arg or "").strip().strip("\"'")
        if not path:
            return "Укажи путь к изображению: /see photo.png"
        # вытаскиваем имя файла из фразы «что на картинке shot.png» / «посмотри на скрин.jpg»
        m = re.search(r"([\w.\-/\\]+\.(?:png|jpg|jpeg|bmp|webp))", path, re.IGNORECASE)
        if m:
            path = m.group(1)
        return self.vision.predict(path)

    # ------------------------------------------------------------- главный цикл
    def reply(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "Пустой запрос."

        self.mem.add("user", text)
        route = router.route(text)

        if route == "tool":
            out = self._call_tool(text)
        elif route == "vision":
            self._say("route", "-> vision")
            out = self._see(text.split(maxsplit=1)[1] if " " in text else "")
        else:
            self._say("route", f"-> {route}")
            out = self._call_expert(route, text, self.mem.context(4))

        self.mem.add("assistant", out)
        return out

    def run_interactive(self):
        print(f"=== {self.name}: твой ПК-агент ===")
        print("Команды: /list /read /write /mkdir /delete /run /see, "
              "или свободный текст (математика, код, диалог).")
        print("Выход: /exit | /quit | Ctrl+C\n")
        while True:
            try:
                text = input("> ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nПока!")
                break
            if text.lower() in ("/exit", "/quit", "выход"):
                break
            print(self.reply(text))
            print()
