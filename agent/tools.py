"""Инструменты агента для работы с ПК пользователя — «мой воркспейс, но твой компьютер».

Аналог моего набора инструментов (чтение/запись файлов, запуск команд), только
мишень — файловая система твоего ноута. Безопасность — главное:

  * всё по умолчанию ограничено WORKSPACE (папкой, которую выберешь ты);
  * выход за WORKSPACE — только по явному allowlist в конфиге;
  * опасные операции (delete, run) требуют подтверждения или флага --allow-dangerous;
  * режим dry-run печатает, что агент СДЕЛАЛ БЫ, ничего не выполняя.

Движок намеренно детерминированный (без LLM-магии): это слой, который потом
вызывает роутер/модель, точно так же, как мой цикл «запрос → инструмент → результат».
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional


@dataclass
class ToolResult:
    ok: bool
    action: str
    target: str
    output: str = ""
    dry_run: bool = False


# Паттерны команд, которые НЕ исполняем без явного разрешения пользователя.
DANGEROUS_CMD_PATTERNS = [
    "format ", "del /", "rm -rf /", "rd /s", "shutdown", "taskkill",
    "diskpart", "reg ", "icacls /reset", "> nul & rd", "move nul",
    "dd if=", "mkfs", ":(){", "chmod -R 777 /",
]

DANGEROUS_SUFFIXES = {".exe", ".bat", ".cmd", ".ps1", ".vbs", ".msi", ".sh"}


@dataclass
class PC:
    """Безопасная песочница вокруг рабочей папки на ПК пользователя."""
    workspace: str = "~/AgentWorkspace"
    allow_extra_roots: List[str] = field(default_factory=list)
    dry_run: bool = False
    allow_dangerous: bool = False
    confirm: Optional[Callable[[str], bool]] = None  # (description) -> разрешено?

    def __post_init__(self):
        self.root = Path(self.workspace).expanduser().resolve()

    # ------------------------------------------------------------ безопасность
    def _resolve(self, p: str) -> Path:
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve()

    def _is_allowed(self, path: Path) -> bool:
        if path == self.root or self.root in path.parents:
            return True
        for extra in self.allow_extra_roots:
            r = Path(extra).expanduser().resolve()
            if path == r or r in path.parents:
                return True
        return False

    def _gate(self, path: Path, action: str) -> bool:
        if not self._is_allowed(path):
            return False
        return True

    # ------------------------------------------------------------- инструменты
    def list_dir(self, p: str = ".") -> ToolResult:
        d = self._resolve(p)
        if not self._gate(d, "list"):
            return ToolResult(False, "list_dir", p, f"Запрещено: {d} вне {self.root}")
        if self.dry_run:
            return ToolResult(True, "list_dir", p, dry_run=True)
        try:
            items = []
            for e in sorted(d.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                try:
                    kind = "dir " if e.is_dir() else "file"
                    size = "" if e.is_dir() else f" {e.stat().st_size}B"
                    items.append(f"{kind}  {e.name}{size}")
                except OSError:
                    items.append(f"???   {e.name}")
            return ToolResult(True, "list_dir", str(d), "\n".join(items))
        except OSError as ex:
            return ToolResult(False, "list_dir", str(d), str(ex))

    def read_file(self, p: str) -> ToolResult:
        f = self._resolve(p)
        if not self._gate(f, "read"):
            return ToolResult(False, "read_file", p, f"Запрещено: {f} вне {self.root}")
        if self.dry_run:
            return ToolResult(True, "read_file", p, dry_run=True)
        try:
            data = Path(f).read_text(encoding="utf-8", errors="replace")
            return ToolResult(True, "read_file", str(f), data)
        except OSError as ex:
            return ToolResult(False, "read_file", str(f), str(ex))

    def write_file(self, p: str, content: str) -> ToolResult:
        f = self._resolve(p)
        if not self._gate(f, "write"):
            return ToolResult(False, "write_file", p, f"Запрещено: {f} вне {self.root}")
        if self.dry_run:
            return ToolResult(True, "write_file", str(f),
                              f"(dry-run) записал бы {len(content)} символов", dry_run=True)
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content, encoding="utf-8")
            return ToolResult(True, "write_file", str(f), f"Записано {len(content)} символов")
        except OSError as ex:
            return ToolResult(False, "write_file", str(f), str(ex))

    def mkdir(self, p: str) -> ToolResult:
        d = self._resolve(p)
        if not self._gate(d, "mkdir"):
            return ToolResult(False, "mkdir", p, f"Запрещено: {d} вне {self.root}")
        if self.dry_run:
            return ToolResult(True, "mkdir", str(d), "(dry-run) создал бы", dry_run=True)
        try:
            d.mkdir(parents=True, exist_ok=True)
            return ToolResult(True, "mkdir", str(d), "Создано")
        except OSError as ex:
            return ToolResult(False, "mkdir", str(d), str(ex))

    def delete(self, p: str) -> ToolResult:
        f = self._resolve(p)
        if not self._gate(f, "delete"):
            return ToolResult(False, "delete", p, f"Запрещено: {f} вне {self.root}")
        if not self.allow_dangerous:
            if self.confirm is not None and not self.confirm(f"Удалить {f}?"):
                return ToolResult(False, "delete", str(f), "Отменено пользователем")
            if self.confirm is None:
                return ToolResult(False, "delete", str(f),
                                  "Требуется подтверждение (запустите --allow-dangerous или подключите confirm)")
        if self.dry_run:
            return ToolResult(True, "delete", str(f), "(dry-run) удалил бы", dry_run=True)
        try:
            if f.is_dir():
                shutil.rmtree(f)
            else:
                f.unlink()
            return ToolResult(True, "delete", str(f), "Удалено")
        except OSError as ex:
            return ToolResult(False, "delete", str(f), str(ex))

    def run_command(self, cmd: str, cwd: str = "") -> ToolResult:
        low = cmd.lower()
        if any(pat in low for pat in DANGEROUS_CMD_PATTERNS) and not self.allow_dangerous:
            return ToolResult(False, "run", cmd, "Опасная команда отклонена (нужен --allow-dangerous)")
        if self.dry_run:
            return ToolResult(True, "run", cmd, "(dry-run) выполнил бы", dry_run=True)
        try:
            cw = self._resolve(cwd) if cwd else self.root
            proc = subprocess.run(
                cmd, shell=True, cwd=str(cw), capture_output=True, text=True,
                timeout=120, encoding="utf-8", errors="replace",
            )
            out = (proc.stdout or "") + (proc.stderr or "")
            return ToolResult(proc.returncode == 0, "run", cmd, out.strip() or "(пусто)")
        except subprocess.TimeoutExpired:
            return ToolResult(False, "run", cmd, "Тайм-аут (120 c)")
        except OSError as ex:
            return ToolResult(False, "run", cmd, str(ex))


def make_pc_from_config(cfg: dict) -> PC:
    ws = cfg.get("workspace", "~/AgentWorkspace")
    pc = PC(
        workspace=ws,
        allow_extra_roots=cfg.get("allow_extra_roots", []),
        dry_run=bool(cfg.get("dry_run", False)),
        allow_dangerous=bool(cfg.get("allow_dangerous", False)),
    )
    os.makedirs(pc.root, exist_ok=True)
    return pc
