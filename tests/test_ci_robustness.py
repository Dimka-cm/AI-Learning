"""Тесты устойчивости пайплайна: то, на чём уже падал CI.

Первая ошибка: порция демо-корпуса оказалась меньше окна обучения
(«Слишком мало токенов в shard_01: 232 < block_size+2») и шаг умер.

Вторая: в workflow стояло `OMP_NUM_THREADS: "auto"`, и libgomp ругался
«Недопустимое значение переменной окружения OMP_NUM_THREADS».
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from src.data_utils import load_shard, shard_token_count, token_id_dtype
from src.threads import default_threads, fix_thread_env


class Cfg:
    """Минимальный стенд-ин заглушка конфига (нужны только эти поля)."""

    def __init__(self, root: str, block_size: int = 256, dtype: str = "uint16"):
        self.root = root
        self.block_size = block_size
        self.dtype = dtype
        self.tokenized_dir = "shards"
        self.text_dir = "raw"

    def p(self, *parts: str) -> str:
        return os.path.join(self.root, *parts)


def write_shard(cfg: Cfg, step: int, tokens: list[int]) -> None:
    os.makedirs(cfg.p(cfg.tokenized_dir), exist_ok=True)
    arr = np.asarray(tokens, dtype=token_id_dtype(cfg))
    arr.tofile(cfg.p(cfg.tokenized_dir, f"shard_{step:02d}.bin"))


def test_small_shard_is_repeated_instead_of_crashing(tmp_path):
    """232 токена при block_size=256 — именно то, что уронило прогон в Actions."""
    cfg = Cfg(str(tmp_path), block_size=256)
    write_shard(cfg, 1, list(range(232)))

    data = load_shard(cfg, 1)  # раньше здесь был ValueError

    assert data.size >= cfg.block_size + 2
    # данные именно повторены, а не досыпаны нулями
    assert data[:5].tolist() == [0, 1, 2, 3, 4]
    assert data[232:237].tolist() == [0, 1, 2, 3, 4]


def test_normal_shard_is_not_touched(tmp_path):
    cfg = Cfg(str(tmp_path), block_size=8)
    write_shard(cfg, 2, list(range(100)))

    data = load_shard(cfg, 2)

    assert data.size == 100
    assert data.tolist() == list(range(100))


def test_empty_shard_gives_readable_error(tmp_path):
    """Пустая порция — не «232 < 258», а понятная инструкция, что делать."""
    cfg = Cfg(str(tmp_path), block_size=256)
    write_shard(cfg, 3, [])

    with pytest.raises(ValueError) as err:
        load_shard(cfg, 3)

    msg = str(err.value)
    assert "пустая" in msg
    assert "step_00_prepare.py" in msg


def test_missing_shard_says_where_to_get_data(tmp_path):
    cfg = Cfg(str(tmp_path), block_size=256)

    with pytest.raises(FileNotFoundError) as err:
        load_shard(cfg, 9)

    assert "step_00_prepare.py" in str(err.value)


def test_shard_token_count_reports_zero_for_missing(tmp_path):
    cfg = Cfg(str(tmp_path), block_size=256)
    assert shard_token_count(cfg, 5) == 0

    write_shard(cfg, 5, [1, 2, 3])
    assert shard_token_count(cfg, 5) == 3


def test_omp_auto_becomes_a_number(monkeypatch):
    """libgomp не понимает `auto` — переменная должна стать числом."""
    for bad in ("auto", "AUTO", "", "abc", "0", "-3"):
        monkeypatch.setenv("OMP_NUM_THREADS", bad)
        fix_thread_env()
        assert os.environ["OMP_NUM_THREADS"].isdigit(), bad
        assert int(os.environ["OMP_NUM_THREADS"]) >= 1


def test_good_omp_value_is_kept(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    monkeypatch.setenv("MKL_NUM_THREADS", "2")

    fix_thread_env()

    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "2"


def test_missing_thread_vars_are_left_alone(monkeypatch):
    """Чего нет — то не выдумываем: библиотеки сами решат."""
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)

    fix_thread_env()

    assert "OPENBLAS_NUM_THREADS" not in os.environ


def test_num_workers_auto_is_a_number():
    """config понимает NUM_WORKERS=auto (так ставит workflow), torch — нет."""
    from src.config import CFG

    os.environ["NUM_WORKERS"] = "auto"
    try:
        cfg = CFG().apply_env_overrides()
    finally:
        os.environ.pop("NUM_WORKERS", None)

    assert isinstance(cfg.num_workers, int)
    assert cfg.num_workers == default_threads()


def test_drive_size_parser_understands_large_parts():
    from steps.step_00_ci_data import _drive_file_size, _parse_size_bytes
    from types import SimpleNamespace

    assert _parse_size_bytes("460 MB") == 460 * 1024 * 1024
    assert _parse_size_bytes("1.5 GB") == int(1.5 * 1024 ** 3)
    assert _parse_size_bytes("25мб") == 25 * 1024 * 1024
    assert _drive_file_size(SimpleNamespace(size="6.5 GB")) == int(6.5 * 1024 ** 3)
    assert _drive_file_size(SimpleNamespace(name="part.txt")) is None


def test_drive_filter_selects_one_part_naturally():
    from steps.step_00_ci_data import _filter_drive_txts
    from types import SimpleNamespace

    files = [
        SimpleNamespace(name="part_10.txt"),
        SimpleNamespace(name="notes.md"),
        SimpleNamespace(name="part_2.txt"),
        SimpleNamespace(name="part_1.txt"),
    ]

    assert [f.name for f in _filter_drive_txts(files)] == ["part_1.txt", "part_2.txt", "part_10.txt"]
    assert [f.name for f in _filter_drive_txts(files, include_regex="part_2", max_files=1)] == ["part_2.txt"]
    assert _filter_drive_txts(files, include_regex="missing") == []


def test_resume_from_latest_uses_newest_checkpoint(monkeypatch, tmp_path):
    from src.data_utils import find_resume_checkpoint

    cfg = Cfg(str(tmp_path), block_size=8)
    cfg.checkpoints_dir = "checkpoints"
    cfg.shards = 10

    p4 = tmp_path / "checkpoints" / "step_04"
    p1 = tmp_path / "checkpoints" / "step_01"
    p4.mkdir(parents=True)
    p1.mkdir(parents=True)
    old = p4 / "last.pt"
    new = p1 / "last.pt"
    old.write_bytes(b"old")
    new.write_bytes(b"new")
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))

    monkeypatch.setenv("RESUME_FROM_LATEST", "1")

    assert find_resume_checkpoint(cfg, 2) == str(new)


def test_default_resume_does_not_jump_to_future_step(monkeypatch, tmp_path):
    from src.data_utils import find_resume_checkpoint

    cfg = Cfg(str(tmp_path), block_size=8)
    cfg.checkpoints_dir = "checkpoints"
    cfg.shards = 10

    p4 = tmp_path / "checkpoints" / "step_04"
    p1 = tmp_path / "checkpoints" / "step_01"
    p4.mkdir(parents=True)
    p1.mkdir(parents=True)
    future = p4 / "last.pt"
    prev = p1 / "last.pt"
    future.write_bytes(b"future")
    prev.write_bytes(b"prev")
    os.utime(future, (3000, 3000))
    os.utime(prev, (1000, 1000))

    monkeypatch.delenv("RESUME_FROM_LATEST", raising=False)

    assert find_resume_checkpoint(cfg, 2) == str(prev)
