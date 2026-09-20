from __future__ import annotations

from io import StringIO

from steps.step_00_prepare import iter_training_units


def texts(src: str) -> list[str]:
    return [item for item, _ in iter_training_units(StringIO(src))]


def test_question_answer_pairs_stay_in_one_training_unit():
    units = texts(
        "Пользователь: Сколько будет 2+2?\n"
        "Агент: 2+2=4.\n"
        "Пользователь: Что такое Mixin?\n"
        "Агент: Это способ внедриться в код Minecraft.\n"
    )

    assert len(units) == 2
    assert "Сколько будет" in units[0] and "2+2=4" in units[0]
    assert "Что такое Mixin" in units[1] and "Minecraft" in units[1]


def test_inline_question_answer_lines_are_separate_units():
    units = texts(
        "Пользователь: Привет. Агент: Привет, создатель.\n"
        "Пользователь: Не знаешь? Агент: Скажу честно и предложу поискать.\n"
    )

    assert len(units) == 2
    assert units[0].startswith("Пользователь: Привет")
    assert units[1].startswith("Пользователь: Не знаешь")


def test_plain_text_without_speaker_markers_stays_linewise():
    units = texts("Первая строка прозы.\nВторая строка прозы.\n")

    assert units == ["Первая строка прозы.", "Вторая строка прозы."]


def test_blank_line_finishes_dialogue_block():
    units = texts(
        "Вопрос: Как дела?\n"
        "Ответ: Нормально.\n"
        "\n"
        "Вопрос: Что делать?\n"
        "Ответ: Проверить логи.\n"
    )

    assert len(units) == 2
    assert "Как дела" in units[0] and "Нормально" in units[0]
    assert "Что делать" in units[1] and "Проверить логи" in units[1]


def test_append_token_batch_uses_vectorized_tokenizer():
    from steps.step_00_prepare import append_token_batch

    class FakeBatchTokenizer:
        def __init__(self):
            self.calls = []

        def __call__(self, units, add_special_tokens=False):
            assert add_special_tokens is False
            self.calls.append(list(units))
            return {"input_ids": [[len(x)] for x in units]}

    tok = FakeBatchTokenizer()
    buffer = {1: [], 2: []}
    flushed = []

    def flush(shard):
        flushed.append(shard)

    append_token_batch(tok, [(1, "привет"), (2, "мир"), (1, "длинная строка")], 0, buffer, flush)

    assert tok.calls == [["привет", "мир", "длинная строка"]]
    assert buffer[1] == [6, 0, 14, 0]
    assert buffer[2] == [3, 0]
    assert flushed == []
