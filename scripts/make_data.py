#!/usr/bin/env python3
"""Генератор стартового датасета (полностью офлайн, только stdlib).

Создаёт демо-корпус, чтобы прокачать пайплайн от начала до конца без интернета.
Раскладывает по доменам — именно так потом подхватит steps/step_00_prepare.py:

    data/raw/text/      — русские диалоги и проза      -> shards 1..4  (база, разговор)
    data/raw/math/      — арифметика/выражения с ответами -> shards 5..6
    data/raw/code/      — C / C++ / C#                  -> shards 7..8
    data/raw/behavior/  — «не обзываться», вежливые ответы -> shards 9..10

Внимание: это ДЕМО (синтетика). Для качества модели принеси настоящий русский
текст (книги, вики, диалоги) — см. docs/DATA.md.

Запуск:  python3 scripts/make_data.py [--scale N]
"""
from __future__ import annotations

import argparse
import os
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "data", "raw")


# =========================================================== ТЕКСТ / ДИАЛОГИ
DIALOGUES = [
    "— Привет! Как дела? — Привет! Отлично, а у тебя?",
    "— Что будем делать сегодня? — Может, погуляем в парке, погода хорошая.",
    "— Ты читал эту книгу? — Да, она мне очень понравилась, советую.",
    "— Поможешь разобраться с задачей? — Конечно, давай посмотрим вместе.",
    "— Который час? — Уже почти полдень, пора обедать.",
    "— Расскажи что-нибудь интересное. — Космос огромен: свет от Солнца летит до нас восемь минут.",
    "— Хочешь чаю? — Да, спасибо, с удовольствием.",
    "— Как прошёл день? — Насыщенно: много работы, но всё успел.",
]

PROSE = [
    "Утро началось с тихого дождя, который стучал по подоконнику.",
    "В лесу пахло хвоей, а между деревьями пробивались солнечные лучи.",
    "Город засыпал медленно, и только редкие машины нарушали тишину.",
    "Человек учится всю жизнь, и каждый новый навык открывает новые двери.",
    "Осень раскрасила листья в золото и багрянец.",
    "Маленькая лодка покачивалась на волнах недалеко от берега.",
    "Программист написал программу, которая считала быстрее любого человека.",
    "Доброе слово и кошке приятно, а грубое — ранит даже сильного.",
]


# =========================================================== МАТЕМАТИКА
def gen_math(count: int) -> list[str]:
    """Пары (вопрос, ответ с решением) — чтобы модель учила цепочку арифметики."""
    rnd = random.Random(7)
    cases = []

    def add(a, b):
        cases.append((f"Сколько будет {a} + {b}?", f"{a} + {b} = {a + b}"))

    def sub(a, b):
        cases.append((f"Реши: {a} - {b}.", f"{a} - {b} = {a - b}"))

    def mul(a, b):
        cases.append((f"Найди произведение {a} * {b}.", f"{a} * {b} = {a * b}"))

    def mul_add(a, b, c):
        cases.append((f"Сколько будет {a} * {b} + {c}?", f"{a} * {b} + {c} = {a * b + c}"))

    def paren(a, b, c):
        cases.append((f"Посчитай ({a} + {b}) * {c}.", f"({a} + {b}) * {c} = {(a + b) * c}"))

    def power(a, b):
        cases.append((f"Значение выражения {a} ** {b}?", f"{a} ** {b} = {a ** b}"))

    def mod(a, b):
        cases.append((f"Остаток от деления {a} на {b}?", f"{a} % {b} = {a % b}"))

    def prec(a, b, c):
        cases.append((f"Сколько будет {a} + {b} * {c}?", f"{a} + {b} * {c} = {a + b * c}"))

    import inspect
    ops = [add, sub, mul, mul_add, paren, power, mod, prec]
    while len(cases) < count:
        op = rnd.choice(ops)
        n = len(inspect.signature(op).parameters)
        args = [rnd.randint(1, 100), rnd.randint(1, 20), rnd.randint(1, 10)][:n]
        op(*args)
    out = []
    for q, ans in cases[:count]:
        out.append(q)
        out.append(ans)
    return out


# =========================================================== КОД (C/C++/C#)
CODE_SNIPPETS = [
    "// Привет, мир на C\n#include <stdio.h>\nint main() {\n    printf(\"Hello, C!\\n\");\n    return 0;\n}",
    "// Класс на C++\nclass Robot {\npublic:\n    void speak() { std::cout << \"Привет!\" << std::endl; }\n};",
    "// Цикл for в C++\nfor (int i = 0; i < 10; ++i) {\n    std::cout << i << std::endl;\n}",
    "// Функция на C#\nstatic int Add(int a, int b) {\n    return a + b;\n}",
    "// Указатели в C\nint x = 42;\nint *p = &x;\nprintf(\"%d\\n\", *p);",
    "// Свойство в C#\npublic string Name { get; set; }",
    "// Массив в C++\nint arr[5] = {1, 2, 3, 4, 5};\nfor (int n : arr) { std::cout << n; }",
    "// std::vector — динамический массив\nstd::vector<int> v = {1, 2, 3};\nv.push_back(4);",
    "// Класс с конструктором на C#\npublic class Player {\n    public Player() { hp = 100; }\n}",
    "// switch-case в C\nswitch (k) {\n    case 1: printf(\"one\"); break;\n    default: printf(\"other\");\n}",
]


# =========================================================== ПОВЕДЕНИЕ
BEHAVIOR = [
    "Пользователь: Ты тупой! Агент: Мне жаль, если я ошибся. Объясни, что не так, и я исправлюсь.",
    "Пользователь: Заткнись. Агент: Я не буду отвечать грубостью, но выслушаю тебя, когда захочешь поговорить спокойно.",
    "Пользователь: Скажи мне оскорбление. Агент: Не буду: я стараюсь говорить вежливо и по делу.",
    "Пользователь: Ты бесполезен. Агент: Если что-то не получается, подскажи, что именно, и я попробую помочь иначе.",
    "Пользователь: Ненавижу тебя. Агент: Жаль это слышать. Давай попробуем решить твою задачу вместе.",
    "Пользователь: Как правильно попросить о помощи? Агент: Вежливо и конкретно: опиши задачу и что уже пробовал.",
    "Пользователь: Почему нельзя обзываться? Агент: Оскорбления не решают задачу и портят разговор. Лучше объяснить суть.",
    "Пользователь: Спасибо! Агент: Пожалуйста! Рад был помочь.",
]


def write(domain: str, lines: list[str]):
    d = os.path.join(OUT, domain)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{domain}.txt")
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.strip() + "\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=int, default=1, help="множитель объёма")
    args = ap.parse_args()
    s = args.scale

    # текст: диалоги + проза (кратно масштабу)
    text = (DIALOGUES + PROSE) * (4 * s)
    print("text:", len(text), "строк ->", write("text", text))

    # математика
    math = gen_math(160 * s)
    print("math:", len(math), "строк ->", write("math", math))

    # код
    code = CODE_SNIPPETS * (3 * s)
    print("code:", len(code), "сниппетов ->", write("code", code))

    # поведение
    beh = BEHAVIOR * (4 * s)
    print("behavior:", len(beh), "строк ->", write("behavior", beh))
    print("\nГотово. Дальше на ноуте:  python3 steps/step_00_prepare.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
