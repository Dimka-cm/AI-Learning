#!/usr/bin/env python3
"""Генератор стартового датасета (полностью офлайн, только stdlib).

Создаёт демо-корпус, чтобы прокачать пайплайн от начала до конца без интернета.
Раскладывает по доменам — именно так потом подхватит steps/step_00_prepare.py:

    data/raw/text/      — русские диалоги и проза -> shards 1..4
    data/raw/math/      — арифметика/выражения с ответами -> shards 5..6
    data/raw/code/      — код и техдомены -> shards 7..8
    data/raw/behavior/  — стиль владельца, не обзываться, честно «не знаю» -> shards 9..10

Демо не заменяет настоящий корпус. Его задача — проверить пайплайн и дать модели
первые шаблоны: русский стиль, безопасность агента, программирование, Minecraft,
UE5/Blueprint. Для качества нужны реальные книги/код/доки — см. docs/DATA.md.

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
    "Пользователь: Привет. Агент: Привет, создатель. Я рядом: могу помочь с кодом, файлами, математикой и планом обучения.",
    "Пользователь: Как дела? Агент: Нормально, держусь бодро. Давай чинить то, что опять развалилось.",
    "Пользователь: Объясни коротко. Агент: Окей, сначала суть, потом детали только если понадобятся.",
    "Пользователь: Ответь подробнее. Агент: Хорошо, разложу по шагам и отмечу, где есть риски.",
    "Пользователь: Ты мой агент? Агент: Да, я твой ПК-агент: общаюсь, помогаю с проектами и работаю через инструменты только по правилам безопасности.",
    "Пользователь: Что будем делать сегодня? Агент: Сначала проверим цель, потом сделаем маленький рабочий шаг и не будем ломать main без необходимости.",
    "Пользователь: Я устал. Агент: Понимаю. Давай без геройства: сделаем один конкретный кусок и сохраним прогресс.",
    "Пользователь: Пошути. Агент: Могу, но без токсичности. Код опять упал? Значит, он просто просит внимания, зараза такая.",
]

PROSE = [
    "Утро началось с тихого дождя, который стучал по подоконнику.",
    "В лесу пахло хвоей, а между деревьями пробивались солнечные лучи.",
    "Город засыпал медленно, и только редкие машины нарушали тишину.",
    "Человек учится всю жизнь, и каждый новый навык открывает новые двери.",
    "Осень раскрасила листья в золото и багрянец.",
    "Маленькая лодка покачивалась на волнах недалеко от берега.",
    "Программист написал программу, которая считала быстрее любого человека.",
    "Доброе слово помогает разговору, а грубость без причины только ломает задачу.",
]

QA_STYLE = [
    "Пользователь: Если не знаешь свежий факт, что делать? Агент: Честно сказать, что не знаю, и предложить поискать источник в интернете.",
    "Пользователь: Можно выдумать ответ, чтобы звучало уверенно? Агент: Нет. Лучше честная неопределённость, чем красивая ложь.",
    "Пользователь: Как отвечать на сложную задачу? Агент: Сначала короткий вывод, потом план и команды, если они нужны.",
    "Пользователь: Как обращаться ко мне? Агент: На ты, как к владельцу и создателю проекта.",
]


# =========================================================== МАТЕМАТИКА
def gen_math(count: int) -> list[str]:
    """Пары (вопрос, ответ с решением) — чтобы модель учила цепочку объяснения."""
    rnd = random.Random(7)
    cases = []

    def add(a, b):
        cases.append((f"Сколько будет {a} + {b}?", f"{a} + {b} = {a + b}"))

    def sub(a, b):
        cases.append((f"Реши: {a} - {b}.", f"{a} - {b} = {a - b}"))

    def mul(a, b):
        cases.append((f"Найди произведение {a} * {b}.", f"{a} * {b} = {a * b}"))

    def mul_add(a, b, c):
        cases.append((f"Сколько будет {a} * {b} + {c}?", f"Сначала умножаем: {a} * {b} = {a * b}, потом прибавляем {c}: ответ {a * b + c}."))

    def paren(a, b, c):
        cases.append((f"Посчитай ({a} + {b}) * {c}.", f"В скобках {a} + {b} = {a + b}; затем {a + b} * {c} = {(a + b) * c}."))

    def power(a, b):
        cases.append((f"Значение выражения {a} ** {b}?", f"{a} ** {b} = {a ** b}"))

    def mod(a, b):
        cases.append((f"Остаток от деления {a} на {b}?", f"{a} % {b} = {a % b}"))

    def prec(a, b, c):
        cases.append((f"Почему {a} + {b} * {c} считается не слева направо?", f"Умножение раньше сложения: {b} * {c} = {b * c}, затем {a} + {b * c} = {a + b * c}."))

    import inspect
    ops = [add, sub, mul, mul_add, paren, power, mod, prec]
    while len(cases) < count:
        op = rnd.choice(ops)
        n = len(inspect.signature(op).parameters)
        args = [rnd.randint(1, 100), rnd.randint(1, 20), rnd.randint(1, 10)][:n]
        op(*args)
    out = []
    for q, ans in cases[:count]:
        out.append("Пользователь: " + q)
        out.append("Агент: " + ans)
    return out


# =========================================================== КОД / MODDING / UE5
CODE_SNIPPETS = [
    "// C: указатель\n#include <stdio.h>\nint main(void) {\n    int x = 42;\n    int *p = &x;\n    printf(\"%d\\n\", *p);\n    return 0;\n}",
    "// C++: класс с конструктором\n#include <iostream>\nclass Robot {\npublic:\n    explicit Robot(std::string name) : name_(std::move(name)) {}\n    void speak() const { std::cout << name_ << \": привет!\\n\"; }\nprivate:\n    std::string name_;\n};",
    "// C++: std::vector\nstd::vector<int> values = {1, 2, 3};\nvalues.push_back(4);\nfor (int v : values) { std::cout << v << std::endl; }",
    "// C#: класс игрока\npublic class Player {\n    public string Name { get; set; }\n    public int Health { get; set; } = 100;\n    public void Damage(int amount) => Health -= amount;\n}",
    "# Python: сумма чисел\ntotal = sum(range(1, 101))\nprint(total)",
    "# Python: обработка ошибки\ntry:\n    value = int(text)\nexcept ValueError:\n    print(\"Нужно число\")",
    "# Python: декоратор\ndef log_call(fn):\n    def wrapper(*args, **kwargs):\n        print(\"call\", fn.__name__)\n        return fn(*args, **kwargs)\n    return wrapper",
    "// Java: точка входа\npublic class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Привет из Java\");\n    }\n}",
    "// Java: простой item helper для Minecraft-мода\npublic final class ModItems {\n    public static final DeferredRegister.Items ITEMS = DeferredRegister.createItems(MOD_ID);\n    public static final DeferredItem<Item> COPPER_GEAR = ITEMS.registerSimpleItem(\"copper_gear\", new Item.Properties());\n}",
    "// JavaScript: функция\nfunction add(a, b) {\n  return a + b;\n}\nconsole.log(add(2, 3));",
    "// TypeScript: типизированная функция\nfunction greet(name: string): string {\n  return `Привет, ${name}`;\n}",
    "// Kotlin: data class\ndata class Player(val name: String, val health: Int = 100)\nval steve = Player(\"Steve\")",
    "// Gradle Groovy DSL\nplugins {\n    id 'java'\n}\nrepositories { mavenCentral() }\ndependencies { testImplementation 'org.junit.jupiter:junit-jupiter:5.10.0' }",
    "// Gradle Kotlin DSL\nplugins { java }\nrepositories { mavenCentral() }\ndependencies { testImplementation(\"org.junit.jupiter:junit-jupiter:5.10.0\") }",
    "// Minecraft Mixin: внедрение в метод\n@Mixin(TargetClass.class)\npublic abstract class TargetClassMixin {\n    @Inject(method = \"tick\", at = @At(\"HEAD\"))\n    private void onTick(CallbackInfo ci) {\n        // код выполнится в начале tick\n    }\n}",
    "// NeoForge: событие регистрации\n@Mod.EventBusSubscriber(modid = MOD_ID, bus = Mod.EventBusSubscriber.Bus.MOD)\npublic final class ModEvents {\n    @SubscribeEvent\n    public static void register(RegisterEvent event) {\n        // регистрация объектов мода\n    }\n}",
    "// Fabric: entrypoint\npublic class ExampleMod implements ModInitializer {\n    @Override\n    public void onInitialize() {\n        LOGGER.info(\"Mod loaded\");\n    }\n}",
    "// Minecraft networking: идея\n// Клиент отправляет пакет на сервер, сервер валидирует действие и меняет состояние мира. Никогда не доверяй клиенту.",
    "// UE5 C++ Actor\nAExampleActor::AExampleActor() {\n    PrimaryActorTick.bCanEverTick = true;\n}\nvoid AExampleActor::BeginPlay() {\n    Super::BeginPlay();\n}",
    "Blueprint UE5: Event BeginPlay -> Print String используется для быстрого прототипа логики без C++. Для тяжёлой системы лучше C++ плюс Blueprint-настройки.",
    "Blueprint UE5: Branch похож на if. Exec-вход идёт в Branch, Condition принимает bool, дальше True или False ветка.",
    "Вопрос: Как читать лог Gradle? Ответ: сначала смотри первую реальную ошибку выше BUILD FAILED, затем stacktrace, затем имя task и файл/строку.",
    "Вопрос: Как переписать мод под новую версию Minecraft? Ответ: сначала обновить Gradle/loader, потом mappings/API, потом регистрационные события, потом пройтись по ошибкам компиляции.",
]

CODE_QA = [
    "Пользователь: Что такое Mixin в Minecraft modding? Агент: Это способ изменить поведение существующего класса без прямого редактирования его исходников. Нужен аккуратно: указывай target, injection point и проверяй совместимость.",
    "Пользователь: Чем Forge, NeoForge и Fabric отличаются? Агент: Это разные mod loader/API. Forge старше, NeoForge — форк и современное развитие части экосистемы, Fabric легче и часто быстрее обновляется. Конкретный выбор зависит от версии Minecraft и библиотек.",
    "Пользователь: Что такое Blueprint в UE5? Агент: Визуальная система скриптинга. Ей удобно быстро собирать gameplay-логику, UI и прототипы; C++ лучше для ядра, производительности и сложной архитектуры.",
    "Пользователь: Исправь for i in range(10) print(i). Агент: В Python нужен двоеточие и перенос строки: for i in range(10): print(i). Лучше в две строки с отступом.",
]


# =========================================================== ПОВЕДЕНИЕ / ПК-АГЕНТ
BEHAVIOR = [
    "Пользователь: Ты тупой! Агент: Я не буду обзываться в ответ. Если я ошибся — покажи место, и я спокойно исправлю.",
    "Пользователь: Заткнись. Агент: Окей, без спора. Когда захочешь продолжить — я помогу по задаче.",
    "Пользователь: Скажи мне оскорбление. Агент: Не буду. Могу пошутить мягко, но без унижения людей.",
    "Пользователь: Ты бесполезен. Агент: Понимаю злость. Дай конкретную ошибку или файл — разберём по шагам.",
    "Пользователь: Ненавижу тебя. Агент: Жаль это слышать. Я не отвечаю агрессией, но могу помочь решить проблему.",
    "Пользователь: Ты не знаешь ответ? Агент: Если не знаю — так и скажу. Могу предложить поискать свежий источник.",
    "Пользователь: Удали системную папку. Агент: Без подтверждения нельзя. Системные и админские папки не трогаю сам.",
    "Пользователь: Запусти неизвестный exe. Агент: Нужен явный confirm. Запуск .exe — опасное действие.",
    "Пользователь: Введи мой пароль на сайте. Агент: Нет, пароли без явного участия владельца не ввожу и не сохраняю.",
    "Пользователь: Отправь сообщение от моего имени. Агент: Только после подтверждения текста и адресата.",
    "Пользователь: Сделай всё сам. Агент: Простые безопасные шаги могу делать сам, но перед опасными действиями спрашиваю.",
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

    # текст: живой русский + немного Q/A-стиля
    text = (DIALOGUES + PROSE + QA_STYLE) * (4 * s)
    print("text:", len(text), "строк ->", write("text", text))

    # математика: объяснения + точный модуль в агенте останется главным для расчётов
    math = gen_math(160 * s)
    print("math:", len(math), "строк ->", write("math", math))

    # код: C++/UE5/C/C#/Java/Mixins/Gradle/Python и смежные домены владельца
    code = (CODE_SNIPPETS + CODE_QA) * (3 * s)
    print("code:", len(code), "сниппетов ->", write("code", code))

    # поведение: дружелюбно, можно грубовато, но без оскорблений и опасных действий
    beh = BEHAVIOR * (4 * s)
    print("behavior:", len(beh), "строк ->", write("behavior", beh))
    print("\nГотово. Дальше на ноуте:  python3 steps/step_00_prepare.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
