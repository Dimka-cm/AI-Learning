# Обучение на ноуте (Gigabyte G5 KD, RTX 3060)

Твой ноут — это наша «бесплатная GPU-станция». GitHub Actions (CPU-раннер)
для реального дообучения не хватает, поэтому учим локально, а Actions
используем как диспетчер + хранилище чекпоинтов + витрину прогресса.

## Что у тебя есть (и как мы этим пользуемся)

| Ресурс | Значение | Роль |
| --- | --- | --- |
| CPU | i5-11400H, 6 ядер / 12 потоков | 5 потоков на загрузку данных (`OMP_NUM_THREADS=5`) |
| GPU 0 | Intel UHD (встроенная) | **не трогаем** — она рисует экран |
| **GPU 1** | **RTX 3060 Laptop 6 ГБ** | **вот на ней учим** (`CUDA_VISIBLE_DEVICES=1`) |
| RAM | 16 ГБ DDR4 | хватает (данные читаем порциями по 20–40 МБ) |
| SSD | 512 ГБ | корпуса + чекпоинты |

> ⚠️ **Ловушка нумерации GPU.** В Windows «Диспетчер задач» пишет GPU 0 = Intel UHD,
> GPU 1 = RTX 3060. Но **CUDA (а значит PyTorch) видит только NVIDIA-карты** —
> Intel UHD для CUDA не существует. Поэтому RTX 3060 имеет CUDA-индекс **0**.
> Мой пайплайн это разруливает сам: `--gpu auto` находит дискретную карту по имени,
> а если напишешь `--gpu 1` (по привычке из Windows) — он поймёт, предупредит
> и всё равно возьмёт RTX 3060. Главное, что iGPU остаётся нетронутой.

## Вариант A — просто запустить локально (минимальный путь)

Windows (PowerShell, WSL2 или любой Linux) — одинаково:

```bash
cd AI-Learning
py -m venv .venv                        # Windows / WSL
.venv\Scripts\activate                  # Windows PowerShell
# или Linux/WSL:  source .venv/bin/activate

pip install -r requirements.txt

# CUDA-сборка PyTorch под RTX 3060:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# 1) положи русский текст в data/raw/*.txt (код на C/C++/C# — туда же, на шаг «кода»)
# 2) токенизация в 10 порций:
python steps/step_00_prepare.py

# 3) обучение: шаги 1..10, на RTX 3060, по 3 эпохи за шаг (суммарно до 30)
python orchestrator.py --start 1 --end 10 --device cuda --gpu auto --evaluate
# (--gpu 3060 или --gpu 0 тоже сработают; "1" из Windows-нумерации разрулится сам)
```

Проверка, что учится именно на RTX 3060:

```bash
nvidia-smi                 # здесь видно ТОЛЬКО NVIDIA: одна строка, индекс 0 = RTX 3060
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
# ожидание: True 1 NVIDIA GeForce RTX 3060 Laptop GPU
```

Запуск с явным выбором карты (модель/имя вместо индекса):

```bash
python orchestrator.py --start 1 --end 10 --device cuda --gpu auto --evaluate
# или --gpu 3060 (подстрока имени) / --gpu 0 (CUDA-индекс)
```

## Вариант B — ноут как self-hosted runner (автоматизация)

Тогда сам GitHub запускает обучение на твоём ноуте, и прогресс
(таблица loss + график) виден прямо в интерфейсе Actions (Job Summary
и артефакт `training-progress`).

1. **Скачай и установи runner** (Windows x64), привяжи к репозиторию:
   - репозиторий → **Settings → Actions → Runners → New self-hosted runner**;
   - скачай архив, распакуй, выполни `config.cmd` с `--labels self-hosted,windows,gpu`

2. **Запусти runner** и проверь, что он виден как `Idle`.

3. **Ограничь раннер на GPU 1** — чтобы Actions-обучение случайно не село на GPU 0
   или не вспомнило iGPU. В workflow уже задаётся `CUDA_VISIBLE_DEVICES: 1`.

4. **Запускай вручную**: репозиторий → **Actions → Train + Progress → Run workflow**
   и выбери `runner = self-hosted`. Обучение стартует на ноуте, прогресс — в интерфейсе.

> ⚠️ GPU-hosted раннеры GitHub — платные и только на Team/Enterprise.
> Поэтому GPU мы берём с твоего ноута, а Actions лишь дирижирует.

## Про эпохи: сколько реально сыпется на RTX 3060

Дано: модель `rugpt3small` (125M), смешанная точность (fp16), `block_size = 1024`,
`batch_size = 4`, `grad_accum = 8` (эффективный батч 32), gradient checkpointing.

- Скорость RTX 3060 Laptop (~10 TFLOPS FP16): **~3–6 тыс. токенов/с**.
- Порция 100 МБ текста ≈ ~25–35 млн токенов → **1 эпоха ≈ 2–3 часа**.
- **20–30 эпох суммарно = 45–90 часов** (это 2–4 суток, не «за один вечер»).

Поэтому правильный цикл — **не 30 эпох за раз**, а:
**запуск = 2–3 эпохи (~6–9 часов) → чекпоинт → возобновление со следующего**.
Тёплый ноут, безопасный нагрев, нет потери при сбое.

Именно так устроен пайплайн: каждый шаг резюмится с `checkpoints/step_{N-1}/last.pt`.

## Нагрев / режим ноута

- Держи блок питания подключённым (на батарее CUDA режется).
- Поставь охлаждение: подставка, не на одеяле; следи за `nvidia-smi` → temp < 85 °C.
- Если хочешь фоновый прогон, пока работаешь — это ок: мы трогаем только GPU 1
  и 5 CPU-потоков, GPU 0 (экран) и 1 ядро остаются тебе.
