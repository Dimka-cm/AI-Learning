# Пакет исходников пайплайна обучения.
#
# Санитизация окружения (OMP_NUM_THREADS=auto ломает libgomp) делается при
# импорте пакета — это первое, что подтягивают шаги обучения. См. src/threads.py.
from .threads import fix_thread_env

fix_thread_env()
