#!/usr/bin/env bash
# Train exactly one Google Drive corpus part: library/merged_data_N.txt.
# Expects a resume checkpoint already restored under checkpoints/** when this is
# not the first part.
set -Eeuo pipefail

PART="${1:?part number required}"
DRIVE_FOLDER_URL="${DRIVE_FOLDER_URL:?DRIVE_FOLDER_URL is required}"
DATA_LIMIT_MB="${DATA_LIMIT_MB:-2000}"
DRIVE_BUDGET_S="${DRIVE_BUDGET_S:-3600}"
START_STEP="${START_STEP:-1}"
END_STEP="${END_STEP:-4}"
MAX_STEPS="${MAX_STEPS:-300}"
EVAL_LIMIT="${EVAL_LIMIT:-4}"

export DATA_SOURCE=drive
export DRIVE_FOLDER_URL
export DRIVE_INCLUDE_REGEX="^merged_data_${PART}\\.txt$"
export DRIVE_MAX_FILES=1
export LIMIT_MB="$DATA_LIMIT_MB"
export DRIVE_BUDGET_S
export REQUIRE_REAL_DATA=1

export START_STEP
export END_STEP
export MAX_STEPS
export EPOCHS_PER_STEP=1
export LOG_EVERY_STEPS=5
export NUM_WORKERS=auto
export RESUME_FROM_LATEST=1

# Fresh data/shards/logs for this exact part; keep checkpoints for resume.
rm -rf data/raw data/shards logs
mkdir -p logs

echo "[chain] part=${PART} regex=${DRIVE_INCLUDE_REGEX} limit=${LIMIT_MB}MB steps=${START_STEP}..${END_STEP} max_steps=${MAX_STEPS}"
echo "[chain] existing checkpoints before data:"
find checkpoints -maxdepth 3 -type f -name 'last*.pt' -printf '  %p %s bytes\n' 2>/dev/null || true

python steps/step_00_ci_data.py

NPROC="$(nproc)"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NPROC}"
echo "[chain] runner: ${NPROC} CPU, $(free -m | awk '/Mem:/{print $2}') MB RAM"
python orchestrator.py --start "$START_STEP" --end "$END_STEP" --device cpu --evaluate

# Не тратим много времени на eval после каждой части, но короткий отчёт полезен.
if compgen -G "checkpoints/step_*/last.pt" > /dev/null; then
  python scripts/evaluate_prompts.py --limit "$EVAL_LIMIT" --out logs/eval_answers.md || true
fi

python - <<'PY'
from pathlib import Path
import tarfile

cands = sorted(Path('checkpoints').glob('step_*/last.pt'), key=lambda p: p.stat().st_mtime)
if not cands:
    raise SystemExit('[chain] no checkpoints/step_*/last.pt to package')
latest = cands[-1]
print(f'[chain] latest checkpoint: {latest} ({latest.stat().st_size / 1e6:.1f} MB)')
with tarfile.open('chain-checkpoint.tar.gz', 'w:gz') as tar:
    tar.add(latest, arcname=str(latest))
PY

ls -lh chain-checkpoint.tar.gz
