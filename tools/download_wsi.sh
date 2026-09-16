#!/bin/bash
#SBATCH --job-name=download_wsi
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --output=logs/download-wsi_%j.out
#SBATCH --error=logs/download-wsi_%j.out

# Open TG-GATEs の Liver WSI を追加取得するジョブ。
# tools/uv_sync.sh と同じく、SLURM_SUBMIT_DIR をプロジェクトルートとして
# Apptainer 内で走らせる（GPUは不要なので --nv は付けない）。
#
# Usage:
#   sbatch --partition=x-large-andre01 tools/download_wsi.sh [-- <script args>]
#
# 追加引数はそのまま scripts/download_tggates_liver_wsi.py に渡す。
#   例: sbatch --partition=x-large-andre01 tools/download_wsi.sh --dose-levels High
#
# 途中で落ちても .part から再開できるので、同じコマンドで再投入してよい。

set -uo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR}"

if [ -f "${PROJECT_ROOT}/.env" ]; then
    set -a
    source "${PROJECT_ROOT}/.env"
    set +a
fi
SIF_PATH="${SIF_PATH:-${PROJECT_ROOT}/env/env.sif}"

# `--periods "4 day"` のように空白を含む引数が来るので、bash -c に渡す前に
# printf %q でクォートしておく（素の $* だと単語分割されて壊れる）。
SCRIPT_ARGS=$(printf '%q ' "$@")

echo "=========================================="
echo "Downloading TG-GATEs Liver WSIs"
echo "Project Root: ${PROJECT_ROOT}"
echo "Args        : $*"
echo "Started     : $(date --iso-8601=seconds)"
echo "=========================================="

if apptainer exec --bind "${PROJECT_ROOT}" "${SIF_PATH}" bash -c "
    set -euo pipefail
    cd ${PROJECT_ROOT}
    source ${PROJECT_ROOT}/.venv/bin/activate
    python scripts/download_tggates_liver_wsi.py ${SCRIPT_ARGS}
"; then
    echo "=========================================="
    echo "Download finished: $(date --iso-8601=seconds)"
    df -h "${PROJECT_ROOT}/data" | tail -1
    echo "=========================================="
else
    exit_code=$?
    echo "❌ Download failed (exit code: ${exit_code})." >&2
    echo "   同じコマンドで再投入すれば、取得済みはスキップされ .part から再開されます。" >&2
    exit "${exit_code}"
fi
