#!/bin/bash
#SBATCH --job-name=0001_20260907_extract_wsi_embeddings
#SBATCH --partition=x-large-andre01
#SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0001_20260907_extract_wsi_embeddings/%j_0001_20260907_extract_wsi_embeddings.out
#SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0001_20260907_extract_wsi_embeddings/%j_0001_20260907_extract_wsi_embeddings.out
#SBATCH --signal=B:USR1@1728
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=110g
#SBATCH --time=48:00:00
# --time lowered from the 196h template default to a more realistic budget for
# one TRIDENT pass over 219 slides on 1 GPU. Bucket is still >240min so
# --partition stays x-large-andre01 (unchanged); --signal recomputed for the
# new --time via tools/create_exp.sh's own formula (total_seconds/100, min 30):
# 48*3600/100 = 1728.

# 他の実験のジョブに依存させたい場合、有効化してjob_idを埋める
# （job_idは outputs/{依存先exp}/latest_job_id.txt を参照。投入のたびに
#  変わりうる値なので、都度手動で書き換えること）:
# #SBATCH --dependency=afterok:<job_id>

# Array run にする場合、上の3行の --output/--error/この直後の --array を
# 以下の2行に置き換える（%j→%A_%a、--array=0-N を追加。Nの決め方は下記参照）:
# #SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0001_20260907_extract_wsi_embeddings/%A_%a_0001_20260907_extract_wsi_embeddings.out
# #SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0001_20260907_extract_wsi_embeddings/%A_%a_0001_20260907_extract_wsi_embeddings.out
# #SBATCH --array=0-N
#
# ⚠️ 注意: リソース(--gres/--cpus-per-task/--mem/--time)を変更したら、
#          --partition と --signal のマージンも合わせて手動で見直すこと
#          （make create_exp 実行時に一度だけ計算されたもので、自動追従しない）。
# ⚠️ 注意: シェル上での for/while ループによる複数組み合わせ実行は推奨しない。
#          下記の Array run / Seq run の使用を推奨。

export PROJECT_ROOT="/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector"
export EXP_NAME="0001_20260907_extract_wsi_embeddings"

# =====================================================
# Storage
# /workspace はNFS（遅い）、/scratch はノード付属のm.2 SSD（速い・ジョブ終了時に
# 自動削除）。デフォルトで有効。NFS越しに直接読み書きしたい場合のみ0にする
# （例: 出力を実行中にリアルタイムで/workspace側から監視したい等）。
#
# このexperimentでは両方0にオーバーライドしている:
# - INPUT: data/ が156GB。ノードのローカルSSD容量が未知数のため、
#   rsyncでの一括コピーは避けてNFSから直接読む（TRIDENTのタイル読み出しは
#   逐次アクセスなのでNFS越しでもボトルネックになりにくい想定）。
# - OUTPUT: TRIDENTは job_dir 配下に per-slide の進捗（wsi_states/*.json・
#   .lockファイル）を書き、再実行時にそこを見て完了済みslideをスキップする
#   （smart resume）。OUTPUT_ROOT がジョブ実行毎に異なる新規scratchディレクトリ
#   になると、タイムアウトで死んだ場合に前回の進捗を認識できずsmart resumeが
#   機能しない。48時間ジョブでの再実行耐性を優先し、job_dirは常にNFS上の
#   固定パス（outputs/{exp_name}/trident_job/）に直接書く。
# =====================================================

USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=0

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Single run（デフォルト）
# =====================================================

RUN_MODE="single"
RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

# =====================================================
# Array run にしたい場合
#
# 1. 上の RUN_MODE="single" と RUN_COMMAND=... をコメントアウトする
# 2. 下のブロックを有効化する
# 3. ファイル先頭の --output/--error/--array の3行を%A_%a版に切り替える
#    （Nは GRID_VALUES の組み合わせ数-1。make preflight が一致を検証する）
#
# GRID_ARGS[i] と GRID_VALUES[i] が対応し、直積が CONFIGS として展開される。
# 例:
#   GRID_ARGS=("--model" "--dataset")
#   GRID_VALUES=("bert roberta" "pubmed pmc")
#   → --model bert --dataset pubmed / --model bert --dataset pmc / ...
# =====================================================

# RUN_MODE="array"
# BASE_COMMAND="python ${PYTHON_PATH}"
# GRID_ARGS=(
#     "--model"
#     "--dataset"
# )
# GRID_VALUES=(
#     "google/gemma-4-31b-it meta-llama/Llama-3-8b-it"
#     "BC5CDR BIORED"
# )

# =====================================================
# Seq run にしたい場合（1ジョブ内でGRIDを順次実行）
#
# 上と同様に RUN_MODE="seq" にし、BASE_COMMAND/GRID_ARGS/GRID_VALUES を設定する。
# こちらは #SBATCH --array は不要（1ジョブでループするため）。
# =====================================================

# RUN_MODE="seq"
# BASE_COMMAND="python ${PYTHON_PATH}"
# GRID_ARGS=(
#     "--model"
# )
# GRID_VALUES=(
#     "bert roberta"
# )

# =====================================================
# Entry point
# =====================================================

source "${PROJECT_ROOT}/scripts/slurm_entry.sh"
