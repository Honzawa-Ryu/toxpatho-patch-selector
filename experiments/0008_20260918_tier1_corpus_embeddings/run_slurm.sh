#!/bin/bash
#SBATCH --job-name=0008_20260918_tier1_corpus_embeddings
#SBATCH --partition=x-large-andre01
#SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0008_20260918_tier1_corpus_embeddings/%A_%a_0008_20260918_tier1_corpus_embeddings.out
#SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0008_20260918_tier1_corpus_embeddings/%A_%a_0008_20260918_tier1_corpus_embeddings.out
#SBATCH --array=0-3
#SBATCH --signal=B:USR1@1080
#SBATCH --export=ALL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=55g
#SBATCH --time=30:00:00
# tier1未埋め込みの2,755枚を4チャンク(689/689/689/688枚)に分け、
# x-large-andre01の4GPUに1チャンクずつ割り当てて並列実行する。
# ノードはGPU4基だがCPU32基/メモリ240gしかないため（0007と同じ理由）、
# 8cpu/55g×4タスク=32cpu/220gに抑える。
# 実績値（0001: 371枚/10.5h/1GPU・20cpu）から見積もると1チャンク689枚は
# 1GPUあたり約20時間。30hの余裕を見て確保。TRIDENTは per-slide の
# smart-resumeがあるので、時間切れで打ち切られても同じ設定で再投入すれば
# 完了済み分はスキップして続きから進む。
# x-large-andre01はtime limitが無制限のパーティションなので、
# --time=30:00:00 でも --partition はそのままでよい。

# 他の実験のジョブに依存させたい場合、有効化してjob_idを埋める
# （job_idは outputs/{依存先exp}/latest_job_id.txt を参照。投入のたびに
#  変わりうる値なので、都度手動で書き換えること）:
# #SBATCH --dependency=afterok:<job_id>

# Array run にする場合、上の3行の --output/--error/この直後の --array を
# 以下の2行に置き換える（%j→%A_%a、--array=0-N を追加。Nの決め方は下記参照）:
# #SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0008_20260918_tier1_corpus_embeddings/%A_%a_0008_20260918_tier1_corpus_embeddings.out
# #SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0008_20260918_tier1_corpus_embeddings/%A_%a_0008_20260918_tier1_corpus_embeddings.out
# #SBATCH --array=0-N
#
# ⚠️ 注意: リソース(--gres/--cpus-per-task/--mem/--time)を変更したら、
#          --partition と --signal のマージンも合わせて手動で見直すこと
#          （make create_exp 実行時に一度だけ計算されたもので、自動追従しない）。
# ⚠️ 注意: シェル上での for/while ループによる複数組み合わせ実行は推奨しない。
#          下記の Array run / Seq run の使用を推奨。

export PROJECT_ROOT="/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector"
export EXP_NAME="0008_20260918_tier1_corpus_embeddings"

# =====================================================
# Storage
# /workspace はNFS（遅い）、/scratch はノード付属のm.2 SSD（速い・ジョブ終了時に
# 自動削除）。デフォルトで有効。NFS越しに直接読み書きしたい場合のみ0にする
# （例: 出力を実行中にリアルタイムで/workspace側から監視したい等）。
# =====================================================

# data/ 全体が2.6TBあり、この実験は各チャンク689枚しか読まないので
# ローカルSSDへの全量rsyncは無駄（0007と同じ理由）。NFS越しに直接読む。
USE_LOCAL_SSD_INPUT=0
USE_LOCAL_SSD_OUTPUT=1

# =====================================================
# python path
# =====================================================

PYTHON_PATH="${PROJECT_ROOT}/experiments/${EXP_NAME}/experiment.py"

# =====================================================
# Single run（デフォルト）
# =====================================================

#RUN_MODE="single"
#RUN_COMMAND="python ${PYTHON_PATH} --config config.yml"

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

RUN_MODE="array"
BASE_COMMAND="python ${PYTHON_PATH} --config config.yml"
GRID_ARGS=(
    "--chunk_id"
)
GRID_VALUES=(
    "0 1 2 3"
)

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
