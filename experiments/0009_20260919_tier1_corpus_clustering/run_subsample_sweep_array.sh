#!/bin/bash
#SBATCH --job-name=0009_subsample_sweep
#SBATCH --partition=x-large-andre01
#SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0009_20260919_tier1_corpus_clustering/%A_%a_subsample_sweep.out
#SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0009_20260919_tier1_corpus_clustering/%A_%a_subsample_sweep.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32g
#SBATCH --time=06:00:00
#SBATCH --array=0-2

# k=100固定で、サブサンプル数だけ200,000(既存結果)→50万→100万→200万と振る。
# 「kを増やしても直らなかった。サンプリング密度の方はどうか」を切り分ける実験。
# 既にキャッシュ済みのembeddingを使うので3,441スライドの再読み込みは発生しない。
# propagate_labelsのコストはn_sub線形なので、200万は200kの約10倍時間がかかる
# 想定（GPUメモリ対策でpropagate_chunkもconfig側でサイズに応じて縮めてある）。

set -euo pipefail
cd /workspace/andre01/honzawa/02-playground/toxpatho-patch-selector
export PROJECT_ROOT="$(pwd)"
export EXP_NAME="0009_20260919_tier1_corpus_clustering"

CONFIGS=(config_subsample_500k.yml config_subsample_1m.yml config_subsample_2m.yml)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
.venv/bin/python experiments/${EXP_NAME}/experiment.py --config "${CONFIG}"
