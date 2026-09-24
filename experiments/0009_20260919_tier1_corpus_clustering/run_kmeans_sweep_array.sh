#!/bin/bash
#SBATCH --job-name=0009_kmeans_sweep
#SBATCH --partition=x-large-andre01
#SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0009_20260919_tier1_corpus_clustering/%A_%a_kmeans_sweep.out
#SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0009_20260919_tier1_corpus_clustering/%A_%a_kmeans_sweep.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32g
#SBATCH --time=06:00:00
#SBATCH --array=0-3

# kmeans_k=100..2000(100刻み,20通り)を4パート(5個ずつ)に分け、ノード内の
# 4GPUにそれぞれ1パートを割り当てて並列実行する。各パートは
# run_precompute_cache.sh が書いたembedding cacheを読むだけなので、
# 3,441スライドの読み込み+PCA投影(本体run実測で約1.5h・GPU未使用)を
# 4回繰り返さずに済む。

set -euo pipefail
cd /workspace/andre01/honzawa/02-playground/toxpatho-patch-selector
export PROJECT_ROOT="$(pwd)"
export EXP_NAME="0009_20260919_tier1_corpus_clustering"

CONFIG="config_kmeans_k_sweep_part${SLURM_ARRAY_TASK_ID}.yml"
.venv/bin/python experiments/${EXP_NAME}/experiment.py --config "${CONFIG}"
