#!/bin/bash
#SBATCH --job-name=0009_precompute_cache
#SBATCH --partition=x-large-andre01
#SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0009_20260919_tier1_corpus_clustering/%j_precompute_cache.out
#SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0009_20260919_tier1_corpus_clustering/%j_precompute_cache.out
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=96g
#SBATCH --time=04:00:00

# kmeans_kスイープ(config_kmeans_k_sweep_part*.yml)が共有するembedding cacheを
# 事前に1回だけ計算する。GPU不要（PCA fit/transformはCPU実装のsklearn、
# h5読み込みも純粋にCPU/IO）。28プロセスで並列化しているのが本体run
# (experiment.py、シングルスレッドで約1.5h)との違い。

set -euo pipefail
cd /workspace/andre01/honzawa/02-playground/toxpatho-patch-selector
.venv/bin/python scripts/precompute_tier1_kmeans_cache.py
