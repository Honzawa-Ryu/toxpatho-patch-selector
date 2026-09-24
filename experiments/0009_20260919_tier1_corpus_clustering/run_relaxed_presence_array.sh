#!/bin/bash
#SBATCH --job-name=0009_relaxed_presence
#SBATCH --partition=x-large-andre01
#SBATCH --output=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0009_20260919_tier1_corpus_clustering/%A_%a_relaxed_presence.out
#SBATCH --error=/workspace/andre01/honzawa/02-playground/toxpatho-patch-selector/logs/0009_20260919_tier1_corpus_clustering/%A_%a_relaxed_presence.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32g
#SBATCH --time=06:00:00
#SBATCH --array=0-2

# min_treated_presence を 0.05 / 0.02 / 0.01 の3水準、kを300/500/1000の3水準で振る。
#
# 背景: 投与群2,570枚/44化合物なので1化合物あたり2.27%しかなく、従来の0.1は
# 「4〜5化合物にまたがること」を要求していた(0006の219枚・8化合物コーパスでは
# 1化合物≒11%だったので0.1はほぼ1化合物分の意味だった)。所見特異的なクラスタ
# ほど落ちる設計になっていたのを、化合物比率に合わせて緩める。
#
# kは幾何学的には決まらない(embedding空間が連続体でエルボーが立たない)ため、
# LOCO AUROC=未知化合物への転移性能が最大になるkを事後的に選ぶ。

set -euo pipefail
cd /workspace/andre01/honzawa/02-playground/toxpatho-patch-selector
export PROJECT_ROOT="$(pwd)"
export EXP_NAME="0009_20260919_tier1_corpus_clustering"

CONFIGS=(config_relaxed_p05.yml config_relaxed_p02.yml config_relaxed_p01.yml)
CONFIG="${CONFIGS[$SLURM_ARRAY_TASK_ID]}"
.venv/bin/python experiments/${EXP_NAME}/experiment.py --config "${CONFIG}"
