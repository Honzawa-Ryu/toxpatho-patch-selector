#!/bin/bash
# 0012の全段を依存関係つきで投入する。各段は run_slurm.sh（テンプレートのarray機構・
# Apptainer経由）をそのまま使い、段ごとのリソースだけをsbatchのCLIで上書きする。
#
# タスク数は `experiment.py --stage count` の出力と一致させること（不一致なら
# experiment.py側がN_TASKSを照合して停止する）:
#   prepare 7 (0-5: 部分集合S0-S4,L0 / 6: reference) / cluster 100 (0-83: k-means系,
#   84-98: Leiden, 99: Ward) / consensus 5 / evaluate 111 / summarize 1
#
# 使い方: bash experiments/0012_.../submit_all.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../../logs/0012_20260928_compare_clustering_methods_at_scale

P=x-large-andre01
sub() {  # sub <STAGE> <N_TASKS> <sbatch options...>  → job idを返す
    local stage=$1 n=$2; shift 2
    sbatch --parsable -p "$P" --export=ALL,STAGE="$stage",N_TASKS="$n" \
        --job-name="0012_${stage}" "$@" run_slurm.sh
}

# 1. 部分集合＋全patch kNN（GPU）と、reference（CPU）
PREP=$(sub prepare 7 --array=0-5 --gres=gpu:1 -c 2 --mem=16g --time=03:00:00)
REF=$(sub prepare 7 --array=6 -c 1 --mem=16g --time=01:00:00)

# 2. クラスタリング: k-means系とWardはGPU、LeidenはCPUのみ
KM=$(sub cluster 100 --array=0-83,99 --gres=gpu:1 -c 2 --mem=16g --time=04:00:00 --dependency=afterok:$PREP)
LD=$(sub cluster 100 --array=84-98%2 -c 1 --mem=12g --time=06:00:00 --dependency=afterok:$PREP)

# 3. コンセンサス（S0上のk-means全60runが揃ってから）
CO=$(sub consensus 5 --array=0-4%2 -c 1 --mem=12g --time=06:00:00 --dependency=afterok:$KM)

# 4. 評価（上流が一部失敗しても残りは評価する。欠けた上流は各タスクが明示的に止まる）
EV=$(sub evaluate 111 --array=0-110%4 -c 1 --mem=16g --time=03:00:00 \
    --dependency=afterany:$KM:$LD:$CO:$REF)

# 5. 集約とseed間安定性
SU=$(sub summarize 1 --array=0 -c 1 --mem=16g --time=06:00:00 --dependency=afterany:$EV)

echo "prepare=$PREP reference=$REF kmeans+ward=$KM leiden=$LD consensus=$CO evaluate=$EV summarize=$SU"
echo "$PREP $REF $KM $LD $CO $EV $SU" > ../../outputs/0012_20260928_compare_clustering_methods_at_scale/latest_job_ids.txt
