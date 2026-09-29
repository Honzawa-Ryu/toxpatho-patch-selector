#!/bin/bash
# 2026-09-29追加分: Leiden・コンセンサスの細粒度（r=512,1024）だけを投入する。
# タスク番号はconfig.ymlの現在の展開（cluster 106 / consensus 7 / evaluate 119）に対応する:
#   cluster 99-104 = leiden r512/r1024 × seed0-2, consensus 5-6 = r512/r1024,
#   evaluate 100-105 = 上記leiden, 117-118 = 上記consensus
# 既存の段は再実行しない。同時実行は最大3枠に収める（各ジョブがGPU1枚を占有するため）。
set -euo pipefail
cd "$(dirname "$0")"
P=x-large-andre01
sub() {  # sub <STAGE> <N_TASKS> <sbatch options...>
    local stage=$1 n=$2; shift 2
    sbatch --parsable -p "$P" --export=ALL,STAGE="$stage",N_TASKS="$n" --job-name="0012f_${stage}" "$@" run_slurm.sh
}
LD=$(sub cluster 106 --array=99-104%2 -c 1 --mem=12g --time=08:00:00)
CO=$(sub consensus 7 --array=5-6%1 -c 1 --mem=12g --time=06:00:00)
EV1=$(sub evaluate 119 --array=100-105%1 -c 1 --mem=16g --time=03:00:00 --dependency=afterany:$LD)
EV2=$(sub evaluate 119 --array=117-118%1 -c 1 --mem=16g --time=03:00:00 --dependency=afterany:$CO)
SU=$(sub summarize 1 --array=0 -c 1 --mem=16g --time=06:00:00 --dependency=afterany:$EV1:$EV2)
echo "leiden=$LD consensus=$CO eval_leiden=$EV1 eval_consensus=$EV2 summarize=$SU"
echo "$LD $CO $EV1 $EV2 $SU" > ../../outputs/0012_20260928_compare_clustering_methods_at_scale/latest_job_ids_fine.txt
