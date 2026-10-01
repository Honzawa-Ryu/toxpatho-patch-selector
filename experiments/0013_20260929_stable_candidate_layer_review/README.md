# 0013: run横断の投票率による候補patchの層分けと目視確認 — 2026-10-01

## 要点

- 0012の110 run（reference以外）で、各patchが同じ所見の候補クラスタに入ったrunの割合を「投票率」とした（分母は110で固定）。最多の所見をそのpatchの所見とする。
- **選定の指標は投票率（分母110）だけにする。** 所見ごとに閾値を決める方針で、閾値はまだ決めていない。
- q値（候補クラスタのq値のrun間中央値）と投票率は全体で順位相関 -0.42（patch単位）、-0.23（slide単位）。投票率はクラスタの大きさとも +0.44 で相関し、両者は分離できていない。所見ごとに向きも違う。目視でもq値の高低で像がはっきり分かれる所見はなく、**q値は選定に使わない。**
- 一貫性（最多所見の回数 / 何らかの候補になった回数）も見た。所見と多少対応はするが切り分けは弱く、指標がわかりにくくなるため**採用しない**。シートの色分けとしてだけ残す。

## 条件

- 投票の対象は0012の固定100万patch標本（`prepare_reference/stability_idx.npy`）。集計とシートは所見ラベル確定済みスライド（0010の`manifest_known_unique.parquet`）に限る。
- 各runの候補は0012の方式A（`eval_*/candidates.csv`）。1 runでは1クラスタに1所見なので、1 runでの所見は1 patchにつき1つ。run間で複数所見に票が入りうる。同票はargmaxによりアルファベット順で先の所見になる。
- シートはスライドを一様に選んでから1スライド最大2枚を取る。枠の色は緑＝所見ありスライド、赤＝なし。
- 組織割合フィルタはかけていない（`tiles.csv`に記録のみ）。

## 実行モード

`EXP_ARGS` で引数を渡す（未指定なら `--layer ${LAYER:-L1}`）。

| モード | 実行例 | 出力 |
|---|---|---|
| 層 | `LAYER=L2 sbatch run_slurm.sh` | `L1/`, `L2/`: 層 [lo,hi) ごとの所見別シート24枚 |
| q値 | `EXP_ARGS="--analysis q_vote" sbatch run_slurm.sh` | `q_vote/`: 相関表、投票率区間ごとのq分布、低q/高q四分位のシート |
| 0.1刻み | `EXP_ARGS="--analysis vote_steps" sbatch run_slurm.sh` | `vote_steps/`: 所見×区間の表（区間・累積）、1所見1枚・1区間12枚のシート（見出しの帯が一貫性） |

L3（0.05〜0.5）は未実行。`vote_steps` がこの範囲を含むため不要と考えている。

## 限界

- 110 runは手法・粒度が混在しており（MiniBatch 48, Lloyd 36, Leiden 15, Ward 6, コンセンサス 5）、投票率は同一条件の再現率ではない。k-means系の影響が大きい。
- 方式Aは化合物数を制約しないため、化合物を識別しているだけのクラスタも候補になる（脂肪変性≒CCl4、核の変化＝ethambutol）。
- 閾値を同じデータの目視で決めるため、決めた閾値は探索的なもの。別の化合物・実験での確認が必要。

## 成果物

`outputs/0013_20260929_stable_candidate_layer_review/`

- `L1/`, `L2/`: `layer_summary.csv`, `tiles.csv`, `sheets/`。`L2/finding_threshold_table.csv` は閾値0.9〜0.1の累積集計（L1の後に追加したのでL1にはない）
- `q_vote/`: `q_vote_correlation.csv`, `q_by_vote_bin.csv`, `tiles.csv`, `sheets/`
- `vote_steps/`: `vote_step_table.csv`, `tiles.csv`（一貫性・2番目の所見つき）, `sheets/`

q_voteの初回（job 11881）はq値をfloat32に入れて1e-38未満が0になっていたため、log10にしてから保存するよう直して再実行した（job 11884）。
