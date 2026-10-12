# 保存済み集計によるVoteFractionヒストグラム

0013の `vote_steps/vote_step_table.csv` を図にした。
投票の再計算や画像の再抽出は行っていない。

- [全17所見を合計したヒストグラム](vote_fraction_pooled.png) / [PDF](vote_fraction_pooled.pdf)
- [所見別のパッチ数](vote_fraction_by_finding_counts.png) / [PDF](vote_fraction_by_finding_counts.pdf)
- [所見別の割合](vote_fraction_by_finding_percent.png) / [PDF](vote_fraction_by_finding_percent.pdf)

VoteFractionは、1パッチについて最も票を集めた所見の票数 / 110 run。
所見別のパネルは、その最多所見によってパッチを振り分けている。
スライドの所見ラベルによる分類や、パッチの確定診断ではない。

対象は0012の固定100万パッチ標本のうち、監査済みスライドに属し、VoteFractionが0.05以上のパッチ。
さらに、VoteFraction 0.1以上のパッチが100個以上ある17所見だけが保存済み表に含まれる。
全体図も、この17所見の合計であり、全パッチ・全所見の分布ではない。

パッチ数の図は所見ごとに縦軸を変えている。所見間の形を比較する場合は割合の図を使う。
割合の分母は、その所見で表示されるVoteFraction 0.05以上のパッチ数。
横軸の最初の区間は0.05〜0.1、その後は0.1刻み。
最後の区間は保存済み表の `[0.9,1.01)` を0.9〜1.0として表示し、1.0を含む。
棒の高さは区間内の件数または割合で、区間幅で割った密度ではない。

110 runには手法とkの異なる設定が混在する。VoteFractionを同一条件の再現率や診断の確信度として扱わない。
閾値はこの図から自動で決めていない。元の分析条件は [0013のREADME](../README.md) を参照。

`finding_summary.csv` に所見別の表示パッチ数と、VoteFraction 0.5以上・0.9以上の件数と割合を保存。
`source_bin_counts.csv` に入力集計、`pooled_bin_counts.csv` に合計集計を保存。
各所見の区間件数を保存済み累積件数と照合し、合計図と所見別図の総件数の一致も確認した。
入力とスクリプトのSHA256、対象範囲は `provenance.json` に記録している。

再現コマンド：

```bash
.venv/bin/python scripts/plot_vote_fraction_histograms.py
```
