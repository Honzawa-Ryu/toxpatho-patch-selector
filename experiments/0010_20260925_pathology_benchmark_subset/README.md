# 病理所見評価用subset v1 — 2026-09-25

追記: レビュー後のラベル監査・修正・同用量subset v2・一次目視の結果は
[FOLLOWUP.md](FOLLOWUP.md) を参照。以下はv1作成時点の記録。

## 結論と到達点

候補3所見をtier1の3,441スライドで監査し、**顆粒状好酸性変性
（Degeneration, granular, eosinophilic）@ Hepatocyte**の比較用90枚を固定した。
新しいクラスタリングは実行していない。候補の選択・slide選定にはクラスタ番号や性能を使っていない。

これは**メタデータに基づく暫定評価subset**であり、交絡のないbenchmarkとしての最終承認ではない。
compound / experiment の単一条件への集中は抑えたが、facilityは不明、用量・併存所見の交絡は残る。
元のDone Criteriaのうち「施設への極端な依存がないことの確認」は未達。
また、スライドの既存ラベルはパッチの診断や目視正解ラベルを意味しない。

## 候補の比較

以下は元コーパスのユニークslide数。病理所見の行数ではない。
同じ所見が複数部位に記録される場合もslideは1枚と数えた。

| Finding | Compound数 | Experiment数 | 陽性slides | 最大compound比率 | 判断 |
|---|---:|---:|---:|---:|---|
| Necrosis | 23 | 23 | 160 | 19.4% | 保留。肝細胞・胆管細胞・部位不詳等が混在し、解剖学的定義を先に絞る必要がある |
| Degeneration, fatty | 2 | 2 | 80 | 75.0% | 保留。CCl4はCentrilobular、cholesterolはPeripheralでcompoundと部位が完全に対応 |
| Degeneration, granular, eosinophilic | 6 | 6 | 202 | 29.7% | 暫定採用。Hepatocyteの記録に限定し、同一実験・時点の比較群が得られるものを選定 |

「cytoplasmic degeneration」のような広い呼称へ複数ラベルを併合せず、元の正確な所見名を使った。
NecrosisとSingle cell necrosisも別ラベルとして扱った。
facility数はいずれも不明。TOPOGRAPHY_TYPEは臓器内の部位・細胞種であり、施設ではない。

## 固定したslide群

| Compound | Experiment | 陽性 | 投与群・対象所見の記録なし | Control・対象所見の記録なし |
|---|---:|---:|---:|---:|
| fenofibrate | 434 | 10 | 10 | 10 |
| gemfibrozil | 184 | 10 | 10 | 10 |
| lomustine | 311 | 10 | 10 | 10 |
| 合計 | 3実験 | 30 | 30 | 30 |

全体でも各role内でも、各compound / experimentの比率は1/3。
各match_idは同じexperiment・sacrifice_periodの3枚（各role 1枚）を表す。
陽性は対象所見の記録部位がHepatocyteのみの投与slide。
陰性は対象所見がどの部位にも記録されていないslideであり、「病理的に正常」とは断定しない。
他所見のラベルをslides.csvのfinding_types / finding_sitesとcofindings.csvに保持した。

選定は次の固定ルールによる。

1. 元ラベルとmanifestを整数化したexperiment・group・individual IDで結合し、対象所見の一致を確認。
2. 用量不明を除外。同じ個体の複数画像はslide ID順の最初のみ選定対象にする。
3. 実験・時点ごとに3群の最小数までtripletを作る。陽性は重症度・用量、陰性は用量を交互に抽出し、同順位は数値slide ID順。
4. 10組以上作れるcompoundのみ残し、時点・重症度を交互に選んで各compound 10組に揃える。
5. 個体重複、role、実験・時点の対応、画像・特徴量ファイルの存在を検証する。

Hepatocyteで記録のあるWY-14643は60枚の投与slideすべてが陽性で、
同じ実験内の投与群陰性を用意できないため不採用。
thioacetamideは部位不詳、clofibrateはCentrilobularのため今回の定義から除外した。
元コーパスには同一個体の画像重複（acetaminophen、5個体）と用量不明のslideも存在した。
全3,441枚の採否概要をeligibility_audit.csv、各実験・時点の候補数をmatching_strata.csvに残した。

## 解消していない交絡と利用上の区別

- **施設**: data/correctedの10ファイルの列を確認したがfacility / laboratory / institutionのIDはなかった。
  全slideにfacility_id=unknownを明記。施設の対応表が得られるまで施設独立性は評価できない。
- **compoundとexperiment**: 元コーパスの44化合物中43は1実験、1化合物は2実験。
  全45実験がそれぞれ1化合物に対応し、採用subsetでも両者は1対1。
  複数実験に分散させてもcompound効果とexperiment効果を別々に推定することはできない。
  同一実験内の陽性・陰性比較は可能だが、実験内の他の差まで解消したことにはならない。
- **用量**: 時点は一致させたが用量は一致させていない。
  lomustine陽性は全10枚High、陰性はLow/Middle。fenofibrate陰性は全10枚Low。
  単なる投与影響・用量差との区別は別途必要。
- **重症度**: 陽性にはminimal/slight/moderateの3段階があるが化合物間で分布は異なる。
  fenofibrate=5/4/1、gemfibrozil=8/2/0、lomustine=0/2/8。重症度は対象所見について集約しており、全所見の最大重症度を代用していない。
- **併存所見**: lomustine陽性10枚すべてにProliferation, Kupffer cellがある。
  肥大・壊死・線維化等も混ざるため、このcompoundのクラスタ対応を対象所見だけの証拠として数えない。
  fenofibrate陽性では3/10にIncreased mitosis、gemfibrozil陽性では1/10にGranulomaがある。
  初回の目視ではfenofibrate/gemfibrozilの60枚を先に確認し、lomustineの30枚は併存所見の影響を調べる比較群として扱う。
  60枚だけを採る場合は2化合物の検証に留まることを明示する。
- **目視未確認**: 元ラベルには誤り・見落としがありうる。現在のstatusは全件source_annotation_unreviewed。
  SP_FLGも保持したが今回の陽性定義の除外条件には使っていない。

## 成果物と再現

- `snapshot_v1/candidate_summary.csv`: 3候補の分布・判断。
- `snapshot_v1/candidate_finding_rows.csv`: 3候補の全病理記録。slide ID・compound・experiment・dose・時点・grade・topography・SP_FLGを含む。
- `snapshot_v1/candidate_distribution.csv`: 候補ごとのcompound・experiment・dose・時点・部位・grade別のslide数。
- `snapshot_v1/slides.csv`: 固定90枚のmetadata、role、match_id、holdout_group、画像・特徴量パス、選定理由。
- `snapshot_v1/{positive,treated_negative,control}_slide_ids.txt`: 各30枚のID一覧。
- `snapshot_v1/subset_counts.csv`, `subset_dose_time.csv`, `subset_severity.csv`, `cofindings.csv`: subsetの構成と残る偏り。
- `snapshot_v1/eligibility_audit.csv`, `matching_strata.csv`: 選定対象外を含む監査表。
- `snapshot_v1/provenance.json`: 入力・スクリプト・成果物のSHA-256、選定ルール、施設・目視検証の未完了状態。

これらはoutputs/ではなく本ディレクトリに保存し、Gitで追跡できるようにした。
生成スクリプトは既存snapshotを上書きしない。比較・再生成は別ディレクトリへ出力する。

```bash
.venv/bin/python scripts/build_pathology_benchmark_subset.py --out /tmp/pathology_benchmark_reproduction
```

元の病理CSVへの独立した再結合でも陽性IDと対象部位が一致し、
全tripletのrole・experiment・時点、個体の一意性、成果物チェックサムを確認した。

## 次の作業

1. 同じ実験・時点の3群を画像で比較し、対象所見・併存所見・アーチファクトを区別して記録する。
2. 施設の対応表を入手して再監査する。得られなければfacility未検証の制約を維持する。
3. 必要ならNecrosisを対象細胞・部位で絞った第2候補を作り、より清潔な第3化合物を探す。
4. 選定を確定した後に、既存のクラスタリング手順で各role・compound・experimentごとの構成比を見る。
   patch数の多いslideに支配されないようslide単位で集計する。
   分割はslide/個体・compound/experimentを保ち、未知化合物の評価ではPCA・クラスタ学習も学習側だけで行う。
   今回の90枚で繰り返し調整した性能を独立した最終test性能として扱わない。
