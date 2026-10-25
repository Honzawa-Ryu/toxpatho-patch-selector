# 0018 結果まとめ（2026-10-24時点）

0017の最大の塊で、全体でのクラスタリングを「正常patchを省く」大まかな選別（1段目）に使い、残ったpatchだけでコアを作り直す（2段目）。設計は`plan.md`。
出力は`outputs/0018_20261022_ctrl_filter_then_cores/<run>/`（gitの管理外）。runの名前は `<0017のvariant>[_thr<閾値>][_s2<2段目の0017 variant>]`。

## 方式

- 1段目: 0017の60 runそれぞれで、patchが入ったクラスタのControl割合を出し、runで平均する（Control近さのスコア）。閾値以下を残す。
  閾値の既定は「所見ありslideのpatchが残る割合 − Controlのpatchが残る割合」が最大の点。`--threshold`で固定もできる。
- 2段目: 残したpatchでk-means（k=10/20/40 × 20 run、80%で学習）、kごとにコンセンサス（τ=0.8、Leiden）。比較に1回だけのk-means。
  埋め込みは0017のPCA64（pca0017）か、残したpatchで学習し直したPCA64（pca_refit）。`--stage2_variant`で別の0017 variantの埋め込みを使える。
- slideの地図（`scripts/ctrl_filter_slide_maps.py`）: 1段目のスコアは0017の重心で全patchに計算し直し、コアは残した評価用patchの近傍15点の多数決で付ける。

## 主な結果

| run | 1段目の閾値 | 残る: 全体 / Control / 投与・所見なし / 所見あり | 2段目（k=20コンセンサス）: コア数 / 最多化合物の割合（中央値） / 一致度（中央値） |
|---|---|---|---|
| `block6_all` | 0.246（差最大） | 28.5% / 13.2% / 22.6% / 60.0% | 29 / 0.19 / 0.96 |
| `block6_clean` | 0.254（差最大） | 32.0% / 14.6% / 25.0% / 64.2% | 27 / 0.22 / 0.93 |
| `block6_clean_thr0.07` | 0.07（所見ありslideのスコア分布の谷） | 8.3% / 0.2% / 1.6% / 30.1% | 19 / 0.67 / — |
| `block6_clean_thr0.03` | 0.03 | 5.2% / 0.0% / 0.3% / 20.6% | 23 / 0.77 / — |
| `block6_clean_centered` | 0.248（差最大） | 27.3% / 16.1% / 20.9% / 51.7% | 21 / 0.13 / 0.96 |
| `block6_clean_thr0.07_s2block6_clean_centered` | 0.07、2段目は平均を引いた埋め込み | 8.3% / 0.2% / 1.6% / 30.1% | 23 / 0.35 / 0.72 |

- **1段目の通過率は病変の重さに比例する**（`pass_vs_pathology/`）。slideの最大重症度ごとの通過率の中央値（閾値0.07）: Control 0.00、所見なし 0.00、minimal 0.02、slight 0.13、moderate 0.54、severe 0.85。
  「陽性slideのほぼ全部が通る」のは主にmoderate・severeのslide。小葉中心性の所見は通過率が低め。
- **閾値を厳しくするとControlと縞は消えるが、slide全体が変わる化合物（PPARα作動薬、nitrosodiethylamine、acetamidofluorene）ではslide全体が残る**。コアも1化合物に寄る。
- **組み合わせ版**（1段目は平均を引かない閾値0.07、2段目は平均を引いた埋め込み）では、Controlをほぼ含まず、化合物をまたぐコアが出た。
  例: コア0・3（肝細胞内の空胞、ethinylestradiol・imipramine・amitriptyline・hydroxyzine・diltiazem）、コア2（好酸性顆粒状、WY-14643・gemfibrozil）、コア13（大型で核の目立つ肝細胞、acetamidofluorene・nitrosodiethylamine）。
  ただし一致度は下がり（k=20で0.72）、slideの地図では、均一なはずの組織がスキャナのストライプに沿って別のコアに割れた（WY-14643・High）。
- **所見の側から見ると、コアは所見に明確には対応しない**（`finding_to_core/`）。同じ試験の中で陽性・陰性slideを占有率で分けるAUCは、主要7所見のうち5つで「1段目の通過率」のほうが最良のコアより高い。
  1つのコアが同じ化合物の複数の所見を受け持つ。コアは「異常の量」×「化合物」に近い。平均を引いた版が、所見とコアの対応は最も1対1に近い（空胞化 コア3 0.90、好酸性顆粒状変性 コア1 0.95、壊死 コア2 0.89）。
- **縦の縞はぼやけでは説明できない**（`block6_clean/focus_maps/`）。nitrosodiethylamineのControl（52759）の帯は、スキャナのストライプ（2,032 px）と同じ間隔だが、patchのスコアとピント（log10 var(Laplacian)）の順位相関は0.09。
  比較用のslide（52693）のほうがピントの周期はむしろ強い。

## 限界

- 所見はslide単位の記録。patchが所見を写しているとは限らない（→ 0019のBayerアノテーションで場所を確かめられる）。
- 試験＝化合物。PPARα作動薬3つが同じ塊にあり、それらに共通のコアは所見と作用機序を分けられない。
- 同じ試験の中の比較でも、陽性slideは高用量に偏るので、AUCには用量による変化の量が入る。
- A_cleanに絞ったため、SS1598の試験の多くが欠けている（5試験は0枚）。

## 残っていること・Issue

- ストライプごとに平均を引く、近傍のslide・ストライプの偏りを測る（未着手）
- #12 埋め込みの事後調整でslide間差の方向をつぶす、#13 UNIに事後でDANNなど

## 関連する図・表（`outputs/0018_20261022_ctrl_filter_then_cores/`）

- `<run>/figures/`: S1（1段目のスコア分布・閾値・群ごとの通過率・画像シート）、T・T2（残したpatchのt-SNE）、F（コア×所見・化合物）、`sheets/<組み合わせ>/`（コアの画像シート）
- `<run>/slide_maps_pca0017_consensus_k20[_ctrlkeep*]/`: 所見ごとのslideの地図
- `block6_clean/focus_maps/`: ピントの地図とpatchのスコアの比較
- `pass_vs_pathology/`: 1段目の通過率と重症度・分布（`scripts/stage1_pass_vs_pathology.py`）
- `finding_to_core/`: 所見→コアのAUC（`scripts/finding_to_core_map.py`）
