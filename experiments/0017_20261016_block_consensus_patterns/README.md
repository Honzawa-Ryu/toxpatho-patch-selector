# 0017 結果まとめ（2026-10-19時点）

塊（系列C、cut 0.72）の中で、k-means複数runの共所属からpatchのコアを取り出せるかのテスト。設計は`plan.md`。
出力は`outputs/0017_20261016_block_consensus_patterns/<variant>/`（gitの管理外）。

## 対象

| variant | 塊 | 試験 | slide | うちControl | 投与群で所見あり |
|---|---|---:|---:|---:|---:|
| `pilot` | 最大の塊（44試験）の手元分 | 13 | 1,020 | 260 | 452 |
| `block3` | 塊3（23試験）の手元分 | 11 | 853 | 220 | 225 |

特徴量は0016のMacenko正規化・全patch。60 run（k=50/100/200 × 抽出seed 5 × 初期化seed 4）、評価用patch 1枚50（空白除外後 pilot 47,514 / block3 40,362）。

## 本体の結果（kを混ぜた共所属、Leiden res 1.0、τ=0.8）

| | pilot | block3 |
|---|---:|---:|
| コア数 | 81 | 76 |
| 覆う割合 | 62% | 60% |
| Controlのpatchが5%未満のコア | 25 | 15 |
| 一致度（runを半分ずつ）: Control<5% / それ以外 | 0.77 / 0.61 | 0.84 / 0.60 |

- 連結成分ではτ=0.7〜0.8で巨大な1成分にまとまり使えない。
- Control<5%のコアは所見と対応するが、多くは1〜3化合物に偏る（所見の偏りから見て仕方ない、とユーザー判断）。
  pilot: PPARα作動薬（顆粒状好酸性変性・核分裂増加）、acetamidofluorene・DEN（卵円形細胞増生・Swelling）、amitriptyline・diltiazem（細胞質空胞化）、methylene dianiline（浮腫）、ethinylestradiol（好酸性変化）。
  block3: methapyrilene（大小不同核・胆管増生）、ibuprofen・naproxen（髄外造血）、monocrotaline（出血・線維化・壊死）、ethambutol（核の変化）。
- 所見名の付け方（`top_lift_findings`）は2026-10-17修正: 投与群slideの30%以上に出る所見の中から、13試験の投与群全体比の倍率順に最大3つ。slide単位の記録との対応であり、patchの診断ではない。

## 診断（`scripts/diagnose_consensus_cores.py`、pilot）

- patchの安定度（近傍との平均共所属率）は0.5付近の単峰。安定な塊と境目は明確に分かれない（埋め込みは連続体）。
- 安定度の高い側の裾は、所見の記録があるslideのpatchだけ。未割当はControlと混ざる中央の領域に多い（Control 47%、投与所見なし 42%、所見あり 30%）。
- kで共所属率の性質が大きく違う（辺の共所属率の中央値 k50 0.70 / k100 0.55 / k200 0.45）。

## Leiden解像度の比較（`scripts/sweep_consensus_resolution.py`、pilot、τ=0.8）

解像度0.003〜2.0でコア数13〜102。覆う割合は約63%で不変。Controlと混ざるコアの一致度はどの解像度でも0.4〜0.66。
Control<5%のコアは解像度を下げると少し安定（0.005で5コア・0.93）するが化合物をまたいで合わさり、ethinylestradiolの島は背景とつながった。
→ コア数を減らす方針は採らない（2026-10-18ユーザー判断）。

## kごとのコンセンサス（案2、`scripts/consensus_per_k.py`）

| | k=50 | k=100 | k=200 | 混ぜた場合 |
|---|---:|---:|---:|---:|
| pilot コア数 / 覆う割合 / 一致度 | 49 / 92% / 0.87 | 80 / 82% / 0.79 | 132 / 64% / 0.72 | 81 / 62% / 0.66 |
| block3 コア数 / 覆う割合 / 一致度 | 47 / 91% / 0.87 | 79 / 82% / 0.79 | 132 / 60% / 0.64 | 76 / 60% / 0.69 |

- kを混ぜたことが不安定さの大きな原因だった。kごとに作ると覆う割合も一致度も上がる。
- 細かいkほど「Control<5%のコアは安定、Controlと混ざるコアは不安定」の差がはっきりする（pilot k200: 0.84 / 0.63）。k=50では背景も安定に見える。
- kの間の入れ子はきれいではない（細かいコアが1つ粗いkの1コアに7割以上入るのは37〜46%）。
- 全粒度（k200→k100→k50）で入れ子になり、Control<5%のコア: pilot 11、block3 12。画像シートは`<variant>/sheets/per_k200_nested_ctrl_lt0.05_tau0.8/`。
- 方針（2026-10-18ユーザー了承）: 今後はkを混ぜず、kごとにコンセンサスを取る。0017本体への組み込みは次回の実行時。

## ユーザーの見方（議論メモにも記録）

- patchが潜在空間に連続的に広がるのはSSLの性質上当然。最終的にはk-means等で区切る場面も来る。
- 「安定なコア（Controlに出ない）＝所見」という見方には納得感がある。ただし安定さは病変の証明ではない（組織の端・ぼけなど画像として特徴的なものも安定になりうる）ので画像で確認する。

## 関連する図・表

- `visualization_20261017/`: pilotのt-SNE（A）、試験ごと（B）、コアマップ（C）
- `outputs/.../<variant>/sheets/`: コアの画像シート（上位40、Control<5%、全粒度で入れ子）
- `outputs/.../pilot/slide_maps_leiden_tau0.8/`: slide全体のpatchをコアで色分けした図（13試験×High・Control）
- `outputs/.../<variant>/diagnostics/`、`per_k/`: 診断・解像度比較・kごとの結果

## 最大の塊全体（2026-10-20〜23）

kごとのコンセンサス（`per_k: true`）を本体に組み込み、cut 0.72の最大の塊44試験すべてで回した。続く2段階の方式は0018。

| variant | 内容 | slide / 試験 | 評価用patch | k=200のコア / Control<5% / 全粒度で入れ子かつControl<5% |
|---|---|---|---:|---|
| `block6_all` | 44試験すべて | 3,468 / 44 | 161,250 | 140 / 20 / 9 |
| `block6_clean` | QC目録（tggate-wsi-qc）でA_clean（ボケ・縦縞なし）のみ。試験の欠け・偏りはそのまま（2026-10-22ユーザー決定） | 2,271 / 39 | 105,410 | 133 / 27 / 10 |
| `block6_clean_centered` | block6_cleanで、各slideの平均（1536次元、1枚約350 patch）を引いてからPCA（2026-10-23ユーザー提案） | 2,271 / 39 | 105,410 | 122 / 5 / 1 |

- block6_cleanはblock6_allのキャッシュから抜き出し（`derive_cache_from`）、centeredはblock6_cleanのキャッシュを共有（`cache_variant`）。
- 概要図: `visualization_<日付>_<variant>/`（O1〜O4: 所見・試験・scannerの内訳、A〜E: t-SNEとコア）。所見ありslideのpatchは所見を問わず1か所に集まった（block6_all）。
- block6_cleanにしても、Controlのslide（nitrosodiethylamineなど）に縦の帯が残った。0018で調べた結果、スキャナのストライプ（2,032 px）と同じ間隔だが、ピント（ラプラシアン分散）では説明できない（ぼやけ以外のスキャン由来の影響）。
- 平均を引くと、コアは化合物をまたぐようになる（コア内で最多の化合物の割合の中央値 0.2前後→0.12〜0.16）が、Controlとの分離は弱くなる。
