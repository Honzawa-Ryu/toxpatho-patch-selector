# 0002 plan — control contrast patch scoring

## 問い

Control群のpatch embedding分布からの逸脱で、病理所見を含むpatchをどの程度拾えるか。
最終目標（所見に対応する視覚トークン・病理パッチ集の構築）に対して、
「Ctrlとの比較」というアプローチが実用になるかどうかの判定がこのフェーズの成果物。

## 入力（0001の出力）

- `outputs/0001_20260907_extract_wsi_embeddings/trident_job/20x_256px_0px_overlap/features_uni_v2/<slide_id>.h5`
  - `features` (n_patch, 1536) float32 と `coords` (n_patch, 2) を持つ
  - 219枚合計 3,260,413 patch（Control 2,280,571 / High 979,842）、1枚平均 14.9k
- `outputs/0001_.../trident_all_slides/slide_manifest.parquet`（219枚、compound/dose/timepoint）

## データ地形（確認済み）

| | Control | High |
|---|---|---|
| スライド数 | 160 | 59 |
| 所見あり | 3 | 58 |
| 所見なし | 157 | 1 |

- 8化合物 × 4時点（4/8/15/29 day）、32群すべてにcontrolが5枚ずつ存在する
- 所見: Hypertrophy 37, Degeneration granular eosinophilic 20, Necrosis 14, Fibrosis 13, Ground glass 10 …
- 所見ありスライドの最大grade: severe 41 / moderate 17 / slight 2 / minimal 1

## 手法

### Step A: 所見ラベル付きmanifest

`lib/tggates_metadata.py` に `open_tggates_pathology.csv`（Liver）を結合する関数を追加する。
キーは `(EXP_ID, GROUP_ID, INDIVIDUAL_ID)`。スライド単位に集約して
`has_finding` / `finding_types`（list） / `max_grade` を持たせる。

### Step B: patch異常スコア

control bankからの逸脱度をpatchごとに算出する。手法は複数出して比較する。

- kNN距離（bank内の最近傍k個との平均距離）
- Mahalanobis距離（bankの平均・共分散）
- ベースライン: ランダムスコア、全patch平均からの距離（`global_mean_dist`）
  - TRIDENTのpatchは既に組織領域だけなので「組織面積」はベースラインにならない。
    代わりに、control情報を使わない素朴な外れ値度を対照に置いた。

bankの単位も比較軸に含める。

- 同一 EXP_ID × 同一時点のcontrol 5枚（約7万patch）
- 全control 160枚（228万patch）

前者と後者の差が、そのままbatch effect（染色ロット・スキャン条件）の寄与の大きさになる。

control slideを採点する際は自分自身をbankから除く（leave-one-out）。

計算量: 全control bankは 228万 × 1536 × 4B ≈ 14GB。素朴な全探索kNNは重いので
FAISS、またはPCAでの次元削減＋bankのサブサンプリングで落とす。

### Step C: 評価（「どの程度有効か」の判定）

patch単位の正解アノテーションが無いため、3層で間接的に測る。

1. **slide-level**: patchスコア上位k%の平均をslide異常スコアとし、Control vs High のAUROC。
   k は複数（1%, 5%, 10%）試す。
2. **順序性**: High群内で、slide異常スコアと `max_grade` の順序相関（Spearman）。
   ただし現状severeに潰れているため、この指標はLow/Middle追加後に本番になる。
3. **偽陽性**: control slideのスコア上位patchが何を拾っているかを座標つきで出力する。
   染色ムラ・組織辺縁・アーチファクトばかりなら、Control比較は前処理の追加なしでは使えない。
   ここが実質的な有効性判定になる。

## 出力

- `patch_scores.parquet`: slide_id, x, y, score, method, bank_unit
- `slide_scores.parquet`: slide_id, method, bank_unit, top_k, slide_score + manifestのラベル
- `results.json`: AUROC / Spearman を method × bank_unit × k で

可視化（上位patchの切り出し画像）とクラスタリングによる視覚トークン化は0003に回す。
`coords` を持っているので、openslideで後から切り出せる。

## 未決・リスク

- **データが不完全**: Highは150枚中59枚しか無い（4day 0/40, 8day 15/40, 15day 19/39, 29day 25/31）。
  Low/Middleは0枚。追加DL（High補完91枚、Low+Middle 280枚）を検討中。
  0002は現状219枚で回し、DL完了後に同じコードで再実行できる設計にしておく。
- **High = 所見あり がほぼ同義**（58/59）。現データではスコアが所見を見ているのか投与群を
  見ているのかを分離できない。Low/Middleが入って初めて切り分けられる。
- GRADE_TYPEは学習の教師には使えない（severeに偏りすぎ）。評価軸としてのみ使う。

## 結果（2026-09-12 / job 10638 / 219枚）

`outputs/0002_20260912_control_contrast_patch_scoring/control_contrast/results.json`

| method | top_k% | AUROC (所見あり) | AUROC (投与群) | Spearman (grade) |
|---|---|---|---|---|
| knn_matched | 10 | **0.879** | 0.890 | 0.20 (p=0.13) |
| knn_global | 10 | 0.879 | 0.876 | 0.18 (p=0.16) |
| maha_matched | 10 | 0.824 | 0.862 | 0.19 (p=0.14) |
| maha_global | 10 | 0.691 | 0.684 | 0.04 (p=0.74) |
| random | 10 | 0.509 | 0.502 | 0.01 (p=0.96) |
| global_mean_dist | 10 | 0.305 | 0.317 | 0.17 (p=0.19) |

読み取れたこと:

- kNNはcontrol比較として機能している（AUROC 0.88、randomは0.50）。
- **bank単位（matched 5枚 vs global 160枚）でほぼ差が出ない**（0.879 vs 0.879）。
  同一実験のcontrolに揃えることの利得は、少なくともslide-levelでは見えない。
  バッチ効果の寄与が小さいか、kNNが元々それに鈍いかのどちらか。
- `global_mean_dist` のAUROCが0.5を大きく下回る（0.31）。全patch平均から遠いpatchを
  多く持つのはcontrol側ということで、素朴な外れ値度は所見ではなくアーチファクトや
  組織辺縁を見ている可能性が高い。control比較はそれとは別のものを見ている。
- **化合物ごとに成績が大きく違う**（High群スライドの平均順位、1が最も異常）:
  ethambutol 5.0 / promethazine 13.2 / WY-14643 23.2 / monocrotaline 34.4 /
  cholesterol 44.7 / vitamin A 76.6 / thioacetamide 78.8 / **phenobarbital 166.6**。
  phenobarbitalは5枚すべて120位以下で、moderate/severeの所見があるのにほぼ最下位。
  びまん性の小葉中心性肥大のように「全体が少しずつ変わる」所見は、上位k%patchの
  逸脱としては出てこない。これがこの手法の系統的な盲点。
- 偽陽性の上位2枚（15290: vitamin A の control, 36115: ethambutol の control）は
  所見なしで4位・8位。0003で実際のpatchを見る対象。
- controlの自然発生所見3枚（glycogen deposit ×2, 最小限のnecrosis ×1）は
  70/95/148位。所見はあるが投与由来ではないものを高く採点してはいない。

次に効きそうな修正:

- AUROCを化合物横断でプールしているのが甘い。化合物ごと（自群のcontrolに対するHigh 5枚）に
  出し直すと、実際の使い方に近い評価になる。
- びまん性所見には上位k%平均ではなく分布全体のシフトを見る集約が要る
  （control分布との距離の中央値、あるいは分布間距離）。
