# 0022 外部（エーザイ・シオノギ）のWSIのMacenko正規化UNI2-h特徴量（全patch）

2026-10-25 ユーザー: 「外部データはベンチマークのリポ（tggate-bench）にある製薬企業のデータ。好きに使っていい。学習に使うのも構わないが、リークに十分気をつけて」。
層別化（形態・小葉の位置・領域など）が現実の組織とどう結びつくかを、TG-GATEsの外でも確かめるために使う。

## 入力（読むだけ、tggate-benchは変更しない）

- WSI: `tggate-bench/data/g_drive/`（78枚: エーザイ18、シオノギ60）。一覧と所見・用量は `tggate-bench/outputs/0013_20261012_extract_external_tiles/default/slides.csv`
- 肝臓の範囲: `tggate-bench/docs/external/liver_roi.json`（レビュー済み。番号付きの組織片 73枚、手描きの多角形 5枚）と、組織片の番号 `outputs/0012_.../<segmentation>/labels/*.png`
- 所見: `tggate-bench/outputs/external_data_audit/findings_raw.csv`

## 方法（0016と同じ条件にそろえる）

- 物理視野をTG-GATEsにそろえる: patchは 256 px × 0.4916 µm/px ≒ 126 µm 四方。level 0 で round(126 / mpp) px を読み、エンコーダの前処理で縮める（シオノギ mpp 0.442 → 285 px、エーザイ 0.503 → 250 px）。
- 肝臓の範囲の中を、重ならない格子で全部切り出す（縮小画像で組織割合0.5以上の格子だけ）。
- Macenko正規化: 参照は0016の `reference.json`（TG-GATEsの全Controlから推定）。各slideから40 patchで染色を推定。
- エンコーダ: UNI2-h（0016と同じ）。出力はh5（features、coords、attrsにpatch_size_level0・mpp）。

## リーク対策（以後の解析での約束）

- 外部は評価・適用だけに使う。プロトタイプ・LDA・小葉の軸など、学習するものは一切外部で学習しない。
- 外部で閾値や設定を選ばない（TG-GATEsで決めたものをそのまま当てる）。

## 限界

- 企業ごとにスキャナ・染色・標本作製が違う。Macenko正規化とslide内の順位で揃えるが、残る差は結果に出る。
- エーザイは画像単位（同じ動物かは未確認）、シオノギは動物単位（追加切片3枚あり）。所見名・程度の尺度は企業ごとに違う。
