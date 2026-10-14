# 0016 反復投与Liverの全ControlとtierI投与群のMacenko正規化埋め込み

染色正規化は基本必須（2026-10-13ユーザー方針）。既存特徴量（0001/0007/0008のTRIDENT出力）は正規化されていない。
系列C（反復投与の全試験Controlで、どの試験を一緒に扱えるかを判定する大規模調査）を先に始められるよう、
Controlを先に処理する（2026-10-13ユーザー了承）。**準備のみ。投入時期はユーザーが決める。**

## 対象（計5,420枚）

| phase | 対象 | 枚数 |
|---|---|---:|
| `coords` | 追加取得した反復投与Control（99化合物、座標なし） | 1,979 |
| `ctrl` | 全Control = tier1のControl 911（individual非対応を含む）＋追加1,979 | 2,890（144実験） |
| `treated` | tier1コーパスの投与群（44化合物） | 2,530 |

追加Controlは別マシン（preproc01）から`rsync --files-from outputs/target_slide_list/rsync_files_ctrl_repeat.txt`で手動転送（2026-10-13開始）。
一覧は`scripts/build_ctrl_slide_list.py`で作成。

## 手順

1. `coords`: 追加Controlに TRIDENT seg → coords（0008と同一: hest、20x、256px、overlap 0）。特徴量抽出はしない。
   出力は`outputs/0016.../trident_job_chunk{0-3}/20x_256px_0px_overlap/patches/<slide_id>_patches.h5`。
2. `ctrl`: 参照染色を全Controlから推定（各実験のControlから2枚・各20 patch、seed固定）。最初に計算した値を
   `outputs/0016.../reference.json`に保存し、以後のchunk・phaseはそれを読む。各slideをMacenko正規化してUNI2-hで抽出。
   **高速化（2026-10-14、方法はユーザー一任）**: ctrl phaseは1枚あたり無作為2,048 patchだけ抽出し、
   `features_uni_v2_macenko_sub2048/`に保存する（h5に元のpatch数と抽出の有無を記録）。0015の占有率は2,048 patchでも十分推定できる。
   全patchが必要になったら同じslideを`features_uni_v2_macenko/`へ抽出し直す。
3. `treated`: 同じ参照で投与群を抽出（全patch）。

- 座標は既存のTRIDENT出力を再利用（tier1分）。セグメンテーションとタイル分割はやり直さない。
- 各slideの染色推定はslide内40 patch、乱数はslide IDから決める（chunk分割に依存しない）。
- エンコーダ・推論経路は0005と同一（UNI2-h、TRIDENTと同じautocast）。抽出部品は0005の`experiment.py`から読み込む。
- 染色推定に失敗したslideは正規化なしで混ぜず、スキップして`results.json`に記録する。
- 出力: `outputs/0016.../20x_256px_0px_overlap/features_uni_v2_macenko/<slide_id>.h5`（0001と同じレイアウト）。
- 参照が0005（8実験のControl 16枚）と異なるため、0005の219枚も含めて作り直す。

## 実行

`run_slurm.sh`の`GRID_VALUES`のphaseを`coords` → `ctrl` → `treated`の順に書き換えて、1段ずつ`runx 16`で投入する。
各段はarray 4本（chunk 0〜3）、GPU 1枚・8 CPU・55GB・上限36時間。

- 見込み: 抽出は0005の実績（約1分強/枚）から、ctrl 2,890枚で1本あたり約720枚・12〜16時間、treated 2,530枚で同程度。coordsは0008の実績から数時間。
- 途中で止まっても、作成済みの`.h5`は飛ばして再開できる（`.h5.tmp`から原子的に置換）。

## 後続（別途）

- 正規化版のPCA64キャッシュを作る（`scripts/precompute_tier1_kmeans_cache.py`相当）。
- 系列C: 全Control（144実験）で0015相当の解析と、試験の塊の判定（`scripts/preview_exp_blocks.py`）。

## 未決

- 参照の抽出をControlに限るか（現状はControlのみ）。
- 追加Controlに単回投与・72時間など反復以外は含めない（2026-10-13決定: 反復投与のみ）。
