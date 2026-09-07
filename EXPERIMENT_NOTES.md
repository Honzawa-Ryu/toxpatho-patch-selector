# EXPERIMENT_NOTES.md

実験固有の注意事項をIDごとに記録する。plan-next-experimentでplan.mdを書く際、
review-expで結果を集約する際、debug-experimentで調査する際は、対象実験のIDに
該当する節があればまず読むこと。

## 0001_20260907_extract_wsi_embeddings

- TRIDENT（mahmoodlab/TRIDENT, v0.3.2固定）でLiver WSI（`data/TGGATEs/WSI/Liver/{4,8,15,29}_day/*.svs`,
  219枚）に対してtissue segmentation → patch座標 → patch embeddings（UNI2-h, `uni_v2`）を
  1回のパイプラインで実行する。PyPIの`trident`パッケージ（天体物理シミュレーション用、無関係）とは
  別物なので、依存関係は必ず`pyproject.toml`の`[tool.uv.sources]`経由（git pin）で入れること。
- RUN_MODE="single"（219枚をTRIDENT内部でループ・resumeさせる設計。テンプレートの
  GRID_ARGS/GRID_VALUES機構は直積50件が上限のため、219枚のarray分割には使っていない）。
- `USE_LOCAL_SSD_INPUT`/`USE_LOCAL_SSD_OUTPUT`は両方0にオーバーライドしている。
  理由: dataが156GBでノードのローカルSSD容量が未知数な点、およびTRIDENTのsmart resume
  （`job_dir/wsi_states/`を見て完了済みslideをスキップ）が48時間ジョブのタイムアウト時に
  機能するためには`job_dir`が常にNFS上の固定パスである必要がある点（scratch経由だと
  ジョブ再投入のたびに空のディレクトリになり進捗が失われる）。
- TRIDENTの出力（tissue mask/contours/patch座標/embeddings）は
  `outputs/0001_20260907_extract_wsi_embeddings/trident_job/` に直接書かれる
  （`lib/output_utils.get_run_dir()`のvariant_key配下ではない、TRIDENT自身のディレクトリ構造）。
- スライドごとのcompound/dose/timepointメタデータは `lib/tggates_metadata.build_liver_slide_manifest()`
  が`data/corrected/open_tggates_pathological_image.csv`（Liverでフィルタ、FILE_LOCATIONの
  basenameでローカルslide_idと突合）と`open_tggates_individual.csv`（DOSE_LEVEL等を追加）を
  結合して作る。出力は`outputs/.../trident_all_slides/slide_manifest.parquet`。
  後続フェーズ（patch selection/学習）で正解ラベルや層別化に使う場合はここを参照する。
- UNI2-h（`uni_v2`）はHuggingFace上でgatedなモデル。`HF_TOKEN`環境変数（既にシェルにexport済み）が
  `#SBATCH --export=ALL`経由でジョブに引き継がれる前提。トークンにUNI2-hへのアクセスが
  承認されていないと `run_batch_of_slides` がダウンロード段階で失敗する。
