# EXPERIMENT_NOTES.md

## 2026-09-25: 評価前に参照する監査

0009の元manifestにはindividual metadata未対応80枚があり、旧解析で投与群・所見なしに
含まれていた。新しい評価はunknownと重複個体を除いた3,357枚を参照する。
保存済みクラスタでの候補選定LOCOは0.8707→0.8587（何らかの所見ありのpooled AUROC）。
k2000品質表のgrade_passed/best_rhoはk1000からの誤結合だったため、原本退避後に除去済み。
詳細・参照manifest・採用条件は `experiments/0010_20260925_pathology_benchmark_subset/FOLLOWUP.md`
と `analysis_protocol.json`。旧計画の性能値は当時の条件であり、所見別精度として引用しない。

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
- TRIDENT v0.3.2は`run_batch_of_slides`/`run_single_slide`/`trident batch`/`trident single`の
  console scriptが全て壊れている（`pyproject.toml`の`[tool.poetry.scripts]`がルート直下の
  `run_batch_of_slides.py`等をentry pointに指定しているが、poetryのビルドがそのファイルを
  パッケージに含めていないため、インストール後は`ModuleNotFoundError: No module named
  'run_batch_of_slides'`で必ず落ちる。アップストリームのパッケージングバグ、v0.3.2固定なら
  再発する）。回避策として`run_batch_of_slides.py`（`trident`パッケージ以外への依存なし）を
  `libraries/trident_run_batch_of_slides.py`にvendoringし、`.venv`のpythonで直接実行している
  （`experiment.py`の`run_trident_pipeline()`参照）。今後TRIDENTを呼ぶ実験でも同じ回避策が要る。

## 0002_20260912_control_contrast_patch_scoring

- 設計は `experiments/0002_20260912_control_contrast_patch_scoring/plan.md` にある。
- **join keyのゼロ埋めに注意**: `open_tggates_pathology.csv` / `open_tggates_pathological_image.csv` を
  `dtype=str` で読むと `EXP_ID="0117"`, `GROUP_ID="01"` のようにゼロ埋めされた文字列になるが、
  `slide_manifest.parquet` 側は int（`117`, `1`）で入っている。素朴に文字列同士でmergeすると
  1行もマッチせず「全219枚が所見なし」という無言の誤った結果になる。両側を int に正規化してから結合すること。
- **手元のWSIは反復投与のControl群のみ完備**。Highは150枚中59枚（4day 0/40, 8day 15/40,
  15day 19/39, 29day 25/31）、Low/Middleは0枚。`data/corrected/open_tggates_pathological_image.csv` の
  `FILE_LOCATION`（ftp URL）から追加DL可能。DLする場合は 4/8/15/29 day の Liver のみで、
  既存の `data/TGGATEs/WSI/Liver/<N>_day/` 構造に合わせて置くこと。
- 現データではHigh群の58/59が所見ありで、Control群の157/160が所見なし。つまり
  「所見の有無」と「投与群」がほぼ同義になっており、スコアが所見を捉えているのか
  投与群のbatch差を捉えているのかを分離できない。Low/Middle追加が本質的な解決になる。
- CPUだけの軽い確認を `srun` で流す場合、このプロジェクトは `/workspace/andre01` 配下なので
  `small-andre01` 等の andre01 パーティションを使う。`*-o` 系（small-creator-o 等）は
  `/workspace/filesrv01|02` からしか投入できず `Error: *-o jobs must run from ...` で弾かれる。

## 0003_20260912_control_absent_cluster_mining

- 設計は `experiments/0003_20260912_control_absent_cluster_mining/plan.md`。
  「patch embeddingをクラスタリングし、Controlにほぼ現れないクラスタを引けば所見特有の
  形態が取れるのではないか」という仮説の検証。0002（距離ベース）とは別物で、
  0002が苦手なびまん性所見（phenobarbital等）を頻度ベースで拾えるかが焦点。
- **PCAをcontrol patchのみで学習してはいけない**（0002はそうしている）。所見特有の方向は
  controlでの分散が小さく、control限定のPCAでは削られる可能性がある。拾いたいものを
  前処理で消すことになるので、controlと投与群を均等にサンプルして学習する。
- **クラスタ占有率の群間比較はスライド単位で行う**。同一スライドのpatchは独立ではないので、
  patch単位で検定するとn=数百万になり些細な差でも有意になる（pseudo-replication）。
- control欠損クラスタには染色ムラ・気泡・ペンマーク等の単一スライド由来アーチファクトが
  必ず混ざる。クラスタごとの寄与スライド数・寄与化合物数を必ず出し、1〜2枚に集中している
  ものは候補から外すこと。

## 0005_20260913_stain_normalized_embeddings

- 0004で、病変の無いcontrol同士が由来する実験で有意に分離することが分かった
  （silhouette 0.175, z=31.9）。染色バッチが埋め込みに乗っているため、Macenko正規化
  （`lib/stain.py`）してから同じエンコーダで埋め込み直す。座標は0001の計算結果を再利用する
  （画素値を変えてもセグメンテーションとタイル分割は変わらないため）。
- **抽出経路は0001と完全に一致していることを確認済み**: 正規化を切って抽出した特徴量と
  0001の特徴量のcos類似度が1.0000。`trident.patch_encoder_models.load.encoder_factory`
  と `encoder.eval_transforms` をそのまま使い、autocastの精度もTRIDENTと同じにしてある。
  ここがずれると「正規化の効果」と「抽出経路の差」が混ざるので、変更時は必ず再確認すること。
- **Macenkoの実装で踏んだ罠2つ**:
  1. 固有ベクトルの符号は不定。第1軸が平均OD方向と逆を向くと射影角が±πの切れ目をまたぎ、
     パーセンタイルが分布の両端ではなく切れ目の両側を拾う。結果、染色ベクトル2本がほぼ平行に
     潰れて `max_c` が発散する（実際に 9.9 / 29.1 という値が出た）。`fit_macenko` で第1軸を
     平均OD方向に揃えて回避している。cos(H,E)>0.99 で例外を投げる保険も入れた。
  2. 文献既定の参照濃度 `MAXC_REF=[1.97, 1.03]` はこの切片群（control 16枚から推定して
     [2.56, 1.69]）より淡く、そこに合わせると全体が褪色する。好酸性の強弱そのものが
     所見の手がかりなので、参照はデータ内のcontrolスライドから推定して使うこと。
- 正規化ありとなしの埋め込みのcos類似度は0.958。効いてはいるが埋め込みを壊してはいない。
- **読み方**: 正規化後に期待するのは「controlを実験でラベル付けしたときの分離度が下がる」
  ことと「投与群 vs control の分離は保たれる」ことの両方。後者まで下がるなら、
  正規化がバッチと一緒に病変の手がかり（好酸性の強弱など）も消している。
- 0003/0004は `variant_key` をconfigから取るようにしてあるので、`features_dir` と
  `variant_key` を差し替えれば正規化版を別ランとして残せる（完了ガードはvariant_key単位）。

## 0006_20260915_clustering_method_comparison

- 0003のMiniBatchKMeansを kNNグラフ+Leiden / DBSCAN に差し替えて比較した。
  解析側は `lib/cluster_analysis.py`（0003から移設）で3手法完全に共通。
- **結論: この用途ではk-meansが最も良い**（LOCO 0.747 vs Leiden 0.65-0.68 vs DBSCAN 0.51-0.54）。
  patch埋め込み空間が離散的な塊ではなく連続体なので、「自然な切れ目」を探す手法は
  切れ目を見つけられない。Leidenはr=2.0でも43クラスタ止まり、DBSCANはノイズ率24-66%。
- **粒度を揃えた比較になっていない**点に注意（k-means 100 vs Leiden 20-43）。
  差が手法由来か粒度由来かは未分離。Leidenの解像度を上げた追試が要る。
- `leidenalg` / `igraph` を pyproject.toml に追加済み。kNNグラフはGPUで厳密計算
  （`lib/clustering.knn_graph`）しているので faiss/pynndescent は入れていない。
- Leidenは20万点・約250万エッジで6〜17分かかる。326万patch全体には適用できないため、
  部分集合でクラスタを決めてkNN多数決で全体に伝播する構成にしてある。


## 0011_20260925_stratified_candidate_selection

- k=2000を固定し、全体投与対Control（A）、実験・時点内比較＋化合物内Control希少性（B1）、同じ比較＋従来の全体Control希少性（B2）を比較。
- 組織割合フィルタありの候補数はA312/B1 303/B2 382。脂肪変性関連の旧候補につながる1356/846/703/535はB1/B2で回収できた。
- B2でも旧29・81由来のパッチ保持が低下。所見保有内率中央値も改善せず、従来設定の全面置換はしない。背景候補が戻るため組織割合フィルタは維持する方針。
- 3方式の和集合489クラスタ×24枚=11,736パッチの比較一覧を作成。`outputs/0011_20260925_stratified_candidate_selection/gallery/index.html`。座標・メタデータ・判定記入用CSVあり、切り出しエラー0。
- 探索的比較であり、施設調整・独立評価・全画像の所見確定は未実施。条件と結果は `experiments/0011_20260925_stratified_candidate_selection/README.md`。

## 0012_20260928_compare_clustering_methods_at_scale

- 候補選定（方式A）を固定し、クラスタリングだけを振った（MiniBatch / Lloyd / Leiden / Ward / コンセンサス、k=250〜10000、seed・部分集合）。
- **個々の候補クラスタはseed間で再現しない**（k=2000で最良対応Jaccardの中央値0.25〜0.33）。候補patchの集合全体はJaccard約0.72で保たれる。クラスタIDはrun固有のものとして扱い、seedをまたいで同じIDを比べないこと。
- MiniBatchはLloydより不安定だが、タスク指標はほぼ同じ。Leiden・コンセンサスは同じ粒度で安定性が高い（約3000クラスタでARI約0.5）。
- kを上げたときの保有内率の上昇は、単一化合物への偏りと連動している。最適なkを決める指標はない。旧42/53の脂肪変性patchは、どの設定でも候補にほぼ入らない。
- このクラスタは job_submit/lua が全ジョブに gpu:1 を付ける。`--gres=none` は効かない。
- 詳細は `experiments/0012_20260928_compare_clustering_methods_at_scale/README.md`。
