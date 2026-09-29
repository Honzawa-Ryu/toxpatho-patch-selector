# レビュー後の対応 — 2026-09-25

## 1. ラベル監査

元3,441枚のうち80枚がindividual metadataに非対応。
CCl4 exp67:20枚、acetaminophen exp42:21枚、exp707:39枚。
画像metadataでdose=0の20枚も含まれていた。
これらを病理陰性と断定せず、評価対象外のunknownとして扱う。

共通metadata処理はindividualの結合状態を保存し、非対応のhas_finding/countをNAにする。
既存manifestでも既知でないdoseがある場合は共通評価関数が停止する。
元manifestは保持。監査結果は `label_audit/slide_label_audit.csv`、
評価対象3,357枚のmanifestは `label_audit/manifest_known_unique.parquet` に保存した。
80枚除外後は3,361枚、残る同一個体画像の重複4枚を除いて3,357枚になる。

保存済みk=2000の割当を固定し、旧条件・unknown除外・unknownと個体重複除外の3条件を比較。
結果は `label_audit/sensitivity.csv`、化合物ごとの値は各 `*_loco.csv`。

| 条件 | slides | 候補数 | pooled AUROC | 化合物別AUROCの平均 |
|---|---:|---:|---:|---:|
| 旧条件 | 3,441 | 329 | 0.8707 | 0.8706 |
| unknown除外 | 3,361 | 331 | 0.8586 | 0.8680 |
| unknown・個体重複除外（今後の参照） | 3,357 | 330 | 0.8587 | 0.8679 |

これは候補選定LOCOで、PCA/クラスタ作成までholdoutした性能ではない。
旧条件は再現監査専用の明示的な参照関数で実行し、通常の評価でunknownを許容しない。

## 2. 集計・レビュー保存の修正

- k=1000のgrade指標をk=2000へ結合する処理を修正。grade出力名・列にkを付け、照合する。
- 保存済みk2000品質表はbefore_20260925_fix.csvへ退避した上で、無効なgrade_passed/best_rhoを除去。
  tissue_frac_ccは保持。k=1000の元grade表は残し、明示的なk付きファイルも保存した。
- ranked画像生成はk別品質表の欠損・k不一致・候補測定値欠損をエラーにする。
- 副所見リンクの既定足切りは保有内率0.7。ファイル名はリンク先の所見の率を使う。
  0.7は表示用の操作的基準で、検証済みの病理正解率の基準ではない。
- ranked/reviewとも既存出力への上書きを拒否。--limitによる判定消失や、seed変更後の旧判定流用を防ぐ。
  古い311シートとsymlinkは履歴として保持しており、修正後の基準で再生成したパッチ集とは扱わない。
- 旧0009/config.ymlは過去比較の記録として保存。現在の評価条件と再実行入口はanalysis_protocol.jsonに固定。

## 3. subsetの再検討

同じ実験・時点・用量で陽性/陰性が混在する層を、3候補の各部位について全件集計した。
`snapshot_v2/candidate_matching_capacity.csv` に0件の組合せも含めて保存。

顆粒状好酸性変性 @ Hepatocyteはfenofibrate 1組、gemfibrozil 2組、計3組しかなく、
lomustineでは同用量ペアを作れない。旧90枚はこの用量差を許容した探索用として保持。

代替は **Necrosis @ Hepatocyte**。
6化合物×（陽性3・同用量の投与陰性3・同実験/時点のControl 3）=54枚をsnapshot_v2に固定。
compoundはcholesterol混合物、WY-14643、amitriptyline、ethinylestradiol、gemfibrozil、monocrotaline。
3組という上限は小規模な均等レビュー用であり、統計的な検出力の根拠ではない。
陽性と投与陰性は他所見の集合の差が小さくなる相手を優先したが、完全一致ではない。
Controlも同一個体を重複利用していない。個体ID・画像パス・match_idを保存。
特徴量は既存features_tier1_corpus内のslide_id.h5を参照する。

**v2も最終benchmarkとはしない。** 陽性18枚中12枚はSP_FLG=Trueで自然発生と記録され、
4化合物はminimal、残る2化合物はslightで、重症度・由来に化合物差が残る。
今回の対象は「治療関連壊死のみ」ではなく自然発生の記録も含む壊死。
治療関連に限定すれば6化合物という条件は満たさない。施設は全件unknown。

## 4. 一次目視

54枚から各6patch、計324patchをseed固定で抽出し、ラベルを隠した6ページで全件を確認した。
patchは既存tissue座標からランダムに候補抽出し、連結背景率による組織割合0.85以上を採用。
画像タイトルにはcompound/roleを表示せず、対応キーは別CSV。
元画像のラベル位置・clusterによって「それらしい部分」を選んでいない。

結果: 抽出失敗や大半が背景だけのシートは見られず、染色濃度・細胞間/細胞内の白い空隙の差はある。
ただし**6ランダムpatchで巣状壊死を確認/否定できないため全54件はuncertain**。
壊死の正解アノテーションとして採用できる件数は、この確認だけでは0件。
元ラベルと一致したとは主張しない。

`assistant_screening_v2.csv` に実際の観察を保存。
これはCodexによる一次スクリーニングで、病理専門家の判定ではない。
人が入力する `outputs/0010_20260925_pathology_benchmark_subset/review_v2/review.csv` は空欄を保ち、
assistant_screening.csvと混ぜない。各画像のhash・slide座標・level0の切出し幅を保存。
view用の `page_1.jpg`〜`page_6.jpg` と `R001.png`〜`R054.png` を同ディレクトリに保存。

## 検証と残作業

実データ再結合でunknownは80枚ちょうど、既知のラベルは変更なし。
unknownを陰性として使わない/レビューを消さない回帰テスト3件、修正スクリプトの構文確認、
既存reviewへの再出力が拒否されること、v1/v2のchecksum、v2の個体重複なし・同用量対応を確認した。

次の必要作業は**WSI上の壊死ROIの位置確認**と施設対応表の補完。
ランダムpatchをさらに増やすだけでは陰性の根拠にならない。
病理所見の判定を伴うROIが得られたら、保有内率や化合物の偏りと独立に、
陽性/陰性の対応画像で所見を確認して最終subsetへ進む。
現段階のv1/v2で新しいクラスタリングやK探索を始める必要はない。
