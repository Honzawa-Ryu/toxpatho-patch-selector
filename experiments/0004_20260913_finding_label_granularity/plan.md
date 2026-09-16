# 0004 plan — 所見名の粒度と形態の粒度のずれ

## 問い

毒性病理の所見名は、広範な概念を1つの名前に押し込めていることが多い。
patch embeddingのクラスタ（＝形態の型）と所見名の対応を取ったとき、ずれはどこに、
どちら向きに出るか。そしてそのずれは `TOPOGRAPHY_TYPE`（病変の部位）で説明できるか。

0003の結果から、ずれは既に両方向に観測されている。

- **1つの名前 → 複数の形態**: 手元8化合物の `Hypertrophy` は、部位で割ると
  胆管上皮・肝細胞・星細胞(Ito)・Kupffer細胞の肥大が同居している。別の細胞が
  肥大しているものを1語でまとめている。
- **1つの形態 → 複数の名前**: 0003のcluster 45は WY-14643 と phenobarbital の
  スライドにまたがる単一の形態（好酸性顆粒状の肝細胞肥大）だが、ラベルは
  WY-14643側が `Degeneration, granular, eosinophilic`、phenobarbital側が
  `Hypertrophy, Centrilobular` と別の名前になっている。

## 手元データで決着が付く仮説

`Hypertrophy` の部位内訳（手元219枚）:

| 化合物 | 部位 | 個体数 |
|---|---|---|
| phenobarbital | Centrilobular | 5 |
| promethazine | Centrilobular | 5 |
| thioacetamide | Centrilobular | 5 |
| WY-14643 | Bile duct, interlobular | 13 |
| vitamin A | Ito cell | 5 |
| monocrotaline | Kupffer cell / Peripheral | 4 / 2 |

- **H1**: Centrilobular Hypertrophy を持つ3化合物（phenobarbital / promethazine /
  thioacetamide）は共通のクラスタを持つ。
- **H2**: 同じ `Hypertrophy` でも部位が違う化合物（WY-14643 / vitamin A /
  monocrotaline）はそのクラスタを持たない。

両方成り立てば「所見名だけでは形態と対応しないが、部位を足すと対応する」。
成り立たなければ、ずれは部位でも説明できない＝もっと根深いということになる。
どちらでも情報が出る。

**交絡の明示**: 手元では部位と化合物がほぼ交絡している（胆管肥大はWY-14643のみ、
Ito細胞肥大はvitamin Aのみ）。H1が3化合物で成立するかどうかだけが、化合物交絡を
免れて検証できる部分。それ以外の部位については「分かれた」としても化合物由来と
区別できないので、そう明記して報告する。

## 手順

0003の `cluster_assignments.parquet` をそのまま使う。再抽出・再クラスタリングは不要。

1. manifestに `finding_sites`（`FINDING_TYPE @ TOPOGRAPHY_TYPE`）を持たせる
   （`lib/tggates_metadata.attach_pathology_findings` に追加済み）。
2. スライド×クラスタの占有率行列を作る（0003と同じ）。
3. **H1/H2の直接検定**: Centrilobular Hypertrophy 陽性スライド群 vs それ以外の
   投与スライド群で、クラスタごとに `scipy.stats.mannwhitneyu` → BH補正。
   有意なクラスタについて、部位別の占有率を並べる。
4. **対応の良さの比較**: 「所見名」でスライドをラベル付けした場合と
   「所見名@部位」でラベル付けした場合で、クラスタ割当との
   `sklearn.metrics.normalized_mutual_info_score` を比較する。
   部位を足して上がるなら、ずれの一部は既存メタデータで復元できるということ。
   - 注意: ラベル体系を細かくするとNMIは機械的に上がりうる。部位ラベルを
     ランダムに入れ替えた対照（permutation）を必ず並べて、増分が偶然でないことを見る。
5. **可視化**: `Hypertrophy` の部位ごとに代表patchを並べる
   （`scripts/make_cluster_exemplars.py`）。胆管上皮・肝細胞・Ito細胞・Kupffer細胞が
   実際に写り分けているかを目で確認する。ここが一番効く。

## 脂肪変性の軸（今回は片側のみ）

`Degeneration, fatty` はTG-GATEsのLiver全体で**2化合物しか出さない**:

| 化合物 | 部位 | 個体数 | 手元 |
|---|---|---|---|
| carbon tetrachloride | **Centrilobular** | 67 | ✗ 未取得 |
| 1% cholesterol + 0.25% sodium cholate | **Peripheral** | 20 | ✓ |

同じ所見名で小葉内分布が反対という、細分化を問うのに理想的な対比。
0003のcluster 87（ランダム36枚すべてが微小脂肪空胞で均一）は cholesterol 側＝
辺縁性だけを見ている。**carbon tetrachloride を1化合物追加するだけでこの軸が立つ**。

今回は片側しか無いので、0004では「cluster 87が辺縁性脂肪変性に対応している」ことの
確認までにとどめ、細分化の検証はCCl4取得後に回す。

なお脂肪・空胞系はラベル自体が5つの所見名に散らばっている
（`Degeneration, fatty` 2化合物 / `Vacuolization, cytoplasmic` 18化合物9部位 /
`Degeneration, vacuolar` 1 / `Deposit, lipid` 1 / `Vacuolization, nuclear` 2）。
細分化だけでなく、名前の境界が形態と合っているかという統合側の問いもある。

## 今後のデータ拡張（メモ）

所見特異性の検証は枚数ではなく**化合物数が律速**。反復投与Liverで所見を出す化合物は
全115、所見は49種。追加DLの優先順位:

1. **carbon tetrachloride** — 脂肪変性の細分化。1化合物で軸が立つ。最小コスト最大情報量
2. `Vacuolization, cytoplasmic` 系（amiodarone / ethionamide / hexachlorobenzene 等）
   — 脂肪変性との名前の境界を問う
3. `Hypertrophy` Centrilobular系（全41化合物）— 部位を揃えた上での形態の共通性

コスト目安: 1化合物あたり反復投与のみで約80枚・64GB・埋め込み抽出3時間弱。
`scripts/download_tggates_liver_wsi.py` は現状「手元にある化合物」しか対象にしないので、
化合物を指定する引数の追加が要る。

## 出力

- `slide_manifest_with_sites.parquet`: finding_sites 付きmanifest
- `site_cluster_stats.parquet`: (所見名@部位) × クラスタ の富化統計
- `granularity_comparison.json`: 所見名 vs 所見名@部位 のNMI（permutation対照込み）
- `exemplars_by_site/`: 部位ごとの代表patch
- `results.json`

## 結果（2026-09-13 / job 10659 / 219枚 / k=100）

### 検定の設計ミスと訂正

最初の H1 検定は「Centrilobular Hypertrophy の投与スライド vs 他の化合物の投与スライド」で
実装した。これは誤り。control基準が無いため、病変ではなく**化合物・実験ごとのバッチ署名**を
拾う。富化クラスタが23個も出たのがその兆候だった。

実例: cluster 63 は phenobarbital の投与群で富化して見えたが、同じ化合物の
**control 20/20枚にも平均0.037で出ており、High群(0.033)とほぼ同じ**。用量に依存しない
＝病変ではない。

**正しい比較は「その化合物のHigh vs その化合物自身のControl」**（同一実験内比較）。

### H2: 成立

部位が違う化合物（WY-14643=胆管 / vitamin A=Ito細胞 / monocrotaline=Kupffer細胞）は、
Centrilobular群の富化クラスタの占有率がすべて 0.000。部位が違えば形態も違う。

### H1: 棄却

化合物ごとに自群controlと比べた富化クラスタ（上位4）:

| 化合物 | 最大富化クラスタ | High平均 | Control平均 | 上位4クラスタ |
|---|---|---|---|---|
| thioacetamide | 42 | 0.744 | 0.000 | 42, 40, 25, 8 |
| monocrotaline | 12 | 0.501 | 0.0002 | 12, 76, 62, 6 |
| promethazine | 43 | 0.376 | 0.0001 | 43, 27, 86, 18 |
| phenobarbital | 16 | 0.284 | 0.0005 | 16, 56, 1, 34 |
| WY-14643 | 30 | 0.149 | 0.000 | 30, 45, 97, 55 |
| vitamin A | 36 | 0.166 | 0.029 | 36, 68, 90, 7 |

Centrilobular Hypertrophy を共有する3化合物の重なりは**完全に空**:

```
phenobarbital ∩ promethazine : []
phenobarbital ∩ thioacetamide: []
promethazine  ∩ thioacetamide: []
```

phenobarbital と promethazine に至ってはラベルが完全に同一
（`Ground glass appearance @ Centrilobular` + `Hypertrophy @ Centrilobular`）なのに、
占有クラスタが排他的（cluster 16: pheno 0.284 / prom 0.000、cluster 43: pheno 0.000 / prom 0.376）。

### ただしバッチ効果があり、棄却の解釈は一意でない

所見が一切無い**control群160枚だけ**を実験(化合物)でラベル付けした分離度:

| ラベル | silhouette | 帰無平均 | z |
|---|---|---|---|
| control群を実験で | **0.175** | -0.095 | 31.9 |
| control群を時点で | -0.029 | -0.040 | 2.7 |
| （参考）投与群を化合物で | **0.566** | -0.144 | 55.4 |

**病変が無いcontrol同士が、由来する実験で有意に分離する**（z=31.9）。染色・スキャン条件の
バッチ署名が実在する。時点では分離しない（z=2.7）ので、バッチの単位は実験＝化合物。

control群で他実験に出ない「専有クラスタ」数: WY-14643 **12個**、cholesterol 6個、
thioacetamide 1、vitamin A 1、他4化合物は0。WY-14643の実験は特にバッチ署名が強く、
この化合物由来のクラスタは割り引いて読む必要がある。

ただし規模は control 0.175 に対し投与群 0.566 で、処置由来の分離の方がはるかに大きい。
バッチは存在するが支配的ではない。

### 結論

- 所見名だけでは形態と対応しない（H2が示すとおり、部位を足すと改善する）
- **部位を足しても足りない**。同じ `Hypertrophy @ Centrilobular` の3化合物が
  互いに素なクラスタを占める（H1棄却）
- ただしバッチ署名が実在するため、「互いに素」の一部は染色差由来の可能性が残る。
  統計だけでは分離できず、画像を見ても色調差が大きくて判別できなかった

### 次にやること

1. **染色正規化してから埋め込み直す**（Macenko / Reinhard等）。その上でH1を再検定する。
   これが決着を付ける唯一の道。視覚トークンがバッチ不変でないなら、語彙そのものが
   染色条件で汚染されていることになるので、目標に対して本質的な問題。
2. 0003/0004の結論はすべて「バッチ効果の上限 silhouette 0.175」を添えて読む。
3. 590枚（Low/Middle込み）で再実行。同一実験内に4用量が揃うので、
   「用量とともにクラスタ占有率が単調に増えるか」はバッチに影響されない検証軸になる。
   バッチは用量と直交するので、これは今ある中で最もバッチに頑健な検証。

## 追記（2026-09-14）: 染色正規化後の再検証

0005でMacenko正規化して埋め込み直し、0003→0004を再実行した結果
（`variant_key: cluster_mining_macenko` / `label_granularity_macenko`）。

- **H1は棄却されたまま**。3化合物の富化クラスタの重なりは
  phenobarbital ∩ thioacetamide が cluster 94 を1つ共有しただけで、他2ペアは空。
- バッチの上限（control 160枚を実験で分離）は 0.175 → 0.153 でほぼ変わらない。
  **残るバッチ署名は色ではない**（切片の厚み・固定・スキャナ光学など）。
- 部位 vs 化合物の分離度はどちらも上がった（部位 0.337→0.439、化合物 0.606→0.717）が、
  化合物が部位を大きく上回る関係は変わらない。

つまり0004の結論「所見名も部位も、形態の粒度には足りない」は、染色差では説明できない。
詳細は `experiments/0005_20260913_stain_normalized_embeddings/plan.md`。
