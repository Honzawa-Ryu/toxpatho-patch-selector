"""Does each cluster↔finding association survive a severity check?

finding_associations は「そのクラスタを持つスライドは、その所見が記録された
スライドに偏っているか」しか見ていない。1スライドには複数所見が併記されるので、
同じスライド上に居るだけの所見とも有意に結びつく。

そのクラスタが本当にその所見の組織を捉えているなら、**その所見が重症なスライド
ほど占有率が高い**はず。TG-GATEsはGRADE_TYPE(minimal/slight/moderate/severe)を
所見ごとに持っているので、所見あり群の中だけで occupancy と等級の順位相関を取る。
これはスライド単位の共起では説明できない量なので、対応の厳密さの直接検証になる。

SP_FLG=true（自然発生・背景病変、Liver 6,831行中1,201行）も分けて見る。
現在の finding_types はこれらを含んでおり、投与と無関係な所見がラベルに
混ざっている。

Usage: python scripts/grade_response_check.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reanalyze_cluster_thresholds import MANIFEST, ROOT, load_assignments  # noqa: E402

K = 1000
PRESENCE_EPS = 0.001
MIN_TREATED_PRESENCE = 0.0
Q_THRESHOLD = 0.05
N_FINDINGS = 36
GRADE_ORDER = {"minimal": 1, "slight": 2, "moderate": 3, "severe": 4}
MIN_SLIDES_FOR_RHO = 15


class _Q:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def per_finding_grades(manifest: pd.DataFrame) -> pd.DataFrame:
    """(slide_id, finding_type) -> grade_ord, is_spontaneous."""
    from lib.tggates_metadata import _as_int_key

    path = pd.read_csv(PROJECT_ROOT / "data/corrected/open_tggates_pathology.csv")
    path = path[path["ORGAN"] == "Liver"].copy()
    path = path.rename(columns={"EXP_ID": "exp_id", "GROUP_ID": "group_id",
                                "INDIVIDUAL_ID": "individual_id"})
    key = ["exp_id", "group_id", "individual_id"]
    for c in key:
        path[c] = _as_int_key(path[c])
    path["grade_ord"] = path["GRADE_TYPE"].map(GRADE_ORDER)
    path["spontaneous"] = path["SP_FLG"].astype(str).str.lower() == "true"

    man = manifest[["slide_id"] + key].copy()
    for c in key:
        man[c] = _as_int_key(man[c])
    merged = man.merge(path, on=key, how="inner")
    # 同一所見が複数トポグラフィで記録される場合は最重症を採る
    g = (merged.groupby(["slide_id", "FINDING_TYPE"])
         .agg(grade_ord=("grade_ord", "max"), spontaneous=("spontaneous", "all"))
         .reset_index()
         .rename(columns={"FINDING_TYPE": "finding_type"}))
    return g


def main() -> None:
    from lib.cluster_analysis import (
        cluster_statistics,
        finding_associations,
        occupancy_matrix,
        select_candidates,
    )

    manifest = pd.read_parquet(MANIFEST)
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest.sort_values("slide_id").reset_index(drop=True)
    pos_of = {s: i for i, s in enumerate(manifest["slide_id"])}

    grades = per_finding_grades(manifest)
    grades["slide_id"] = grades["slide_id"].astype(str)
    print(f"所見×スライドの等級レコード: {len(grades)}件 "
          f"(うち自然発生 {int(grades['spontaneous'].sum())}件)\n")

    slide_index, assignments = load_assignments(K, manifest)
    n_clusters = int(assignments.max()) + 1
    occ = occupancy_matrix(slide_index, assignments, len(manifest), n_clusters)
    stats = cluster_statistics(
        occ, manifest, presence_eps=PRESENCE_EPS,
        min_treated_presence=MIN_TREATED_PRESENCE, logger=_Q(),
    )
    cand = select_candidates(stats, q_threshold=Q_THRESHOLD, min_compounds=1)
    assoc = finding_associations(
        occ, manifest, cand, presence_eps=PRESENCE_EPS, max_finding_types=N_FINDINGS
    )
    sig = assoc[assoc["q_value"] < Q_THRESHOLD].copy()
    print(f"k={K}: 候補{len(cand)}クラスタ / 有意な(クラスタ,所見)組 {len(sig)}\n")

    rows = []
    for r in sig.itertuples():
        g = grades[grades["finding_type"] == r.finding_type]
        idx = np.array([pos_of[s] for s in g["slide_id"] if s in pos_of])
        gr = g.loc[[s in pos_of for s in g["slide_id"]], "grade_ord"].to_numpy()
        ok = ~np.isnan(gr)
        idx, gr = idx[ok], gr[ok]
        if len(idx) < MIN_SLIDES_FOR_RHO or len(np.unique(gr)) < 2:
            continue
        rho, p = spearmanr(occ[idx, r.cluster_id], gr)
        rows.append({
            "cluster_id": r.cluster_id, "finding_type": r.finding_type,
            "odds_ratio": r.odds_ratio, "assoc_q": r.q_value,
            "n_slides_with_finding": len(idx), "rho": rho, "rho_p": p,
        })

    gr_df = pd.DataFrame(rows)
    from scipy.stats import false_discovery_control
    gr_df["rho_q"] = false_discovery_control(np.clip(gr_df["rho_p"].to_numpy(), 0, 1), method="bh")
    gr_df.insert(0, "k", K)
    gr_df.to_csv(ROOT / f"grade_response_k{K}.csv", index=False)

    passed = gr_df[(gr_df["rho"] > 0) & (gr_df["rho_q"] < 0.05)]
    print("=== 等級との用量反応チェック ===")
    print(f"検定できた(クラスタ,所見)組: {len(gr_df)}")
    print(f"等級と正の相関あり(q<0.05): {len(passed)} ({len(passed)/len(gr_df):.1%})")
    print(f"相関rhoの分布: 中央値 {gr_df['rho'].median():.3f} / "
          f"rho>0 は {(gr_df['rho']>0).mean():.1%}\n")

    print("-- 所見ごとの通過率 --")
    per_f = gr_df.assign(passed=(gr_df["rho"] > 0) & (gr_df["rho_q"] < 0.05)).groupby("finding_type").agg(
        組数=("rho", "size"), 通過=("passed", "sum"), rho中央値=("rho", "median")
    )
    per_f["通過率"] = (per_f["通過"] / per_f["組数"]).round(2)
    print(per_f.sort_values("組数", ascending=False).round(3).to_string())

    print("\n-- 通過したクラスタ数（＝等級反応で裏付けられた候補）--")
    cl_pass = passed["cluster_id"].nunique()
    print(f"{cl_pass} / {len(cand)} クラスタ ({cl_pass/len(cand):.1%})")

    print("\n-- 等級反応が最も強い組 上位15 --")
    with pd.option_context("display.width", 200):
        print(passed.nlargest(15, "rho")[
            ["cluster_id", "finding_type", "odds_ratio", "n_slides_with_finding", "rho", "rho_q"]
        ].round(4).to_string(index=False))

    print(f"\nwrote {ROOT / f'grade_response_k{K}.csv'}")


if __name__ == "__main__":
    main()
