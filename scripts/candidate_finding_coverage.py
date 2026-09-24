"""How trustworthy is exp0009's candidate set, and how much of the corpus's
finding vocabulary does it actually cover?

Three questions:
  1. 候補クラスタのうち何割が所見と有意に対応づくか（precision側）
  2. 対応づかない候補（orphan）は何者か。閾値を緩めるほど増えるなら緩すぎる
  3. コーパスに実在する所見のうち何割を拾えているか、拾えない所見は何か（recall側）

Note: 本編のfinding_associationsは max_finding_types=15 で上位15所見しか
検定していない。全46種のうち31種が未検定のまま「対応なし」に数えられて
しまうので、ここでは5枚以上ある36種すべてを検定対象にする。

Usage: python scripts/candidate_finding_coverage.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reanalyze_cluster_thresholds import (  # noqa: E402
    MANIFEST,
    ROOT,
    compound_balance,
    load_assignments,
)

K_VALUES = [1000, 2000]
MIN_TREATED_PRESENCE = [0.0, 0.02, 0.05, 0.1]
MIN_COMPOUNDS = 1
Q_THRESHOLD = 0.05
PRESENCE_EPS = 0.001
MIN_SLIDES_PER_FINDING = 5


class _QuietLogger:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


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
    quiet = _QuietLogger()

    finding_counts = manifest.explode("finding_types")["finding_types"].dropna().value_counts()
    testable = finding_counts[finding_counts >= MIN_SLIDES_PER_FINDING]
    n_test = len(testable)
    print(f"検定対象の所見: {n_test}種（{MIN_SLIDES_PER_FINDING}枚以上）/ 全{len(finding_counts)}種\n")

    precision_rows, coverage_rows, orphan_rows = [], [], []
    for k in K_VALUES:
        slide_index, assignments = load_assignments(k, manifest)
        n_clusters = int(assignments.max()) + 1
        occ = occupancy_matrix(slide_index, assignments, len(manifest), n_clusters)
        bal = compound_balance(slide_index, assignments, manifest, n_clusters)

        for mtp in MIN_TREATED_PRESENCE:
            stats = cluster_statistics(
                occ, manifest, presence_eps=PRESENCE_EPS,
                min_treated_presence=mtp, logger=quiet,
            )
            cand = select_candidates(stats, q_threshold=Q_THRESHOLD, min_compounds=MIN_COMPOUNDS)
            if len(cand) == 0:
                precision_rows.append({"k": k, "mtp": mtp, "n_candidates": 0})
                continue

            assoc = finding_associations(
                occ, manifest, cand, presence_eps=PRESENCE_EPS, max_finding_types=n_test
            )
            sig = assoc[assoc["q_value"] < Q_THRESHOLD]
            matched = set(sig["cluster_id"].unique())
            orphans = [c for c in cand if c not in matched]

            info = stats.merge(bal, on="cluster_id").set_index("cluster_id")
            m_rows = info.loc[sorted(matched)] if matched else info.iloc[:0]
            o_rows = info.loc[sorted(orphans)] if orphans else info.iloc[:0]

            precision_rows.append({
                "k": k, "mtp": mtp,
                "n_candidates": len(cand),
                "所見と対応": len(matched),
                "対応率": round(len(matched) / len(cand), 3),
                "orphan": len(orphans),
                "拾えた所見数": sig["finding_type"].nunique(),
                f"所見カバー率(/{n_test})": round(sig["finding_type"].nunique() / n_test, 3),
            })

            def med(df, col):
                return round(float(df[col].median()), 3) if len(df) else None

            orphan_rows.append({
                "k": k, "mtp": mtp,
                "対応_中央値_投与スライド数": med(m_rows, "n_treated_slides"),
                "orphan_中央値_投与スライド数": med(o_rows, "n_treated_slides"),
                "対応_中央値_実験数": med(m_rows, "n_experiments"),
                "orphan_中央値_実験数": med(o_rows, "n_experiments"),
                "対応_中央値_patch数": med(m_rows, "n_treated_patches"),
                "orphan_中央値_patch数": med(o_rows, "n_treated_patches"),
                "orphan_単一実験の割合": round(float((o_rows["n_experiments"] <= 1).mean()), 3) if len(o_rows) else None,
                "対応_単一実験の割合": round(float((m_rows["n_experiments"] <= 1).mean()), 3) if len(m_rows) else None,
            })

            if mtp == 0.0:
                covered = set(sig["finding_type"].unique())
                for f, n in testable.items():
                    hit = sig[sig["finding_type"] == f]
                    coverage_rows.append({
                        "k": k, "finding_type": f, "スライド数": int(n),
                        "拾えた": f in covered,
                        "紐づいたクラスタ数": int(hit["cluster_id"].nunique()),
                        "最大OR": round(float(hit["odds_ratio"].max()), 1) if len(hit) else None,
                    })
        print(f"k={k} done", flush=True)

    prec = pd.DataFrame(precision_rows)
    print("\n=== Q1/Q3: 候補の対応率と所見カバー率 ===")
    print(prec.to_string(index=False))

    orph = pd.DataFrame(orphan_rows)
    print("\n=== Q2: orphan（所見と対応しない候補）の素性 ===")
    print(orph.to_string(index=False))

    cov = pd.DataFrame(coverage_rows)
    cov.to_csv(ROOT / "finding_coverage.csv", index=False)
    print("\n=== Q3: 所見ごとの拾えた/拾えない（mtp=0.0）===")
    for k in K_VALUES:
        sub = cov[cov["k"] == k].sort_values("スライド数", ascending=False)
        miss = sub[~sub["拾えた"]]
        print(f"\n-- k={k}: {int(sub['拾えた'].sum())}/{len(sub)}種を捕捉。拾えなかった{len(miss)}種 --")
        print(miss[["finding_type", "スライド数"]].to_string(index=False))
    print(f"\nwrote {ROOT / 'finding_coverage.csv'}")


if __name__ == "__main__":
    main()
