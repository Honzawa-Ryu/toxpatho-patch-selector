"""Compare the two ways of defining the reference ("control") side of the
candidate-selection test, and sweep the thresholds that control how many
clusters survive.

  A. dose_level == "Control"（従来）
     投与群2,570枚のうち1,323枚は所見なし＝形態的に正常なので、陽性側に
     半分ノイズが混ざる。逆にcontrol 871枚中40枚は自然発生病変あり。
  B. has_finding == False（所見なしを対照にする）
     「病変の形態を拾う」という目的に直接対応する対比。

注意: Bでは選定に使うラベルとLOCOの評価ラベル(has_finding)が同じになる。
化合物単位でhold-outしているので循環ではないが、Aの数値(選定=dose_level、
評価=has_finding という別ラベル同士)とは意味が違うので直接比較しないこと。

Usage: python scripts/compare_reference_groups.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reanalyze_cluster_thresholds import MANIFEST, ROOT, load_assignments  # noqa: E402

K = 1000
PRESENCE_EPS = [0.001, 0.005, 0.01]
MIN_TREATED_PRESENCE = [0.0, 0.02, 0.05]
Q_THRESHOLD = 0.05
N_FINDINGS = 36


class _Q:
    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


def main() -> None:
    from lib.cluster_analysis import (
        cluster_statistics,
        control_group_by_dose,
        control_group_by_finding,
        finding_associations,
        leave_one_compound_out_auroc,
        occupancy_matrix,
        select_candidates,
    )

    manifest = pd.read_parquet(MANIFEST)
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest.sort_values("slide_id").reset_index(drop=True)
    quiet = _Q()

    counts = manifest.explode("finding_types")["finding_types"].dropna().value_counts()
    n_test = int((counts >= 5).sum())

    slide_index, assignments = load_assignments(K, manifest)
    n_clusters = int(assignments.max()) + 1
    occ = occupancy_matrix(slide_index, assignments, len(manifest), n_clusters)

    refs = {"A_dose_level": control_group_by_dose, "B_has_finding": control_group_by_finding}
    rows = []
    for ref_name, ref in refs.items():
        for eps in PRESENCE_EPS:
            stats = cluster_statistics(
                occ, manifest, presence_eps=eps, min_treated_presence=0.0,
                logger=quiet, reference_selector=ref,
            )
            for mtp in MIN_TREATED_PRESENCE:
                st = cluster_statistics(
                    occ, manifest, presence_eps=eps, min_treated_presence=mtp,
                    logger=quiet, reference_selector=ref,
                ) if mtp else stats
                cand = select_candidates(st, q_threshold=Q_THRESHOLD, min_compounds=1)
                row = {"参照群": ref_name, "presence_eps": eps, "mtp": mtp, "候補数": len(cand)}
                if len(cand):
                    assoc = finding_associations(
                        occ, manifest, cand, presence_eps=eps, max_finding_types=n_test
                    )
                    sig = assoc[assoc["q_value"] < Q_THRESHOLD]
                    row["所見と対応"] = sig["cluster_id"].nunique()
                    row["対応率"] = round(sig["cluster_id"].nunique() / len(cand), 3)
                    row["拾えた所見数"] = sig["finding_type"].nunique()
                    row[f"カバー率(/{n_test})"] = round(sig["finding_type"].nunique() / n_test, 3)
                    pooled, _ = leave_one_compound_out_auroc(
                        occ, manifest, presence_eps=eps, min_treated_presence=mtp,
                        q_threshold=Q_THRESHOLD, min_compounds=1, logger=quiet,
                        reference_selector=ref,
                    )
                    row["auroc_loco"] = round(pooled, 4) if pooled is not None else None
                rows.append(row)
                print(f"{ref_name} eps={eps} mtp={mtp}: {len(cand)}候補", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "reference_group_comparison.csv", index=False)
    print("\n=== 参照群の定義 × 閾値 （k=1000, min_compounds撤廃）===")
    with pd.option_context("display.width", 220):
        print(df.to_string(index=False))
    print(f"\nwrote {ROOT / 'reference_group_comparison.csv'}")


if __name__ == "__main__":
    main()
