"""0017のコンセンサスの診断（k-meansは回し直さない）。保存済みの辺の共所属率・コア割り当てだけを使う。

1. patchごとの安定度（kNN近傍との平均共所属率）の分布。全体／kごと／slideの群（Control・投与群所見なし・所見あり）ごと
2. kごとの辺の共所属率の分布と、τを動かしたときに残る辺・割り当てられるpatchの割合
3. 未割当patchの中身（群・試験・組織割合）を、コアに入ったpatchと比べる
4. t-SNE（可視化で作った2万点）を安定度で色付け

usage: .venv/bin/python scripts/diagnose_consensus_cores.py [variant] [method] [tau]   既定: pilot leiden 0.8
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/0017_20261016_block_consensus_patterns"
GROUPS = [("Control", "#9aa3ad"), ("treated, no finding", "#5b9be6"), ("treated, with finding", "#e34948")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 150})


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    method = sys.argv[2] if len(sys.argv) > 2 else "leiden"
    tau = sys.argv[3] if len(sys.argv) > 3 else "0.8"
    run = ROOT / "outputs/0017_20261016_block_consensus_patterns" / variant
    out = run / "diagnostics"
    out.mkdir(exist_ok=True)
    cfg = yaml.safe_load(open(EXP / "config.yml"))

    ev = pd.read_parquet(run / "eval_patches.parquet")
    e = pd.read_parquet(run / "edges.parquet")
    a, b = e.a.to_numpy(), e.b.to_numpy()
    n = len(ev)
    m = pd.read_parquet(ROOT / cfg["manifest_parquet"])
    m["slide_id"] = m.slide_id.astype(str)
    s = ev[["slide_id"]].merge(m[["slide_id", "exp_id", "compound_name", "dose_level", "has_finding"]], on="slide_id", how="left")
    grp = np.where(s.dose_level == "Control", "Control", np.where(s.has_finding, "treated, with finding", "treated, no finding"))
    core = ev[f"{method}_tau{tau}"].to_numpy()

    def node_mean(r):
        return (np.bincount(a, weights=r, minlength=n) + np.bincount(b, weights=r, minlength=n)) / \
               np.maximum(np.bincount(a, minlength=n) + np.bincount(b, minlength=n), 1)

    rate_cols = ["rate"] + sorted(c for c in e.columns if c.startswith("rate_k"))
    stab = {c: node_mean(e[c].to_numpy()) for c in rate_cols}
    pd.DataFrame({"slide_id": ev.slide_id, "group": grp, "core": core, **{f"stab_{c}": v for c, v in stab.items()}}) \
        .to_parquet(out / "patch_stability.parquet", index=False)

    # 1・2: 分布
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    bins = np.linspace(0, 1, 41)
    ax = axes[0, 0]
    for c in rate_cols:
        ax.hist(e[c], bins=bins, histtype="step", lw=1.6, density=True, label=c.replace("rate", "all runs") if c == "rate" else c)
    ax.set_title("edge co-assignment rate (kNN pairs), by k", fontsize=11)
    ax.set_xlabel("fraction of runs in which the two neighbours share a cluster")
    ax.legend(frameon=False)
    ax = axes[0, 1]
    for c in rate_cols:
        ax.hist(stab[c], bins=bins, histtype="step", lw=1.6, density=True, label=c)
    ax.set_title("patch stability = mean co-assignment rate with its neighbours", fontsize=11)
    ax.legend(frameon=False)
    ax = axes[0, 2]
    taus = np.linspace(0.5, 0.95, 10)
    for c in rate_cols:
        ax.plot(taus, [(e[c] >= t).mean() for t in taus], marker="o", ms=3, label=f"{c}: edges kept")
    ax.set_xlabel("tau")
    ax.set_ylabel("fraction of kNN edges with rate >= tau")
    ax.set_title("how many edges survive a threshold", fontsize=11)
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    for g, col in GROUPS:
        ax.hist(stab["rate"][grp == g], bins=bins, histtype="step", lw=1.8, density=True, color=col, label=f"{g} ({(grp == g).sum():,})")
    ax.set_title("patch stability (all runs) by slide group", fontsize=11)
    ax.legend(frameon=False)
    ax = axes[1, 1]
    for lab, mask, col in [("in a core", core >= 0, "#1baf7a"), ("unassigned", core < 0, "#eb6834")]:
        ax.hist(stab["rate"][mask], bins=bins, histtype="step", lw=1.8, density=True, color=col, label=f"{lab} ({mask.sum():,})")
    ax.set_title(f"patch stability: in a core vs unassigned ({method}, tau={tau})", fontsize=11)
    ax.legend(frameon=False)
    ax = axes[1, 2]
    tab = pd.crosstab(grp, np.where(core >= 0, "in a core", "unassigned"), normalize="index")
    tab.plot.barh(stacked=True, ax=ax, color=["#1baf7a", "#eb6834"], width=.6)
    ax.set_xlabel("fraction of eval patches")
    ax.set_title("unassigned fraction by slide group", fontsize=11)
    ax.legend(frameon=False, loc="lower right")
    fig.suptitle(f"0017 consensus diagnostics ({variant}, {n:,} eval patches, {len(e):,} kNN edges)", fontsize=13, y=1.0)
    fig.savefig(out / "diagnostics_distributions.png")
    plt.close(fig)

    # 3: 未割当の中身（試験・群ごと）
    df = pd.DataFrame({"compound": s.compound_name, "group": grp, "unassigned": core < 0, "tissue_frac": ev.tissue_frac,
                       "stability": stab["rate"]})
    by_exp = df.groupby("compound").agg(n=("unassigned", "size"), unassigned_frac=("unassigned", "mean"),
                                        stability_median=("stability", "median")).sort_values("unassigned_frac")
    by_grp = df.groupby("group").agg(n=("unassigned", "size"), unassigned_frac=("unassigned", "mean"),
                                     stability_median=("stability", "median"), tissue_frac_median=("tissue_frac", "median"))
    by_exp.to_csv(out / "unassigned_by_compound.csv")
    by_grp.to_csv(out / "unassigned_by_group.csv")
    summ = pd.DataFrame([{"rate": c, "edge_rate_median": float(e[c].median()), "patch_stability_median": float(np.median(stab[c])),
                          **{f"edges_ge_{t}": float((e[c] >= t).mean()) for t in [0.7, 0.8, 0.9]}} for c in rate_cols])
    summ.to_csv(out / "rate_summary.csv", index=False)

    # 4: t-SNE上の安定度
    vis = sorted(EXP.glob("visualization_*"))
    tsne = vis[-1] / f"tsne_{method}_tau{tau}.parquet" if vis else None
    if variant == "pilot" and tsne and tsne.exists():
        t = pd.read_parquet(tsne)
        key = ev.assign(stab=stab["rate"], i=np.arange(n))
        # t-SNEは評価用patchの部分集合（同じ順の抽出）なので、slideとコア番号の並びで突き合わせる代わりに保存した順を使う
        sub = np.sort(np.random.default_rng(0).choice(n, size=len(t), replace=False))
        if (ev.slide_id.to_numpy()[sub] == t.slide_id.to_numpy()).all():
            fig, axes = plt.subplots(1, 2, figsize=(16, 7.5))
            sc = axes[0].scatter(t.tsne_x, t.tsne_y, c=stab["rate"][sub], cmap="viridis", s=1.5, vmin=0.3, vmax=1, rasterized=True)
            fig.colorbar(sc, ax=axes[0], fraction=.04).set_label("patch stability")
            axes[0].set_title("t-SNE colored by patch stability", fontsize=11)
            un = core[sub] < 0
            axes[1].scatter(t.tsne_x[~un], t.tsne_y[~un], color="#cfd3d8", s=1, rasterized=True, label="in a core")
            axes[1].scatter(t.tsne_x[un], t.tsne_y[un], color="#eb6834", s=1.5, rasterized=True, label="unassigned")
            axes[1].legend(markerscale=6, frameon=False)
            axes[1].set_title(f"unassigned patches ({method}, tau={tau})", fontsize=11)
            for ax in axes:
                ax.set_xticks([]), ax.set_yticks([])
            fig.savefig(out / "diagnostics_tsne.png")
            plt.close(fig)
    print(out)
    print(summ.round(3).to_string(index=False))
    print(by_grp.round(3).to_string())
    print(by_exp.round(3).to_string())


if __name__ == "__main__":
    main()
