"""0017（塊内のコンセンサスコア）の全体像を平面図にする。

A. 評価用patchのt-SNE（コア / 用量 / 所見 / 時点 / scanner）
B. 試験ごとのハイライト（13試験）
C. コアの地図（1点=1コア: x=Controlのpatch割合、y=上位1試験のシェア、大きさ=patch数、色=半分ずつのrunでの一致度）

評価用patchの埋め込みは0017と同じ手順・同じseedで作り直し、0017の eval_patches.parquet と座標で突き合わせる。
usage: .venv/bin/python scripts/visualize_consensus_cores.py [method] [tau]   既定: leiden 0.8
"""
import colorsys
import importlib.util
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments/0017_20261016_block_consensus_patterns"
RUN = ROOT / "outputs/0017_20261016_block_consensus_patterns/pilot"
DOSES = ["Control", "Low", "Middle", "High"]
DOSE_COLORS = ["#9aa3ad", "#a8cbf2", "#5b9be6", "#1f4f8f"]
DAYS = ["4 day", "8 day", "15 day", "29 day"]
DAY_COLORS = ["#a8cbf2", "#5b9be6", "#2a78d6", "#1f4f8f"]
SCAN_COLORS = {"SS1601": "#2a78d6", "SS1598": "#eb6834", "SS1224": "#1baf7a"}
GREY = "#dde1e5"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 150})

spec = importlib.util.spec_from_file_location("exp0017", EXP / "experiment.py")
exp0017 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp0017)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def palette(k):
    return np.asarray([colorsys.hsv_to_rgb((i * .61803398875) % 1, .55 + .3 * ((i % 3) / 2), .6 + .3 * ((i % 5) / 4)) for i in range(k)])


def rebuild_embedding(cfg):
    """0017のmain()と同じ手順で評価用patchの64次元埋め込みを作り直す（slide順・seedも同じ）。"""
    m = pd.read_parquet(ROOT / cfg["manifest_parquet"])
    m["slide_id"] = m.slide_id.astype(str)
    slides = m[m.exp_id.isin(cfg["exp_ids"])].sort_values(["exp_id", "dose", "sacrifice_period", "individual_id"]).reset_index(drop=True)
    scan = pd.read_csv(ROOT / "experiments/0010_20260925_pathology_benchmark_subset/label_audit/slide_scanner_meta.csv",
                       dtype={"slide_id": str})[["slide_id", "aperio.ScanScope ID"]].rename(columns={"aperio.ScanScope ID": "scanner"})
    slides = slides.merge(scan, on="slide_id", how="left", validate="one_to_one")

    class Quiet:
        def info(self, *_):
            pass
    feats, rows, xy = exp0017.load_subsets(slides, ROOT / cfg["features_dir"], cfg, Quiet())
    is_ctrl = (slides.dose_level == "Control").to_numpy()
    c = np.flatnonzero(is_ctrl[rows["pca"]])
    t = np.random.default_rng(cfg["seed"]).choice(np.flatnonzero(~is_ctrl[rows["pca"]]), size=len(c), replace=False)
    pca = PCA(n_components=cfg["pca_components"], svd_solver="randomized", random_state=cfg["seed"]).fit(feats["pca"][np.r_[c, t]])
    return slides, pca.transform(feats["eval"]).astype(np.float32), rows["eval"], xy


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "leiden"
    tau = sys.argv[2] if len(sys.argv) > 2 else "0.8"
    col = f"{method}_tau{tau}"
    cfg = yaml.safe_load(open(EXP / "config.yml"))
    out = EXP / f"visualization_{date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)

    import time
    t0 = time.time()
    cache = out / "eval_embedding.npz"
    if cache.exists():   # 埋め込みの作り直し（h5 1,020本の読み込み）は1回だけにする
        z = np.load(cache, allow_pickle=True)
        slides = pd.read_parquet(out / "slides.parquet")
        ev, erow, exy = z["ev"], z["erow"], z["exy"]
    else:
        slides, ev, erow, exy = rebuild_embedding(cfg)
        np.savez_compressed(cache, ev=ev, erow=erow, exy=exy)
        slides.drop(columns=["finding_types", "finding_sites"]).to_parquet(out / "slides.parquet", index=False)
    print(f"embedding ready: {len(ev):,} patches, {time.time() - t0:.0f}s", flush=True)
    kept = pd.read_parquet(RUN / "eval_patches.parquet")
    key = pd.DataFrame({"slide_id": slides.slide_id.to_numpy()[erow], "x": exy[:, 0], "y": exy[:, 1], "i": np.arange(len(erow))})
    j = kept.merge(key, on=["slide_id", "x", "y"], how="left", validate="one_to_one")
    if j.i.isna().any():
        raise ValueError("eval patches could not be matched to the rebuilt embedding")
    idx = j.i.to_numpy().astype(int)
    ev, row = ev[idx], erow[idx]
    core = j[col].to_numpy()
    s = slides.iloc[row].reset_index(drop=True)

    # t-SNEは4.75万点だと1時間を超えたため、2万点を無作為に抜いて描く（2026-10-17）
    sub = np.sort(np.random.default_rng(0).choice(len(ev), size=min(20000, len(ev)), replace=False))
    ev, row, core, s = ev[sub], row[sub], core[sub], s.iloc[sub].reset_index(drop=True)
    t0 = time.time()
    with threadpool_limits(limits=8):
        xy = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", max_iter=1000,
                  random_state=0, n_jobs=8).fit_transform(ev)
    print(f"t-SNE done: {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame({"slide_id": s.slide_id, "core": core, "tsne_x": xy[:, 0], "tsne_y": xy[:, 1]}) \
        .to_parquet(out / f"tsne_{col}.parquet", index=False)
    order = np.random.default_rng(0).permutation(len(xy))

    # A. t-SNE 6枚
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    ax = axes[0, 0]
    un = core < 0
    ax.scatter(*xy[un].T, color=GREY, s=1, linewidths=0, rasterized=True)
    pal = palette(core.max() + 1)
    o = order[~un[order]]
    ax.scatter(*xy[o].T, c=pal[core[o]], s=1.5, linewidths=0, rasterized=True)
    for c in range(core.max() + 1):
        cx, cy = np.median(xy[core == c], axis=0)
        ax.text(cx, cy, str(c), fontsize=6.5, ha="center", va="center", color="black",
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=.7))
    ax.set_title(f"cores ({method}, tau={tau}): {core.max() + 1} cores, grey = unassigned ({un.mean():.0%})", fontsize=11)
    for ax, title, values, cats, colors in [
            (axes[0, 1], "dose group", s.dose_level.to_numpy(), DOSES, DOSE_COLORS),
            (axes[1, 0], "time point", s.sacrifice_period.to_numpy(), DAYS, DAY_COLORS),
            (axes[1, 1], "scanner", s.scanner.to_numpy(), list(SCAN_COLORS), list(SCAN_COLORS.values()))]:
        for c_, col_ in zip(cats, colors):
            mm = values[order] == c_
            ax.scatter(*xy[order][mm].T, color=col_, s=1.2, alpha=.7, linewidths=0, rasterized=True, label=c_)
        ax.legend(markerscale=6, frameon=False, loc="lower left")
        ax.set_title(title, fontsize=11)
    ax = axes[0, 2]
    grp = np.where(s.dose_level == "Control", "Control", np.where(s.has_finding, "treated, slide with finding", "treated, no finding recorded"))
    for g, c_ in [("Control", "#9aa3ad"), ("treated, no finding recorded", "#a8cbf2"), ("treated, slide with finding", "#e34948")]:
        mm = grp[order] == g
        ax.scatter(*xy[order][mm].T, color=c_, s=1.2, alpha=.7, linewidths=0, rasterized=True, label=g)
    ax.legend(markerscale=6, frameon=False, loc="lower left")
    ax.set_title("slide-level finding record (not patch diagnosis)", fontsize=11)
    ax = axes[1, 2]
    exps = sorted(s.exp_id.unique())
    epal = palette(len(exps))
    for e, c_ in zip(exps, epal):
        mm = s.exp_id.to_numpy()[order] == e
        ax.scatter(*xy[order][mm].T, color=c_, s=1.2, alpha=.7, linewidths=0, rasterized=True,
                   label=f"e{e} {s[s.exp_id == e].compound_name.iloc[0][:18]}")
    ax.legend(markerscale=6, frameon=False, loc="lower left", fontsize=7, ncol=2)
    ax.set_title("experiment (13 colors; see B for one panel per experiment)", fontsize=11)
    for a in axes.flat:
        bare(a)
    fig.suptitle(f"0017: {len(xy):,} eval patches (random subset) from 13 experiments (largest Ctrl block), one t-SNE", fontsize=14, y=1.0)
    fig.savefig(out / f"A_tsne_overview_{col}.png")
    plt.close(fig)

    # B. 試験ごと
    fig, axes = plt.subplots(3, 5, figsize=(22, 13))
    for ax, e in zip(axes.flat, exps):
        mm = s.exp_id.to_numpy() == e
        ax.scatter(*xy[~mm].T, color=GREY, s=.4, linewidths=0, rasterized=True)
        for d, c_ in zip(DOSES, DOSE_COLORS):
            md = mm & (s.dose_level.to_numpy() == d)
            ax.scatter(*xy[md].T, color=c_, s=1.2, linewidths=0, rasterized=True)
        ax.set_title(f"e{e} {s[s.exp_id == e].compound_name.iloc[0][:22]}", fontsize=9)
    for ax in axes.flat:
        bare(ax)
    axes.flat[-1].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c_, label=d) for d, c_ in zip(DOSES, DOSE_COLORS)],
                         loc="center", frameon=False, fontsize=11)
    fig.suptitle("Each experiment on the same t-SNE, colored by dose group (grey = other experiments)", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out / f"B_tsne_by_experiment_{col}.png")
    plt.close(fig)

    # C. コアの地図
    tab = pd.read_csv(RUN / f"cores_{col}.csv")
    fig, ax = plt.subplots(figsize=(13, 9))
    sc = ax.scatter(tab.ctrl_patch_frac, tab.top_exp_share, s=tab.n_patches / 3, c=tab.split_half_jaccard_median,
                    cmap="viridis", vmin=0, vmax=1, alpha=.8, edgecolors="white", linewidths=.6)
    for r in tab.itertuples():
        lab = str(r.core)
        if isinstance(r.top_lift_findings, str) and r.top_lift_findings:
            lab += " " + r.top_lift_findings.split(";")[0].replace("Degeneration, granular, eosinophilic", "gran. eos. degen.")[:34]
        ax.annotate(lab, (r.ctrl_patch_frac, r.top_exp_share), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("fraction of the core's patches that come from Control slides")
    ax.set_ylabel("share of the core's patches from its top experiment")
    fig.colorbar(sc, ax=ax, fraction=.03).set_label("split-half Jaccard (stability)")
    ax.set_title(f"Core map ({method}, tau={tau}): one point per core, size = patches; "
                 "labels: core id and the finding most enriched among its treated slides (x fold, if >=30% of slides)", fontsize=10)
    fig.savefig(out / f"C_core_map_{col}.png")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
