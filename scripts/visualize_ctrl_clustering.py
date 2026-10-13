"""0015（無所見Ctrlの共通クラスタリング）の様子を、variantを並べて図にする。

1a. patchのt-SNE（クラスタ / 時点 / scanner）。variantごとに1行
1b. 試験ごとのハイライト。variantごとに1行
2.  slide × クラスタの占有率ヒートマップ。variantを横に並べる
3.  slide間JS距離のMDS（時点 / scanner / 試験）。variantごとに1行

各variantの代表run（k=100、抽出seed 0、初期化seed 0）の重心で割り当てる（0015本体と同じ割当）。
t-SNE・MDSはvariantごとに別々に計算するので、variant間で点の位置は対応しない。
usage: .venv/bin/python scripts/visualize_ctrl_clustering.py [variant ...]   既定: s219_raw s219_macenko
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
from matplotlib.lines import Line2D
from scipy.spatial.distance import cdist
from sklearn.manifold import MDS, TSNE
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/0015_20261012_ctrl_batch_structure"
RUNS = ROOT / "outputs/0015_20261012_ctrl_batch_structure"
REP, K = "k100_s0_i0", 100
PER_SLIDE = 40
SEED = 0
DAYS = ["4 day", "8 day", "15 day", "29 day"]
DAY_COLORS = ["#a8cbf2", "#5b9be6", "#2a78d6", "#1f4f8f"]          # 時点: 単色の明→暗
SCAN_COLORS = {"SS1601": "#2a78d6", "SS1598": "#eb6834", "SS1224": "#1baf7a"}
EXP_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GREY = "#cfd3d8"
TITLE = {"main": "no stain normalization, 830 slides", "s219_raw": "no stain normalization",
         "s219_macenko": "Macenko stain normalization"}

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.dpi": 160})

spec = importlib.util.spec_from_file_location("exp0015", EXP / "experiment.py")
exp0015 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp0015)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def palette(k):
    return np.asarray([colorsys.hsv_to_rgb((i * .61803398875) % 1, .55 + .3 * ((i % 3) / 2),
                                           .65 + .25 * ((i % 5) / 4)) for i in range(k)])


def load_variant(variant, config):
    """0015本体と同じ入力（同じ前処理・同じPCA）を作り、各slideからPER_SLIDE個を抽出する。"""
    run = RUNS / variant
    slides = pd.read_csv(run / "slides.csv", dtype={"slide_id": str})
    vcfg = config["variants"][variant]
    if vcfg["source"] == "cache":
        feats, _, counts = exp0015.load_from_cache(ROOT / config["embedding_cache_dir"], slides.slide_id)
    else:
        feats, _, counts, _ = exp0015.load_from_h5(ROOT / vcfg["features_dir"], slides.slide_id, vcfg)
    assert np.array_equal(counts, slides.n_patches.to_numpy()), "patch counts differ from the 0015 run"
    rng = np.random.default_rng(SEED)
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    pick = np.concatenate([o + np.sort(rng.choice(c, size=min(PER_SLIDE, c), replace=False)) for o, c in zip(offsets, counts)])
    row = np.repeat(np.arange(len(slides)), [min(PER_SLIDE, c) for c in counts])
    sample = feats[pick]
    centers = np.load(run / "centers.npz")[REP]
    labels = cdist(sample, centers).argmin(1)
    with threadpool_limits(limits=8):
        xy = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", max_iter=1000,
                  random_state=SEED, n_jobs=8).fit_transform(sample)
    occ = np.load(run / "occupancy.npz")[REP]
    bias = pd.read_csv(run / "cluster_bias.csv").query("k == @K and sample_seed == 0 and init_seed == 0")
    outlier = pd.read_csv(run / "exp_stability.csv").query("k == @K")
    return dict(slides=slides, xy=xy, labels=labels, row=row, occ=occ, bias=bias, outlier=outlier)


def fig_tsne(data, out):
    fig, axes = plt.subplots(len(data), 3, figsize=(18, 6 * len(data)), squeeze=False)
    for r, (v, d) in enumerate(data.items()):
        xy, row, s = d["xy"], d["row"], d["slides"]
        order = np.random.default_rng(SEED).permutation(len(xy))
        ax = axes[r]
        ax[0].scatter(*xy[order].T, c=palette(K)[d["labels"][order]], s=2, alpha=.7, linewidths=0, rasterized=True)
        ax[0].set_title(f"{TITLE[v]}\ncluster (k={K}, colors identify clusters only)", fontsize=11)
        day = s.sacrifice_period.to_numpy()[row]
        for dd, c in zip(DAYS, DAY_COLORS):
            m = day[order] == dd
            ax[1].scatter(*xy[order][m].T, color=c, s=2, alpha=.7, linewidths=0, rasterized=True, label=dd)
        ax[1].set_title(f"{TITLE[v]}\ntime point", fontsize=11)
        sc = s.scanner.to_numpy()[row]
        for name, c in SCAN_COLORS.items():
            m = sc[order] == name
            if m.any():
                ax[2].scatter(*xy[order][m].T, color=c, s=2, alpha=.7, linewidths=0, rasterized=True,
                              label=f"{name} ({(s.scanner == name).sum()} slides)")
        ax[2].set_title(f"{TITLE[v]}\nscanner", fontsize=11)
        for a in ax[1:]:
            a.legend(markerscale=4, frameon=False, loc="lower left", fontsize=8)
        for a in ax:
            bare(a)
    n = len(next(iter(data.values()))["slides"])
    fig.suptitle(f"No-finding Ctrl patches, t-SNE per row ({n} slides x {PER_SLIDE} patches). "
                 "Layouts are computed separately per row.", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out / "1a_tsne_cluster_time_scanner.png")
    plt.close(fig)


def fig_tsne_by_exp(data, out):
    exps = sorted(next(iter(data.values()))["slides"].exp_id.unique())
    many = len(exps) > len(EXP_COLORS)
    if many:  # 試験が多いときは1試験1パネル（外れ度順）
        for v, d in data.items():
            o = d["outlier"].sort_values("mean_score", ascending=False)
            cols = 8
            fig, axes = plt.subplots(int(np.ceil(len(o) / cols)), cols, figsize=(cols * 2.2, np.ceil(len(o) / cols) * 2.35))
            e = d["slides"].exp_id.to_numpy()[d["row"]]
            for ax, r in zip(axes.flat, o.itertuples()):
                m = e == r.exp_id
                ax.scatter(*d["xy"][~m].T, color=GREY, s=.4, alpha=.35, linewidths=0, rasterized=True)
                ax.scatter(*d["xy"][m].T, color="#2a78d6", s=1.2, alpha=.8, linewidths=0, rasterized=True)
                ax.set_title(f"e{r.exp_id} {r.compound[:18]}\nscore {r.mean_score:.2f}", fontsize=7)
            for ax in axes.flat:
                bare(ax)
            fig.suptitle(f"{TITLE[v]}: each experiment's Ctrl patches (blue), sorted by outlier score", fontsize=11, y=1.0)
            fig.tight_layout()
            fig.savefig(out / f"1b_tsne_by_experiment_{v}.png")
            plt.close(fig)
        return
    fig, axes = plt.subplots(1, len(data), figsize=(9 * len(data), 8), squeeze=False)
    for ax, (v, d) in zip(axes[0], data.items()):
        e = d["slides"].exp_id.to_numpy()[d["row"]]
        comp = d["slides"].groupby("exp_id").compound_name.first()
        order = np.random.default_rng(SEED).permutation(len(e))
        for ex, c in zip(exps, EXP_COLORS):
            m = e[order] == ex
            ax.scatter(*d["xy"][order][m].T, color=c, s=2, alpha=.7, linewidths=0, rasterized=True, label=f"e{ex} {comp[ex][:24]}")
        ax.set_title(f"{TITLE[v]}: experiment", fontsize=12)
        ax.legend(markerscale=5, frameon=False, fontsize=8, loc="lower left")
        bare(ax)
    fig.suptitle("Same t-SNE as 1a, colored by experiment (= compound)", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out / "1b_tsne_by_experiment.png")
    plt.close(fig)


def heatmap_panel(fig, gs, d, title, show_labels):
    slides, occ, bias = d["slides"], d["occ"], d["bias"]
    rows = slides.assign(_d=slides.sacrifice_period.map({x: i for i, x in enumerate(DAYS)})) \
        .sort_values(["exp_id", "_d", "individual_id"]).index.to_numpy()
    exp_order = {e: i for i, e in enumerate(slides.exp_id.iloc[rows].unique())}
    b = bias.assign(_o=bias.top1_exp.map(exp_order)).sort_values(["_o", "top1_share"], ascending=[True, False])
    m = np.sqrt(occ[np.ix_(rows, b.cluster.to_numpy())])
    sub = gs.subgridspec(1, 3, width_ratios=[0.25, 0.25, 12], wspace=0.02)
    ax_d, ax_s, ax = fig.add_subplot(sub[0]), fig.add_subplot(sub[1]), fig.add_subplot(sub[2])
    im = ax.imshow(m, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=np.quantile(m, .995))
    day = slides.sacrifice_period.iloc[rows].map({x: i for i, x in enumerate(DAYS)}).to_numpy()
    ax_d.imshow(np.array([matplotlib.colors.to_rgb(DAY_COLORS[i]) for i in day])[:, None, :], aspect="auto")
    ax_s.imshow(np.array([matplotlib.colors.to_rgb(SCAN_COLORS[s]) for s in slides.scanner.iloc[rows]])[:, None, :], aspect="auto")
    ex = slides.exp_id.iloc[rows].to_numpy()
    bounds = np.flatnonzero(np.r_[True, ex[1:] != ex[:-1], True])
    for y in bounds[1:-1]:
        ax.axhline(y - .5, color="white", lw=.8)
    comp = slides.groupby("exp_id").compound_name.first()
    ax_d.set_yticks((bounds[:-1] + bounds[1:]) / 2 - .5,
                    [f"e{e} {comp[e][:20]}" for e in ex[bounds[:-1]]] if show_labels else [], fontsize=7)
    ax_d.set_xticks([0], ["time"], fontsize=7)
    ax_s.set_yticks([])
    ax_s.set_xticks([0], ["scan"], fontsize=7)
    biased = b.top1_share.to_numpy() >= 0.5
    ax.set_xticks(np.flatnonzero(biased), [""] * biased.sum())
    ax.tick_params(axis="x", length=4, color="#eb6834", width=1.5)
    ax.set_yticks([])
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(f"{K} clusters ordered by top-1 experiment (orange ticks: top-1 share >= 50%)")
    for a in (ax, ax_d, ax_s):
        for s in a.spines.values():
            s.set_visible(False)
    return im, ax


def fig_heatmap(data, out):
    n = len(next(iter(data.values()))["slides"])
    fig = plt.figure(figsize=(9 * len(data), max(8, n / 55)))
    gs = fig.add_gridspec(1, len(data), wspace=0.12)
    for i, (v, d) in enumerate(data.items()):
        im, ax = heatmap_panel(fig, gs[i], d, TITLE[v], show_labels=True)
        fig.colorbar(im, ax=ax, fraction=.03, pad=.01).set_label("sqrt(occupancy)")
    handles = [Line2D([], [], marker="s", ls="", color=c, label=d) for d, c in zip(DAYS, DAY_COLORS)] + \
              [Line2D([], [], marker="s", ls="", color=c, label=s) for s, c in SCAN_COLORS.items()]
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(.5, .97))
    fig.suptitle(f"Slide x cluster occupancy ({n} no-finding Ctrl slides, representative run {REP})", fontsize=14, y=1.0)
    fig.savefig(out / "2_occupancy_heatmap.png")
    plt.close(fig)


def fig_mds(data, out):
    fig, axes = plt.subplots(len(data), 3, figsize=(18, 6 * len(data)), squeeze=False)
    for r, (v, d) in enumerate(data.items()):
        s, ax = d["slides"], axes[r]
        dist = exp0015.js_distance_matrix(d["occ"])
        xy = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED, n_init=4,
                 normalized_stress="auto").fit_transform(dist)
        for dd, c in zip(DAYS, DAY_COLORS):
            m = s.sacrifice_period.to_numpy() == dd
            ax[0].scatter(*xy[m].T, color=c, s=18, alpha=.85, linewidths=0, label=dd)
        for name, c in SCAN_COLORS.items():
            m = s.scanner.to_numpy() == name
            if m.any():
                ax[1].scatter(*xy[m].T, color=c, s=18, alpha=.85, linewidths=0, label=name)
        exps = sorted(s.exp_id.unique())
        comp = s.groupby("exp_id").compound_name.first()
        if len(exps) <= len(EXP_COLORS):
            for ex, c in zip(exps, EXP_COLORS):
                m = s.exp_id.to_numpy() == ex
                ax[2].scatter(*xy[m].T, color=c, s=22, alpha=.9, linewidths=.5, edgecolors="white", label=f"e{ex} {comp[ex][:20]}")
            ax[2].set_title(f"{TITLE[v]}\nexperiment (= compound)", fontsize=11)
        else:
            top = d["outlier"].sort_values("mean_score", ascending=False).head(5)
            rest = ~s.exp_id.isin(top.exp_id).to_numpy()
            ax[2].scatter(*xy[rest].T, color=GREY, s=12, alpha=.7, linewidths=0, label="other experiments")
            for t, c in zip(top.itertuples(), EXP_COLORS):
                m = s.exp_id.to_numpy() == t.exp_id
                ax[2].scatter(*xy[m].T, color=c, s=22, alpha=.95, linewidths=.5, edgecolors="white", label=f"e{t.exp_id} {t.compound[:18]}")
            ax[2].set_title(f"{TITLE[v]}\ntop 5 experiments by outlier score", fontsize=11)
        ax[0].set_title(f"{TITLE[v]}\ntime point", fontsize=11)
        ax[1].set_title(f"{TITLE[v]}\nscanner", fontsize=11)
        for a in ax:
            a.legend(frameon=False, fontsize=7.5, loc="lower left")
            bare(a)
    fig.suptitle(f"Slides placed by Jensen-Shannon distance between cluster occupancies (MDS per row, run {REP})",
                 fontsize=14, y=1.0)
    fig.tight_layout()
    fig.savefig(out / "3_slide_mds.png")
    plt.close(fig)


def main():
    variants = sys.argv[1:] or ["s219_raw", "s219_macenko"]
    out = EXP / f"clustering_visualization_{date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(open(EXP / "config.yml"))
    data = {v: load_variant(v, config) for v in variants}
    for v, d in data.items():
        pd.DataFrame({"slide_id": d["slides"].slide_id.to_numpy()[d["row"]], "cluster": d["labels"],
                      "tsne_x": d["xy"][:, 0], "tsne_y": d["xy"][:, 1]}).to_parquet(out / f"sample_layout_{v}.parquet", index=False)
    fig_tsne(data, out)
    fig_tsne_by_exp(data, out)
    fig_heatmap(data, out)
    fig_mds(data, out)
    print(out)


if __name__ == "__main__":
    main()
