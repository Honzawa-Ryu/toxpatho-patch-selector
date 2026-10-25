import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

import torch
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_config(exp_dir: Path) -> dict:
    """Load config.yml from the experiment directory."""
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")   # run_slurm.shが渡す。読むのは常に実験ディレクトリのconfig.yml
    parser.add_argument("--source_variant", default=None)   # 0017のvariant（既定はconfig.ymlのsource_variant）
    parser.add_argument("--threshold", type=float, default=None)   # 1段目の閾値を固定する（既定はa-b最大。2026-10-23）
    parser.add_argument("--stage2_variant", default=None)   # 2段目の埋め込みだけ別の0017 variantから取る（同じslide・評価用patch。2026-10-23）
    return parser.parse_args()


DOSES = ["Control", "Low", "Middle", "High"]
DOSE_COLORS = ["#9aa3ad", "#a8cbf2", "#5b9be6", "#1f4f8f"]
DAYS = ["4 day", "8 day", "15 day", "29 day"]
SCAN_COLORS = {"SS1601": "#2a78d6", "SS1598": "#eb6834", "SS1224": "#1baf7a"}
GROUPS = [("Control", "#9aa3ad"), ("treated, no finding", "#5b9be6"), ("treated, finding", "#e34948")]
GREY = "#dde1e5"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 150})


def load_exp0017(project_root: Path, name: str):
    spec = importlib.util.spec_from_file_location("exp0017", project_root / "experiments" / name / "experiment.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def as_list(v):
    return [] if v is None else sorted(set(v))


def short(name, n=34):
    return name.replace("Degeneration, granular, eosinophilic", "gran. eos. degen.")[:n]


def palette(k):
    import colorsys
    return np.asarray([colorsys.hsv_to_rgb((i * .61803398875) % 1, .55 + .3 * ((i % 3) / 2), .6 + .3 * ((i % 5) / 4)) for i in range(k)])


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def tsne(x, seed):
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=16):
        return TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", max_iter=1000, random_state=seed, n_jobs=16).fit_transform(x)


def ctrl_score(labels: np.ndarray, is_ctrl: np.ndarray) -> np.ndarray:
    """各runで自分が入ったクラスタのControl割合を出し、runで平均する（低いほどControlから遠い）。"""
    out = np.zeros(labels.shape[1])
    for lab in labels:
        n = np.bincount(lab)
        c = np.bincount(lab, weights=is_ctrl.astype(float), minlength=len(n))
        out += (c / np.maximum(n, 1))[lab]
    return out / len(labels)


def pick_one_per_slide(order, row, n):
    picks, seen = [], set()
    for p in order:
        if row[p] not in seen:
            picks.append(p)
            seen.add(row[p])
        if len(picks) == n:
            break
    return picks


def stage1_figures(score, thr, curve, kept, group, sl, out):
    """1段目の確認用の図（スコアの分布、閾値の曲線、群ごとの残る割合）。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    ax = axes[0]
    bins = np.linspace(0, score.max(), 80)
    for g, c in GROUPS:
        m = group == g
        ax.hist(score[m], bins=bins, histtype="step", lw=1.8, color=c, density=True, label=f"{g} ({m.sum():,} patches)")
    ax.axvline(thr, color="black", ls="--", lw=1)
    ax.set_xlabel("Control-likeness score (mean Control fraction of the patch's cluster over 60 runs)")
    ax.set_ylabel("density")
    ax.legend(frameon=False)
    ax.set_title(f"score distribution; dashed = threshold {thr:.3f} (keep patches at or below)", fontsize=10)
    ax = axes[1]
    ax.plot(curve.threshold, curve.a_finding, color="#e34948", label="a: finding-slide patches kept")
    ax.plot(curve.threshold, curve.b_control, color="#9aa3ad", label="b: Control patches kept")
    ax.plot(curve.threshold, curve.treated_no_finding, color="#5b9be6", ls=":", label="treated, no finding: kept")
    ax.plot(curve.threshold, curve.J, color="black", lw=2, label="a - b")
    ax.axvline(thr, color="black", ls="--", lw=1)
    best = curve.loc[curve.J.idxmax()]
    ax.set_title(f"threshold sweep: max a-b = {best.J:.3f} (a {best.a_finding:.2f}, b {best.b_control:.2f}, "
                 f"all kept {best.kept_all:.1%})", fontsize=10)
    ax.set_xlabel("threshold on the score")
    ax.legend(frameon=False)
    fig.savefig(out / "S1_score_and_threshold.png")
    plt.close(fig)

    # 群ごとの残る割合
    fig, axes = plt.subplots(1, 4, figsize=(30, 12), gridspec_kw={"width_ratios": [1.2, 2, 0.8, 2], "wspace": .45})
    ax = axes[0]
    m_ = pd.crosstab(sl.dose_level, sl.sacrifice_period, values=kept, aggfunc="mean").reindex(index=DOSES, columns=DAYS)
    ax.imshow(m_.to_numpy(), cmap="Reds", vmin=0)
    for (i, j), v in np.ndenumerate(m_.to_numpy()):
        ax.text(j, i, f"{v:.0%}", ha="center", va="center")
    ax.set_xticks(range(4), DAYS)
    ax.set_yticks(range(4), DOSES)
    ax.set_title("kept fraction: dose x time", fontsize=10)
    ax = axes[1]
    e = pd.DataFrame({"exp": [f"e{a} {b[:20]}" for a, b in zip(sl.exp_id, sl.compound_name)], "exp_id": sl.exp_id.to_numpy(),
                      "kept": kept, "group": group, "scanner": sl.scanner.fillna("unknown").to_numpy()})
    g = e.pivot_table(index=["exp_id", "exp"], columns="group", values="kept", aggfunc="mean").reindex(columns=[n for n, _ in GROUPS])
    g = g.join(e.groupby(["exp_id", "exp"]).scanner.agg(lambda v: v.mode().iloc[0])).reset_index()
    y = np.arange(len(g))
    for off, (lab, c) in zip([-.27, 0, .27], GROUPS):
        ax.barh(y + off, g[lab].fillna(0), height=.27, color=c, label=lab)
    for i, r in g.iterrows():
        ax.text(-0.01, i, "■", color=SCAN_COLORS.get(r.scanner, "black"), ha="right", va="center", fontsize=9)
    ax.set_yticks(y, g.exp, fontsize=7.5)
    ax.tick_params(axis="y", pad=12)
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right")
    ax.set_title("kept fraction per experiment (square = main scanner: blue SS1601, orange SS1598)", fontsize=10)
    ax = axes[2]
    for i, (sc, c) in enumerate(SCAN_COLORS.items()):
        for j, (lab, gc) in enumerate(GROUPS):
            m = (e.scanner == sc) & (group == lab)
            if m.any():
                ax.bar(i + (j - 1) * .27, kept[m].mean(), width=.27, color=gc)
    ax.set_xticks(range(3), list(SCAN_COLORS))
    ax.set_title("kept fraction per scanner\n(grey Control, blue treated no finding, red finding)", fontsize=10)
    ax = axes[3]
    fl = sl.finding_types.apply(as_list).to_numpy()
    recs = []
    for f in pd.Series([f for v in fl for f in v]).value_counts().index:
        m = np.array([f in v for v in fl])
        recs.append({"finding": f, "kept": kept[m].mean(), "n_slides": sl[m].slide_id.nunique(), "n_exp": sl[m].exp_id.nunique()})
    ft = pd.DataFrame(recs).sort_values("kept")
    ax.barh(np.arange(len(ft)), ft.kept, color=["#e34948" if n > 1 else "#f0a8a7" for n in ft.n_exp])
    for i, r in enumerate(ft.itertuples()):
        ax.text(r.kept, i, f" {r.n_slides} sl / {r.n_exp} exp", va="center", fontsize=7)
    ax.axvline(kept[group == "Control"].mean(), color="#9aa3ad", ls="--")
    ax.set_yticks(np.arange(len(ft)), [short(f, 40) for f in ft.finding], fontsize=7.5)
    ax.set_title("kept fraction of patches from slides with each finding\n(pale = 1 experiment only; dashed = Control)", fontsize=10)
    fig.suptitle("S1. Stage 1: which patches are kept (score at or below the threshold)", fontsize=13)
    fig.savefig(out / "S1_kept_fraction_by_group.png")
    plt.close(fig)
    ft.to_csv(out / "stage1_kept_by_finding.csv", index=False)
    g.to_csv(out / "stage1_kept_by_experiment.csv", index=False)


def stage2_embedding(name, ev, feats, kept, cfg):
    if name == "pca0017":
        return ev[kept]
    from sklearn.decomposition import PCA
    return PCA(n_components=cfg["pca_components"], svd_solver="randomized", random_state=cfg["seed"]).fit_transform(feats[kept]).astype(np.float32)


def kmeans_runs(x, cfg, device, kmeans_lloyd, assign):
    """kごとに 抽出seed × 初期化seed のrun。各runは80%で学習し、全patchに割り当てる。"""
    labels, meta = [], []
    xt = torch.from_numpy(x).to(device)
    for k in cfg["ks"]:
        for s in cfg["sample_seeds"]:
            sel = np.random.default_rng([cfg["seed"], s]).choice(len(x), size=int(cfg["train_frac"] * len(x)), replace=False)
            for init in cfg["init_seeds"]:
                _, c, n_iter = kmeans_lloyd(x[sel], k, init, device)
                labels.append(assign(xt, c))
                meta.append({"run": f"k{k:03d}_s{s}_i{init}", "k": k, "sample_seed": s, "init_seed": init, "n_iter": n_iter})
    return np.stack(labels), pd.DataFrame(meta)


def finding_matrix(core, row, slides, kept_base):
    """コア×所見: コアのpatchのうち、その所見が記録されたslideのものの割合と、残したpatch全体での割合との比。"""
    fl = slides.finding_types.apply(as_list)
    fs = kept_base.index
    has = np.array([[f in v for f in fs] for v in fl], dtype=float)   # slide × 所見
    share = np.stack([has[row[core == c]].mean(0) for c in range(core.max() + 1)])
    return pd.DataFrame(share, columns=fs), pd.DataFrame(share / kept_base.to_numpy()[None, :], columns=fs)


def core_figures(tag, core, row, slides, kept_base, out, n_total):
    """コア×所見（割合と比）とコア×試験の図。"""
    if core.max() < 0:
        return
    share, lift = finding_matrix(core, row, slides, kept_base)
    share.assign(core=range(len(share))).to_csv(out / f"finding_share_{tag}.csv", index=False)
    sl = slides.iloc[row]
    ex = pd.crosstab(core[core >= 0], sl.compound_name.to_numpy()[core >= 0], normalize="index")
    ex = ex[ex.max().sort_values(ascending=False).index]
    n = np.bincount(core[core >= 0])
    cols = [f for f in kept_base.index if kept_base[f] > 0][:30]
    fig, axes = plt.subplots(1, 3, figsize=(30, 0.32 * len(n) + 4), gridspec_kw={"width_ratios": [3, 3, 3]})
    ax = axes[0]
    v = share[cols].to_numpy()
    ax.imshow(v, cmap="Reds", aspect="auto", vmin=0, vmax=1)
    for (i, j), x in np.ndenumerate(v):
        if x >= .1:
            ax.text(j, i, f"{x:.0%}"[:-1], ha="center", va="center", fontsize=6, color="white" if x > .6 else "black")
    ax.set_xticks(range(len(cols)), [short(f, 26) for f in cols], rotation=90, fontsize=7)
    ax.set_yticks(range(len(n)), [f"{c} ({k}p)" for c, k in enumerate(n)], fontsize=7)
    ax.set_title("share of the core's patches from slides recording the finding (%, shown if >=10)", fontsize=10)
    ax = axes[1]
    v = np.log2(np.clip(lift[cols].to_numpy(), 1 / 16, 16))
    ax.imshow(v, cmap="RdBu_r", aspect="auto", vmin=-4, vmax=4)
    for (i, j), x in np.ndenumerate(lift[cols].to_numpy()):
        if x >= 2 and share[cols].to_numpy()[i, j] >= .1:
            ax.text(j, i, f"{x:.0f}", ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(cols)), [short(f, 26) for f in cols], rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_title("enrichment vs all kept patches (log2 color; number = fold, if >=2 and share >=10%)", fontsize=10)
    ax = axes[2]
    top = ex.iloc[:, :30]
    ax.imshow(top.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=1)
    for (i, j), x in np.ndenumerate(top.to_numpy()):
        if x >= .1:
            ax.text(j, i, f"{x:.0%}"[:-1], ha="center", va="center", fontsize=6, color="white" if x > .6 else "black")
    ax.set_xticks(range(top.shape[1]), [c[:18] for c in top.columns], rotation=90, fontsize=7)
    ax.set_yticks([])
    ax.set_title("share of the core's patches per compound (%)", fontsize=10)
    fig.suptitle(f"F {tag}: {len(n)} cores covering {(core >= 0).sum():,} of {n_total:,} kept patches", fontsize=13)
    fig.savefig(out / f"F_{tag}.png")
    plt.close(fig)


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.clustering import _undirected_pairs, kmeans_lloyd, knn_graph
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    args = parse_args()
    exp_name = os.environ["EXP_NAME"]
    cfg = load_config(Path(__file__).parent)
    cfg["source_variant"] = args.source_variant or cfg["source_variant"]
    variant_key = (cfg["source_variant"] + (f"_thr{args.threshold:g}" if args.threshold is not None else "")
                   + (f"_s2{args.stage2_variant}" if args.stage2_variant else ""))
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, config=cfg)
    logger.info(f"Starting: {exp_name} / {variant_key} (threshold {args.threshold})")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 0017の結果と埋め込み ---
    ex17 = load_exp0017(project_root, cfg["source_exp"])
    cfg17 = ex17.load_config(project_root / "experiments" / cfg["source_exp"])
    cfg17.update(cfg17["variants"][cfg["source_variant"]])
    src = project_root / "outputs" / cfg["source_exp"] / cfg["source_variant"]
    t0 = time.time()
    slides, ev, feats, row, xy = ex17.cached_eval_embedding(cfg17, cfg["source_variant"], project_root)
    ev2, feats2 = ev, feats   # 2段目の埋め込み
    if args.stage2_variant:
        cfg17b = ex17.load_config(project_root / "experiments" / cfg["source_exp"])
        cfg17b.update(cfg17b["variants"][args.stage2_variant])
        slides_b, ev2, feats2, row_b, xy_b = ex17.cached_eval_embedding(cfg17b, args.stage2_variant, project_root)
        if not ((slides_b.slide_id.to_numpy()[row_b] == slides.slide_id.to_numpy()[row]).all() and (xy_b == xy).all()):
            raise ValueError(f"{args.stage2_variant} eval patches are not the same as {cfg['source_variant']}")
    kept17 = pd.read_parquet(src / "eval_patches.parquet")
    if not ((slides.slide_id.to_numpy()[row] == kept17.slide_id.to_numpy()).all() and (xy[:, 0] == kept17.x.to_numpy()).all()):
        raise ValueError("rebuilt eval patches do not match 0017 eval_patches.parquet")
    labels = np.load(src / "eval_labels.npy")
    logger.info(f"loaded 0017 {cfg['source_variant']}: {len(ev):,} eval patches, {len(labels)} runs; {time.time() - t0:.0f}s")
    sl = slides.iloc[row].reset_index(drop=True)
    is_ctrl = (sl.dose_level == "Control").to_numpy()
    group = np.where(is_ctrl, "Control", np.where(sl.has_finding, "treated, finding", "treated, no finding"))

    # --- 1段目 ---
    score = ctrl_score(labels, is_ctrl)
    ts = np.unique(np.quantile(score, np.linspace(0, 1, cfg["threshold_grid"] + 1)))
    m_f, m_c, m_n = group == "treated, finding", group == "Control", group == "treated, no finding"
    curve = pd.DataFrame([{"threshold": t, "a_finding": (score[m_f] <= t).mean(), "b_control": (score[m_c] <= t).mean(),
                           "treated_no_finding": (score[m_n] <= t).mean(), "kept_all": (score <= t).mean()} for t in ts])
    curve["J"] = curve.a_finding - curve.b_control
    thr = float(curve.loc[curve.J.idxmax(), "threshold"]) if args.threshold is None else args.threshold
    kept = score <= thr
    curve.to_csv(run_dir / "stage1_threshold_curve.csv", index=False)
    kept17.assign(score=score, kept=kept, group=group).to_parquet(run_dir / "stage1_scores.parquet", index=False)
    logger.info(f"stage 1: threshold {thr:.4f}, kept {kept.sum():,} ({kept.mean():.1%}); "
                + ", ".join(f"{g} {kept[group == g].mean():.1%}" for g, _ in GROUPS))
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    stage1_figures(score, thr, curve, kept, group, sl, out)

    size0 = {}
    for sid in slides.slide_id:
        with h5py.File(ex17_features(cfg17, project_root, sid), "r") as h:
            size0[sid] = int(h["coords"].attrs.get("patch_size_level0", 256))

    def lab(s_, p):
        return (f"e{s_.exp_id} {s_.compound_name[:13]} {s_.dose_level[0]}", f"{s_.sacrifice_period} {s_.scanner} {'F' if s_.has_finding else '-'} s{score[p]:.2f}")
    rng = np.random.default_rng(cfg["seed"])
    near_in = np.flatnonzero(kept)[np.argsort(-score[kept])]
    near_out = np.flatnonzero(~kept)[np.argsort(score[~kept])]
    rows = [("most distant from Control (lowest score)", pick_one_per_slide(np.argsort(score), row, cfg["sheet_tiles"])),
            ("kept, just inside the threshold", pick_one_per_slide(near_in, row, cfg["sheet_tiles"])),
            ("excluded, just outside the threshold", pick_one_per_slide(near_out, row, cfg["sheet_tiles"])),
            ("Control patches that were kept (random)", pick_one_per_slide(rng.permutation(np.flatnonzero(kept & m_c)), row, cfg["sheet_tiles"])),
            ("finding-slide patches that were excluded (random)", pick_one_per_slide(rng.permutation(np.flatnonzero(~kept & m_f)), row, cfg["sheet_tiles"])),
            ("finding-slide patches that were kept (random)", pick_one_per_slide(rng.permutation(np.flatnonzero(kept & m_f)), row, cfg["sheet_tiles"]))]
    ex17.draw_sheet(f"0018 stage 1 (threshold {thr:.3f}); label: exp, compound, dose, time, scanner, F = slide has a finding, s = score",
                    [(h, ex17.read_tiles(p, row, xy, slides, size0, lab)) for h, p in rows], out / "S1_sheet.jpg")

    # 1段目のt-SNE（全体から無作為に）
    sub = np.sort(rng.choice(len(ev), size=min(cfg["tsne_max_points"], len(ev)), replace=False))
    t0 = time.time()
    xy1 = tsne(ev[sub], cfg["seed"])
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    o = rng.permutation(len(sub))
    sc = axes[0].scatter(*xy1[o].T, c=score[sub][o], cmap="viridis_r", s=1.2, linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=axes[0], fraction=.03).set_label("Control-likeness score")
    axes[0].set_title("score (yellow = far from Control)", fontsize=12)
    axes[1].scatter(*xy1[~kept[sub]].T, color=GREY, s=1, linewidths=0, rasterized=True)
    axes[1].scatter(*xy1[kept[sub]].T, color="#e34948", s=1.5, linewidths=0, rasterized=True)
    axes[1].set_title(f"kept (red, {kept.mean():.1%} of all)", fontsize=12)
    for g_, c_ in GROUPS:
        m = group[sub][o] == g_
        axes[2].scatter(*xy1[o][m].T, color=c_, s=1.2, alpha=.7, linewidths=0, rasterized=True, label=g_)
    axes[2].legend(markerscale=6, frameon=False)
    axes[2].set_title("slide group", fontsize=12)
    for a in axes:
        bare(a)
    fig.suptitle(f"S1. Stage 1 on a t-SNE of {len(sub):,} random eval patches (0017 PCA64)", fontsize=14)
    fig.savefig(out / "S1_tsne.png")
    plt.close(fig)
    logger.info(f"stage 1 figures done; t-SNE {time.time() - t0:.0f}s")

    # --- 2段目 ---
    kidx = np.flatnonzero(kept)
    krow, kxy = row[kidx], xy[kidx]
    ksl = sl.iloc[kidx].reset_index(drop=True)
    kept_base = pd.Series([f for v in ksl.finding_types.apply(as_list) for f in v]).value_counts() / len(ksl)
    base_rate = slides[slides.dose_level != "Control"].finding_types.explode().dropna().value_counts() / (slides.dose_level != "Control").sum()
    summary = []
    for emb in cfg["embeddings"]:
        t0 = time.time()
        x = stage2_embedding(emb, ev2, feats2, kept, cfg)
        d = run_dir / "stage2" / emb
        d.mkdir(parents=True, exist_ok=True)
        lab2, meta = kmeans_runs(x, cfg, device, kmeans_lloyd, ex17.assign)
        np.save(d / "labels.npy", lab2)
        meta.to_csv(d / "runs.csv", index=False)
        nn, _ = knn_graph(x, cfg["knn"], device)
        pairs = _undirected_pairs(nn)
        co = lab2[:, pairs[:, 0]] == lab2[:, pairs[:, 1]]
        rate_k = {k: co[meta.k.to_numpy() == k].mean(0) for k in cfg["ks"]}
        cores_k, tables_k, chains = ex17.per_k_consensus(lab2, meta, pairs, rate_k, krow, slides, base_rate, cfg, cfg["tau"], logger)
        chains.to_csv(d / "chains.csv", index=False)
        single = {k: lab2[np.flatnonzero((meta.k == k) & (meta.sample_seed == 0) & (meta.init_seed == 0))[0]] for k in cfg["ks"]}
        assign_tab = kept17.iloc[kidx][["slide_id", "x", "y"]].reset_index(drop=True)
        for k in cfg["ks"]:
            assign_tab[f"consensus_k{k}"] = cores_k[k]
            assign_tab[f"single_k{k}"] = single[k]
        assign_tab.to_parquet(d / "assignments.parquet", index=False)
        logger.info(f"stage 2 {emb}: k-means + consensus done; {time.time() - t0:.0f}s")

        # t-SNE（残したpatch）
        sub2 = np.sort(rng.choice(len(x), size=min(cfg["tsne_max_points"], len(x)), replace=False))
        xy2 = tsne(x[sub2], cfg["seed"])
        o2 = rng.permutation(len(sub2))
        s2 = ksl.iloc[sub2].reset_index(drop=True)
        fig, axes = plt.subplots(3, 3, figsize=(30, 30))
        for ax, (method, ass) in zip(axes[:2].flat, [(f"consensus k={k}", cores_k[k]) for k in cfg["ks"]] + [(f"single k-means k={k}", single[k]) for k in cfg["ks"]]):
            c_ = ass[sub2]
            on = c_ >= 0
            ax.scatter(*xy2[~on].T, color=GREY, s=1, linewidths=0, rasterized=True)
            pal = palette(max(c_.max() + 1, 1))
            oo = o2[on[o2]]
            ax.scatter(*xy2[oo].T, c=pal[c_[oo]], s=1.5, linewidths=0, rasterized=True)
            for c in range(c_.max() + 1):
                if (c_ == c).any():
                    ax.text(*np.median(xy2[c_ == c], axis=0), str(c), fontsize=7, ha="center", va="center",
                            bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=.7))
            ax.set_title(f"{method}: {ass.max() + 1} clusters/cores, unassigned {(ass < 0).mean():.0%}", fontsize=12)
        ax = axes[2, 0]
        for dd, c_ in zip(DOSES, DOSE_COLORS):
            m = s2.dose_level.to_numpy()[o2] == dd
            ax.scatter(*xy2[o2][m].T, color=c_, s=1.2, linewidths=0, rasterized=True, label=f"{dd} ({m.mean():.0%})")
        ax.legend(markerscale=6, frameon=False)
        ax.set_title("dose group", fontsize=12)
        ax = axes[2, 1]
        for scn, c_ in SCAN_COLORS.items():
            m = s2.scanner.to_numpy()[o2] == scn
            ax.scatter(*xy2[o2][m].T, color=c_, s=1.2, linewidths=0, rasterized=True, label=f"{scn} ({m.mean():.0%})")
        ax.legend(markerscale=6, frameon=False)
        ax.set_title("scanner", fontsize=12)
        ax = axes[2, 2]
        cmps = s2.compound_name.value_counts().index
        cp = palette(len(cmps))
        for cm, c_ in zip(cmps, cp):
            m = s2.compound_name.to_numpy()[o2] == cm
            ax.scatter(*xy2[o2][m].T, color=c_, s=1.2, linewidths=0, rasterized=True, label=f"{cm[:18]} ({m.mean():.0%})")
        ax.legend(markerscale=6, frameon=False, fontsize=6, ncol=2, loc="lower left")
        ax.set_title("compound", fontsize=12)
        for a in axes.flat:
            bare(a)
        fig.suptitle(f"T {emb}: t-SNE of {len(sub2):,} of {len(x):,} kept patches", fontsize=15)
        fig.savefig(out / f"T_{emb}.png")
        plt.close(fig)

        # 所見ごと（残したpatchのt-SNE上）
        fl2 = s2.finding_types.apply(as_list)
        top = kept_base.index[:20]
        fig, axes = plt.subplots(4, 5, figsize=(25, 20))
        for ax, f in zip(axes.flat, top):
            m = fl2.apply(lambda v: f in v).to_numpy()
            ax.scatter(*xy2[~m].T, color=GREY, s=.4, linewidths=0, rasterized=True)
            ax.scatter(*xy2[m].T, color="#e34948", s=1.5, linewidths=0, rasterized=True)
            ax.set_title(f"{short(f, 38)}\n{m.sum():,} pts, {s2[m].slide_id.nunique()} slides, {s2[m].exp_id.nunique()} exp", fontsize=9)
            bare(ax)
        fig.suptitle(f"T2 {emb}: kept patches from slides recording each finding (red), top 20 findings among kept patches", fontsize=13, y=1.0)
        fig.tight_layout()
        fig.savefig(out / f"T2_{emb}_by_finding.png")
        plt.close(fig)

        # コアごとの表・図・画像シート
        for k in cfg["ks"]:
            for method, ass in [("consensus", cores_k[k]), ("single", single[k])]:
                tag = f"{emb}_{method}_k{k}"
                core_figures(tag, ass, krow, slides, kept_base, out, len(x))
                if ass.max() < 0:
                    continue
                cen = np.zeros(len(x))
                for c in range(ass.max() + 1):
                    m = ass == c
                    cen[m] = -np.linalg.norm(x[m] - x[m].mean(0), axis=1)
                tab = ex17.core_table(ass, krow, slides, cen, base_rate)
                if method == "consensus":
                    tab = tab.merge(tables_k[k][["core", "split_half_J"]], on="core")
                tab.to_csv(d / f"cores_{method}_k{k}.csv", index=False)
                share = pd.read_csv(out / f"finding_share_{tag}.csv").set_index("core")
                sheet_rows = []
                for r in tab.sort_values("n_patches", ascending=False).itertuples():
                    members = np.flatnonzero(ass == r.core)
                    picks = pick_one_per_slide(members[np.argsort(-cen[members])], krow, cfg["sheet_tiles"])
                    tiles = ex17.read_tiles(picks, krow, kxy, slides, size0,
                                            lambda s_, p: (f"e{s_.exp_id} {s_.compound_name[:13]} {s_.dose_level[0]}",
                                                           f"{s_.sacrifice_period} {s_.scanner} {'F' if s_.has_finding else '-'}"))
                    fs = share.loc[r.core].sort_values(ascending=False).head(3)
                    cmp = slides.iloc[krow[members]].compound_name.value_counts(normalize=True).head(3)
                    header = (f"core {r.core}: {r.n_patches} p / {r.n_slides} sl / {r.n_exp} exp | Ctrl {r.ctrl_patch_frac:.2f}"
                              + (f" | J {r.split_half_J:.2f}" if method == "consensus" else "") + " | "
                              + ", ".join(f"{short(f, 22)} {v:.0%}" for f, v in fs.items()) + " | "
                              + ", ".join(f"{c[:12]} {v:.0%}" for c, v in cmp.items()))
                    sheet_rows.append((header, tiles))
                for f0 in range(0, len(sheet_rows), 10):
                    ex17.draw_sheet(f"0018 stage 2 {tag}, sorted by size; tiles nearest the core mean, one per slide; F = slide has a finding; "
                                    f"finding % = share of the core's patches from slides recording it",
                                    sheet_rows[f0:f0 + 10], out / "sheets" / tag / f"cores_{f0 // 10 + 1:02d}.jpg")
                summary.append({"embedding": emb, "method": method, "k": k, "n_cores": int(ass.max() + 1),
                                "assigned_frac": float((ass >= 0).mean()),
                                "median_top_exp_share": float(tab.top_exp_share.median()),
                                "split_half_J_median": float(tab.split_half_J.median()) if method == "consensus" else None})
                logger.info(f"stage 2 {tag}: {summary[-1]}")

    pd.DataFrame(summary).to_csv(run_dir / "stage2_summary.csv", index=False)
    (run_dir / "results.json").write_text(json.dumps(
        {"source_variant": cfg["source_variant"], "stage2_variant": args.stage2_variant or cfg["source_variant"], "n_eval_patches": int(len(ev)), "threshold": thr, "n_kept": int(kept.sum()),
         "kept_frac": {g: float(kept[group == g].mean()) for g, _ in GROUPS}, "max_J": float(curve.J.max()),
         "stage2": summary}, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info("Done.")


def ex17_features(cfg17, project_root, sid):
    """0017と同じ規則: 全patch版を優先し、なければ2,048 patch版。"""
    for d in cfg17.get("features_dirs", [cfg17["features_dir"]]):
        p = project_root / d / f"{sid}.h5"
        if p.exists():
            return p
    raise FileNotFoundError(sid)


if __name__ == "__main__":
    main()
