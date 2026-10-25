"""0017（塊内のコンセンサスコア）の全体像を平面図にする。

O1〜O4. 対象slide全体の概要（所見ごとのslide数、試験ごとの用量×時点・所見あり・scanner、試験×所見）とCSV
A. 評価用patchのt-SNE（コア / 用量 / 所見 / 時点 / scanner）、A2. 所見ごとのハイライト
B. 試験ごとのハイライト
C. コアの地図（1点=1コア: x=Controlのpatch割合、y=上位1試験のシェア、大きさ=patch数、色=半分ずつのrunでの一致度）

per_kの結果がある場合（block6_all）: C2. kごとのコアの地図、D. kごとのコアのt-SNE、D2. 入れ子・Control<5%のコア、
D3. Control<5%のコアを1つずつ（親コアも重ねる）、E. Control<5%のコアの試験・用量・時点・scannerの内訳

評価用patchの埋め込みは0017と同じ手順・同じseedで作り直し（pilot以外は本体のキャッシュから）、0017の eval_patches.parquet と座標で突き合わせる。
usage: python scripts/visualize_consensus_cores.py [variant] [method] [tau]   既定: pilot leiden 0.8
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
RUNS = ROOT / "outputs/0017_20261016_block_consensus_patterns"
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


def as_list(v):
    return [] if v is None else sorted(set(v))   # 同じ所見が部位違いで重複していても1枚として数える


def short(name, n=34):
    return name.replace("Degeneration, granular, eosinophilic", "gran. eos. degen.")[:n]


def data_overview(slides, row, out):
    """対象slide全体の概要（所見・試験・scanner）。表はCSV、図はO*.png。"""
    s = slides.assign(findings=slides.finding_types.apply(as_list), scanner=slides.scanner.fillna("unknown"),
                      n_eval=np.bincount(row, minlength=len(slides)))
    trt = s[s.dose_level != "Control"]
    ex = s[["slide_id", "exp_id", "compound_name", "dose_level", "sacrifice_period", "scanner", "findings"]].explode("findings").dropna(subset=["findings"]).reset_index(drop=True)
    counts = ex.findings.value_counts()
    fs = counts.index.tolist()
    head = (f"{len(s):,} slides ({(s.dose_level == 'Control').sum():,} Control) from {s.exp_id.nunique()} experiments | "
            f"treated slides with any recorded finding: {trt.has_finding.sum():,} / {len(trt):,} ({trt.has_finding.mean():.0%}) | "
            f"Control slides with a finding: {s[s.dose_level == 'Control'].has_finding.sum():,}")

    # 表
    ftab = pd.DataFrame({"finding": fs, "n_slides": counts.values,
                         "n_experiments": ex.groupby("findings").exp_id.nunique().reindex(fs).values})
    for col_, cats in [("dose_level", DOSES), ("sacrifice_period", DAYS), ("scanner", sorted(s.scanner.unique()))]:
        ftab = ftab.join(pd.crosstab(ex.findings, ex[col_]).reindex(index=fs, columns=cats, fill_value=0).add_prefix(f"{col_}:"), on="finding")
    ftab.to_csv(out / "overview_findings.csv", index=False)
    etab = s.groupby(["exp_id", "compound_name"]).agg(n_slides=("slide_id", "size"), n_eval_patches=("n_eval", "sum")).reset_index()
    for col_, cats in [("dose_level", DOSES), ("sacrifice_period", DAYS), ("scanner", sorted(s.scanner.unique()))]:
        etab = etab.join(pd.crosstab(s.exp_id, s[col_]).reindex(columns=cats, fill_value=0).add_prefix(f"{col_}:"), on="exp_id")
    g = trt.groupby("exp_id").has_finding
    etab["n_treated"] = etab.n_slides - etab["dose_level:Control"]
    etab["treated_with_finding"] = etab.exp_id.map(g.sum()).fillna(0).astype(int)
    etab["treated_with_finding_frac"] = etab.exp_id.map(g.mean())
    etab.to_csv(out / "overview_experiments.csv", index=False)
    fx = pd.crosstab(ex.findings, ex.exp_id).reindex(index=fs, columns=etab.exp_id, fill_value=0)
    fx.to_csv(out / "overview_finding_by_experiment.csv")

    # O1. 所見ごとのslide数（用量別・時点別・scanner別）
    show = fs[:45]
    fig, axes = plt.subplots(1, 3, figsize=(24, 0.3 * len(show) + 2.5), sharey=True)
    for ax, col_, cats, colors in [(axes[0], "dose_level", DOSES, DOSE_COLORS), (axes[1], "sacrifice_period", DAYS, DAY_COLORS),
                                   (axes[2], "scanner", list(SCAN_COLORS), list(SCAN_COLORS.values()))]:
        ct = pd.crosstab(ex.findings, ex[col_]).reindex(index=show[::-1], columns=cats, fill_value=0)
        left = np.zeros(len(ct))
        for c_, k_ in zip(cats, colors):
            ax.barh(np.arange(len(ct)), ct[c_], left=left, color=k_, label=c_)
            left += ct[c_].to_numpy()
        ax.set_yticks(np.arange(len(ct)), [short(f, 45) for f in ct.index], fontsize=8)
        ax.legend(frameon=False, loc="lower right")
        ax.set_xlabel("slides")
        ax.set_title(f"by {col_}", fontsize=11)
    for i, f in enumerate(show[::-1]):
        axes[0].text(counts[f], i, f"  {counts[f]} ({ftab.set_index('finding').n_experiments[f]} exp)", va="center", fontsize=7)
    fig.suptitle(f"O1. Slides per recorded finding (top {len(show)} of {len(fs)}; a slide can have several)\n{head}", fontsize=12)
    fig.savefig(out / "O1_findings.png")
    plt.close(fig)

    # O2. 試験ごと: 用量×時点のslide数、所見ありの割合、scanner
    lab = [f"e{e} {c[:22]}" for e, c in zip(etab.exp_id, etab.compound_name)]
    grid = pd.crosstab(s.exp_id, [s.dose_level, s.sacrifice_period]).reindex(
        index=etab.exp_id, columns=pd.MultiIndex.from_product([DOSES, DAYS]), fill_value=0)
    fig, axes = plt.subplots(1, 3, figsize=(24, 0.28 * len(etab) + 3), gridspec_kw={"width_ratios": [3, 1, 1]}, sharey=True)
    ax = axes[0]
    ax.imshow(grid.to_numpy(), cmap="Blues", aspect="auto", vmin=0, vmax=10)
    for (i, j), v in np.ndenumerate(grid.to_numpy()):
        ax.text(j, i, v, ha="center", va="center", fontsize=6.5, color="black" if v == 5 else "#c0392b", fontweight="normal" if v == 5 else "bold")
    ax.set_xticks(np.arange(16), [f"{d[0]} {t.split()[0]}d" for d, t in grid.columns], fontsize=7, rotation=90)
    ax.set_yticks(np.arange(len(lab)), lab, fontsize=7.5)
    ax.set_title("slides per dose group (C/L/M/H) x time point (red = not 5)", fontsize=10)
    ax = axes[1]
    y = np.arange(len(etab))
    ax.barh(y, etab.treated_with_finding_frac, color="#e34948")
    for i, r in enumerate(etab.itertuples()):
        ax.text(r.treated_with_finding_frac, i, f" {r.treated_with_finding}/{r.n_treated}", va="center", fontsize=6.5)
    ax.set_xlim(0, 1.25)
    ax.set_title("treated slides with any finding", fontsize=10)
    ax = axes[2]
    left = np.zeros(len(etab))
    for c_, k_ in SCAN_COLORS.items():
        v = etab.get(f"scanner:{c_}", pd.Series(0, index=etab.index)).to_numpy() / etab.n_slides.to_numpy()
        ax.barh(y, v, left=left, color=k_, label=c_)
        left += v
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.02), ncol=3)
    ax.set_title("scanner share", fontsize=10)
    fig.suptitle(f"O2. Experiments in {head.split(' | ')[0]}", fontsize=12)
    fig.savefig(out / "O2_experiments.png")
    plt.close(fig)

    # O3. 試験×所見（slide数）
    sub = fx.iloc[:40]
    fig, ax = plt.subplots(figsize=(0.32 * len(etab) + 6, 0.3 * len(sub) + 3))
    v = sub.to_numpy()
    ax.imshow(np.log1p(v), cmap="Reds", aspect="auto")
    for (i, j), n_ in np.ndenumerate(v):
        if n_:
            ax.text(j, i, n_, ha="center", va="center", fontsize=6)
    ax.set_xticks(np.arange(len(lab)), lab, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(sub)), [short(f, 45) for f in sub.index], fontsize=7.5)
    ax.set_title(f"O3. Slides with each finding, per experiment (top {len(sub)} findings; color = log count)", fontsize=11)
    fig.savefig(out / "O3_finding_by_experiment.png")
    plt.close(fig)

    # O4. 上位所見の用量×時点
    top = fs[:20]
    fig, axes = plt.subplots(4, 5, figsize=(20, 14))
    for ax, f in zip(axes.flat, top):
        e_ = ex[ex.findings == f]
        m_ = pd.crosstab(e_.dose_level, e_.sacrifice_period).reindex(index=DOSES, columns=DAYS, fill_value=0).to_numpy()
        ax.imshow(m_, cmap="Reds", vmin=0)
        for (i, j), n_ in np.ndenumerate(m_):
            ax.text(j, i, n_, ha="center", va="center", fontsize=8, color="white" if n_ > .6 * m_.max() else "black")
        ax.set_xticks(np.arange(4), [d.split()[0] + "d" for d in DAYS], fontsize=7)
        ax.set_yticks(np.arange(4), [d[0] for d in DOSES], fontsize=7)
        ax.set_title(f"{short(f, 30)}\n{counts[f]} slides, {e_.exp_id.nunique()} exp", fontsize=8.5)
    fig.suptitle("O4. Top 20 findings: slides per dose group (rows C/L/M/H) x time point (columns)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "O4_findings_dose_time.png")
    plt.close(fig)
    return head


def label_medians(ax, xy, core, ids, fontsize=6.5, names=None):
    for c in ids:
        if (core == c).any():
            cx, cy = np.median(xy[core == c], axis=0)
            ax.text(cx, cy, names.get(c, str(c)) if names else str(c), fontsize=fontsize, ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=.7))


def scatter_cores(ax, xy, core, order, ids=None, size=1.5):
    """ids（既定は全コア）を色分けし、それ以外と未割当は灰色。"""
    ids = np.arange(core.max() + 1) if ids is None else np.asarray(ids)
    on = np.isin(core, ids)
    ax.scatter(*xy[~on].T, color=GREY, s=1, linewidths=0, rasterized=True)
    pal = palette(core.max() + 1)
    o = order[on[order]]
    ax.scatter(*xy[o].T, c=pal[core[o]], s=size, linewidths=0, rasterized=True)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    method = sys.argv[2] if len(sys.argv) > 2 else "leiden"
    tau = sys.argv[3] if len(sys.argv) > 3 else "0.8"
    col = f"{method}_tau{tau}"
    cfg = yaml.safe_load(open(EXP / "config.yml"))
    cfg.update(cfg.get("variants", {}).get(variant, {}))
    run = RUNS / variant
    out = EXP / (f"visualization_{date.today():%Y%m%d}" + ("" if variant == "pilot" else f"_{variant}"))
    out.mkdir(parents=True, exist_ok=True)

    import time
    t0 = time.time()
    cache = out / "eval_embedding.npz"
    if variant == "pilot":
        if cache.exists():   # 埋め込みの作り直し（h5 1,020本の読み込み）は1回だけにする
            z = np.load(cache, allow_pickle=True)
            slides = pd.read_parquet(out / "slides.parquet")
            ev, erow, exy = z["ev"], z["erow"], z["exy"]
        else:
            slides, ev, erow, exy = rebuild_embedding(cfg)
            np.savez_compressed(cache, ev=ev, erow=erow, exy=exy)
            slides.drop(columns=["finding_types", "finding_sites"]).to_parquet(out / "slides.parquet", index=False)
    else:
        slides, ev, _, erow, exy = exp0017.cached_eval_embedding(cfg, variant, ROOT)
    print(f"embedding ready: {len(ev):,} patches, {time.time() - t0:.0f}s", flush=True)
    kept = pd.read_parquet(run / "eval_patches.parquet")
    key = pd.DataFrame({"slide_id": slides.slide_id.to_numpy()[erow], "x": exy[:, 0], "y": exy[:, 1], "i": np.arange(len(erow))})
    j = kept.merge(key, on=["slide_id", "x", "y"], how="left", validate="one_to_one")
    if j.i.isna().any():
        raise ValueError("eval patches could not be matched to the rebuilt embedding")
    idx = j.i.to_numpy().astype(int)
    ev, row = ev[idx], erow[idx]
    core = j[col].to_numpy()
    pk_path = run / "per_k" / f"assignments_tau{tau}.parquet"
    pk = pd.read_parquet(pk_path) if pk_path.exists() else None
    if pk is not None and not (pk.slide_id.to_numpy() == kept.slide_id.to_numpy()).all():
        raise ValueError("per-k assignments are not in the order of eval_patches.parquet")
    ks = sorted(int(c[6:]) for c in pk.columns if c.startswith("core_k")) if pk is not None else []
    has_findings = "finding_types" in slides
    head = data_overview(slides, row, out) if has_findings else ""
    print(f"overview done", flush=True)

    # t-SNEは4.75万点だと1時間を超えたため、無作為に抜いて描く（pilot 2万点、2026-10-17。block6_allは3万点）
    n_tsne = 20000 if variant == "pilot" else 30000
    sub = np.sort(np.random.default_rng(0).choice(len(ev), size=min(n_tsne, len(ev)), replace=False))
    ev_s, row_s, core_s = ev[sub], row[sub], core[sub]
    s = slides.iloc[row_s].reset_index(drop=True)
    pkc = {k: pk[f"core_k{k}"].to_numpy()[sub] for k in ks}
    t0 = time.time()
    done = out / f"tsne_{col}.parquet"
    if done.exists() and len(pd.read_parquet(done)) == len(sub):   # 図だけ描き直すときはt-SNEを使い回す
        xy = pd.read_parquet(done)[["tsne_x", "tsne_y"]].to_numpy()
    else:
        with threadpool_limits(limits=16):
            xy = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto", max_iter=1000,
                      random_state=0, n_jobs=16).fit_transform(ev_s)
    print(f"t-SNE done: {time.time() - t0:.0f}s", flush=True)
    pd.DataFrame({"slide_id": s.slide_id, "core": core_s, **{f"core_k{k}": v for k, v in pkc.items()},
                  "tsne_x": xy[:, 0], "tsne_y": xy[:, 1]}).to_parquet(out / f"tsne_{col}.parquet", index=False)
    order = np.random.default_rng(0).permutation(len(xy))
    exps = sorted(s.exp_id.unique())
    sub_note = f"{len(xy):,} of {len(ev):,} eval patches (random), {len(exps)} experiments, one t-SNE"

    # A. t-SNE 6枚
    fig, axes = plt.subplots(2, 3, figsize=(21, 14))
    ax = axes[0, 0]
    scatter_cores(ax, xy, core_s, order)
    label_medians(ax, xy, core_s, range(core_s.max() + 1))
    ax.set_title(f"cores ({method}, tau={tau}): {core.max() + 1} cores, grey = unassigned ({(core < 0).mean():.0%})", fontsize=11)
    for ax, title, values, cats, colors in [
            (axes[0, 1], "dose group", s.dose_level.to_numpy(), DOSES, DOSE_COLORS),
            (axes[1, 0], "time point", s.sacrifice_period.to_numpy(), DAYS, DAY_COLORS),
            (axes[1, 1], "scanner", s.scanner.to_numpy(), list(SCAN_COLORS), list(SCAN_COLORS.values()))]:
        for c_, col_ in zip(cats, colors):
            mm = values[order] == c_
            ax.scatter(*xy[order][mm].T, color=col_, s=1.2, alpha=.7, linewidths=0, rasterized=True, label=f"{c_} ({mm.mean():.0%})")
        ax.legend(markerscale=6, frameon=False, loc="lower left")
        ax.set_title(title, fontsize=11)
    ax = axes[0, 2]
    grp = np.where(s.dose_level == "Control", "Control", np.where(s.has_finding, "treated, slide with finding", "treated, no finding recorded"))
    for g, c_ in [("Control", "#9aa3ad"), ("treated, no finding recorded", "#a8cbf2"), ("treated, slide with finding", "#e34948")]:
        mm = grp[order] == g
        ax.scatter(*xy[order][mm].T, color=c_, s=1.2, alpha=.7, linewidths=0, rasterized=True, label=f"{g} ({mm.mean():.0%})")
    ax.legend(markerscale=6, frameon=False, loc="lower left")
    ax.set_title("slide-level finding record (not patch diagnosis)", fontsize=11)
    ax = axes[1, 2]
    epal = palette(len(exps))
    for e, c_ in zip(exps, epal):
        mm = s.exp_id.to_numpy()[order] == e
        ax.scatter(*xy[order][mm].T, color=c_, s=1.2, alpha=.7, linewidths=0, rasterized=True,
                   label=f"e{e} {s[s.exp_id == e].compound_name.iloc[0][:18]}")
    ax.legend(markerscale=6, frameon=False, loc="lower left", fontsize=6 if len(exps) > 20 else 7, ncol=2 if len(exps) <= 20 else 3)
    ax.set_title(f"experiment ({len(exps)} colors; see B for one panel per experiment)", fontsize=11)
    for a in axes.flat:
        bare(a)
    fig.suptitle(f"0017 {variant}: {sub_note}", fontsize=14, y=1.0)
    fig.savefig(out / f"A_tsne_overview_{col}.png")
    plt.close(fig)

    # A2. 所見ごと（その所見が記録されたslideのpatchを赤で）
    if has_findings:
        fl = s.finding_types.apply(as_list)
        top = pd.Series([f for v in slides[slides.dose_level != "Control"].finding_types.apply(as_list) for f in v]).value_counts().index[:20]
        fig, axes = plt.subplots(4, 5, figsize=(25, 20))
        for ax, f in zip(axes.flat, top):
            mm = fl.apply(lambda v: f in v).to_numpy()
            ax.scatter(*xy[~mm].T, color=GREY, s=.4, linewidths=0, rasterized=True)
            ax.scatter(*xy[mm].T, color="#e34948", s=1.5, linewidths=0, rasterized=True)
            ax.set_title(f"{short(f, 38)}\n{mm.sum():,} pts, {s[mm].slide_id.nunique()} slides, {s[mm].exp_id.nunique()} exp", fontsize=9)
            bare(ax)
        fig.suptitle(f"A2. Patches from slides recording each finding (red; slide-level record, not patch diagnosis), top 20 findings. {sub_note}",
                     fontsize=13, y=1.0)
        fig.tight_layout()
        fig.savefig(out / f"A2_tsne_by_finding.png")
        plt.close(fig)

    # B. 試験ごと
    nc = 5 if len(exps) <= 14 else 8
    nr = int(np.ceil((len(exps) + 1) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(4.4 * nc, 4.3 * nr))
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
    tab = pd.read_csv(run / f"cores_{col}.csv")
    fig, ax = plt.subplots(figsize=(13, 9))
    sc = ax.scatter(tab.ctrl_patch_frac, tab.top_exp_share, s=tab.n_patches / (3 if variant == "pilot" else 10),
                    c=tab.split_half_jaccard_median, cmap="viridis", vmin=0, vmax=1, alpha=.8, edgecolors="white", linewidths=.6)
    for r in tab.itertuples():
        lab = str(r.core)
        if isinstance(r.top_lift_findings, str) and r.top_lift_findings:
            lab += " " + short(r.top_lift_findings.split(";")[0])
        ax.annotate(lab, (r.ctrl_patch_frac, r.top_exp_share), fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("fraction of the core's patches that come from Control slides")
    ax.set_ylabel("share of the core's patches from its top experiment")
    fig.colorbar(sc, ax=ax, fraction=.03).set_label("split-half Jaccard (stability)")
    ax.set_title(f"Core map ({method}, tau={tau}): one point per core, size = patches; "
                 "labels: core id and the finding most enriched among its treated slides (x fold, if >=30% of slides)", fontsize=10)
    fig.savefig(out / f"C_core_map_{col}.png")
    plt.close(fig)
    if not ks:
        print(out)
        return

    # --- ここからkごとのコンセンサス（per_k） ---
    kf = max(ks)
    tabs = {k: pd.read_csv(run / "per_k" / f"cores_k{k}_tau{tau}.csv") for k in ks}
    chains = pd.read_csv(run / "per_k" / f"chains_tau{tau}.csv")
    nested = chains[chains.nested_all_levels][f"core_k{kf}"].to_numpy()
    lowc = tabs[kf][tabs[kf].ctrl_patch_frac < 0.05].sort_values("n_patches", ascending=False).core.to_numpy()
    nested_low = [c for c in lowc if c in set(nested)]

    # C2. kごとのコアの地図
    fig, axes = plt.subplots(1, len(ks), figsize=(9 * len(ks), 8), sharey=True)
    for ax, k in zip(axes, ks):
        t = tabs[k]
        sc = ax.scatter(t.ctrl_patch_frac, t.top_exp_share, s=t.n_patches / 10, c=t.split_half_J, cmap="viridis", vmin=0, vmax=1,
                        alpha=.8, edgecolors="white", linewidths=.6)
        if k == kf:
            m_ = t.core.isin(nested_low)
            ax.scatter(t.ctrl_patch_frac[m_], t.top_exp_share[m_], s=t.n_patches[m_] / 10, facecolors="none", edgecolors="#e34948", linewidths=1.8)
        for r in t.itertuples():
            ax.annotate(str(r.core), (r.ctrl_patch_frac, r.top_exp_share), fontsize=6, xytext=(2, 2), textcoords="offset points")
        ax.axvline(.05, color="#9aa3ad", lw=.8, ls="--")
        ax.set_xlabel("Control patch fraction")
        ax.set_title(f"k={k}: {len(t)} cores, assigned {(pk[f'core_k{k}'] >= 0).mean():.0%}, "
                     f"Control<5%: {(t.ctrl_patch_frac < .05).sum()}", fontsize=11)
    axes[0].set_ylabel("share of the core's patches from its top experiment")
    fig.colorbar(sc, ax=axes, fraction=.015).set_label("split-half J")
    fig.suptitle(f"C2. Per-k core maps (tau={tau}); dashed = Control 5%; red ring (k={kf}) = nested across all k and Control<5%", fontsize=13)
    fig.savefig(out / "C2_core_map_per_k.png")
    plt.close(fig)

    # D. kごとのコアをt-SNE上に
    fig, axes = plt.subplots(1, len(ks), figsize=(10 * len(ks), 10))
    for ax, k in zip(axes, ks):
        c_ = pkc[k]
        scatter_cores(ax, xy, c_, order)
        label_medians(ax, xy, c_, range(c_.max() + 1), fontsize=6 if k < kf else 5)
        ax.set_title(f"k={k}: {c_.max() + 1} cores, grey = unassigned ({(pk[f'core_k{k}'] < 0).mean():.0%})", fontsize=12)
        bare(ax)
    fig.suptitle(f"D. Per-k consensus cores (tau={tau}) on the same t-SNE; {sub_note}", fontsize=14)
    fig.savefig(out / "D_tsne_per_k_cores.png")
    plt.close(fig)

    # D2. 入れ子のコア（全粒度）と、そのうちControl<5%
    names = {r.core: f"{r.core} {r.top_exp[:10]}" for r in tabs[kf].itertuples()}
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    for ax, ids, title in [(axes[0], nested, f"nested across k={ks} ({len(nested)} of {len(tabs[kf])} k={kf} cores)"),
                           (axes[1], lowc, f"Control<5% ({len(lowc)} k={kf} cores)"),
                           (axes[2], nested_low, f"nested AND Control<5% ({len(nested_low)}; image sheet per_k{kf}_nested_ctrl_lt0.05)")]:
        scatter_cores(ax, xy, pkc[kf], order, ids, size=2.5)
        label_medians(ax, xy, pkc[kf], ids, fontsize=7, names=names)
        ax.set_title(title, fontsize=12)
        bare(ax)
    fig.suptitle(f"D2. k={kf} cores highlighted (label: core id, top compound); other cores and unassigned in grey", fontsize=14)
    fig.savefig(out / "D2_tsne_nested_and_low_control.png")
    plt.close(fig)

    # D3. Control<5%のk=200コアを1つずつ（親のk=100・k=50コアも重ねて入れ子を見せる）
    ch = chains.set_index(f"core_k{kf}")
    t_ = tabs[kf].set_index("core")
    coarse = [k for k in ks if k < kf][::-1]
    shades = ["#f5c48a", "#fbe7b5"]
    nc = 5
    nr = int(np.ceil(len(lowc) / nc))
    fig, axes = plt.subplots(nr, nc, figsize=(5.2 * nc, 5.6 * nr), squeeze=False)
    for ax, c in zip(axes.flat, lowc):
        ax.scatter(*xy.T, color=GREY, s=.4, linewidths=0, rasterized=True)
        for k, sh in list(zip(coarse, shades))[::-1]:
            par = ch.loc[c, f"parent_k{k}"]
            if par >= 0:
                m_ = pkc[k] == par
                ax.scatter(*xy[m_].T, color=sh, s=1.2, linewidths=0, rasterized=True)
        m_ = pkc[kf] == c
        ax.scatter(*xy[m_].T, color="#e34948", s=3, linewidths=0, rasterized=True)
        r = t_.loc[c]
        nest = "NESTED" if ch.loc[c, "nested_all_levels"] else "not nested"
        ax.set_title(f"k{kf} core {c} ({nest}): {r.n_patches} p / {r.n_slides} sl / {r.n_exp} exp, J {r.split_half_J:.2f}\n"
                     f"top {r.top_exp[:18]} {r.top_exp_share:.0%}, Ctrl {r.ctrl_patch_frac:.1%}\n"
                     f"{short(str(r.top_lift_findings) if isinstance(r.top_lift_findings, str) else '-', 44)}", fontsize=8)
    for ax in axes.flat:
        bare(ax)
    fig.suptitle(f"D3. Each k={kf} core with Control<5% (red), its parent cores at k={coarse} (orange, light yellow). "
                 f"p = patches, sl = slides; J = split-half Jaccard", fontsize=13, y=1.0)
    fig.tight_layout()
    fig.savefig(out / "D3_tsne_low_control_cores_each.png")
    plt.close(fig)

    # E. Control<5%のk=200コアの中身（全評価用patchで集計）
    allc = pk[f"core_k{kf}"].to_numpy()
    sl = slides.iloc[row]
    comp = {}
    for name, values, cats in [("experiment", sl.compound_name.to_numpy(), None), ("dose", sl.dose_level.to_numpy(), DOSES),
                               ("time", sl.sacrifice_period.to_numpy(), DAYS), ("scanner", sl.scanner.fillna("unknown").to_numpy(), list(SCAN_COLORS))]:
        comp[name] = pd.crosstab(allc[np.isin(allc, lowc)], values[np.isin(allc, lowc)], normalize="index").reindex(lowc)
        if cats:
            comp[name] = comp[name].reindex(columns=cats, fill_value=0)
    cmp_names = sorted(sl.compound_name.unique())
    cpal = dict(zip(cmp_names, palette(len(cmp_names))))
    fig, axes = plt.subplots(1, 4, figsize=(26, 0.42 * len(lowc) + 3), sharey=True, gridspec_kw={"width_ratios": [3, 1, 1, 1]})
    y = np.arange(len(lowc))
    for ax, name in zip(axes, comp):
        ct = comp[name]
        left = np.zeros(len(ct))
        if name == "experiment":
            for i, (c, r) in enumerate(ct.iterrows()):
                l_ = 0
                for cmpd, v in r.sort_values(ascending=False).items():
                    if v <= 0:
                        break
                    ax.barh(i, v, left=l_, color=cpal[cmpd], edgecolor="white", linewidth=.3)
                    if v >= .08:
                        ax.text(l_ + v / 2, i, cmpd[:12], ha="center", va="center", fontsize=6.5)
                    l_ += v
        else:
            colors = {"dose": DOSE_COLORS, "time": DAY_COLORS, "scanner": list(SCAN_COLORS.values())}[name]
            for c_, k_ in zip(ct.columns, colors):
                ax.barh(y, ct[c_], left=left, color=k_, label=c_)
                left += ct[c_].to_numpy()
            ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.02), ncol=2, fontsize=8)
        ax.set_title(f"{name} share of patches", fontsize=10)
    axes[0].set_yticks(y, [f"{c} {'N' if c in set(nested) else ' '} {t_.loc[c].n_patches}p" for c in lowc], fontsize=8)
    axes[0].invert_yaxis()
    fig.suptitle(f"E. Composition of the {len(lowc)} k={kf} cores with Control<5% (N = nested across all k; p = patches)", fontsize=13)
    fig.savefig(out / "E_low_control_core_composition.png")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()
