"""0017のコンセンサスをkごとに分けて作り直し、粗いkのコアに細かいkのコアが入れ子になっているかを見る（案2、k-meansは回し直さない）。

各k（50/100/200）について、そのkの20 runだけで数えた共所属率（edges.parquetの rate_k*）から、0017と同じ手順（τ・Leiden・最小サイズ）でコアを作る。
一致度も、そのkの20 runを半分ずつに分けて測る。
入れ子: 細かいkの各コアについて、1つ粗いkのどのコアにpatchの何割が入るか（包含率）。k200→k100→k50と全段で包含率が高いものを「全粒度で残るコア」とみなす。

usage: .venv/bin/python scripts/consensus_per_k.py [variant] [tau]   既定: pilot 0.8
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments/0017_20261016_block_consensus_patterns"
CONTAIN = 0.7      # 「入れ子」とみなす包含率（目安）
spec = importlib.util.spec_from_file_location("exp0017", EXP / "experiment.py")
exp0017 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp0017)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 150})


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    run = ROOT / "outputs/0017_20261016_block_consensus_patterns" / variant
    out = run / "per_k"
    out.mkdir(exist_ok=True)
    cfg = yaml.safe_load(open(EXP / "config.yml"))
    cfg.update(cfg.get("variants", {}).get(variant, {}))

    labels = np.load(run / "eval_labels.npy")
    runs = pd.read_csv(run / "runs.csv")
    e = pd.read_parquet(run / "edges.parquet")
    pairs = e[["a", "b"]].to_numpy()
    ev = pd.read_parquet(run / "eval_patches.parquet")
    n = len(ev)
    m = pd.read_parquet(ROOT / cfg["manifest_parquet"])
    m["slide_id"] = m.slide_id.astype(str)
    scan = pd.read_csv(ROOT / "experiments/0010_20260925_pathology_benchmark_subset/label_audit/slide_scanner_meta.csv",
                       dtype={"slide_id": str})[["slide_id", "aperio.ScanScope ID"]].rename(columns={"aperio.ScanScope ID": "scanner"})
    if "scanner" not in m:   # block6のmanifestには既にscanner列がある（二重に結合するとscanner_x/_yになる）
        m = m.merge(scan, on="slide_id", how="left")
    slides = m[m.exp_id.isin(cfg["exp_ids"])].reset_index(drop=True)
    row = ev.slide_id.map({s: i for i, s in enumerate(slides.slide_id)}).to_numpy()
    treated = slides[slides.dose_level != "Control"]
    base_rate = treated.finding_types.explode().dropna().value_counts() / len(treated)
    a, b = pairs[:, 0], pairs[:, 1]

    cores, tables, summary = {}, {}, []
    rng = np.random.default_rng(cfg["seed"])
    for k in sorted(runs.k.unique()):
        idx = np.flatnonzero(runs.k.to_numpy() == k)
        rate = e[f"rate_k{k}"].to_numpy()
        core = exp0017.cores_from_edges(n, pairs, rate, tau, "leiden", row, cfg)
        js = []
        for _ in range(cfg["n_split_repeats"]):
            p = rng.permutation(idx)
            for h in (p[: len(p) // 2], p[len(p) // 2:]):
                r = (labels[h][:, a] == labels[h][:, b]).mean(0)
                js.append(exp0017.best_jaccard(core, exp0017.cores_from_edges(n, pairs, r, tau, "leiden", row, cfg)))
        jmed = np.median(np.stack(js), axis=0)
        same = (core[a] == core[b]) & (core[a] >= 0)
        cs = np.bincount(a[same], weights=rate[same], minlength=n) + np.bincount(b[same], weights=rate[same], minlength=n)
        cn = np.bincount(a[same], minlength=n) + np.bincount(b[same], minlength=n)
        tab = exp0017.core_table(core, row, slides, np.where(cn > 0, cs / np.maximum(cn, 1), 0.0), base_rate)
        tab["split_half_J"] = tab.core.map(dict(enumerate(jmed)))
        cores[k], tables[k] = core, tab
        lo = tab.ctrl_patch_frac < 0.05
        summary.append({"k": k, "n_cores": len(tab), "assigned_frac": float((core >= 0).mean()),
                        "J_median": float(tab.split_half_J.median()), "n_ctrl_lt5pct": int(lo.sum()),
                        "J_median_ctrl_lt5pct": float(tab[lo].split_half_J.median()) if lo.any() else np.nan,
                        "J_median_ctrl_ge5pct": float(tab[~lo].split_half_J.median()) if (~lo).any() else np.nan})
        tab.sort_values("n_patches", ascending=False).to_csv(out / f"cores_k{k}_tau{tau:g}.csv", index=False)
        print(summary[-1], flush=True)
    pd.DataFrame(summary).to_csv(out / f"summary_tau{tau:g}.csv", index=False)
    ev.assign(**{f"core_k{k}": c for k, c in cores.items()})[["slide_id", "x", "y"] + [f"core_k{k}" for k in cores]] \
        .to_parquet(out / f"assignments_tau{tau:g}.parquet", index=False)

    # 入れ子: 細かいk → 1つ粗いk
    ks = sorted(cores)
    parent = {}
    for fine, coarse in zip(ks[::-1][:-1], ks[::-1][1:]):
        recs = []
        for c in range(cores[fine].max() + 1):
            mm = cores[fine] == c
            pc = pd.Series(cores[coarse][mm]).value_counts(normalize=True)
            best = pc.drop(-1, errors="ignore")
            recs.append({f"core_k{fine}": c, f"parent_k{coarse}": int(best.index[0]) if len(best) else -1,
                         "contained_frac": float(best.iloc[0]) if len(best) else 0.0,
                         "frac_unassigned_at_coarse": float(pc.get(-1, 0.0))})
        parent[(fine, coarse)] = pd.DataFrame(recs)
        parent[(fine, coarse)].to_csv(out / f"nesting_k{fine}_in_k{coarse}_tau{tau:g}.csv", index=False)

    # 全粒度で残るコア: k200のコアから親をたどる
    f, mid, c = ks[2], ks[1], ks[0]
    p1, p2 = parent[(f, mid)].set_index(f"core_k{f}"), parent[(mid, c)].set_index(f"core_k{mid}")
    chains = []
    for r in tables[f].itertuples():
        q = p1.loc[r.core]
        rec = {f"core_k{f}": r.core, "n_patches": r.n_patches, "ctrl_frac": r.ctrl_patch_frac, "top_exp": r.top_exp,
               "top_exp_share": r.top_exp_share, f"J_k{f}": r.split_half_J, "findings": exp0017.lift_text(r),
               f"parent_k{mid}": q[f"parent_k{mid}"], f"contained_in_k{mid}": q.contained_frac}
        if q[f"parent_k{mid}"] >= 0:
            q2 = p2.loc[q[f"parent_k{mid}"]]
            rec.update({f"parent_k{c}": q2[f"parent_k{c}"], f"contained_in_k{c}": q2.contained_frac,
                        f"J_k{mid}": tables[mid].set_index("core").split_half_J.get(q[f"parent_k{mid}"], np.nan),
                        f"J_k{c}": tables[c].set_index("core").split_half_J.get(q2[f"parent_k{c}"], np.nan)})
        rec["nested_all_levels"] = bool(rec.get(f"contained_in_k{mid}", 0) >= CONTAIN and rec.get(f"contained_in_k{c}", 0) >= CONTAIN)
        chains.append(rec)
    chains = pd.DataFrame(chains).sort_values(["nested_all_levels", "ctrl_frac"], ascending=[False, True])
    chains.to_csv(out / f"chains_k{f}_to_k{c}_tau{tau:g}.csv", index=False)

    # 図: kごとのコアマップ ＋ 包含率の分布
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), gridspec_kw={"width_ratios": [1, 1, 1, .8]})
    for ax, k in zip(axes, ks):
        t = tables[k]
        sc = ax.scatter(t.ctrl_patch_frac, t.top_exp_share, s=t.n_patches / 4, c=t.split_half_J, cmap="viridis", vmin=0, vmax=1,
                        alpha=.85, edgecolors="white", linewidths=.5)
        ax.set_title(f"k={k}: {len(t)} cores, {(t.ctrl_patch_frac < .05).sum()} with Control < 5%", fontsize=11)
        ax.set_xlabel("Control patch fraction")
        ax.set_ylabel("top-1 experiment share")
        ax.set_xlim(-.02, .7)
        ax.set_ylim(0, 1.02)
    fig.colorbar(sc, ax=axes[2], fraction=.04).set_label("split-half Jaccard (within k)")
    ax = axes[3]
    for (fine, coarse), d in parent.items():
        ax.hist(d.contained_frac, bins=np.linspace(0, 1, 21), histtype="step", lw=1.8, label=f"k{fine} core in a k{coarse} core")
    ax.axvline(CONTAIN, color="#eb6834", ls="--", lw=1)
    ax.set_xlabel("fraction of the fine core's patches inside its best coarse core")
    ax.set_title("nesting across k", fontsize=11)
    ax.legend(frameon=False)
    fig.suptitle(f"0017 {variant}: consensus built separately for each k (tau={tau:g}, Leiden res {cfg['leiden_resolution']})", fontsize=13, y=1.02)
    fig.savefig(out / f"per_k_overview_tau{tau:g}.png")
    plt.close(fig)
    print(out)
    print("nested at all levels:", int(chains.nested_all_levels.sum()), "/", len(chains), "k200 cores;",
          "Control<5% & nested:", int((chains.nested_all_levels & (chains.ctrl_frac < .05)).sum()))


if __name__ == "__main__":
    main()
