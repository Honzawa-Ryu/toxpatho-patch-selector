import argparse
import importlib.util
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

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
    """Define CLI args for all variable dimensions (used in GRID_ARGS / RUN_COMMAND)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")   # run_slurm.shが渡す。読むのは常に実験ディレクトリのconfig.yml
    parser.add_argument("--variant", default="v1")
    return parser.parse_args()


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 150})
_W = {}   # ワーカーごとに1回だけ読むモデル


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def select_slides(cfg, project_root):
    """対象slide: 塊ごとの試験（block6はQC目録でA_cleanのみ）で、特徴量があるもの。メタデータはQC目録から。"""
    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    q = q[q.EXP_ID.notna() & (q.EXP_ID != "")]
    q["exp"] = q.EXP_ID.astype(int)
    parts = []
    for name, b in cfg["blocks"].items():
        t = q[q.exp.isin(b["exp_ids"])]
        if b.get("qc_tiers"):
            t = t[t.tier.isin(b["qc_tiers"])]
        parts.append(t.assign(block=name))
    s = pd.concat(parts, ignore_index=True)
    h5 = []
    for sid in s.slide_id:
        hit = [project_root / d / f"{sid}.h5" for d in cfg["features_dirs"] if (project_root / d / f"{sid}.h5").exists()]
        h5.append(str(hit[0]) if hit else "")
    s["h5"] = h5
    s = s[s.h5 != ""].reset_index(drop=True)
    s["is_ctrl"] = s.DOSE_LEVEL == "Control"
    s["svs_path"] = [str(project_root / cfg["wsi_dir"] / str(p).replace(" ", "_") / f"{sid}.svs") for p, sid in zip(s.SACRIFICE_PERIOD, s.slide_id)]
    s = s.rename(columns={"COMPOUND_NAME": "compound", "DOSE_LEVEL": "dose", "SACRIFICE_PERIOD": "period"})
    return s


def _init_worker(models_path, cen_paths, cfg):
    import torch
    torch.set_num_threads(1)
    _W["models"] = pickle.loads(Path(models_path).read_bytes())
    _W["cen"] = {k: torch.from_numpy(np.load(v)) for k, v in cen_paths.items()}
    _W["cfg"] = cfg


def _assign_one(h5_path):
    """1枚の全patchを、slide内の順位 → 各版の表現 → 最も近いプロトタイプ、に割り当てる（0021・0023と同じ手順）。"""
    import torch
    ex21 = sys.modules["exp0021"]
    with h5py.File(h5_path, "r") as h:
        f = h["features"][:].astype(np.float32)
        xy = h["coords"][:]
        size0 = int(h["coords"].attrs.get("patch_size_level0", 256))
    rk = sys.modules["exp0023"].slide_rank(f)
    out = {"xy": xy, "size0": size0}
    for name, cen in _W["cen"].items():
        Z = torch.from_numpy(ex21.transform(_W["models"], rk, name, _W["cfg"]))
        out[name] = torch.cdist(Z, cen).argmin(1).numpy().astype(np.int16)
    return out


def slide_findings(cfg, project_root, slides):
    """所見表（自然発生を除く）から、slideごと・所見ごとの最大の程度（0 = 記録なし）。"""
    p = pd.read_csv(project_root / cfg["pathology_csv"], dtype=str)
    p = p[(p.ORGAN == "Liver") & (p.SP_FLG.str.lower() != "true")]
    key = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"]
    sk = slides[["slide_id"] + key].copy()
    for c in key:
        sk[c] = pd.to_numeric(sk[c], errors="coerce")
        p[c] = pd.to_numeric(p[c], errors="coerce")
    t = sk.merge(p, on=key)
    t["g"] = t.GRADE_TYPE.map(cfg["grade_map"]).fillna(1)
    G = t.pivot_table(index="slide_id", columns="FINDING_TYPE", values="g", aggfunc="max").reindex(slides.slide_id).fillna(0)
    comp = t.groupby("FINDING_TYPE").COMPOUND_NAME.nunique()
    npos = (G > 0).sum()
    keep = [f for f in G.columns if (npos[f] >= cfg["min_pos"] and comp[f] >= cfg["min_compounds"]) or f in cfg["force_findings"]]
    return G[keep], npos[keep], comp[keep]


def within_exp_delta(share, slides, min_ctrl):
    """各slideのプロトタイプ割合から、同じ試験のControl slideの平均を引く。Controlがmin_ctrl枚未満の試験はNaN。"""
    d = np.full_like(share, np.nan)
    for e, idx in slides.groupby("exp").indices.items():
        c = idx[slides.is_ctrl.to_numpy()[idx]]
        if len(c) >= min_ctrl:
            d[idx] = share[idx] - share[c].mean(0)
    return d


def finding_assoc(delta, G, slides, cfg):
    """所見 × プロトタイプ: 所見のある試験（Controlが十分）のslideで、程度と「Controlとの差」の順位相関。
    あわせて、所見のある試験ごとに「陽性slideの差の平均 > 0」かを数える（複数の化合物で同じ向きか）。"""
    from scipy.stats import spearmanr
    rows = []
    exp_arr = slides.exp.to_numpy()
    for f in G.columns:
        g = G[f].to_numpy()
        exps = np.unique(exp_arr[(g > 0) & ~np.isnan(delta[:, 0])])
        m = np.isin(exp_arr, exps) & ~np.isnan(delta[:, 0])
        for k in range(delta.shape[1]):
            rho = spearmanr(g[m], delta[m, k])[0] if m.sum() > 5 and len(np.unique(g[m])) > 1 else np.nan
            up = [float(delta[(exp_arr == e) & (g > 0), k].mean() > 0) for e in exps]
            rows.append({"finding": f, "prototype": k, "rho": rho, "n_slides": int(m.sum()), "n_pos": int(((g > 0) & m).sum()),
                         "n_exps": len(exps), "exps_up": int(sum(up)), "frac_exps_up": float(np.mean(up)) if up else np.nan,
                         "mean_delta_pos": float(delta[(g > 0) & m, k].mean())})
    return pd.DataFrame(rows)


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()

    # Build variant_key from all variable dimensions so each config gets its own directory.
    # Include every arg that changes the result (model, seed, prompt, ...).
    # Example:
    #   model_short = args.model.replace("/", "-")
    #   variant_key = f"{model_short}__{args.seed}"
    variant_key = args.variant

    # Initialize output directory (exits immediately if already completed).
    # Written on OUTPUT_ROOT (scratch) when set; scripts/slurm_entry.sh
    # rsyncs it back to project_root/outputs/ at job end.
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    import time
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    ex21 = load_module(project_root / "experiments" / config["exp0021"] / "experiment.py", "exp0021")
    load_module(project_root / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex17 = load_module(project_root / "experiments/0017_20261016_block_consensus_patterns/experiment.py", "exp0017")
    src = project_root / "outputs" / config["exp0021"] / config["models_from"]
    reps = config["representations"]
    slides = select_slides(config, project_root)
    logger.info(f"{len(slides)} slides ({slides.block.value_counts().to_dict()}), {slides.exp.nunique()} experiments, Control {int(slides.is_ctrl.sum())}")
    slides.drop(columns=["h5"]).to_csv(run_dir / "slides.csv", index=False)

    # 1. 全patchをプロトタイプに割り当てる（slideごとに並列）
    cache = run_dir / "assignments.npz"
    if cache.exists():
        z = np.load(cache)
        A = {r: z[r] for r in reps}
        row, xy, size0 = z["row"], z["xy"], dict(zip(slides.slide_id, z["size0"]))
    else:
        cen_paths = {r: str(src / f"centroids_{r}.npy") for r in reps}
        A, rows_, xys, s0 = {r: [] for r in reps}, [], [], []
        with ProcessPoolExecutor(max_workers=config["workers"], initializer=_init_worker,
                                 initargs=(str(src / "models.pkl"), cen_paths, {"pca_components": config["pca_components"], "combo_resid_weight": config["combo_resid_weight"]})) as ex:
            for i, o in enumerate(ex.map(_assign_one, slides.h5, chunksize=4)):
                for r in reps:
                    A[r].append(o[r])
                rows_.append(np.full(len(o["xy"]), i, dtype=np.int32))
                xys.append(o["xy"])
                s0.append(o["size0"])
                if (i + 1) % 200 == 0:
                    logger.info(f"  assigned {i + 1}/{len(slides)} slides; {time.time() - t0:.0f}s")
        A = {r: np.concatenate(v) for r, v in A.items()}
        row, xy = np.concatenate(rows_), np.concatenate(xys)
        np.savez_compressed(cache, row=row, xy=xy, size0=np.array(s0), **A)
        size0 = dict(zip(slides.slide_id, s0))
    k = int(A[reps[0]].max()) + 1
    n_per = np.bincount(row, minlength=len(slides))
    logger.info(f"{len(row):,} patches assigned; median {np.median(n_per):.0f} per slide; {time.time() - t0:.0f}s")

    # 2. 所見と、試験内のControlとの差
    G, npos, ncomp = slide_findings(config, project_root, slides)
    logger.info("findings: " + ", ".join(f"{f} ({npos[f]} slides, {ncomp[f]} compounds)" for f in G.columns))
    G.to_csv(run_dir / "slide_findings.csv")
    tabs, shares = {}, {}
    for r in reps:
        sh = np.zeros((len(slides), k))
        np.add.at(sh, (row, A[r].astype(int)), 1)
        sh /= np.maximum(n_per[:, None], 1)
        shares[r] = sh
        pd.DataFrame(sh, index=slides.slide_id, columns=[f"p{j}" for j in range(k)]).to_csv(run_dir / f"slide_shares_{r}.csv")
        tb = finding_assoc(within_exp_delta(sh, slides, config["min_ctrl"]), G, slides, config)
        lab = pd.read_csv(src / f"clusters_{r}.csv").set_index("cluster")
        tb["bayer_top_label"] = tb.prototype.map(lab.top_label)
        tb["bayer_purity"] = tb.prototype.map(lab.top_label_purity)
        tb.to_csv(run_dir / f"assoc_{r}.csv", index=False)
        tabs[r] = tb

    # 3. 版ごとの要約: 所見ごとに、程度と最も強く正に相関するプロトタイプ（複数の試験で同じ向きのもの）
    summ = []
    for r, tb in tabs.items():
        ok = tb[(tb.n_exps >= config["min_exps"])]
        for f, t in tb.groupby("finding"):
            b = t.sort_values("rho", ascending=False).iloc[0]
            summ.append({"rep": r, "finding": f, "best_proto": int(b.prototype), "best_rho": b.rho, "exps_up": f"{int(b.exps_up)}/{int(b.n_exps)}",
                         "bayer_top_label": b.bayer_top_label, "n_pos": int(b.n_pos)})
    summ = pd.DataFrame(summ)
    summ.to_csv(run_dir / "summary_best.csv", index=False)
    logger.info("\n" + summ.pivot(index="finding", columns="rep", values="best_rho").round(3).to_string())

    # 4. 図
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    findings = list(G.columns)
    fig, axes = plt.subplots(1, len(reps), figsize=(6.2 * len(reps), 0.42 * len(findings) + 2.2), squeeze=False)
    for ax, r in zip(axes[0], reps):
        M = tabs[r].pivot(index="finding", columns="prototype", values="rho").reindex(findings)
        im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
        U = tabs[r].pivot(index="finding", columns="prototype", values="frac_exps_up").reindex(findings).to_numpy()
        for i, j in zip(*np.where((M.to_numpy() >= 0.3) & (U >= 2 / 3))):
            ax.text(j, i, "*", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(k), range(k), fontsize=6)
        ax.set_yticks(range(len(findings)), [f"{f} (n={npos[f]})" for f in findings] if r == reps[0] else [], fontsize=7)
        ax.set_title(f"{r}", fontsize=10)
        ax.set_xlabel("prototype")
    fig.colorbar(im, ax=axes[0], shrink=0.6, label="Spearman: grade vs share minus same-experiment Control mean")
    fig.suptitle("U1. Findings without Bayer labels vs prototype share (within-experiment).  * rho >= 0.3 and up in >= 2/3 of the experiments", fontsize=10)
    fig.savefig(out / "U1_assoc_heatmaps.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 0.38 * len(findings) + 1.5))
    piv = summ.pivot(index="finding", columns="rep", values="best_rho").reindex(findings)
    mk = {"unsup": ("o", "#0072B2"), "weak": ("s", "#E69F00"), "combo": ("D", "#009E73")}
    for r in reps:
        ax.scatter(piv[r], range(len(findings)), marker=mk[r][0], color=mk[r][1], label=r, s=28, zorder=3)
    ax.set_yticks(range(len(findings)), findings, fontsize=7.5)
    ax.invert_yaxis()
    ax.axvline(0, color="0.6", lw=0.8)
    ax.set_xlabel("best Spearman over prototypes (grade vs within-experiment share difference)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("U2. Best-matching prototype per finding: unsupervised vs weakly supervised", fontsize=10)
    fig.savefig(out / "U2_best_rho_by_rep.png")
    plt.close(fig)

    # 程度ごとの散布（版ごとの上位の組）と、画像シート
    rng = np.random.default_rng(seed)
    delta = {r: within_exp_delta(shares[r], slides, config["min_ctrl"]) for r in reps}
    for r in reps:
        top = tabs[r][(tabs[r].n_exps >= config["min_exps"]) & (tabs[r].frac_exps_up >= 2 / 3)].sort_values("rho", ascending=False)
        top = top.drop_duplicates("finding").head(config["top_pairs"])
        if not len(top):
            continue
        fig, axes = plt.subplots(2, (len(top) + 1) // 2, figsize=(3.3 * ((len(top) + 1) // 2), 6.4), squeeze=False)
        sheet_rows = []
        for ax, t in zip(axes.flat, top.itertuples()):
            g = G[t.finding].to_numpy()
            exps = np.unique(slides.exp[(g > 0)])
            m = slides.exp.isin(exps).to_numpy() & ~np.isnan(delta[r][:, 0])
            cmps = slides.compound[m].to_numpy()
            pal = {c: plt.get_cmap("tab10")(i % 10) for i, c in enumerate(sorted(set(cmps)))}
            ax.scatter(g[m] + rng.uniform(-.18, .18, m.sum()), delta[r][m, t.prototype], s=7, c=[pal[c] for c in cmps], alpha=.7, linewidths=0)
            ax.axhline(0, color="0.6", lw=0.7)
            ax.set_xticks(range(5), ["0", "min", "sli", "mod", "sev"], fontsize=7)
            ax.set_title(f"{t.finding[:28]}\nproto {t.prototype} ({t.bayer_top_label}) rho {t.rho:.2f}, up {t.exps_up}/{t.n_exps} exp", fontsize=7.5)
            ax.set_ylabel("share - Control mean (same exp)", fontsize=7)
            # 画像: その所見の程度が高いslideから、そのプロトタイプのpatchを1枚ずつ（程度の高い順）
            pos = np.flatnonzero(g > 0)
            pos = pos[np.argsort(-g[pos], kind="stable")]
            picks = []
            for si in pos:
                cand = np.flatnonzero((row == si) & (A[r] == t.prototype))
                if len(cand):
                    picks.append(int(rng.choice(cand)))
                if len(picks) == config["sheet_tiles"]:
                    break
            if picks:
                tiles = ex17.read_tiles(picks, row, xy, slides, size0,
                                        lambda s_, p: (f"e{s_.exp} {str(s_.compound)[:13]} {str(s_.dose)[0]}", f"{s_.period} grade {int(G[t.finding].iloc[row[p]])}"))
                sheet_rows.append((f"{t.finding} | {r} prototype {t.prototype} (Bayer: {t.bayer_top_label}) | rho {t.rho:.2f}, up in {t.exps_up}/{t.n_exps} experiments", tiles))
        for ax in list(axes.flat)[len(top):]:
            ax.axis("off")
        fig.suptitle(f"U3. {r}: top finding-prototype pairs (colour = compound)", fontsize=10)
        fig.savefig(out / f"U3_top_pairs_{r}.png")
        plt.close(fig)
        for f0 in range(0, len(sheet_rows), 4):
            ex17.draw_sheet(f"0024 {r}: patches of the prototype in slides with the finding (highest grade first, one per slide)",
                            sheet_rows[f0:f0 + 4], out / "sheets" / f"{r}_{f0 // 4 + 1:02d}.jpg")
        logger.info(f"{r}: figures done; {time.time() - t0:.0f}s")

    results: dict = {"n_slides": len(slides), "n_patches": int(len(row)), "findings": {f: int(npos[f]) for f in findings},
                     "best_rho": piv.round(4).to_dict()}

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()