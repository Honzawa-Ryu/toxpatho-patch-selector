import argparse
import importlib.util
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

# --- Deep learning (uncomment if needed) ---
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset

# --- Visualization (uncomment if needed) ---
# import matplotlib.pyplot as plt
# import seaborn as sns


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


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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
    from concurrent.futures import ProcessPoolExecutor
    import torch
    from scipy.stats import spearmanr
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = {**config, **config.get("variants", {}).get(args.variant, {})}
    t0 = time.time()
    load_module(project_root / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    load_module(project_root / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex26 = load_module(project_root / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(project_root / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    cfg20 = yaml.safe_load(open(project_root / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    run26 = project_root / "outputs" / cfg["exp0026"]

    # 1. 0026 tree_v3と同じLDA（Bayerの230枚、脂肪滴系の21枚のラベルを除く）
    src = project_root / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    Ka, laba = c["K"], c["lab"].astype(np.int64)
    keep = [k for k in np.unique(laba) if k >= 0 and (laba == k).sum() >= cfg20["min_label_patches"]]
    laba = np.where(np.isin(laba, keep), laba, -1)
    lab_c = np.where(laba >= 0, np.searchsorted(keep, np.maximum(laba, 0)), -1)
    bayer = pd.read_csv(src / "slides.csv", dtype=str)
    ex_ids = pd.read_csv(project_root / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab_c = np.where(np.isin(bayer.slide_id.to_numpy()[c["row"]], ex_ids), -1, lab_c)
    tf = ex26.fit_combo(Ka, lab_c, cfg, cfg["seed"])
    L = len(keep) - 1
    del Ka

    # 2. 対象のslide: 対象の所見がある試験の全slide（Bayerの230枚と脂肪変性の3試験は除く）
    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    q = q[q.EXP_ID.notna() & ~q.slide_id.isin(bayer.slide_id)]
    q = q[~q.EXP_ID.astype(int).isin(cfg["exclude_exps"])].copy()
    def h5_of(sid):
        for d_ in cfg["features_dirs"]:
            if (project_root / d_ / f"{sid}.h5").exists():
                return str(project_root / d_ / f"{sid}.h5")
        return ""
    q["h5"] = [h5_of(x) for x in q.slide_id]
    q = q[q.h5 != ""]
    pth = pd.read_csv(project_root / cfg["pathology_csv"], dtype=str)
    pth = pth[(pth.ORGAN == "Liver") & (pth.SP_FLG.str.lower() != "true") & pth.FINDING_TYPE.isin(list(cfg["findings"]))]
    key = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"]
    for k_ in key:
        q[k_] = pd.to_numeric(q[k_], errors="coerce")
        pth[k_] = pd.to_numeric(pth[k_], errors="coerce")
    pth["g"] = pth.GRADE_TYPE.map(cfg["grade_map"]).fillna(1)
    pos = q.merge(pth, on=key)
    exps = sorted(set(pos.EXP_ID))
    sl = q[q.EXP_ID.isin(exps)].reset_index(drop=True)
    G = pos.pivot_table(index="slide_id", columns="FINDING_TYPE", values="g", aggfunc="max").reindex(sl.slide_id).fillna(0)
    G = G.reindex(columns=list(cfg["findings"]), fill_value=0)
    sl["is_ctrl"] = sl.DOSE_LEVEL == "Control"
    sl["bayer_exp"] = sl.EXP_ID.isin(set(bayer.EXP_ID.dropna().astype(int)))
    logger.info(f"slides: {len(sl)} in {len(exps)} experiments ({sl.COMPOUND_NAME.nunique()} compounds), controls {int(sl.is_ctrl.sum())}, "
                f"in Bayer experiments {int(sl.bayer_exp.sum())}; positives per finding: {(G > 0).sum().to_dict()}; {time.time() - t0:.0f}s")

    # 3. 各slideを2,048 patchに間引き、slide内の順位 → PCA256 → U（64）・LDA（L）
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=ex27._init_ctrl, initargs=(str(project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl"), cfg["patches_per_slide"])) as ex:
        Us, Ls = [], []
        for K in ex.map(ex27._load_ctrl, sl.h5, chunksize=4):
            Us.append(K[:, :64])
            Ls.append(tf(K)[:, :L].astype(np.float32))
    row = np.concatenate([np.full(len(u), i) for i, u in enumerate(Us)])
    U, Lp = np.concatenate(Us), np.concatenate(Ls)
    del Us, Ls
    expn = sl.EXP_ID.to_numpy()[row]
    cm = sl.is_ctrl.to_numpy()[row]
    e_fix = np.where(np.isin(expn, expn[cm]), expn, -999)   # 対照群のない試験は全試験の対照群でまとめて白色化
    Sw = ex27.sphering(U, e_fix, np.r_[U[cm], U[cm]], np.r_[expn[cm], np.full(int(cm.sum()), -999)])
    sc = json.loads((run26 / "tree_v3" / "scale_sphering_lda.json").read_text())
    rep_X = {"sphering_lda": lambda: np.concatenate([Sw, sc["lda_beta"] * sc["lda_scale"] * Lp], 1), "sphering": lambda: Sw, "lda": lambda: Lp, "unsup": lambda: U}
    logger.info(f"patches: {len(U):,}; {time.time() - t0:.0f}s")

    # 4. プロトタイプに割り当て、slideごとの構成比（同じ試験の対照群の平均を引く）
    names, shares, deltas = {}, {}, {}
    for mname, (run, rep, k) in cfg["maps"].items():
        cen = torch.from_numpy(np.load(run26 / run / f"centroids_{rep}_k{k}.npy").astype(np.float32))
        X = torch.from_numpy(rep_X[rep]().astype(np.float32))
        cl = torch.cat([torch.cdist(x, cen).argmin(1) for x in X.split(65536)]).numpy()
        sh = np.zeros((len(sl), k))
        np.add.at(sh, (row, cl), 1)
        sh /= sh.sum(1, keepdims=True)
        ctrl_mean = pd.DataFrame(sh[sl.is_ctrl.to_numpy()]).groupby(sl.EXP_ID.to_numpy()[sl.is_ctrl.to_numpy()]).mean()
        base = ctrl_mean.reindex(sl.EXP_ID).to_numpy()
        base = np.where(np.isnan(base), sh[sl.is_ctrl.to_numpy()].mean(0), base)
        shares[mname], deltas[mname] = sh, sh - base
        names[mname] = pd.read_csv(project_root / "outputs" / cfg["exp0029"] / cfg["names_from"] / f"prototype_names_{mname}.csv").set_index("prototype").name.fillna("").reindex(range(k)).to_numpy()
        np.save(run_dir / f"shares_{mname}.npy", sh)
    sl.drop(columns=["h5"]).assign(**{f"grade__{f}": G[f].to_numpy() for f in G}).to_csv(run_dir / "slides.csv", index=False)
    logger.info(f"assigned; {time.time() - t0:.0f}s")

    # 5. 所見の分類（化合物単位の分割）: (a) TG-GATEsの記録で読み出しを学ぶロジスティック回帰、(b) 学習なし: Bayerのラベルの名前が付いたプロトタイプの割合
    comp = sl.COMPOUND_NAME.to_numpy()
    rows = []
    for f, blab in cfg["findings"].items():
        y = (G[f].to_numpy() > 0).astype(int)
        npos_c = len(set(comp[y == 1]))
        for mname in cfg["maps"]:
            D_ = deltas[mname]
            oof = np.full(len(sl), np.nan)
            for tr, te in GroupKFold(n_splits=min(cfg["cv_splits"], npos_c)).split(D_, y, comp):
                if y[tr].sum() == 0:
                    continue
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
                oof[te] = clf.fit(D_[tr], y[tr]).predict_proba(D_[te])[:, 1]
            ok = ~np.isnan(oof)
            r = {"finding": f, "bayer_label": blab or "(none)", "map": mname, "n_pos": int(y.sum()), "n_pos_compounds": npos_c,
                 "AUC_readout": roc_auc_score(y[ok], oof[ok]) if len(set(y[ok])) > 1 else np.nan}
            for part, pm in [("bayer_exp", sl.bayer_exp.to_numpy()), ("new_exp", ~sl.bayer_exp.to_numpy())]:
                m_ = ok & pm
                r[f"AUC_readout_{part}"] = roc_auc_score(y[m_], oof[m_]) if len(set(y[m_])) > 1 else np.nan
            if blab:
                gs = deltas[mname][:, names[mname] == blab].sum(1)
                r["n_named_protos"] = int((names[mname] == blab).sum())
                r["AUC_named_noTGtraining"] = roc_auc_score(y, gs) if (names[mname] == blab).any() else np.nan
                r["rho_named_vs_grade"] = spearmanr(G[f].to_numpy(), gs)[0] if (names[mname] == blab).any() else np.nan
            # プロトタイプ: 記録ありslideで最も増えるもの（対照群との差の平均）と、その名前
            j = int(np.argmax(D_[y == 1].mean(0)))
            r.update({"top_proto": j, "top_proto_name": names[mname][j], "top_proto_delta_pos": float(D_[y == 1, j].mean()), "top_proto_delta_neg": float(D_[(y == 0) & ~sl.is_ctrl.to_numpy(), j].mean())})
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "finding_classification.csv", index=False)
    pd.set_option("display.width", 250)
    logger.info("\n" + res.pivot_table(index="finding", columns="map", values="AUC_readout").round(3).to_string())
    logger.info("named prototypes, no TG-GATEs training:\n" + res.dropna(subset=["AUC_named_noTGtraining"]).pivot_table(index="finding", columns="map", values="AUC_named_noTGtraining").round(3).to_string())

    # 6. 図
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    order = list(cfg["findings"])
    mk = {"D_unsup": ("o", "#999999"), "C_lda": ("s", "#009E73"), "B_sphering": ("^", "#0072B2"), "A_sphering_lda": ("D", "#D55E00")}
    for ax, col, ttl in [(axes[0], "AUC_readout", "readout trained on TG-GATEs records (compound-held-out CV)"), (axes[1], "AUC_named_noTGtraining", "no TG-GATEs training: prototypes named by Bayer labels")]:
        for mname, (mk_, cc) in mk.items():
            v = res[res["map"] == mname].set_index("finding")[col].reindex(order) if col in res else None
            if v is None:
                continue
            ax.scatter(v.to_numpy(), range(len(order)), marker=mk_, color=cc, s=36, label=mname, zorder=3)
        ax.axvline(0.5, color="0.75", lw=0.8)
        ax.set_xlim(0.3, 1.0)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("AUROC (finding recorded vs not)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(range(len(order)), [f"{f} [{cfg['findings'][f] or 'no Bayer label'}]" for f in order], fontsize=8)
    axes[0].invert_yaxis()
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle(f"TG-GATEs slides unseen in training ({len(sl)} slides, {len(exps)} experiments)", fontsize=10)
    fig.tight_layout()
    fig.savefig(run_dir / "F1_finding_auc.png", dpi=150)
    plt.close(fig)

    results: dict = {"n_slides": len(sl), "n_experiments": len(exps), "mean_AUC_readout": res.groupby("map").AUC_readout.mean().to_dict(),
                     "mean_AUC_named": res.groupby("map").AUC_named_noTGtraining.mean().to_dict() if "AUC_named_noTGtraining" in res else {}}

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()