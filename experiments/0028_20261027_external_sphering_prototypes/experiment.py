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


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def proto_table(run_dir26, run, rep, k):
    """0026のプロトタイプのTG-GATEsでの素性（Bayerの最多ラベル、脂肪変性の純度）。"""
    t = run_dir26 / run / "tree" / f"{rep}_k{k}" / "prototypes_in_tree_order.csv"
    if t.exists():
        d = pd.read_csv(t).set_index("prototype")
        return d.rename(columns={"fatty_purity_CCl4": "fat_CCl4", "fatty_purity_cholesterol": "fat_chol"})[["bayer_top_label", "bayer_purity", "fat_CCl4", "fat_chol"]]
    d = pd.read_csv(run_dir26 / run / f"prototypes_{rep}_k{k}.csv").set_index("prototype")
    return d.rename(columns={"frac_from_fatty_CCl4": "fat_CCl4", "frac_from_fatty_cholesterol": "fat_chol"}).assign(bayer_purity=np.nan)[["bayer_top_label", "bayer_purity", "fat_CCl4", "fat_chol"]]


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
    from scipy.stats import spearmanr
    import torch
    cfg = config
    t0 = time.time()
    load_module(project_root / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    ex23 = load_module(project_root / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex26 = load_module(project_root / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(project_root / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    cfg20 = yaml.safe_load(open(project_root / "experiments/0020_20261024_batch_robust_methods/config.yml"))

    # 1. TG-GATEsの土台（0026と同じ: Bayerの230枚で combo を学ぶ。決定的なので0026と同じ写像になる）
    src = project_root / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    Ka, laba = c["K"], c["lab"].astype(np.int64)
    keep = [k for k in np.unique(laba) if k >= 0 and (laba == k).sum() >= cfg20["min_label_patches"]]
    lab_c = np.where(np.isin(laba, keep), np.searchsorted(keep, np.maximum(laba, 0)), -1)
    tf = ex26.fit_combo(Ka, lab_c, cfg, cfg["seed"])
    L = len(keep) - 1
    del Ka
    pre = pickle.loads((project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl").read_bytes())["pre"]
    logger.info(f"TG-GATEs combo refitted; {time.time() - t0:.0f}s")

    # 2. 外部のslide: slide内の順位 → PCA256 → 各表現の元
    ext = project_root / "outputs" / cfg["exp0022"]
    sl = pd.read_csv(ext / "default/slides.csv", dtype=str)
    sl["is_ctrl"] = sl.control_status.isin(cfg["control_status"])
    parts = []
    for r in sl.itertuples():
        with h5py.File(ext / "features_uni_v2_macenko" / f"{r.slide_id.replace(' ', '_')}.h5", "r") as h:
            f = h["features"][:].astype(np.float32)
            xy = h["coords"][:]
            s0 = int(h["coords"].attrs["patch_size_level0"])
        K = pre.transform(ex23.slide_rank(f)).astype(np.float32)
        parts.append({"U": K[:, :64], "C": tf(K), "xy": xy, "size0": s0})
    row = np.concatenate([np.full(len(p_["U"]), i) for i, p_ in enumerate(parts)])
    U = np.concatenate([p_["U"] for p_ in parts])
    Cmb = np.concatenate([p_["C"] for p_ in parts])
    study = sl.study.to_numpy()[row]
    logger.info(f"external: {len(sl)} slides, {len(U):,} patches; controls {int(sl.is_ctrl.sum())} ({sl[sl.is_ctrl].groupby('study').size().to_dict()}); {time.time() - t0:.0f}s")

    # 3. 表現とプロトタイプへの割り当て
    run26 = project_root / "outputs" / cfg["exp0026"]
    reps = {}
    for name, (run, rep, k) in cfg["prototypes"].items():
        cen = np.load(run26 / run / f"centroids_{rep}_k{k}.npy")
        if rep == "combo":
            X = Cmb
        else:
            sc = json.loads((run26 / run / f"scale_{rep}.json").read_text())
            base = U
            if rep == "sphering_lda":
                # 外部の各試験の対照群のpatchで白色化（外部で学習するのはこの平均と共分散だけ）
                ctrl_mask = sl.is_ctrl.to_numpy()[row]
                base = ex27.sphering(U, study, U[ctrl_mask], study[ctrl_mask])
            X = np.concatenate([base, sc["lda_beta"] * sc["lda_scale"] * Cmb[:, :L]], axis=1).astype(np.float32)
        Xt, ct = torch.from_numpy(X), torch.from_numpy(cen.astype(np.float32))
        cl = torch.cat([torch.cdist(x, ct).argmin(1) for x in Xt.split(65536)]).numpy()
        reps[name] = (cl, k, proto_table(run26, run, rep, k))
        logger.info(f"{name}: assigned to {k} TG-GATEs prototypes; {time.time() - t0:.0f}s")

    # 4. 外部の所見との対応（試験ごとに、対照群の平均との差 × 程度の順位相関）
    fr = pd.read_csv(Path(cfg["bench_root"]) / cfg["external_findings"], encoding="utf-8-sig", dtype=str)
    fr["grade"] = pd.to_numeric(fr.grade_ordinal_within_source, errors="coerce").fillna(0)
    rows = []
    for name, (cl, k, tab) in reps.items():
        sh = np.zeros((len(sl), k))
        np.add.at(sh, (row, cl), 1)
        sh /= sh.sum(1, keepdims=True)
        pd.DataFrame(sh, index=sl.slide_id).to_csv(run_dir / f"shares_{name}.csv")
        for st, idx in sl.groupby("study").indices.items():
            cidx = idx[sl.is_ctrl.to_numpy()[idx]]
            if len(cidx) < 2:
                continue
            dlt = sh[idx] - sh[cidx].mean(0)
            f_ = fr[fr.study == st]
            for fn, g in f_.groupby("finding_raw"):
                gm = g.groupby("label_unit").grade.max()
                grade = sl.label_unit.iloc[idx].map(gm).fillna(0).to_numpy()
                if (grade > 0).sum() < cfg["min_pos"] or len(np.unique(grade)) < 2:
                    continue
                rho = np.array([spearmanr(grade, dlt[:, j])[0] if dlt[:, j].std() > 0 else np.nan for j in range(k)])
                j = int(np.nanargmax(rho))
                rows.append({"rep": name, "study": st, "finding": fn, "n_slides": len(idx), "n_pos": int((grade > 0).sum()), "best_rho": float(rho[j]),
                             "prototype": j, "tg_bayer_label": tab.bayer_top_label.get(j, ""), "tg_fat_CCl4": tab.fat_CCl4.get(j, np.nan), "tg_fat_chol": tab.fat_chol.get(j, np.nan)})
    res = pd.DataFrame(rows)
    res.to_csv(run_dir / "external_findings.csv", index=False)
    piv = res.pivot_table(index=["study", "finding"], columns="rep", values="best_rho")
    logger.info("\n" + piv.round(2).to_string())
    logger.info("mean best rho by rep: " + res.groupby("rep").best_rho.mean().round(3).to_dict().__repr__())

    # 5. 壊死: TG-GATEsで壊死のプロトタイプ（Bayerの最多ラベルが壊死）の割合 × 外部の中心静脈側の壊死の程度
    ne = []
    for name, (cl, k, tab) in reps.items():
        nec = [j for j in tab.index if tab.bayer_top_label.get(j, "") == "necrosis"]
        share = np.bincount(row, weights=np.isin(cl, nec).astype(float), minlength=len(sl)) / np.bincount(row, minlength=len(sl))
        g = []
        for r in sl.itertuples():
            f_ = fr[(fr.study == r.study) & (fr.label_unit == str(r.label_unit)) & (fr.grade > 0)]
            g.append(ex23.max_grade(f_, "ecrosis"))
        g = np.array(g)
        ne.append({"rep": name, "n_necrosis_prototypes": len(nec), "spearman_share_vs_necrosis_grade": float(spearmanr(g, share)[0]),
                   "median_share_ctrl": float(np.median(share[sl.is_ctrl])), "median_share_necrosis": float(np.median(share[g > 0]))})
    pd.DataFrame(ne).to_csv(run_dir / "external_necrosis.csv", index=False)
    logger.info("necrosis: " + pd.DataFrame(ne).round(3).to_string(index=False))

    # 6. 図: 所見ごとの最良の順位相関を表現ごとに
    out = run_dir / "figures"
    out.mkdir(exist_ok=True)
    piv2 = piv.dropna(how="all")
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(piv2) + 1.5))
    mk = {"sphering_lda": ("D", "#0072B2"), "unsup_lda1": ("s", "#D55E00"), "combo": ("o", "#999999")}
    for name in reps:
        if name in piv2:
            ax.scatter(piv2[name], range(len(piv2)), marker=mk.get(name, ("o", "k"))[0], color=mk.get(name, ("o", "k"))[1], label=name, s=26, zorder=3)
    ax.set_yticks(range(len(piv2)), [f"{a} | {b[:40]}" for a, b in piv2.index], fontsize=6.5)
    ax.invert_yaxis()
    ax.axvline(0, color="0.7", lw=0.7)
    ax.set_xlabel("best Spearman over TG-GATEs prototypes (grade vs share minus study-control mean)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("X1. External findings (Eisai, Shionogi) vs TG-GATEs prototypes, no external training", fontsize=9)
    fig.savefig(out / "X1_external_findings.png")
    plt.close(fig)

    results: dict = {"n_slides": len(sl), "mean_best_rho": res.groupby("rep").best_rho.mean().to_dict(), "necrosis": ne}

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()