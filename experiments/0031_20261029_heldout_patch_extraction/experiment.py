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


def name_prototypes(cl, lab, nlab, k, nm):
    """学習側のラベル付きpatchで、各プロトタイプに濃縮されたラベルの番号を付ける（0029と同じ規則）。名前なしは-1。"""
    m = lab >= 0
    cnt = np.zeros((k, nlab))
    np.add.at(cnt, (cl[m], lab[m]), 1)
    share = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
    lift = share / np.maximum(cnt.sum(0) / cnt.sum(), 1e-12)
    ok = (cnt >= nm["min_n"]) & ((share >= nm["min_share"]) | (lift >= nm["min_lift"]))
    return np.array([int(np.where(ok[j])[0][np.argmax(lift[j, ok[j]])]) if ok[j].any() else -1 for j in range(k)])


def label_metrics(y, p, nlab):
    """ラベル付きpatchの正解 y と予測 p（-1は名前なし）から、ラベルごとの再現率・適合率・F1。"""
    out = np.full((nlab, 3), np.nan)
    for l in range(nlab):
        tp = np.sum((y == l) & (p == l))
        n_true, n_pred = np.sum(y == l), np.sum(p == l)
        rec = tp / n_true if n_true else np.nan
        prec = tp / n_pred if n_pred else (0.0 if n_true else np.nan)
        f1 = 2 * rec * prec / (rec + prec) if n_true and (rec + prec) > 0 else (0.0 if n_true else np.nan)
        out[l] = rec, prec, f1
    return out


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
    from sklearn.model_selection import GroupKFold
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cfg = {**config, **config.get("variants", {}).get(args.variant, {})}
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_module(project_root / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    load_module(project_root / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex25 = load_module(project_root / "experiments" / cfg["exp0025"] / "experiment.py", "exp0025")
    ex26 = load_module(project_root / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(project_root / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    cfg20 = yaml.safe_load(open(project_root / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    prio = cfg20["label_priority"]

    # 1. Bayerの230枚（0025 cache）。脂肪滴系の21枚のラベルは使わない
    src = project_root / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    K, laba, row = c["K"], c["lab"].astype(np.int64), c["row"]
    slides = pd.read_csv(src / "slides.csv", dtype=str)
    keep = [k_ for k_ in np.unique(laba) if k_ >= 0 and (laba == k_).sum() >= cfg20["min_label_patches"]]
    names_l = [prio[k_] for k_ in keep]
    lab = np.where(np.isin(laba, keep), np.searchsorted(keep, np.maximum(laba, 0)), -1)
    ex_ids = pd.read_csv(project_root / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab = np.where(np.isin(slides.slide_id.to_numpy()[row], ex_ids), -1, lab)
    nlab = len(keep)
    exp_p = slides.EXP_ID.fillna("-1").astype(int).to_numpy()[row]
    comp_s = slides.COMPOUND_NAME.fillna("unknown_" + slides.slide_id).to_numpy()
    comp_p = comp_s[row]
    U = K[:, :64].astype(np.float32)
    logger.info(f"Bayer: {len(slides)} slides, {len(K):,} patches, labeled {(lab >= 0).sum():,} in {nlab} labels; compounds {len(set(comp_s))}; {time.time() - t0:.0f}s")

    # 2. 白色化（B・A）: 各試験のControl slide。Controlのない試験は全試験のControlでまとめて
    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    q = q[(q.DOSE_LEVEL == "Control") & q.EXP_ID.notna() & q.EXP_ID.astype(float).astype(int).isin(set(exp_p))].copy()
    def h5_of(sid):
        for d_ in cfg["ctrl_features_dirs"]:
            if (project_root / d_ / f"{sid}.h5").exists():
                return str(project_root / d_ / f"{sid}.h5")
        return ""
    q["h5"] = [h5_of(x) for x in q.slide_id]
    q = q[q.h5 != ""]
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=ex27._init_ctrl, initargs=(str(project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl"), cfg["ctrl_patches_per_slide"])) as ex:
        parts_c = list(ex.map(ex27._load_ctrl, q.h5))
    Kc = np.concatenate(parts_c)[:, :64]
    exp_c = np.concatenate([np.full(len(p_), int(float(e))) for p_, e in zip(parts_c, q.EXP_ID)])
    e_fix = np.where(np.isin(exp_p, exp_c), exp_p, -999)
    Sw = ex27.sphering(U, e_fix, np.r_[Kc, Kc], np.r_[exp_c, np.full(len(exp_c), -999)])
    del Kc, parts_c
    logger.info(f"Control for sphering: {len(q)} slides; {time.time() - t0:.0f}s")

    # 3. 化合物単位の5分割。分割ごとに学習側だけで LDA → k-means（コンセンサス）→ 名前付け、評価側を割り当て
    pred = {r_: np.full(len(K), -1, dtype=np.int64) for r_ in cfg["reps"]}
    fold_of = np.full(len(K), -1)
    gkf = GroupKFold(n_splits=cfg["folds"])
    for fi, (tr_s, te_s) in enumerate(gkf.split(np.zeros(len(slides)), groups=comp_s)):
        tr, te = np.isin(row, tr_s), np.isin(row, te_s)
        fold_of[te] = fi
        tf = ex26.fit_combo(K[tr], lab[tr], cfg, cfg["seed"] + fi)
        L_f = len(np.unique(lab[tr][lab[tr] >= 0])) - 1   # 学習側にないラベルがあればLDAの次元は減る
        Lall = np.concatenate([tf(b_)[:, :L_f] for b_ in np.array_split(K, 40)]).astype(np.float32)
        fac = float(Sw[tr].std(0).mean() / Lall[tr].std(0).mean())
        reps = {"D_unsup": U, "D13_unsup13": U[:, :L_f], "C_lda": Lall, "B_sphering": Sw, "A_sphering_lda": np.concatenate([Sw, cfg["lda_beta"] * fac * Lall], 1)}
        for r_ in cfg["reps"]:
            X = torch.from_numpy(reps[r_]).to(device)
            _, cen = ex25.consensus_kmeans(X[torch.from_numpy(np.where(tr)[0]).to(device)], cfg["k"], cfg["consensus_runs"], cfg["fit_points"], cfg["seed"] + fi, device)
            cl = torch.cat([torch.cdist(x, cen).argmin(1) for x in X.split(65536)]).cpu().numpy()
            nm = name_prototypes(cl[tr], lab[tr], nlab, cfg["k"], cfg["naming"])
            pred[r_][te] = nm[cl[te]]
            del X
        logger.info(f"fold {fi}: train {len(tr_s)} slides / test {len(te_s)} slides ({len(set(comp_s[te_s]))} compounds); {time.time() - t0:.0f}s")

    # 4. 評価（評価側のラベル付きpatch）。化合物を復元抽出した区間
    ev = lab >= 0
    y, cp = lab[ev], comp_p[ev]
    rows = []
    M = {r_: label_metrics(y, pred[r_][ev], nlab) for r_ in cfg["reps"]}
    for r_ in cfg["reps"]:
        for l in range(nlab):
            rows.append({"rep": r_, "label": names_l[l], "n_test": int((y == l).sum()), "recall": M[r_][l, 0], "precision": M[r_][l, 1], "F1": M[r_][l, 2]})
    res = pd.DataFrame(rows)
    comps = np.unique(cp)
    idx_by_c = {c_: np.where(cp == c_)[0] for c_ in comps}
    rng = np.random.default_rng(cfg["seed"])
    B = {r_: [] for r_ in cfg["reps"]}
    for _ in range(cfg["bootstrap"]):
        ii = np.concatenate([idx_by_c[c_] for c_ in rng.choice(comps, len(comps))])
        for r_ in cfg["reps"]:
            B[r_].append(label_metrics(y[ii], pred[r_][ev][ii], nlab)[:, 2])
    B = {r_: np.array(v) for r_, v in B.items()}
    lo = {r_: np.nanpercentile(B[r_], 2.5, axis=0) for r_ in B}
    hi = {r_: np.nanpercentile(B[r_], 97.5, axis=0) for r_ in B}
    res["F1_lo"] = [lo[r_][names_l.index(l)] for r_, l in zip(res.rep, res.label)]
    res["F1_hi"] = [hi[r_][names_l.index(l)] for r_, l in zip(res.rep, res.label)]
    res.to_csv(run_dir / "heldout_label_metrics.csv", index=False)
    nonpar = [i for i, n in enumerate(names_l) if n != "parenchyma"]
    diffs = []
    for a, b in [(a_, b_) for a_, b_ in [("C_lda", "D_unsup"), ("C_lda", "D13_unsup13"), ("D13_unsup13", "D_unsup"), ("A_sphering_lda", "B_sphering"), ("A_sphering_lda", "D_unsup"), ("B_sphering", "D_unsup")] if a_ in B and b_ in B]:
        d_macro = np.nanmean(B[a][:, nonpar], 1) - np.nanmean(B[b][:, nonpar], 1)
        point = np.nanmean(M[a][nonpar, 2]) - np.nanmean(M[b][nonpar, 2])
        diffs.append({"comparison": f"{a} - {b}", "label": "macro F1 (all but parenchyma)", "diff": point, "lo": np.nanpercentile(d_macro, 2.5), "hi": np.nanpercentile(d_macro, 97.5), "p_le0": float(np.mean(d_macro <= 0))})
        for l in nonpar:
            dl = B[a][:, l] - B[b][:, l]
            diffs.append({"comparison": f"{a} - {b}", "label": names_l[l], "diff": M[a][l, 2] - M[b][l, 2], "lo": np.nanpercentile(dl, 2.5), "hi": np.nanpercentile(dl, 97.5), "p_le0": float(np.nanmean(dl <= 0))})
    diffs = pd.DataFrame(diffs)
    diffs.to_csv(run_dir / "heldout_differences.csv", index=False)
    pd.set_option("display.width", 250)
    logger.info("F1 (held-out compounds):\n" + res.pivot_table(index="label", columns="rep", values="F1").round(3)[cfg["reps"]].to_string())
    logger.info("recall:\n" + res.pivot_table(index="label", columns="rep", values="recall").round(3)[cfg["reps"]].to_string())
    logger.info("precision:\n" + res.pivot_table(index="label", columns="rep", values="precision").round(3)[cfg["reps"]].to_string())
    logger.info("differences:\n" + diffs[diffs.label.str.startswith("macro") | (diffs.comparison == "C_lda - D_unsup")].round(3).to_string(index=False))
    np.savez_compressed(run_dir / "predictions.npz", **{r_: pred[r_] for r_ in cfg["reps"]}, fold=fold_of, lab=lab)

    # 5. 図: ラベルごとのF1（95%区間）
    fig, ax = plt.subplots(figsize=(9, 0.42 * nlab + 1.6))
    mk = {"D_unsup": ("o", "#999999"), "D13_unsup13": ("v", "#555555"), "C_lda": ("s", "#009E73"), "B_sphering": ("^", "#0072B2"), "A_sphering_lda": ("D", "#D55E00")}
    order = [names_l[i] for i in np.argsort([-(y == i).sum() for i in range(nlab)])]
    for k_, r_ in enumerate(cfg["reps"]):
        t_ = res[res.rep == r_].set_index("label").reindex(order)
        yy = np.arange(len(order)) + (k_ - (len(cfg["reps"]) - 1) / 2) * 0.17
        ax.errorbar(t_.F1, yy, xerr=[t_.F1 - t_.F1_lo, t_.F1_hi - t_.F1], fmt=mk[r_][0], color=mk[r_][1], ms=5, lw=1, capsize=0, label=r_)
    ax.set_yticks(range(len(order)), [f"{l} (n={int((y == names_l.index(l)).sum()):,})" for l in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1 on held-out compounds (Bayer polygons as truth); 95% CI, compounds resampled")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Patch-level extraction on compounds unseen in training (prototypes k=100, named from training labels only)", fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "P1_heldout_f1.png", dpi=150)
    plt.close(fig)

    results: dict = {"macro_F1_nonparenchyma": {r_: float(np.nanmean(M[r_][nonpar, 2])) for r_ in cfg["reps"]},
                     "differences_macro": diffs[diffs.label.str.startswith("macro")].to_dict("records")}

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()