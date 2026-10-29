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
    import torch
    from scipy.cluster.hierarchy import fcluster, linkage
    cfg = {**config, **config.get("variants", {}).get(args.variant, {})}
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from lib.clustering import kmeans_lloyd
    load_module(project_root / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    load_module(project_root / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex17 = load_module(project_root / "experiments/0017_20261016_block_consensus_patterns/experiment.py", "exp0017")
    ex26 = load_module(project_root / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex29 = load_module(project_root / "experiments/0029_20261028_external_finding_maps/experiment.py", "exp0029")
    cfg20 = yaml.safe_load(open(project_root / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    prio = cfg20["label_priority"]

    # 1. Bayerの230枚。ラベルは30 patch以上の14種、脂肪滴系の21枚のラベルは除く
    src = project_root / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    K, laba, row, xy = c["K"], c["lab"].astype(np.int64), c["row"], c["xy"]
    slides = pd.read_csv(src / "slides.csv", dtype=str).rename(columns={"EXP_ID": "exp", "COMPOUND_NAME": "compound", "DOSE_LEVEL": "dose", "SACRIFICE_PERIOD": "period"})
    known = {**pd.read_csv(project_root / "outputs" / cfg["exp0021"] / "v3_maps/slides.csv", dtype=str).set_index("slide_id").svs_path.to_dict(),
             **pd.read_csv(project_root / "outputs/target_slide_list/bayer_annotated_all_missing.csv", dtype=str).set_index("slide_id").svs_path.to_dict()}
    slides["svs_path"] = [known.get(s_, p_) for s_, p_ in zip(slides.slide_id, slides.svs_path)]
    size0 = dict(zip(slides.slide_id, c["size0"].tolist()))
    keep = [k_ for k_ in np.unique(laba) if k_ >= 0 and (laba == k_).sum() >= cfg20["min_label_patches"]]
    laba = np.where(np.isin(laba, keep), laba, -1)
    ex_ids = pd.read_csv(project_root / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    laba = np.where(np.isin(slides.slide_id.to_numpy()[row], ex_ids), -1, laba)
    lab_c = np.where(laba >= 0, np.searchsorted(keep, np.maximum(laba, 0)), -1)
    L = len(keep) - 1
    na = len(K)
    logger.info(f"Bayer: {len(slides)} slides, {na:,} patches, labeled {(laba >= 0).sum():,}; {time.time() - t0:.0f}s")

    # 2. 表現: D = PCA256の上位64、C = Bayerのラベルで学んだLDAの13次元
    tf = ex26.fit_combo(K, lab_c, cfg, cfg["seed"])
    reps = {"D_unsup": K[:, :64].astype(np.float32), "D13_unsup13": K[:, :L].astype(np.float32),   # D13: LDAと同じ次元数（13）までPCAで落とした対照（2026-10-29ユーザー）
            "C_lda": np.concatenate([tf(b_)[:, :L] for b_ in np.array_split(K, 40)]).astype(np.float32)}
    del K

    # 3. 表現ごとに: コンセンサスk-means → 全patchの割り当て → Ward法の系統樹 → 名前付け → 見た目
    results = {}
    for rep in cfg["reps"]:
        X = torch.from_numpy(reps[rep]).to(device)
        cents = []
        for m_ in range(cfg["consensus_runs"]):
            rr = np.random.default_rng([cfg["seed"], m_])
            fit = rr.choice(na, size=min(cfg["fit_points"], na), replace=False)
            _, cc, _ = kmeans_lloyd(X[torch.from_numpy(fit).to(device)].cpu().numpy(), cfg["k"], 1000 + m_, device)
            cents.append(cc)
        _, cen, _ = kmeans_lloyd(np.concatenate(cents), cfg["k"], cfg["seed"], device)
        cen_t = torch.from_numpy(cen).to(device)
        cl = torch.cat([torch.cdist(x, cen_t).argmin(1) for x in X.split(65536)]).cpu().numpy()
        dist = torch.cat([torch.cdist(x, cen_t).min(1).values for x in X.split(65536)]).cpu().numpy()
        short = rep.split("_", 1)[1]
        np.save(run_dir / f"centroids_{short}_k{cfg['k']}.npy", cen.astype(np.float32))
        tdir = run_dir / "tree" / f"{short}_k{cfg['k']}"
        (tdir / "img").mkdir(parents=True, exist_ok=True)
        np.save(tdir / "assign_all.npy", cl.astype(np.int16))
        Zl = linkage(cen, method="ward")
        cuts = {cut: fcluster(Zl, cut, criterion="maxclust") for cut in cfg["cuts"]}
        nm = ex29.name_prototypes(cl, laba, prio, cfg["k"], cfg["naming"])
        nm.to_csv(run_dir / f"prototype_names_{rep}.csv", index=False)
        recs = []
        for j in range(cfg["k"]):
            mk = np.where(cl == j)[0]
            lab_j = laba[mk][laba[mk] >= 0]
            lc = pd.Series(lab_j).value_counts()
            recs.append({"prototype": j, "n_patches": len(mk), "n_bayer": len(mk), "n_slides": len(np.unique(row[mk])), "n_experiments": len(np.unique(slides.exp.to_numpy()[row[mk]])),
                         "n_labeled": len(lab_j), "bayer_top_label": prio[lc.index[0]] if len(lc) else "", "bayer_purity": float(lc.iloc[0] / len(lab_j)) if len(lc) else np.nan,
                         "name": nm.name.iloc[j] if isinstance(nm.name.iloc[j], str) else "", **{f"group_{cut}": int(cuts[cut][j]) for cut in cuts}})
            # 見た目: 重心に近い順に、別々のslideから
            picks, seen = [], set()
            for q in mk[np.argsort(dist[mk])]:
                if row[q] not in seen:
                    picks.append(int(q)); seen.add(row[q])
                if len(seen) == cfg["tiles_per_proto"]:
                    break
            if picks:
                tiles = ex17.read_tiles(picks, row, xy, slides, size0, lambda s_, q_: (f"e{s_.exp} {str(s_.compound)[:12]} {str(s_.dose)[0]}", f"{s_.period} {prio[laba[q_]] if laba[q_] >= 0 else ''}"))
                ex17.draw_sheet("", [("", tiles)], tdir / "img" / f"p{j:03d}.jpg")
        tab = pd.DataFrame(recs)
        tab.to_csv(tdir / "prototypes_in_tree_order.csv", index=False)
        results[rep] = {"named": nm.name.fillna("-").value_counts().to_dict()}
        logger.info(f"{rep}: k={cfg['k']}, names {results[rep]['named']}; {time.time() - t0:.0f}s")

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()