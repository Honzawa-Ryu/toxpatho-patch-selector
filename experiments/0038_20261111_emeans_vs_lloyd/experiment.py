import argparse
import json
import logging
import os
import sys
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


def load_module(path, name):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args() -> argparse.Namespace:
    """Define CLI args for all variable dimensions (used in GRID_ARGS / RUN_COMMAND)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")   # run_slurm.shが渡す。読むのは常に実験ディレクトリのconfig.yml
    # These must match GRID_ARGS entries in run_slurm.sh.
    # Example:
    #   parser.add_argument("--model", required=True)
    #   parser.add_argument("--seed",  required=True)
    return parser.parse_args()


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
    variant_key = "default"

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
    import itertools
    import time
    import types
    from concurrent.futures import ProcessPoolExecutor

    import torch
    from sklearn.metrics import adjusted_rand_score
    from lib.clustering import kmeans_lloyd
    cfg = config
    device = torch.device("cuda")
    t0 = time.time()
    for n_, e_ in [("exp0021", cfg["exp0021"]), ("exp0023", "0023_20261025_layer_reality_check")]:
        load_module(project_root / "experiments" / e_ / "experiment.py", n_)
    ex25, ex27, ex30, ex35, ex37 = [load_module(project_root / "experiments" / cfg[k_] / "experiment.py", k_)
                                    for k_ in ("exp0025", "exp0027", "exp0030", "exp0035", "exp0037")]
    # e-means: パッケージの __init__ は faiss を import するので通さず、PyTorch 版だけを読む（faiss は環境にない）
    pkg = types.ModuleType("e_means")
    pkg.__path__ = [cfg["emeans_dir"]]
    sys.modules["e_means"] = pkg
    from e_means.e_means_quantized import QuantizedEmeans
    from e_means.e_means_torch import Emeans
    cfg37 = yaml.safe_load(open(project_root / "experiments" / cfg["exp0037"] / "config.yml"))
    pre_path = str(project_root / "outputs" / cfg37["exp0021"] / "v3_maps/models.pkl")

    def h5_of(sid):
        for d_ in cfg37["features_dirs"]:
            if (project_root / d_ / f"{sid}.h5").exists():
                return str(project_root / d_ / f"{sid}.h5")
        raise FileNotFoundError(sid)

    # 1. 0037 のまとまり 3 の B128（0037 と同じ作り方: slide 内の順位 → PCA256、試験ごとに自分の対照群で白色化 128）
    S = pd.read_csv(project_root / "outputs" / cfg["exp0037"] / "default/slides.csv", dtype={"slide_id": str})
    S = S[S.grp == cfg["group"]].reset_index(drop=True)
    with ProcessPoolExecutor(max_workers=8, initializer=ex27._init_ctrl, initargs=(pre_path, cfg37["patches_per_slide"])) as ex:
        parts = list(ex.map(ex30._load_sub, [h5_of(s_) for s_ in S.slide_id], chunksize=4))
    U = np.concatenate([p_[0][:, :cfg37["sphere_dim"]] for p_ in parts]).astype(np.float32)
    so = np.concatenate([np.full(len(p_[0]), i) for i, p_ in enumerate(parts)])
    del parts
    exp_p, cm = S.exp.to_numpy()[so], S.is_ctrl.to_numpy()[so]
    X = ex27.sphering(U, exp_p, U[cm], exp_p[cm])
    del U
    N, D = X.shape
    Xg = torch.from_numpy(X).to(device)
    findings = cfg37["findings"]
    pos = S[[f"grade__{f}" for f in findings]].to_numpy() > 0
    logger.info(f"group {cfg['group']}: {len(S)} slides, {N:,} patches, d={D}; GPU {torch.cuda.get_device_name(0)}; {time.time() - t0:.0f}s")

    def timed(fn):
        torch.cuda.synchronize()
        t_ = time.time()
        out = fn()
        torch.cuda.synchronize()
        return out, time.time() - t_

    def fit(method, k, s):
        """重心（torch、GPU）と、学習にかかった秒数（最後の全 patch の割り当ては含めない）。"""
        mc = cfg["methods"][method]
        if method == "lloyd_sub":
            sub = np.random.default_rng([cfg["seed"], s]).choice(N, cfg["fit_points"], replace=False)
            (_, cen, _), sec = timed(lambda: kmeans_lloyd(X[sub], k, 1000 + s, device))
            return torch.from_numpy(cen).to(device), sec
        if method == "consensus":
            r_ = np.random.default_rng([cfg["seed"], s, k])
            pool = Xg[torch.from_numpy(r_.choice(N, min(N, 2 * cfg["fit_points"]), replace=False)).to(device)]
            (_, cen), sec = timed(lambda: ex25.consensus_kmeans(pool, k, cfg["consensus_runs"][k], cfg["fit_points"], s, device))
            return cen, sec
        if method == "lloyd_full":
            (_, cen, _), sec = timed(lambda: kmeans_lloyd(X, k, 1000 + s, device))
            return torch.from_numpy(cen).to(device), sec
        em = Emeans(D, k, nepochs=mc["nepochs"], seed=s) if method == "emeans" else QuantizedEmeans(D, k, quantizer=None, nepochs=mc["nepochs"], seed=s)
        (cen, _), sec = timed(lambda: em.train(Xg))
        fit.recipe[(method, k)] = {kk: (vv if isinstance(vv, (int, float, str, bool)) or vv is None else str(vv)) for kk, vv in em.recipe.items()}
        return cen.float(), sec
    fit.recipe = {}

    rows, labels = [], {}
    for k in cfg["k_list"]:
        for method in cfg["methods"]:
            for s in cfg["seeds"]:
                try:
                    cen, sec = fit(method, k, s)
                except Exception as e:   # 例: Triton がこの GPU で動かない。記録して次へ
                    logger.warning(f"{method} k={k} seed={s} failed: {type(e).__name__}: {e}")
                    rows.append({"method": method, "k": k, "seed": s, "failed": f"{type(e).__name__}: {e}"})
                    continue
                cl, dist = ex35.assign(cen, X, device, with_dist=True)
                labels[(method, k, s)] = cl
                cs = ex35.slide_counts(cl, k, so, len(S))
                lo = pd.DataFrame(ex37.loso_auc(cs, pos, S.exp.to_numpy(), S.is_ctrl.to_numpy(), S.COMPOUND_NAME.to_numpy(), findings, cfg37["loso"], {}))
                rows.append({"method": method, "k": k, "seed": s, "fit_sec": sec, "mse": float((dist.astype(np.float64) ** 2).mean()),
                             "n_empty": int(k - len(np.unique(cl))), "max_cluster_share": float(np.bincount(cl, minlength=k).max() / N),
                             "loso_auc_mean": float(lo.groupby("finding").auc_own_neg.mean().mean()),
                             **{f"loso__{f}": v for f, v in lo.groupby("finding").auc_own_neg.mean().items()}})
                logger.info(f"{method} k={k} seed={s}: {sec:.1f}s, mse {rows[-1]['mse']:.4f}, LOSO AUC {rows[-1]['loso_auc_mean']:.3f}; {time.time() - t0:.0f}s")
    runs = pd.DataFrame(rows)
    runs.to_csv(run_dir / "runs.csv", index=False)
    # seed 間の一致度（全 patch の割り当ての ARI、seed の全ての組）
    ari = []
    for (method, k), grp in itertools.groupby(sorted(labels), key=lambda t: t[:2]):
        ss = [t[2] for t in grp]
        for a, b in itertools.combinations(ss, 2):
            ari.append({"method": method, "k": k, "seed_a": a, "seed_b": b, "ari": float(adjusted_rand_score(labels[(method, k, a)], labels[(method, k, b)]))})
    ari = pd.DataFrame(ari)
    ari.to_csv(run_dir / "seed_ari.csv", index=False)
    summ = runs.groupby(["k", "method"]).agg(fit_sec=("fit_sec", "median"),   # 中央値: 最初の seed は Triton のコンパイルを含む
                                             fit_sec_max=("fit_sec", "max"), mse=("mse", "mean"), mse_sd=("mse", "std"), loso_auc=("loso_auc_mean", "mean"),
                                             loso_auc_sd=("loso_auc_mean", "std"), n_empty=("n_empty", "mean")).join(ari.groupby(["k", "method"]).ari.agg(ari_mean="mean", ari_min="min")).reset_index()
    summ.to_csv(run_dir / "summary.csv", index=False)
    pd.set_option("display.width", 220)
    logger.info("summary:\n" + summ.round(4).to_string(index=False))
    logger.info(f"e-means recipes: {fit.recipe}")

    results: dict = {"n_patches": int(N), "d": int(D), "gpu": torch.cuda.get_device_name(0), "summary": summ.round(5).to_dict(orient="records"),
                     "emeans_recipe": {f"{m}_k{k}": v for (m, k), v in fit.recipe.items()}}

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()