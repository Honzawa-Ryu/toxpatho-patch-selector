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
    import time
    from concurrent.futures import ProcessPoolExecutor

    import torch
    from sklearn.metrics import roc_auc_score
    cfg = config
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for n_, e_ in [("exp0021", cfg["exp0021"]), ("exp0023", "0023_20261025_layer_reality_check")]:
        load_module(project_root / "experiments" / e_ / "experiment.py", n_)
    ex25, ex26, ex27, ex30, ex35 = [load_module(project_root / "experiments" / cfg[k_] / "experiment.py", k_)
                                    for k_ in ("exp0025", "exp0026", "exp0027", "exp0030", "exp0035")]
    cfg20 = yaml.safe_load(open(project_root / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    pre_path = str(project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl")
    findings = cfg["findings"]

    def h5_of(sid):
        for d_ in cfg["features_dirs"]:
            if (project_root / d_ / f"{sid}.h5").exists():
                return str(project_root / d_ / f"{sid}.h5")
        return ""

    # 1. まとまり 3・26 の全 slide（対照群＋投与群）。各 2,048 patch、slide 内の順位 → 0021 の PCA256（0030 と同じ間引き）
    E = pd.read_csv(project_root / cfg["groups_csv"])
    E = E[E.group_B128.isin(cfg["groups"])]
    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    q = q[q.EXP_ID.notna() & (q.EXP_ID != "")]
    S = q[q.EXP_ID.astype(int).isin(E.exp_id)][["slide_id", "EXP_ID", "GROUP_ID", "INDIVIDUAL_ID", "COMPOUND_NAME", "DOSE_LEVEL", "SACRIFICE_PERIOD"]].copy()
    S["h5"] = [h5_of(s_) for s_ in S.slide_id]
    logger.info(f"slides in groups {cfg['groups']}: {len(S)}, without features: {(S.h5 == '').sum()} (dropped)")
    S = S[S.h5 != ""].reset_index(drop=True)
    S["exp"] = S.EXP_ID.astype(int)
    S["grp"] = S.exp.map(E.set_index("exp_id").group_B128)
    S["is_ctrl"] = S.DOSE_LEVEL.eq("Control")
    with ProcessPoolExecutor(max_workers=cfg["workers"], initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts = list(ex.map(ex30._load_sub, S.h5, chunksize=4))
    K_all = np.concatenate([p_[0] for p_ in parts]).astype(np.float16)
    slide_of = np.concatenate([np.full(len(p_[0]), i) for i, p_ in enumerate(parts)])
    del parts
    N = len(slide_of)

    # 2. 記録（slide × 所見、程度の最大）。0035 と同じ（肝臓、SP_FLG を除く）
    pth = pd.read_csv(project_root / cfg["pathology_csv"], dtype=str)
    pth = pth[(pth.ORGAN == "Liver") & (pth.SP_FLG.str.lower() != "true") & pth.FINDING_TYPE.isin(findings)].copy()
    key = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"]
    for k_ in key:
        S[k_] = pd.to_numeric(S[k_], errors="coerce")
        pth[k_] = pd.to_numeric(pth[k_], errors="coerce")
    pth["g"] = pth.GRADE_TYPE.map(cfg["grade_map"]).fillna(1)
    G = S[["slide_id"] + key].merge(pth, on=key).pivot_table(index="slide_id", columns="FINDING_TYPE", values="g", aggfunc="max").reindex(S.slide_id).fillna(0).reindex(columns=findings, fill_value=0)
    pos = G.to_numpy() > 0
    S.drop(columns="h5").assign(**{f"grade__{f}": G[f].to_numpy() for f in findings}).to_csv(run_dir / "slides.csv", index=False)
    logger.info(f"{len(S)} slides ({S.is_ctrl.sum()} controls) in {S.exp.nunique()} experiments, {N:,} patches; "
                f"positive slides: {dict(zip(findings, pos.sum(0).tolist()))}; {time.time() - t0:.0f}s")

    # 3. LDA: Bayer の 230 枚のラベル（0035 と同じ扱い。脂肪滴系の 21 枚のラベルは使わない）で学び、全 patch を写す
    src = project_root / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    Kb, laba, row_b = c["K"], c["lab"].astype(np.int64), c["row"]
    bayer = pd.read_csv(src / "slides.csv", dtype=str)
    keep = [k_ for k_ in np.unique(laba) if k_ >= 0 and (laba == k_).sum() >= cfg20["min_label_patches"]]
    lab = np.where(np.isin(laba, keep), np.searchsorted(keep, np.maximum(laba, 0)), -1)
    ex_ids = pd.read_csv(project_root / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab = np.where(np.isin(bayer.slide_id.to_numpy()[row_b], ex_ids), -1, lab)
    tf = ex26.fit_combo(Kb, lab, cfg, cfg["seed"])
    L = len(keep) - 1
    Lw = np.concatenate([tf(b_)[:, :L] for b_ in np.array_split(K_all, max(1, N // 200000))]).astype(np.float32)
    del Kb, c
    logger.info(f"LDA: {L} dims from {(lab >= 0).sum():,} labeled Bayer patches; {time.time() - t0:.0f}s")

    # 4. まとまりごと: 白色化（試験ごとに、その試験の対照群の patch で）→ 表現 → 合議 k-means（k ごと）→ 評価
    fl, lo, ev = cfg["flat"], cfg["loso"], cfg["eval"]
    exp_s, ctrl_s = S.exp.to_numpy(), S.is_ctrl.to_numpy()
    rec_rows, loso_rows = [], []
    for g in cfg["groups"]:
        sg = np.where(S.grp.to_numpy() == g)[0]
        m = np.isin(slide_of, sg)
        so = np.searchsorted(sg, slide_of[m])                 # まとまりの中の slide 番号
        ns = len(sg)
        exp_p = exp_s[slide_of[m]]
        U = K_all[m, :cfg["sphere_dim"]].astype(np.float32)
        cm = ctrl_s[slide_of[m]]
        assert set(np.unique(exp_p)) <= set(np.unique(exp_p[cm])), "every experiment needs controls for sphering"
        Sw = ex27.sphering(U, exp_p, U[cm], exp_p[cm])
        del U
        posg, expg, ctrlg = pos[sg], exp_s[sg], ctrl_s[sg]
        for rep, use_lda in cfg["reps"].items():
            X = Sw
            if use_lda:
                fac = float(Sw.std(0).mean() / Lw[m].std(0).mean())
                X = np.concatenate([Sw, cfg["lda_beta"] * fac * Lw[m]], 1).astype(np.float32)
            for k in fl["k_list"]:
                r_ = np.random.default_rng([cfg["seed"], g, k])
                pool = torch.from_numpy(X[r_.choice(len(X), min(len(X), 2 * fl["fit_points"]), replace=False)]).to(device)
                _, cen = ex25.consensus_kmeans(pool, k, fl["consensus_runs"][k], fl["fit_points"], cfg["seed"], device)
                del pool
                cl = ex35.assign(cen, X, device)
                tag = f"g{g}_{rep}_k{k}"
                np.save(run_dir / f"assign_{tag}.npy", cl.astype(np.int16))
                np.save(run_dir / f"centroids_{tag}.npy", cen.cpu().numpy().astype(np.float32))
                cs = ex35.slide_counts(cl, k, so, ns)                     # k × slides
                # (c) 記録の純度（0035 と同じ定義）
                share, base, lift, njf, n = ex35.record_shares(cs, posg)
                big = n >= ev["min_proto_patches"]
                for i, f in enumerate(findings):
                    if not posg[:, i].any():
                        continue
                    j = int(np.where(big, share[:, i], -1).argmax())
                    hi = big & (share[:, i] >= ev["precision_thr"])
                    rec_rows.append({"group": g, "rep": rep, "k": k, "finding": f, "n_pos_slides": int(posg[:, i].sum()), "base_rate": float(base[i]),
                                     "best_share": float(share[j, i]), "best_lift": float(lift[j, i]), "best_n": int(n[j]),
                                     "n_protos_share_ge_thr": int(hi.sum()), "coverage": float(njf[hi, i].sum() / max(njf[:, i].sum(), 1))})
                # (a) 試験を 1 つ伏せる（0035 scripts/group_crossstudy_findings.py と同じ選び方）
                P = (cs / np.maximum(cs.sum(0, keepdims=True), 1)).T              # slides × k の割合
                lr = lambda a, b: np.log2((P[a].mean(0) + 1e-4) / (P[b].mean(0) + 1e-4))
                for i, f in enumerate(findings):
                    pf = posg[:, i]
                    neg = ~ctrlg & ~pf
                    fx = [e for e in np.unique(expg[pf]) if ((expg == e) & pf).sum() >= lo["min_n"] and ((expg == e) & neg).sum() >= lo["min_n"]]
                    if len(fx) < 2:
                        continue
                    non_f = ~ctrlg & ~np.isin(expg, fx) & ~pf
                    for e in fx:
                        tr = [x for x in fx if x != e]
                        gc = ctrlg & (expg != e)
                        LRc = np.array([lr((expg == x) & pf, gc) for x in tr])
                        LRn = np.array([lr((expg == x) & pf, (expg == x) & neg) for x in tr])
                        LRg = np.array([lr((expg == x) & neg, gc) for x in tr])
                        me = min(2, len(tr))
                        sel = (((LRc >= lo["lr_up"]) & (LRn >= lo["lr_up"])).sum(0) >= me) & (LRn.mean(0) >= lo["lr_up"]) & ((LRg >= lo["lr_up"]).sum(0) < me)
                        score = P[:, sel].sum(1)
                        tgt = (expg == e) & pf
                        refs = {"own_neg": (expg == e) & neg, "own_ctrl": (expg == e) & ctrlg, "group_ctrl": gc, "group_nonfinding_treated": non_f}
                        row = {"group": g, "rep": rep, "k": k, "finding": f, "exp": int(e), "compound": S.COMPOUND_NAME.to_numpy()[sg][expg == e][0],
                               "n_pos": int(tgt.sum()), "n_train_exps": len(tr), "n_selected": int(sel.sum()), "share_pos_in_selected": float(score[tgt].mean())}
                        for rk, mk in refs.items():
                            ok = mk.sum() >= lo["min_n"] and sel.any()
                            row[f"auc_{rk}"] = float(roc_auc_score(np.r_[np.ones(tgt.sum()), np.zeros(mk.sum())], np.r_[score[tgt], score[mk]])) if ok else np.nan
                            row[f"n_{rk}"] = int(mk.sum())
                        loso_rows.append(row)
                logger.info(f"group {g} {rep} k={k}: done; {time.time() - t0:.0f}s")
            del X
        del Sw
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(run_dir / "record_purity.csv", index=False)
    loso = pd.DataFrame(loso_rows)
    loso.to_csv(run_dir / "loso.csv", index=False)
    summ = loso.groupby(["group", "finding", "rep", "k"]).agg(n_exps=("exp", "size"), n_selected=("n_selected", "mean"),
                                                              **{f"auc_{r}": (f"auc_{r}", "mean") for r in ("own_neg", "own_ctrl", "group_ctrl", "group_nonfinding_treated")}).reset_index()
    summ.to_csv(run_dir / "loso_summary.csv", index=False)
    pd.set_option("display.width", 250)
    logger.info("leave-one-experiment-out (mean over held-out experiments):\n" + summ.round(3).to_string(index=False))
    logger.info("record coverage:\n" + rec.pivot_table(index=["group", "finding"], columns=["rep", "k"], values="coverage").round(2).to_string())

    results: dict = {"n_slides": int(len(S)), "n_patches": int(N), "groups": cfg["groups"], "loso_summary": summ.round(4).to_dict(orient="records")}

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()