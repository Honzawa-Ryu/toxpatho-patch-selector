"""0037 の比較対照: まとまりで作り直さない 0035 の全体のプロトタイプ（flat k = 100・200・500・1000、B128 = v2、A128 = v1）を、
0037 と同じ slide・同じ評価（試験を 1 つ伏せる AUC、記録の純度）にかける（2026-11-11 ユーザー: 次の候補 1）。

0035 の重心は保存されていないので、0035 の割当と同じ並びで表現を作り直し、プロトタイプごとの平均を取る（scripts/control_similarity_groups.py と同じやり方）。
A128 の LDA の重みは 0035 の全 patch で計算した値（0036 v5 の a128_scale.json）。0037 の slide は、0037 と同じく試験ごとに自分の対照群で白色化してから割り当てる。
出力: outputs/0037_20261111_group_prototypes_with_treated/default/baseline_0035/{loso.csv, record_purity.csv}
usage: python scripts/group_prototypes_global_baseline.py   （コンテナ内、Slurm経由。CPU 12 コア・48 GB、1 時間程度）
"""
import importlib.util
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
R35 = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin"
R37 = ROOT / "outputs/0037_20261111_group_prototypes_with_treated/default"
OUT = R37 / "baseline_0035"
REPS = {"B128_sphering": ("v2", "B128_sphering"), "A128_sphering_lda": ("v1", "A128_sphering_lda")}   # 0037 の名前: (0035 の run, 0035 の表現)
KS = [100, 200, 500, 1000]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / "experiments/0035_20261102_finer_prototypes_split_rejoin/config.yml"))
    cfg37 = yaml.safe_load(open(ROOT / "experiments/0037_20261111_group_prototypes_with_treated/config.yml"))
    load_module(ROOT / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    load_module(ROOT / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex26 = load_module(ROOT / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(ROOT / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    ex30 = load_module(ROOT / "experiments" / cfg["exp0030"] / "experiment.py", "exp0030")
    ex35 = load_module(ROOT / "experiments/0035_20261102_finer_prototypes_split_rejoin/experiment.py", "exp0035")
    ex37 = load_module(ROOT / "experiments/0037_20261111_group_prototypes_with_treated/experiment.py", "exp0037")
    cfg20 = yaml.safe_load(open(ROOT / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    pre_path = str(ROOT / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl")
    fac = json.loads((ROOT / "outputs/0036_20261103_local_prototype_context/v5/a128_scale.json").read_text())["fac"]
    D = cfg37["sphere_dim"]

    def h5_of(sid):
        for d_ in cfg["features_dirs"]:
            if (ROOT / d_ / f"{sid}.h5").exists():
                return str(ROOT / d_ / f"{sid}.h5")
        return ""

    # 0. Bayer の LDA（0035・0037 と同じ）
    src = ROOT / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    Kb, laba, row_b = c["K"], c["lab"].astype(np.int64), c["row"]
    bayer = pd.read_csv(src / "slides.csv", dtype=str)
    keep = [k_ for k_ in np.unique(laba) if k_ >= 0 and (laba == k_).sum() >= cfg20["min_label_patches"]]
    lab = np.where(np.isin(laba, keep), np.searchsorted(keep, np.maximum(laba, 0)), -1)
    ex_ids = pd.read_csv(ROOT / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab = np.where(np.isin(bayer.slide_id.to_numpy()[row_b], ex_ids), -1, lab)
    tf = ex26.fit_combo(Kb, lab, cfg, cfg["seed"])
    L = len(keep) - 1

    def lda(K_):
        return np.concatenate([tf(b_)[:, :L] for b_ in np.array_split(K_, max(1, len(K_) // 200000))]).astype(np.float32)

    # 1. 0035 の重心を作り直す: 集合ごと（Bayer・脂肪変性 3 試験・未学習 3,319 枚）に白色化して、割当ごとに和を足す
    cl35 = {(r, k): np.load(R35 / run / f"assign_{rep35}_flat_k{k}.npy").astype(np.int64) for r, (run, rep35) in REPS.items() for k in KS}
    sums = {(r, k): np.zeros((k, D + (L if r.startswith("A") else 0))) for r in REPS for k in KS}

    def add(off, Sw, Lw):
        n = len(Sw)
        for (r, k), cl in cl35.items():
            X = Sw if r.startswith("B") else np.concatenate([Sw, cfg["lda_beta"] * fac * Lw], 1)
            sums[(r, k)] += torch.zeros(k, X.shape[1], dtype=torch.float64).index_add_(0, torch.from_numpy(cl[off:off + n]), torch.from_numpy(X).double()).numpy()
        return off + n

    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    q = q[q.EXP_ID.notna() & (q.EXP_ID != "")].copy()
    q["exp"] = q.EXP_ID.astype(int)
    tgt = ex26.target_slides({**cfg, "target_exps": cfg["fatty_exps"], "features_dir": cfg["features_dirs"][0]}, ROOT)
    tgt = tgt[~tgt.slide_id.isin(bayer.slide_id)].reset_index(drop=True)
    exp_b = bayer.EXP_ID.fillna("-1").astype(int).to_numpy()[row_b]
    exp_ts = np.repeat(tgt.exp.to_numpy(), cfg["patches_per_slide"])
    qc0 = q[(q.DOSE_LEVEL == "Control") & q.exp.isin(set(exp_b) | set(exp_ts))].copy()
    qc0["h5"] = [h5_of(x) for x in qc0.slide_id]
    qc0 = qc0[qc0.h5 != ""]
    with ProcessPoolExecutor(max_workers=8, initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts_c = list(ex.map(ex27._load_ctrl, qc0.h5, chunksize=4))
    Kc = np.concatenate(parts_c)[:, :D].astype(np.float32)
    exp_c = np.concatenate([np.full(len(p_), int(e)) for p_, e in zip(parts_c, qc0.EXP_ID)])
    del parts_c

    def sphere_known(U, e):
        e_fix = np.where(np.isin(e, exp_c), e, -999)
        return ex27.sphering(U, e_fix, np.r_[Kc, Kc], np.r_[exp_c, np.full(len(exp_c), -999)]).astype(np.float32)
    off = add(0, sphere_known(Kb[:, :D].astype(np.float32), exp_b), lda(Kb))
    del Kb, c
    with ProcessPoolExecutor(max_workers=8, initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        Kt = np.concatenate([p_[0] for p_ in ex.map(ex30._load_sub, tgt.h5, chunksize=4)]).astype(np.float32)
    off = add(off, sphere_known(Kt[:, :D], exp_ts), lda(Kt))
    del Kt
    z = np.load(ROOT / cfg["unseen_features_from"], allow_pickle=True)
    Ku, row_u = z["K"], z["row"]
    del z
    S35 = pd.read_csv(R35 / "v1/slides.csv", dtype={"slide_id": str})
    sl_u = S35[S35.set == "unseen"].reset_index(drop=True)
    exp_u = sl_u.exp.to_numpy()[row_u]
    cm = sl_u.is_ctrl.to_numpy()[row_u]
    Uu = Ku[:, :D].astype(np.float32)
    e_fix = np.where(np.isin(exp_u, exp_u[cm]), exp_u, -999)
    Sw_u = ex27.sphering(Uu, e_fix, np.r_[Uu[cm], Uu[cm]], np.r_[exp_u[cm], np.full(int(cm.sum()), -999)]).astype(np.float32)
    del Uu
    off = add(off, Sw_u, lda(Ku))
    del Sw_u, Ku
    assert off == len(next(iter(cl35.values()))), "patch order differs from the 0035 assignments"
    cen = {key: torch.from_numpy((s / np.maximum(np.bincount(cl35[key], minlength=key[1]), 1)[:, None]).astype(np.float32)) for key, s in sums.items()}
    print("0035 centroids rebuilt", flush=True)

    # 2. 0037 と同じ slide（0037 の slides.csv の並び）。まとまりごとに、試験ごとに自分の対照群で白色化して、0035 の重心に割り当てる
    S = pd.read_csv(R37 / "slides.csv", dtype={"slide_id": str})
    with ProcessPoolExecutor(max_workers=8, initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts = list(ex.map(ex30._load_sub, [h5_of(s_) for s_ in S.slide_id], chunksize=4))
    K_all = np.concatenate([p_[0] for p_ in parts]).astype(np.float32)
    slide_of = np.concatenate([np.full(len(p_[0]), i) for i, p_ in enumerate(parts)])
    del parts
    Lw_all = lda(K_all)
    findings = cfg37["findings"]
    pos = S[[f"grade__{f}" for f in findings]].to_numpy() > 0
    exp_s, ctrl_s = S.exp.to_numpy(), S.is_ctrl.to_numpy()
    rec_rows, loso_rows = [], []
    for g in cfg37["groups"]:
        sg = np.where(S.grp.to_numpy() == g)[0]
        m = np.isin(slide_of, sg)
        so = np.searchsorted(sg, slide_of[m])
        U = K_all[m, :D]
        exp_p, cmk = exp_s[slide_of[m]], ctrl_s[slide_of[m]]
        Sw = ex27.sphering(U, exp_p, U[cmk], exp_p[cmk])
        for r in REPS:
            X = Sw if r.startswith("B") else np.concatenate([Sw, cfg["lda_beta"] * fac * Lw_all[m]], 1).astype(np.float32)
            for k in KS:
                cl = ex35.assign(cen[(r, k)], X, torch.device("cpu"))
                cs = ex35.slide_counts(cl, k, so, len(sg))
                meta = {"group": g, "rep": r, "k": k}
                rec_rows += ex37.record_purity(ex35, cs, pos[sg], findings, cfg37["eval"], meta)
                loso_rows += ex37.loso_auc(cs, pos[sg], exp_s[sg], ctrl_s[sg], S.COMPOUND_NAME.to_numpy()[sg], findings, cfg37["loso"], meta)
                print(f"group {g} {r} k={k} done", flush=True)
    pd.DataFrame(rec_rows).to_csv(OUT / "record_purity.csv", index=False)
    pd.DataFrame(loso_rows).to_csv(OUT / "loso.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
