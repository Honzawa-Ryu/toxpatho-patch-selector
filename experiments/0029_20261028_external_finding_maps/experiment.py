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


def name_prototypes(cl, lab, prio, k, nm):
    """Bayerのラベル付きpatchで、各プロトタイプに濃縮されたラベルの名前を付ける（割合 >= min_share か、全体の割合の min_lift 倍以上。複数あれば倍率の最大）。"""
    m = lab >= 0
    cnt = np.zeros((k, len(prio)))
    np.add.at(cnt, (cl[m], lab[m]), 1)
    n_lab = cnt.sum(1, keepdims=True)
    share = cnt / np.maximum(n_lab, 1)
    prev = cnt.sum(0) / cnt.sum()
    lift = share / np.maximum(prev, 1e-12)
    ok = (cnt >= nm["min_n"]) & ((share >= nm["min_share"]) | (lift >= nm["min_lift"]))
    rows = []
    for j in range(k):
        cand = np.where(ok[j])[0]
        best = cand[np.argmax(lift[j, cand])] if len(cand) else -1
        rows.append({"prototype": j, "name": prio[best] if best >= 0 else "", "share": share[j, best] if best >= 0 else np.nan,
                     "lift": lift[j, best] if best >= 0 else np.nan, "n_labeled": int(n_lab[j, 0]),
                     "top_label": prio[int(np.argmax(cnt[j]))] if n_lab[j, 0] else "", "n_patches_bayer": int((cl == j).sum())})
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
    import h5py
    import openslide
    import torch
    from PIL import Image, ImageDraw, ImageFont
    cfg = config
    t0 = time.time()
    load_module(project_root / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    ex23 = load_module(project_root / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex26 = load_module(project_root / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(project_root / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    cfg20 = yaml.safe_load(open(project_root / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    prio = cfg20["label_priority"]
    run26 = project_root / "outputs" / cfg["exp0026"]

    # 1. TG-GATEs: 0026 tree_v3と同じLDA（脂肪滴系の21枚のラベルを除く）を学び直す。プロトタイプの名前はBayerのラベルから
    src = project_root / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    Ka, laba = c["K"], c["lab"].astype(np.int64)
    keep = [k for k in np.unique(laba) if k >= 0 and (laba == k).sum() >= cfg20["min_label_patches"]]
    laba = np.where(np.isin(laba, keep), laba, -1)
    lab_c = np.where(laba >= 0, np.searchsorted(keep, np.maximum(laba, 0)), -1)
    ex_ids = pd.read_csv(project_root / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab_c = np.where(np.isin(pd.read_csv(src / "slides.csv", dtype=str).slide_id.to_numpy()[c["row"]], ex_ids), -1, lab_c)
    tf = ex26.fit_combo(Ka, lab_c, cfg, cfg["seed"])
    L = len(keep) - 1
    na = len(Ka)
    del Ka
    pre = pickle.loads((project_root / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl").read_bytes())["pre"]
    names, cens = {}, {}
    for mname, (run, rep, k) in cfg["maps"].items():
        cl_all = np.load(run26 / run / "tree" / f"{rep}_k{k}" / "assign_all.npy").astype(np.int64)
        tab = name_prototypes(cl_all[:na], laba, prio, k, cfg["naming"])
        tab.to_csv(run_dir / f"prototype_names_{mname}.csv", index=False)
        names[mname] = tab.name.to_numpy()
        cens[mname] = np.load(run26 / run / f"centroids_{rep}_k{k}.npy").astype(np.float32)
        logger.info(f"{mname}: names {tab.name.replace('', '-').value_counts().to_dict()}")
    logger.info(f"TG-GATEs side ready; {time.time() - t0:.0f}s")

    # 2. 外部の全slideを写す（Sphering は外部の各試験の対照群で）
    ext = project_root / "outputs" / cfg["exp0022"]
    sl = pd.read_csv(ext / "default/slides.csv", dtype=str)
    sl["is_ctrl"] = sl.control_status.isin(cfg["control_status"])
    parts = []
    for r in sl.itertuples():
        with h5py.File(ext / "features_uni_v2_macenko" / f"{r.slide_id.replace(' ', '_')}.h5", "r") as h:
            f = h["features"][:].astype(np.float32)
            parts.append({"xy": h["coords"][:], "size0": int(h["coords"].attrs["patch_size_level0"])})
        K = pre.transform(ex23.slide_rank(f)).astype(np.float32)
        parts[-1].update(U=K[:, :64], Lp=tf(K)[:, :L])
    row = np.concatenate([np.full(len(p_["U"]), i) for i, p_ in enumerate(parts)])
    U = np.concatenate([p_["U"] for p_ in parts])
    Lp = np.concatenate([p_["Lp"] for p_ in parts])
    study = sl.study.to_numpy()[row]
    cm = sl.is_ctrl.to_numpy()[row]
    Sw = ex27.sphering(U, study, U[cm], study[cm])
    sc = json.loads((run26 / cfg["maps"]["A_sphering_lda"][0] / "scale_sphering_lda.json").read_text())
    X = {"A_sphering_lda": np.concatenate([Sw, sc["lda_beta"] * sc["lda_scale"] * Lp], axis=1).astype(np.float32), "B_sphering": Sw.astype(np.float32)}
    cls = {}
    for mname in cfg["maps"]:
        ct = torch.from_numpy(cens[mname])
        cls[mname] = torch.cat([torch.cdist(x, ct).argmin(1) for x in torch.from_numpy(X[mname]).split(65536)]).numpy()
    logger.info(f"external: {len(sl)} slides, {len(U):,} patches assigned; {time.time() - t0:.0f}s")

    # 3. slideの選び方（約40枚）と、各slideの記録
    fr = pd.read_csv(Path(cfg["bench_root"]) / cfg["external_findings"], encoding="utf-8-sig", dtype=str)
    fr["grade"] = pd.to_numeric(fr.grade_ordinal_within_source, errors="coerce").fillna(0)
    pos = fr[fr.grade > 0]
    se = cfg["selection"]
    is_nec = pos.finding_raw.str.contains(se["necrosis_word"]) & ~pos.finding_raw.str.contains("|".join(se["necrosis_exclude"]))
    key = lambda d: set(zip(d.study, d.label_unit))
    nec_u, inf_u = key(pos[is_nec]), key(pos[pos.finding_raw.str.contains(se["inflammation_word"])])
    sl["unit"] = list(zip(sl.study, sl.label_unit))
    sl["necrosis_recorded"] = sl.unit.isin(nec_u)
    sl["inflammation_recorded"] = sl.unit.isin(inf_u)
    sl["records"] = [" / ".join(f"{g.finding_raw} ({g.grade_raw})" for g in pos[(pos.study == r.study) & (pos.label_unit == r.label_unit)].itertuples()) for r in sl.itertuples()]
    ctrl_pick = sl[sl.is_ctrl].sort_values("slide_id").groupby("study").head(se["n_ctrl_per_study"]).index
    sl["role"] = np.select([sl.necrosis_recorded, sl.index.isin(ctrl_pick), sl.inflammation_recorded & ~sl.necrosis_recorded],
                           ["necrosis", "control", "inflammation_only"], "")
    sel = sl[sl.role != ""].copy()
    logger.info(f"selected {len(sel)} slides: {sel.role.value_counts().to_dict()}")

    # 4. 色の付いた領域の割合（全slide）と地図（選んだslide）
    groups = cfg["groups"]
    share_rows = []
    for i, r in enumerate(sl.itertuples()):
        m = row == i
        d = {"slide_id": r.slide_id, "study": r.study, "role": r.role, "is_ctrl": r.is_ctrl, "necrosis_recorded": r.necrosis_recorded, "inflammation_recorded": r.inflammation_recorded, "records": r.records}
        for mname in cfg["maps"]:
            nm_ = names[mname][cls[mname][m]]
            for g in groups:
                d[f"{mname}__{g}"] = float((nm_ == g).mean())
        share_rows.append(d)
    shares = pd.DataFrame(share_rows)
    shares.to_csv(run_dir / "slide_group_shares.csv", index=False)
    logger.info("median share by role:\n" + shares[shares.role != ""].groupby("role")[[c_ for c_ in shares if "__" in c_]].median().round(4).T.to_string())

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    out = run_dir / "maps"
    out.mkdir(exist_ok=True)
    rgb = {g: tuple(int(h[i:i + 2], 16) for i in (1, 3, 5)) for g, h in groups.items()}
    for i in sel.index:
        r = sl.loc[i]
        w = openslide.OpenSlide(str(Path(cfg["bench_root"]) / r.path))
        th = w.get_thumbnail((cfg["thumb"], cfg["thumb"])).convert("RGB")
        s_ = th.width / w.dimensions[0]
        w.close()
        m = row == i
        xy, b = parts[i]["xy"], max(2, round(parts[i]["size0"] * s_))
        panels = [th]
        for mname in cfg["maps"]:
            ov = Image.new("RGBA", th.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            for (x0, y0), g in zip(xy, names[mname][cls[mname][m]]):
                if g in rgb:
                    d.rectangle([x0 * s_, y0 * s_, x0 * s_ + b, y0 * s_ + b], fill=rgb[g] + (235,))
            panels.append(Image.alpha_composite(th.convert("L").convert("RGBA"), ov).convert("RGB"))
        canvas = Image.new("RGB", (th.width * 3 + 20, th.height + 30), "white")
        dd = ImageDraw.Draw(canvas)
        for j, (pn, ttl) in enumerate(zip(panels, ["original", "A: Sphering + LDA (Bayer labels)", "B: Sphering (no labels)"])):
            canvas.paste(pn, (j * (th.width + 10), 30))
            dd.text((j * (th.width + 10) + 4, 6), ttl, fill="black", font=font)
        canvas.save(out / f"{r.slide_id.replace(' ', '_')}.jpg", quality=85)
    sel.drop(columns=["unit"]).to_csv(run_dir / "selected_slides.csv", index=False)
    logger.info(f"maps: {len(sel)}; {time.time() - t0:.0f}s")

    results: dict = {"n_slides": len(sl), "n_selected": len(sel), "roles": sel.role.value_counts().to_dict(),
                     "named": {k_: pd.Series(v).replace("", "-").value_counts().to_dict() for k_, v in names.items()}}

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()