import argparse
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd
import torch

import h5py


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")
    parser.add_argument("--phase", choices=["coords", "coords_treated", "ctrl", "treated", "full", "full_block3", "full_added_treated"], required=True)
    parser.add_argument("--chunk", type=int, required=True)   # 0..n_chunks-1（array task）
    return parser.parse_args()


def rel_path(period: str, sid: str) -> str:
    return f"{str(period).replace(' ', '_')}/{sid}.svs"


def build_slides(project_root: Path, config: dict) -> pd.DataFrame:
    """全対象slide（slide_id, svs_path, compound_name, exp_id, sacrifice_period, is_ctrl, added）。"""
    t = pd.read_parquet(project_root / config["tier1_manifest"])
    t["slide_id"] = t.slide_id.astype(str)
    t["is_ctrl"] = (t.dose_level == "Control") | (t.dose.astype(float) == 0)
    t["added"] = False
    a = pd.read_csv(project_root / config["added_ctrl_list"], dtype={"slide_id": str}).rename(
        columns={"COMPOUND_NAME": "compound_name", "EXP_ID": "exp_id"})
    wsi = project_root / config["wsi_dir"]
    a["svs_path"] = [str(wsi / rel_path(p, s)) for p, s in zip(a.sacrifice_period, a.slide_id)]
    a["is_ctrl"], a["added"] = True, True
    parts = [t, a]
    # 追加で転送した投与群（2026-10-17: cut 0.72の最大の塊のうち31試験の投与群1,828枚）
    if config.get("added_treated_list"):
        a2 = pd.read_csv(project_root / config["added_treated_list"], dtype={"slide_id": str}).rename(
            columns={"COMPOUND_NAME": "compound_name", "EXP_ID": "exp_id"})
        a2["svs_path"] = [str(wsi / rel_path(p, s)) for p, s in zip(a2.sacrifice_period, a2.slide_id)]
        a2["is_ctrl"], a2["added"] = False, True
        parts.append(a2)
    cols = ["slide_id", "svs_path", "compound_name", "exp_id", "sacrifice_period", "is_ctrl", "added"]
    slides = pd.concat([p_[cols] for p_ in parts], ignore_index=True)
    if slides.slide_id.duplicated().any():
        raise ValueError("duplicated slide_id between tier1 manifest and added Ctrl list")
    return slides.sort_values("slide_id").reset_index(drop=True)


def trident_patches_dir(project_root: Path, config: dict, exp_name: str, chunk: int) -> Path:
    return (project_root / "outputs" / exp_name / f"trident_job_chunk{chunk}"
            / f"{config['mag']:g}x_{config['patch_size']}px_{config['overlap']}px_overlap" / "patches")


def coords_path(project_root: Path, config: dict, exp_name: str, sid: str) -> Path:
    """既存TRIDENT出力（features h5）か、coords phaseで作ったpatches h5。どちらも"coords"データセットを持つ。"""
    p = project_root / config["tier1_coords_dir"] / f"{sid}.h5"
    if p.exists():
        return p
    for c in range(config["n_chunks"]):
        q = trident_patches_dir(project_root, config, exp_name, c) / f"{sid}_patches.h5"
        if q.exists():
            return q
    raise FileNotFoundError(f"no coords for {sid}; run the coords phase first")


def stage(src_paths: list[str], root: Path, rels: list[str]) -> list[Path]:
    """svsをローカルSSDへ丸ごとコピーする（1回の連続読み）。既にあれば飛ばす。0008のstage_slides_locallyと同じ考え方。"""
    out = []
    for src, rel in zip(src_paths, rels):
        dest = root / rel
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".part")
            shutil.copy2(src, tmp)
            tmp.rename(dest)
        out.append(dest)
    return out


def run_coords(project_root: Path, config: dict, exp_name: str, chunk: pd.DataFrame, chunk_id: int,
               run_dir: Path, logger) -> dict:
    """追加Controlに TRIDENT seg → coords（0008と同一設定、特徴量抽出はしない）。"""
    missing = [p for p in chunk.svs_path if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} added slides not transferred yet, e.g. {missing[:3]}")
    job_dir = project_root / "outputs" / exp_name / f"trident_job_chunk{chunk_id}"
    rels = [rel_path(p, s) for p, s in zip(chunk.sacrifice_period, chunk.slide_id)]
    if not rels:
        return {"n_slides": 0, "n_with_coords": 0, "job_dir": str(job_dir)}
    staging = config.get("staging_root")
    # ローカルSSDに置く場合はバッチごとにコピー → seg・coords → 削除。次バッチのコピーは裏で進める（2026-10-18）
    bs = config.get("staging_batch_size", 30) if staging else len(rels)
    batches = [list(range(i, min(i + bs, len(rels)))) for i in range(0, len(rels), bs)]
    root = Path(staging) / f"coords_chunk{chunk_id}" if staging else project_root / config["wsi_dir"]
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(stage, [chunk.svs_path.iloc[j] for j in batches[0]], root, [rels[j] for j in batches[0]]) if staging else None
    for bi, idx in enumerate(batches):
        t0 = time.time()
        staged = fut.result() if staging else []
        t_copy = time.time() - t0
        if staging and bi + 1 < len(batches):
            nxt = batches[bi + 1]
            fut = pool.submit(stage, [chunk.svs_path.iloc[j] for j in nxt], root, [rels[j] for j in nxt])
        custom = run_dir / f"custom_list_of_wsis_batch{bi}.csv"
        pd.DataFrame({"wsi": [rels[j] for j in idx]}).to_csv(custom, index=False)
        t0 = time.time()
        for task in ["seg", "coords"]:
            argv = [sys.executable, str(project_root / "libraries" / "trident_run_batch_of_slides.py"),
                    "--task", task, "--wsi_dir", str(root), "--search_nested",
                    "--custom_list_of_wsis", str(custom), "--job_dir", str(job_dir),
                    "--reader_type", config["reader_type"], "--segmenter", config["segmenter"],
                    "--mag", str(config["mag"]), "--patch_size", str(config["patch_size"]),
                    "--overlap", str(config["overlap"]), "--patch_encoder", config["patch_encoder"], "--gpus", "0"]
            subprocess.run(argv, check=True)
        logger.info(f"batch {bi + 1}/{len(batches)}: {len(idx)} slides, waited {t_copy:.0f}s for staging, "
                    f"seg+coords {time.time() - t0:.0f}s ({(time.time() - t0) / len(idx):.1f}s/slide)")
        for f in staged:
            f.unlink(missing_ok=True)
    pool.shutdown()
    pdir = trident_patches_dir(project_root, config, exp_name, chunk_id)
    n_ok = sum((pdir / f"{s}_patches.h5").exists() for s in chunk.slide_id)
    return {"n_slides": len(chunk), "n_with_coords": int(n_ok), "job_dir": str(job_dir)}


def get_reference(project_root: Path, config: dict, exp_name: str, slides: pd.DataFrame, exp0005, fit_macenko, logger):
    """全Control（実験ごとに一様抽出）から参照染色を推定。最初の1回だけ計算して保存し、以後はそれを読む。"""
    path = project_root / "outputs" / exp_name / "reference.json"
    if path.exists():
        ref = json.loads(path.read_text())
    else:
        rng = np.random.default_rng(config["seed"])
        ctrl = slides[slides.is_ctrl]
        ids = [str(sid) for _, g in ctrl.groupby("exp_id")
               for sid in rng.choice(g.slide_id.to_numpy(), size=min(config["reference_slides_per_exp"], len(g)), replace=False)]
        pixels = []
        for sid in ids:
            coords, attrs = exp0005.read_coords(coords_path(project_root, config, exp_name, sid))
            size = int(attrs.get("patch_size_level0", attrs.get("patch_size", 256)))
            svs = slides.loc[slides.slide_id == sid, "svs_path"].iloc[0]
            pixels.append(exp0005.sample_patches(svs, coords, size, config["reference_patches_per_slide"], rng).reshape(-1, 3))
        he, maxc = fit_macenko(np.concatenate(pixels))
        ref = {"reference_slides": ids, "n_exp": int(ctrl.exp_id.nunique()),
               "reference_he": he.tolist(), "reference_max_c": maxc.tolist()}
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".json.tmp{os.getpid()}")
        tmp.write_text(json.dumps(ref, indent=2))
        tmp.rename(path)   # 同時に走ったchunkも同じseedで同じ値を計算するので、上書きされても内容は同一
    logger.info(f"参照染色（Control {len(ref['reference_slides'])}枚・{ref['n_exp']}実験）: "
                f"max_c={np.round(ref['reference_max_c'], 3).tolist()}")
    return np.asarray(ref["reference_he"]), np.asarray(ref["reference_max_c"])


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata
    from lib.stain import fit_macenko, macenko_matrix

    # 0005の抽出部品（座標読み込み・patch抽出・正規化Dataset・推論）をそのまま使う。
    # TODO: 共通化するならlib/へ移す（未着手）。
    spec = importlib.util.spec_from_file_location(
        "exp0005", project_root / "experiments/0005_20260913_stain_normalized_embeddings/experiment.py")
    exp0005 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp0005)

    args = parse_args()
    exp_name = os.environ["EXP_NAME"]
    config = load_config(Path(__file__).parent)
    variant_key = f"{args.phase}_chunk{args.chunk}"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, n_chunks=config["n_chunks"])
    logger.info(f"Starting: {exp_name} / {variant_key}")

    slides = build_slides(project_root, config)
    target = {"coords": slides[slides.added], "coords_treated": slides[slides.added & ~slides.is_ctrl], "ctrl": slides[slides.is_ctrl], "treated": slides[~slides.is_ctrl],
              "full": slides[slides.exp_id.isin(config.get("full_exp_ids", []))],
              "full_block3": slides[slides.exp_id.isin(config.get("full_exp_ids_block3", []))],
              "full_added_treated": slides[slides.added & ~slides.is_ctrl]}[args.phase]
    chunk = target.iloc[args.chunk::config["n_chunks"]]
    if args.phase in ("coords", "coords_treated"):   # 座標がまだないslideだけ
        # 2026-10-19修正: 以前はchunkに分ける前に絞っていたため、他chunkの進み具合で割り振りがずれ、
        # 遅れて始まったchunkで60枚が抜けた。分けてから絞る。
        def has_coords(sid):
            try:
                coords_path(project_root, config, exp_name, sid)
                return True
            except FileNotFoundError:
                return False
        chunk = chunk[~chunk.slide_id.map(has_coords)]
    logger.info(f"phase {args.phase}: {len(target)} slides in total, {len(chunk)} in this chunk")

    if args.phase in ("coords", "coords_treated"):
        res = run_coords(project_root, config, exp_name, chunk, args.chunk, run_dir, logger)
        (run_dir / "results.json").write_text(json.dumps(res, indent=2))
        if res["n_with_coords"] < res["n_slides"]:
            logger.warning(f"coords missing for {res['n_slides'] - res['n_with_coords']} slides")
        complete_run(run_dir)
        logger.info("Done.")
        return

    he_ref, maxc_ref = get_reference(project_root, config, exp_name, slides, exp0005, fit_macenko, logger)
    from trident.patch_encoder_models.load import encoder_factory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder_factory(config["patch_encoder"])
    encoder.to(device).eval()
    sub_n = config["ctrl_patches_per_slide"] if args.phase == "ctrl" else 0
    features_dir = project_root / "outputs" / exp_name / config["ctrl_features_subdir" if sub_n else "features_subdir"]
    features_dir.mkdir(parents=True, exist_ok=True)

    fit_rows, failed = [], []
    done = skipped = 0
    # ローカルSSDに置く場合: 次のslideのコピーを裏で進め、処理し終えたslideは消す（2026-10-18）
    staging = config.get("staging_root")
    todo = [r for r in chunk.itertuples(index=False) if not (features_dir / f"{r.slide_id}.h5").exists()]
    skipped = len(chunk) - len(todo)
    sroot = Path(staging) / f"{args.phase}_chunk{args.chunk}" if staging else None
    pool = ThreadPoolExecutor(max_workers=1) if staging else None
    pending = {}

    def prefetch(j):
        if staging and j < len(todo) and j not in pending:
            r = todo[j]
            pending[j] = pool.submit(stage, [r.svs_path], sroot, [rel_path(r.sacrifice_period, r.slide_id)])

    prefetch(0)
    for i, row in enumerate(todo, start=1):
        sid = row.slide_id
        out_path = features_dir / f"{sid}.h5"
        t_slide = time.time()
        if staging:
            local = pending.pop(i - 1).result()[0]
            prefetch(i)
            row = row._replace(svs_path=str(local))
        t_wait = time.time() - t_slide
        coords, attrs = exp0005.read_coords(coords_path(project_root, config, exp_name, sid))
        n_total = len(coords)
        if sub_n and n_total > sub_n:   # ctrl phase: slide固有の乱数で無作為抽出（座標順は保つ）
            keep = np.sort(np.random.default_rng([config["seed"], int(sid), 1]).choice(n_total, size=sub_n, replace=False))
            coords = coords[keep]
        patch_size = int(attrs.get("patch_size_level0", attrs.get("patch_size", 256)))
        rng = np.random.default_rng([config["seed"], int(sid)])   # chunk分割に依存しないslide固有の乱数
        try:
            sample = exp0005.sample_patches(row.svs_path, coords, patch_size, config["fit_patches_per_slide"], rng)
            he, max_c = fit_macenko(sample.reshape(-1, 3))
            matrix = macenko_matrix(he, max_c, he_ref=he_ref, maxc_ref=maxc_ref)
        except ValueError as e:
            # 推定に失敗したslideを正規化なしで混ぜると条件が揃わないので、飛ばして記録する（0005と同じ）
            logger.warning(f"{sid}: 染色推定に失敗したためスキップ: {e}")
            failed.append({"slide_id": sid, "reason": str(e)})
            continue
        fit_rows.append({"slide_id": sid, "compound_name": row.compound_name, "exp_id": int(row.exp_id),
                         "is_ctrl": bool(row.is_ctrl), "max_c_h": float(max_c[0]), "max_c_e": float(max_c[1]),
                         "he_cosine": float(abs(he[:, 0] @ he[:, 1]))})
        dataset = exp0005.NormalizedPatchDataset(row.svs_path, coords, patch_size, matrix, encoder.eval_transforms)
        features = exp0005.extract_slide(encoder, device, dataset, config["batch_size"], config["num_workers"])
        tmp_path = out_path.with_suffix(".h5.tmp")
        with h5py.File(tmp_path, "w") as h:
            fset = h.create_dataset("features", data=features.astype(np.float32))
            fset.attrs["encoder"] = config["patch_encoder"]
            fset.attrs["name"] = sid
            fset.attrs["stain_normalization"] = "macenko"
            fset.attrs["n_patches_total"] = n_total
            fset.attrs["subsampled"] = bool(sub_n and n_total > sub_n)
            cset = h.create_dataset("coords", data=coords)
            for k, v in attrs.items():
                cset.attrs[k] = v
        tmp_path.rename(out_path)  # 途中で落ちた.h5を完成品と誤認しないための原子的置換
        if staging:
            Path(row.svs_path).unlink(missing_ok=True)
        done += 1
        logger.info(f"[{i}/{len(todo)}] {sid}: {features.shape} {time.time() - t_slide:.0f}s (staging wait {t_wait:.0f}s)")
        if done % 50 == 0:  # 長時間ジョブなので途中経過も残す
            pd.DataFrame(fit_rows).to_parquet(run_dir / "stain_fits.parquet", index=False)

    if fit_rows:
        pd.DataFrame(fit_rows).to_parquet(run_dir / "stain_fits.parquet", index=False)
    (run_dir / "results.json").write_text(json.dumps(
        {"phase": args.phase, "chunk": args.chunk, "n_slides": len(chunk), "n_extracted": done,
         "n_skipped_existing": skipped, "n_failed": len(failed), "failed": failed,
         "features_dir": str(features_dir)}, indent=2, ensure_ascii=False))
    if failed:
        logger.warning(f"{len(failed)} slides failed (see results.json)")
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
