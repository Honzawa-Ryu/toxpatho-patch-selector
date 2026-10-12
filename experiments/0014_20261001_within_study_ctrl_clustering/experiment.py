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
import torch

import h5py
import openslide
from itertools import combinations
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import false_discovery_control, rankdata


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


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TILE, COLS, ROW_HEADER = 256, 6, 22
BORDER = {"control": "#2166ac", "with": "#1a9850", "without": "#d73027"}


def exact_rank_p(occ: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """片側（posの占有率が高い）の正確な並べ替え検定。統計量はスライド占有率の順位和。

    occ: (スライド, クラスタ)。全ての分け方 C(n, |pos|) を列挙する（20枚なら最大18万通り）。
    0が多く同順位が多いので、正規近似や同順位なしの正確分布は使わない。
    """
    ranks = rankdata(occ, axis=0)
    n, m = len(pos), int(pos.sum())
    combos = np.array(list(combinations(range(n), m)), dtype=np.int64)
    member = np.zeros((len(combos), n), np.float32)
    member[np.arange(len(combos))[:, None], combos] = 1
    null = member @ ranks.astype(np.float32)
    observed = ranks[pos].sum(0)
    return (null >= observed - 1e-6).mean(0)


def sample_by_slide(pool: np.ndarray, slide_of: np.ndarray, rng: np.random.Generator, n: int, cap: int) -> np.ndarray:
    """スライドを一様に選んでからpatchを取る（大きなスライドに偏らせない）。0013と同じ。"""
    order = np.argsort(slide_of[pool], kind="stable")
    slides, starts, counts = np.unique(slide_of[pool][order], return_index=True, return_counts=True)
    options = {int(s): rng.choice(pool[order[a:a + c]], size=min(cap, c), replace=False)
               for s, a, c in zip(slides, starts, counts)}
    chosen: list[int] = []
    for rnd in range(cap):
        for s in rng.permutation(slides):
            if rnd < len(options[int(s)]):
                chosen.append(int(options[int(s)][rnd]))
                if len(chosen) == n:
                    return np.asarray(chosen)
    return np.asarray(chosen)


def draw_cluster_sheet(title: str, rows: list[tuple[str, list[dict]]], path: Path) -> None:
    """1行=1クラスタ。各行の上に見出し、タイルの枠はスライドの群（Control / 所見あり / なし）。"""
    h = TILE + ROW_HEADER
    sheet = Image.new("RGB", (COLS * TILE, 40 + len(rows) * h), "white")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT, 13)
    d.text((6, 4), title, fill="black", font=font)
    for i, (header, tiles) in enumerate(rows):
        y = 40 + i * h
        d.text((4, y + 4), header, fill="black", font=font)
        for j, t in enumerate(tiles):
            x = j * TILE
            sheet.paste(t["image"], (x, y + ROW_HEADER))
            d.rectangle((x, y + ROW_HEADER, x + TILE - 1, y + h - 1), outline=BORDER[t["group"]], width=4)
            d.text((x + 6, y + ROW_HEADER + 4), f"{t['slide_id']} {t['dose']}", fill="black", font=font,
                   stroke_width=2, stroke_fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def finding_tests(occ: np.ndarray, findings_of: list[list[str]], idx: np.ndarray, scope: str) -> list[dict]:
    """idxのスライドを所見あり/なしに分けて、所見ごとにクラスタ占有率を比べる（両群2枚以上の所見だけ）。"""
    out = []
    names = sorted({f for i in idx for f in findings_of[i]})
    for f in names:
        has = np.array([f in findings_of[i] for i in idx])
        if has.sum() < 2 or (~has).sum() < 2:
            continue
        p = exact_rank_p(occ[idx], has)
        out += [{"scope": scope, "finding": f, "n_with": int(has.sum()), "n_without": int((~has).sum()),
                 "cluster": c, "p": float(p[c])} for c in range(occ.shape[1])]
    return out


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.cluster_analysis import occupancy_matrix
    from lib.clustering import kmeans_lloyd
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    config = load_config(Path(__file__).parent)
    seed: int = config["seed"]
    variant_key = "pilot"
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key,
                       compounds=config["compounds"], ks=config["ks"])
    logger.info(f"Starting: {exp_name} / {variant_key}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = project_root / config["embedding_cache_dir"]
    cache_ids = np.load(cache / "slide_ids.npy", allow_pickle=False).astype(str)
    slide_index = np.load(cache / "slide_index.npy", mmap_mode="r")
    embeddings = np.load(cache / "embeddings.npy", mmap_mode="r")
    xy_all = np.load(cache / "xy.npy", mmap_mode="r")
    starts = np.searchsorted(slide_index, np.arange(len(cache_ids) + 1))  # slide_indexは昇順（スライドごとに連続）
    cache_pos = {s: i for i, s in enumerate(cache_ids)}

    manifest = pd.read_parquet(project_root / config["eval_manifest_parquet"])
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    manifest = manifest[manifest.compound_name.isin(config["compounds"]) & manifest.slide_id.isin(cache_pos)]
    feature_dir = project_root / config["feature_dir"]

    rng = np.random.default_rng(seed)
    cluster_rows, test_rows, study_rows = [], [], []
    for (compound, period), study in manifest.groupby(["compound_name", "sacrifice_period"]):
        study = study.sort_values(["dose", "individual_id"]).reset_index(drop=True)
        name = f"{compound}__{period}".replace(" ", "_")
        ranges = [(starts[cache_pos[s]], starts[cache_pos[s] + 1]) for s in study.slide_id]
        local_slide = np.concatenate([np.full(b - a, i, np.int64) for i, (a, b) in enumerate(ranges)])
        feats = np.concatenate([embeddings[a:b] for a, b in ranges])
        xy = np.concatenate([xy_all[a:b] for a, b in ranges])
        treated = (study.dose_level != "Control").to_numpy()
        findings_of = [list(f) for f in study.finding_types]
        study_rows.append({"study": name, "compound": compound, "period": period, "n_slides": len(study),
                           "n_control": int((~treated).sum()), "n_patches": len(feats),
                           "n_treated_with_any_finding": int(study.has_finding[treated].sum())})
        logger.info(f"{name}: {len(study)} slides ({int((~treated).sum())} control), {len(feats)} patches")

        for k in config["ks"]:
            labels, _, n_iter = kmeans_lloyd(feats, k, seed, device)
            occ = occupancy_matrix(local_slide, labels, len(study), k)
            p_ctrl = exact_rank_p(occ, treated)
            tests = (finding_tests(occ, findings_of, np.arange(len(study)), "all")
                     + finding_tests(occ, findings_of, np.flatnonzero(treated), "treated_only"))
            # 試験・kごとに、Control比較と所見比較をまとめてBH補正する
            q = false_discovery_control(np.r_[p_ctrl, [t["p"] for t in tests]], method="bh")
            q_ctrl = q[:k]
            for t, qt in zip(tests, q[k:]):
                t.update(study=name, k=k, q=float(qt))
            test_rows += tests
            tdf = pd.DataFrame(tests)
            best = {s: tdf[tdf.scope == s].sort_values("q").groupby("cluster").first() if len(tdf) else pd.DataFrame()
                    for s in ["all", "treated_only"]}
            eps = config["presence_eps"]
            for c in range(k):
                row = {"study": name, "compound": compound, "period": period, "k": k, "cluster": c,
                       "n_patches": int((labels == c).sum()),
                       "treated_mean_occ": float(occ[treated, c].mean()), "control_mean_occ": float(occ[~treated, c].mean()),
                       "control_max_occ": float(occ[~treated, c].max()),
                       "n_treated_present": int((occ[treated, c] > eps).sum()), "n_control_present": int((occ[~treated, c] > eps).sum()),
                       "p_ctrl": float(p_ctrl[c]), "q_ctrl": float(q_ctrl[c])}
                for s, b in best.items():
                    hit = c in b.index
                    row[f"best_finding_{s}"] = b.loc[c, "finding"] if hit else ""
                    row[f"q_best_{s}"] = float(b.loc[c, "q"]) if hit else np.nan
                cluster_rows.append(row)
            logger.info(f"  k={k}: {n_iter} iter, q_ctrl<0.05 {int((q_ctrl < 0.05).sum())}, "
                        f"finding tests {len(tests)} (q<0.05 {int(sum(t['q'] < 0.05 for t in tests))})")

            # 目視用: Control比較のpが小さい順に最大max_rows行、1行に6枚
            krows = pd.DataFrame(cluster_rows[-k:]).sort_values(["p_ctrl", "treated_mean_occ"], ascending=[True, False])
            sheet_rows, wsis, sizes = [], {}, {}
            for r in krows.head(config["max_rows"]).itertuples():
                picks = sample_by_slide(np.flatnonzero(labels == r.cluster), local_slide, rng, COLS, config["max_per_slide"])
                tiles = []
                for p in picks:
                    m = study.iloc[local_slide[p]]
                    sid = m.slide_id
                    if sid not in wsis:
                        wsis[sid] = openslide.OpenSlide(str(m.svs_path))
                        with h5py.File(feature_dir / f"{sid}.h5", "r") as h:
                            sizes[sid] = int(h["coords"].attrs["patch_size_level0"])
                    size = sizes[sid]
                    img = wsis[sid].read_region((int(xy[p, 0]), int(xy[p, 1])), 0, (size, size)).convert("RGB").resize((TILE, TILE))
                    group = "control" if m.dose_level == "Control" else ("with" if r.best_finding_all in m.finding_types else "without")
                    tiles.append({"image": img, "slide_id": sid, "dose": m.dose_level, "group": group})
                fa = f"{r.best_finding_all[:26]} q={r.q_best_all:.1e}" if r.best_finding_all else "-"
                ft = f"{r.best_finding_treated_only[:26]} q={r.q_best_treated_only:.1e}" if r.best_finding_treated_only else "-"
                header = (f"c{r.cluster} n={r.n_patches} | vs Ctrl q={r.q_ctrl:.1e} (T {r.n_treated_present}/{int(treated.sum())}, "
                          f"C {r.n_control_present}/{int((~treated).sum())}) | finding(all): {fa} | finding(treated): {ft}")
                sheet_rows.append((header, tiles))
            for w in wsis.values():
                w.close()
            title = (f"{compound} {period} k={k} | rows sorted by vs-Ctrl p | border: blue=Control, "
                     f"green=treated with row's best finding(all), red=treated without")
            draw_cluster_sheet(title, sheet_rows, run_dir / "sheets" / name / f"k{k:03d}.jpg")

    pd.DataFrame(study_rows).to_csv(run_dir / "studies.csv", index=False)
    clusters = pd.DataFrame(cluster_rows)
    clusters.to_csv(run_dir / "clusters.csv", index=False)
    pd.DataFrame(test_rows).to_csv(run_dir / "finding_tests.csv", index=False)
    summary = clusters.assign(sig_ctrl=clusters.q_ctrl < 0.05, sig_all=clusters.q_best_all < 0.05,
                              sig_treated=clusters.q_best_treated_only < 0.05).groupby(["study", "k"])[
        ["sig_ctrl", "sig_all", "sig_treated"]].sum().reset_index()
    summary.to_csv(run_dir / "summary.csv", index=False)
    (run_dir / "results.json").write_text(json.dumps(
        {"n_studies": len(study_rows), "ks": config["ks"], "n_clusters": len(clusters)}, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
