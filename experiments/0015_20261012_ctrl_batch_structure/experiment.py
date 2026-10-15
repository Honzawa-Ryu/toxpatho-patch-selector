import argparse
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
import torch

import h5py
import openslide
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import pdist
from scipy.special import xlogy
from scipy.stats import spearmanr


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
    parser.add_argument("--config", default="config.yml")  # run_slurm.shが渡す。読むのは常に実験ディレクトリのconfig.yml
    parser.add_argument("--variant", default="main")       # config.ymlのvariantsのキー
    return parser.parse_args()


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TILE, ROW_HEADER = 256, 22
LEVELS = {"L1_same_exp_same_time": (True, True), "L2_same_exp_diff_time": (True, False),
          "L3_diff_exp_same_time": (False, True), "L4_diff_exp_diff_time": (False, False)}


# TODO: sample_by_slide / シート描画は0013・0014とほぼ同じ。共通化するならlibへ移す（未着手）。
def sample_by_slide(pool: np.ndarray, slide_of: np.ndarray, rng: np.random.Generator, n: int, cap: int) -> np.ndarray:
    """スライドを一様に選んでからpatchを取る（大きなスライドに偏らせない）。0014と同じ。"""
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


def draw_sheet(title: str, rows: list[tuple[str, list[dict]]], cols: int, path: Path) -> None:
    """1行=1クラスタ。枠: 青=そのクラスタの上位1試験のslide、灰=それ以外。"""
    h = TILE + ROW_HEADER
    sheet = Image.new("RGB", (cols * TILE, 40 + len(rows) * h), "white")
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT, 13)
    d.text((6, 4), title, fill="black", font=font)
    for i, (header, tiles) in enumerate(rows):
        y = 40 + i * h
        d.text((4, y + 4), header, fill="black", font=font)
        for j, t in enumerate(tiles):
            x = j * TILE
            sheet.paste(t["image"], (x, y + ROW_HEADER))
            d.rectangle((x, y + ROW_HEADER, x + TILE - 1, y + h - 1), outline="#2166ac" if t["top"] else "#999999", width=4)
            d.text((x + 6, y + ROW_HEADER + 4), t["label"], fill="black", font=font, stroke_width=2, stroke_fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, quality=90)


def js_distance_matrix(p: np.ndarray, chunk: int = 64) -> np.ndarray:
    """slide占有率どうしのJensen–Shannon距離（底2、0〜1）の全対行列。"""
    out = np.empty((len(p), len(p)))
    for s in range(0, len(p), chunk):
        a = p[s:s + chunk, None, :]
        m = (a + p[None, :, :]) / 2
        js = 0.5 * (xlogy(a, a) - xlogy(a, m)).sum(-1) + 0.5 * (xlogy(p[None], p[None]) - xlogy(p[None], m)).sum(-1)
        out[s:s + chunk] = np.sqrt(np.clip(js / np.log(2), 0, None))
    np.fill_diagonal(out, 0)
    return out


def assign(x: torch.Tensor, centers: np.ndarray, chunk: int = 1 << 18) -> np.ndarray:
    """全patchを最近傍重心へ割り当てる（GPU、チャンク処理）。"""
    c = torch.from_numpy(centers).to(x.device)
    out = torch.empty(len(x), dtype=torch.int64, device=x.device)
    for s in range(0, len(x), chunk):
        out[s:s + chunk] = torch.cdist(x[s:s + chunk], c).argmin(dim=1)
    return out.cpu().numpy()


def level_stats(d: np.ndarray, iu: tuple, exp: np.ndarray, period: np.ndarray, rng: np.random.Generator,
                n_perm: int) -> list[dict]:
    """slide間距離（上三角）を4段階に分けて集計。試験ラベルを時点内で並べ替える検定つき。

    統計量: T_indiv = mean(L3) - mean(L1)（試験差 − 個体差）、T_time = mean(L3) - mean(L2)（試験差 − 時点差）。
    """
    same_time = period[iu[0]] == period[iu[1]]

    def stats(e):
        same_exp = e[iu[0]] == e[iu[1]]
        m = {name: d[(same_exp == se) & (same_time == st)] for name, (se, st) in LEVELS.items()}
        means = {k: v.mean() for k, v in m.items()}
        return m, means["L3_diff_exp_same_time"] - means["L1_same_exp_same_time"], \
            means["L3_diff_exp_same_time"] - means["L2_same_exp_diff_time"]

    m, t1, t2 = stats(exp)
    null1, null2 = np.empty(n_perm), np.empty(n_perm)
    groups = [np.flatnonzero(period == p) for p in np.unique(period)]
    for i in range(n_perm):
        e = exp.copy()
        for g in groups:
            e[g] = rng.permutation(e[g])
        _, null1[i], null2[i] = stats(e)
    p1 = (1 + (null1 >= t1).sum()) / (1 + n_perm)
    p2 = (1 + (null2 >= t2).sum()) / (1 + n_perm)
    rows = [{"level": k, "n_pairs": len(v), "mean": float(v.mean()), "median": float(np.median(v))} for k, v in m.items()]
    rows += [{"level": "T_exp_minus_indiv", "mean": float(t1), "p_perm": float(p1)},
             {"level": "T_exp_minus_time", "mean": float(t2), "p_perm": float(p2)}]
    return rows


def scanner_stats(d: np.ndarray, iu: tuple, grp: np.ndarray, scanner: np.ndarray, rng: np.random.Generator,
                  n_perm: int) -> dict:
    """scanner混在の群（試験×時点）の中だけで、別scanner同士と同scanner同士の距離を比べる。"""
    mixed = [g for g in np.unique(grp) if len(np.unique(scanner[grp == g])) > 1]
    in_mixed = np.isin(grp, mixed)
    pair = (grp[iu[0]] == grp[iu[1]]) & in_mixed[iu[0]]
    dp, a, b = d[pair], iu[0][pair], iu[1][pair]

    def diff(s):
        same = s[a] == s[b]
        return dp[~same].mean() - dp[same].mean(), same

    t, same = diff(scanner)
    idx = [np.flatnonzero(grp == g) for g in mixed]
    null = np.empty(n_perm)
    for i in range(n_perm):
        s = scanner.copy()
        for g in idx:
            s[g] = rng.permutation(s[g])
        null[i], _ = diff(s)
    return {"n_mixed_groups": len(mixed), "n_slides_in_mixed": int(in_mixed.sum()),
            "n_pairs_same_scanner": int(same.sum()), "n_pairs_diff_scanner": int((~same).sum()),
            "mean_same_scanner": float(dp[same].mean()), "mean_diff_scanner": float(dp[~same].mean()),
            "T_diff_minus_same": float(t), "p_perm": float((1 + (null >= t).sum()) / (1 + n_perm))}


def exp_outlier(d_full: np.ndarray, exp: np.ndarray, period: np.ndarray) -> pd.DataFrame:
    """試験ごとの外れ度: 同時点の他試験slideとの平均距離 ÷ 同時点の他試験同士の平均距離（全体）。"""
    same_time = period[:, None] == period[None, :]
    diff_exp = exp[:, None] != exp[None, :]
    base = d_full[same_time & diff_exp].mean()
    rows = []
    for e in np.unique(exp):
        s = exp == e
        block = d_full[np.ix_(s, ~s)][same_time[np.ix_(s, ~s)]]
        rows.append({"exp_id": int(e), "n_slides": int(s.sum()), "score": float(block.mean() / base)})
    df = pd.DataFrame(rows)
    df["rank"] = df.score.rank(ascending=False, method="first").astype(int)
    return df


def cluster_bias(occ: np.ndarray, exp: np.ndarray, scanner: np.ndarray, sizes: np.ndarray) -> pd.DataFrame:
    """クラスタごとの試験・scannerへの偏り。試験シェア = 各試験のslide平均占有率を試験間で正規化したもの。"""
    exps = np.unique(exp)
    per_exp = np.stack([occ[exp == e].mean(0) for e in exps])          # (試験, クラスタ)
    share = per_exp / np.clip(per_exp.sum(0, keepdims=True), 1e-12, None)
    order = np.argsort(-share, axis=0)
    sorted_share = np.take_along_axis(share, order, axis=0)
    cum = np.cumsum(sorted_share, axis=0)
    scans = sorted(np.unique(scanner))
    per_scan = {s: occ[scanner == s].mean(0) for s in scans}
    df = pd.DataFrame({
        "cluster": np.arange(occ.shape[1]), "patch_frac": sizes / sizes.sum(),
        "top1_exp": exps[order[0]], "top1_share": sorted_share[0], "top3_share": cum[2],
        "n_exp_80": (cum < 0.8).sum(0) + 1, "n_exp_present": (per_exp > 0).sum(0)})
    for s in scans:
        df[f"mean_occ_{s}"] = per_scan[s]
    if {"SS1598", "SS1601"} <= set(scans):
        df["log2_SS1598_over_SS1601"] = np.log2((per_scan["SS1598"] + 1e-6) / (per_scan["SS1601"] + 1e-6))
    return df


def load_from_cache(cache: Path, slide_ids: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0009のPCA64キャッシュ（全3,441枚で学習、染色正規化なし）から対象slideの行を取り出す。"""
    cache_ids = np.load(cache / "slide_ids.npy", allow_pickle=False).astype(str)
    slide_index = np.load(cache / "slide_index.npy", mmap_mode="r")
    embeddings = np.load(cache / "embeddings.npy", mmap_mode="r")
    xy_all = np.load(cache / "xy.npy", mmap_mode="r")
    starts = np.searchsorted(slide_index, np.arange(len(cache_ids) + 1))  # slide_indexは昇順（スライドごとに連続）
    pos = {s: i for i, s in enumerate(cache_ids)}
    missing = sorted(set(slide_ids) - set(pos))
    if missing:
        raise ValueError(f"{len(missing)} target slides missing from embedding cache: {missing[:5]}")
    ranges = [(starts[pos[s]], starts[pos[s] + 1]) for s in slide_ids]
    feats = np.concatenate([embeddings[a:b] for a, b in ranges])
    xy = np.concatenate([xy_all[a:b] for a, b in ranges])
    return feats, xy, np.array([b - a for a, b in ranges])


def load_from_h5(feature_dir: Path, slide_ids: pd.Series, vcfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """TRIDENTのh5（1536次元）を読み、L2正規化 → 対象slideで学習したPCAに落とす（0009と同じ前処理）。

    正規化あり/なしのvariantで、PCAの学習patchの抽出（seed・枚数）を揃える。
    """
    from sklearn.decomposition import PCA

    def read(sid):
        with h5py.File(feature_dir / f"{sid}.h5", "r") as h:
            f = h["features"][:].astype(np.float32)
            c = h["coords"][:].astype(np.float32)
        return f / np.linalg.norm(f, axis=1, keepdims=True), c

    rng = np.random.default_rng(vcfg["pca_seed"])
    sample = []
    for sid in slide_ids:
        f, _ = read(sid)
        sample.append(f[rng.choice(len(f), size=min(vcfg["pca_patches_per_slide"], len(f)), replace=False)])
    pca = PCA(n_components=vcfg["pca_components"], svd_solver="randomized", random_state=vcfg["pca_seed"])
    pca.fit(np.concatenate(sample))
    feats, xy, counts = [], [], []
    for sid in slide_ids:
        f, c = read(sid)
        feats.append(pca.transform(f).astype(np.float32))
        xy.append(c)
        counts.append(len(f))
    return np.concatenate(feats), np.concatenate(xy), np.array(counts), float(pca.explained_variance_ratio_.sum())


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.cluster_analysis import occupancy_matrix
    from lib.clustering import kmeans_lloyd
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    args = parse_args()
    exp_name = os.environ["EXP_NAME"]
    config = load_config(Path(__file__).parent)
    variant_key = args.variant
    vcfg = config["variants"][variant_key]
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key, variant=vcfg, ks=config["ks"],
                       sample_seeds=config["sample_seeds"], init_seeds=config["init_seeds"],
                       patches_per_slide=config["patches_per_slide"])
    logger.info(f"Starting: {exp_name} / {variant_key} {vcfg}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_perm = vcfg.get("n_perm", config["n_perm"])

    # --- 対象: Controlかつ記録上無所見（variantによっては特定の特徴量ディレクトリにあるslideに限る） ---
    # 全Control調査（系列C）はmanifestを差し替える。label不明・同一個体の重複画像は除く（0010と同じ規則）。
    manifest = pd.read_parquet(project_root / vcfg.get("manifest_parquet", config["eval_manifest_parquet"]))
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    keep = manifest.dose_level == "Control"
    if "pathology_label_status" in manifest:
        keep &= manifest.pathology_label_status != "unknown"
    if "duplicate_animal_image" in manifest:
        keep &= ~manifest.duplicate_animal_image
    if not vcfg.get("include_ctrl_with_findings", False):
        keep &= manifest.has_finding == False  # noqa: E712（nullable boolのため）
    target = manifest[keep]
    if "restrict_to_dir" in vcfg:
        have = {p.stem for p in (project_root / vcfg["restrict_to_dir"]).glob("*.h5")}
        target = target[target.slide_id.isin(have)]
    if "scanner" not in target:
        scan = pd.read_csv(project_root / config["scanner_csv"], dtype={"slide_id": str})
        target = target.merge(scan[["slide_id", "aperio.ScanScope ID"]].rename(columns={"aperio.ScanScope ID": "scanner"}),
                              on="slide_id", how="left", validate="one_to_one")
    if target.scanner.isna().any():
        raise ValueError("scanner missing for some target slides")
    target = target.sort_values(["exp_id", "sacrifice_period", "individual_id"]).reset_index(drop=True)
    target["group"] = target.exp_id.astype(str) + "|" + target.sacrifice_period
    exp = target.exp_id.to_numpy()
    period = target.sacrifice_period.to_numpy()
    grp = target.group.to_numpy()
    scanner = target.scanner.to_numpy()
    compound_of = dict(zip(target.exp_id, target.compound_name))

    t0 = time.time()
    if vcfg["source"] == "cache":
        feats, xy, counts = load_from_cache(project_root / config["embedding_cache_dir"], target.slide_id)
        feature_dir = project_root / config["feature_dir"]
    else:
        feature_dir = project_root / vcfg["features_dir"]
        feats, xy, counts, explained = load_from_h5(feature_dir, target.slide_id, vcfg)
        logger.info(f"PCA{vcfg['pca_components']} fit on {variant_key}: explained variance {explained:.3f}")
    slide_of = np.repeat(np.arange(len(target)), counts)
    target.assign(n_patches=counts).drop(columns=["finding_types", "finding_sites"], errors="ignore").to_csv(run_dir / "slides.csv", index=False)
    logger.info(f"{len(target)} slides, {target.exp_id.nunique()} exp, {target.group.nunique()} groups, "
                f"{len(feats):,} patches (min {counts.min()}/slide), loaded in {time.time() - t0:.0f}s")
    x_gpu = torch.from_numpy(np.ascontiguousarray(feats)).to(device)
    iu = np.triu_indices(len(target), 1)

    # --- f. クラスタを使わない確認: slide平均埋め込み ---
    mean_emb = np.zeros((len(target), feats.shape[1]))
    np.add.at(mean_emb, slide_of, feats)
    mean_emb /= counts[:, None]
    rng_f = np.random.default_rng(12345)
    d_mean = pdist(mean_emb, "euclidean")
    level_rows = [dict(r, k=0, sample_seed=-1, init_seed=-1, measure="mean_embedding_euclid")
                  for r in level_stats(d_mean, iu, exp, period, rng_f, n_perm)]
    scanner_rows = [dict(scanner_stats(d_mean, iu, grp, scanner, rng_f, n_perm), k=0, sample_seed=-1, init_seed=-1,
                         measure="mean_embedding_euclid")]

    # --- 複数run: 抽出seed × 初期化seed × k ---
    outlier_rows, bias_rows, run_rows = [], [], []
    occ_store, center_store, rep_labels = {}, {}, {}
    for s_seed in config["sample_seeds"]:
        rng_s = np.random.default_rng(s_seed)
        train = np.concatenate([a + rng_s.choice(c, size=min(config["patches_per_slide"], c), replace=False)
                                for a, c in zip(np.r_[0, np.cumsum(counts)[:-1]], counts)])
        for k in config["ks"]:
            for i_seed in config["init_seeds"]:
                t0 = time.time()
                _, centers, n_iter = kmeans_lloyd(feats[train], k, i_seed, device)
                labels = assign(x_gpu, centers)
                occ = occupancy_matrix(slide_of, labels, len(target), k)
                key = f"k{k:03d}_s{s_seed}_i{i_seed}"
                occ_store[key], center_store[key] = occ, centers
                if s_seed == config["sheet_seed"] and i_seed == config["sheet_seed"]:
                    rep_labels[k] = labels
                d_full = js_distance_matrix(occ)
                d = d_full[iu]
                rng_p = np.random.default_rng(1000 * s_seed + i_seed + k)
                tag = dict(k=k, sample_seed=s_seed, init_seed=i_seed)
                level_rows += [dict(r, **tag, measure="occupancy_js") for r in level_stats(d, iu, exp, period, rng_p, n_perm)]
                scanner_rows.append(dict(scanner_stats(d, iu, grp, scanner, rng_p, n_perm), **tag, measure="occupancy_js"))
                outlier_rows += [dict(r, **tag) for r in exp_outlier(d_full, exp, period).to_dict("records")]
                bias = cluster_bias(occ, exp, scanner, np.bincount(labels, minlength=k))
                bias_rows += [dict(r, **tag) for r in bias.to_dict("records")]
                run_rows.append(dict(tag, n_iter=n_iter, seconds=round(time.time() - t0, 1),
                                     n_biased=int((bias.top1_share >= config["bias_share"]).sum())))
                logger.info(f"{key}: {n_iter} iter, {time.time() - t0:.0f}s, biased clusters {run_rows[-1]['n_biased']}")

    levels = pd.DataFrame(level_rows)
    levels.to_csv(run_dir / "level_distances.csv", index=False)
    pd.DataFrame(scanner_rows).to_csv(run_dir / "scanner_within_group.csv", index=False)
    outliers = pd.DataFrame(outlier_rows)
    outliers.to_csv(run_dir / "exp_outlier.csv", index=False)
    biases = pd.DataFrame(bias_rows)
    biases.to_csv(run_dir / "cluster_bias.csv", index=False)
    pd.DataFrame(run_rows).to_csv(run_dir / "runs.csv", index=False)
    np.savez_compressed(run_dir / "occupancy.npz", **occ_store)
    np.savez_compressed(run_dir / "centers.npz", **center_store)

    # --- e. run・kをまたいだ安定性（試験単位） ---
    n_runs = outliers.groupby("k")[["sample_seed", "init_seed"]].apply(lambda g: len(g.drop_duplicates()))
    top = outliers[outliers["rank"] <= config["outlier_top"]]
    biased = biases[biases.top1_share >= config["bias_share"]]
    stab = []
    for k in config["ks"]:
        o = outliers[outliers.k == k].groupby("exp_id").agg(mean_score=("score", "mean"), mean_rank=("rank", "mean"),
                                                            n_slides=("n_slides", "first"))
        o["frac_runs_outlier_top"] = top[top.k == k].groupby("exp_id").size().reindex(o.index, fill_value=0) / n_runs[k]
        b = biased[biased.k == k]
        o["frac_runs_top1_of_biased"] = (b.groupby("top1_exp")[["sample_seed", "init_seed"]]
                                         .apply(lambda g: len(g.drop_duplicates())).reindex(o.index, fill_value=0) / n_runs[k])
        o["mean_n_biased_clusters"] = b.groupby("top1_exp").size().reindex(o.index, fill_value=0) / n_runs[k]
        stab.append(o.reset_index().assign(k=k, compound=lambda df: df.exp_id.map(compound_of)))
    pd.concat(stab).sort_values(["k", "mean_score"], ascending=[True, False]).to_csv(run_dir / "exp_stability.csv", index=False)

    # 外れ度の順位がrun間でどれだけ一致するか: 抽出seedが同じ組 vs 異なる組
    agree = []
    for k in config["ks"]:
        piv = outliers[outliers.k == k].pivot_table(index="exp_id", columns=["sample_seed", "init_seed"], values="score")
        cols = list(piv.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                agree.append({"k": k, "same_sample_seed": cols[i][0] == cols[j][0],
                              "spearman": spearmanr(piv[cols[i]], piv[cols[j]]).statistic})
    agree = pd.DataFrame(agree).groupby(["k", "same_sample_seed"]).spearman.agg(["mean", "min", "count"]).reset_index()
    agree.to_csv(run_dir / "outlier_rank_agreement.csv", index=False)

    # --- c. 画像シート（各kの代表run） ---
    rng_sheet = np.random.default_rng(config["sheet_seed"])
    wsis, sizes = {}, {}
    for k, labels in rep_labels.items():
        key = f"k{k:03d}_s{config['sheet_seed']}_i{config['sheet_seed']}"
        b = biases[(biases.k == k) & (biases.sample_seed == config["sheet_seed"]) & (biases.init_seed == config["sheet_seed"])]
        rows = []
        for r in b.sort_values("top1_share", ascending=False).head(config["sheet_rows"]).itertuples():
            picks = sample_by_slide(np.flatnonzero(labels == r.cluster), slide_of, rng_sheet,
                                    config["sheet_cols"], config["max_per_slide"])
            tiles = []
            for p in picks:
                m = target.iloc[slide_of[p]]
                sid = m.slide_id
                if sid not in wsis:
                    wsis[sid] = openslide.OpenSlide(str(m.svs_path))
                    with h5py.File(feature_dir / f"{sid}.h5", "r") as h:
                        sizes[sid] = int(h["coords"].attrs["patch_size_level0"])
                size = sizes[sid]
                img = wsis[sid].read_region((int(xy[p, 0]), int(xy[p, 1])), 0, (size, size)).convert("RGB").resize((TILE, TILE))
                tiles.append({"image": img, "top": m.exp_id == r.top1_exp,
                              "label": f"{sid} e{m.exp_id} {m.sacrifice_period} {m.scanner}"})
            scan_part = (f" | log2 SS1598/SS1601={r.log2_SS1598_over_SS1601:+.1f}"
                         if hasattr(r, "log2_SS1598_over_SS1601") else "")
            header = (f"c{r.cluster} {100 * r.patch_frac:.2f}% | top1 e{r.top1_exp} {compound_of[r.top1_exp][:22]} "
                      f"{100 * r.top1_share:.0f}% | top3 {100 * r.top3_share:.0f}% | n_exp80={r.n_exp_80}{scan_part}")
            rows.append((header, tiles))
        draw_sheet(f"no-finding Ctrl {key} | rows: clusters by top-1 exp share | border blue = top-1 exp slide",
                   rows, config["sheet_cols"], run_dir / "sheets" / f"{key}.jpg")
    for w in wsis.values():
        w.close()

    summary = levels.groupby(["measure", "k", "level"])[["mean", "p_perm"]].agg(["mean", "min", "max"]).reset_index()
    summary.columns = ["_".join(c).rstrip("_") for c in summary.columns]
    summary.to_csv(run_dir / "level_summary.csv", index=False)
    (run_dir / "results.json").write_text(json.dumps(
        {"n_slides": len(target), "n_exp": int(target.exp_id.nunique()), "n_groups": int(target.group.nunique()),
         "n_patches": int(len(feats)), "n_runs": len(run_rows), "ks": config["ks"]}, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
