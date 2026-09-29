"""0012: 大規模版（tier1, 55M patch）でクラスタリング手法・粒度・seedを比較する。

段（--stage）と1タスクの単位:
  prepare    部分集合1つ（抽出＋全patchからのkNN）、またはreference（既存採用クラスタ・
             旧候補patch・固定標本の準備）
  cluster    グリッドの1条件（1手法×1部分集合×1パラメータ×1seed）、またはWard一式
  consensus  コンセンサスの1解像度
  evaluate   1つのクラスタ割当に対する候補選定（採用条件Aで固定）と評価
  summarize  全評価の集約とseed間安定性

候補選定・所見対応・LOCOは lib.cluster_analysis の既存関数をそのまま使い、
analysis_protocol.json（2026-09-25）と同じ条件・同じ3,357枚で評価する。
"""

import argparse
import itertools
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

import numpy as np
import pandas as pd

import torch


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


def load_config(exp_dir: Path, filename: str = "config.yml") -> dict:
    """Load a config file from the experiment directory."""
    config_path = Path(filename)
    if not config_path.is_absolute():
        config_path = exp_dir / config_path
    if not config_path.exists():
        raise FileNotFoundError(f"config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument(
        "--stage", required=True, choices=["prepare", "cluster", "consensus", "evaluate", "summarize", "count"]
    )
    parser.add_argument("--task", type=int, default=0)
    return parser.parse_args()


# ── タスク一覧（configから決定的に展開）────────────────────────────────────────


def grid_runs(config: dict) -> list[dict]:
    runs = []
    for row in config["grid"]:
        pname = "resolution" if row["method"] == "leiden" else "k"
        for sub, param, seed in itertools.product(row["subsample"], row[pname], row["init_seed"]):
            label = f"{row['method']}_{sub}_{pname[0]}{param}_s{seed}"
            runs.append(
                {"kind": "grid", "label": label, "variant": f"cluster_{label}", "file": "labels.npy",
                 "method": row["method"], "subsample": sub, "param_name": pname, "param": param,
                 "init_seed": seed}
            )
    return runs


def ward_variant(config: dict) -> str:
    w = config["ward"]
    return f"cluster_ward_{w['subsample']}_micro{w['micro_k']}_s{w['seed']}"


def build_tasks(config: dict) -> dict[str, list[dict]]:
    ward = config["ward"]
    prepare = [{"name": name} for name in config["subsamples"]] + [{"name": "reference"}]
    cluster = grid_runs(config) + [{"kind": "ward", "variant": ward_variant(config)}]
    consensus = [
        {"kind": "consensus", "label": f"consensus_{config['consensus']['subsample']}_r{r}",
         "variant": f"consensus_r{r}", "file": "labels.npy", "method": "consensus",
         "subsample": config["consensus"]["subsample"], "param_name": "resolution", "param": r,
         "init_seed": config["consensus"]["seed"]}
        for r in config["consensus"]["resolutions"]
    ]
    labelings = (
        [{"kind": "reference", "label": name, "variant": "prepare_reference", "file": f"{name}.npy",
          "method": "reference", "subsample": None, "param_name": None, "param": None, "init_seed": None}
         for name in config["reference_labelings"]]
        + grid_runs(config)
        + [{"kind": "ward_cut", "label": f"ward_{ward['subsample']}_k{c}", "variant": ward_variant(config),
            "file": f"labels_k{c}.npy", "method": "ward", "subsample": ward["subsample"],
            "param_name": "k", "param": c, "init_seed": ward["seed"]} for c in ward["cuts"]]
        + consensus
    )
    return {"prepare": prepare, "cluster": cluster, "consensus": consensus,
            "evaluate": labelings, "summarize": [{"name": "summary"}]}


# ── 共通の入出力 ──────────────────────────────────────────────────────────────


class Ctx:
    def __init__(self, project_root: Path, config: dict, exp_name: str):
        self.root = project_root
        self.config = config
        self.out = project_root / "outputs" / exp_name
        cache = project_root / config["embedding_cache_dir"]
        self.embeddings = np.load(cache / "embeddings.npy", mmap_mode="r")
        self.slide_index = np.load(cache / "slide_index.npy")
        self.cache_slide_ids = np.load(cache / "slide_ids.npy", allow_pickle=False).astype(str)
        # GPUを割り当てられていないタスク（Slurmが CUDA_VISIBLE_DEVICES を設定しない）では、
        # ノード上のGPUが見えても使わない。他ジョブに割り当てられたGPUを横取りしないため。
        use_gpu = torch.cuda.is_available() and bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.device = torch.device("cuda" if use_gpu else "cpu")

    def done(self, variant: str) -> Path:
        """Directory of a finished upstream task; refuse to read unfinished outputs."""
        d = self.out / variant
        meta = json.loads((d / "completion.json").read_text()) if (d / "completion.json").exists() else {}
        if meta.get("status") != "completed":
            raise SystemExit(f"upstream task not completed: {d}")
        return d

    def subsample(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        d = self.done(f"prepare_{name}")
        sub_idx = np.load(d / "sub_idx.npy")
        return sub_idx, np.ascontiguousarray(self.embeddings[sub_idx])


# ── prepare ──────────────────────────────────────────────────────────────────


def run_prepare(ctx: Ctx, task: dict, run_dir: Path, logger: logging.Logger) -> dict:
    from lib.clustering import knn_graph, knn_to_reference

    cfg = ctx.config
    if task["name"] == "reference":
        return prepare_reference(ctx, run_dir, logger)

    spec = cfg["subsamples"][task["name"]]
    rng = np.random.default_rng(spec["seed"])
    # ソートしておくとmemmapからの読み出しが連続アクセスになる。順序はクラスタリングに無関係。
    sub_idx = np.sort(rng.choice(len(ctx.embeddings), size=spec["size"], replace=False))
    np.save(run_dir / "sub_idx.npy", sub_idx)
    sub = np.ascontiguousarray(ctx.embeddings[sub_idx])
    logger.info(f"subsample {task['name']}: {sub.shape} (seed={spec['seed']})")

    started = time.time()
    neighbor_idx = knn_to_reference(ctx.embeddings, sub, cfg["propagate_k"], ctx.device, chunk=cfg["propagate_chunk"])
    np.save(run_dir / "knn_idx.npy", neighbor_idx)
    logger.info(f"全patch→部分集合 kNN: {neighbor_idx.shape} / {time.time() - started:.0f}s")

    if spec.get("graph"):
        neighbors, _ = knn_graph(sub, cfg["knn_k"], ctx.device, chunk=cfg["knn_chunk"])
        np.save(run_dir / "graph_neighbors.npy", neighbors.astype(np.int32))
        logger.info(f"部分集合内kNNグラフ: {neighbors.shape}")
    return {"size": int(spec["size"]), "seed": int(spec["seed"]), "knn_seconds": round(time.time() - started)}


def _patch_keys(slide_codes: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.asarray(x).astype(np.int64)
    y = np.asarray(y).astype(np.int64)
    assert (x >= 0).all() and (y >= 0).all() and (x < 2**21).all() and (y < 2**21).all()
    return (np.asarray(slide_codes).astype(np.int64) << 42) | (x << 21) | y


def prepare_reference(ctx: Ctx, run_dir: Path, logger: logging.Logger) -> dict:
    import pyarrow.parquet as pq

    cfg = ctx.config
    root = ctx.root
    xy = np.load(root / cfg["embedding_cache_dir"] / "xy.npy")
    cache_keys = _patch_keys(ctx.slide_index, xy[:, 0], xy[:, 1])
    order = np.argsort(cache_keys)
    sorted_keys = cache_keys[order]
    assert (np.diff(sorted_keys) > 0).all(), "duplicate patch coordinates in cache"
    code_of = {sid: i for i, sid in enumerate(ctx.cache_slide_ids)}

    def locate(slide_ids: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        codes = np.array([code_of[s] for s in slide_ids])
        keys = _patch_keys(codes, x, y)
        pos = np.searchsorted(sorted_keys, keys)
        assert (pos < len(sorted_keys)).all() and np.array_equal(sorted_keys[pos], keys), "patch not in cache"
        return order[pos]

    info = {}
    # 既存の採用クラスタ: 0009は同じキャッシュ順で書き出しているはずだが、座標で照合して確かめる。
    for name, ref in cfg["reference_labelings"].items():
        labels = np.full(len(ctx.embeddings), -1, dtype=np.int32)
        for batch in pq.ParquetFile(root / ref["parquet"]).iter_batches(
            batch_size=2_000_000, columns=["slide_id", "x", "y", ref["column"]]
        ):
            frame = batch.to_pandas()
            where = locate(frame["slide_id"].astype(str).to_numpy(), frame["x"].to_numpy(), frame["y"].to_numpy())
            labels[where] = frame[ref["column"]].to_numpy()
        assert (labels >= 0).all(), f"{name}: some cache patches have no label"
        np.save(run_dir / f"{name}.npy", labels)
        info[name] = int(labels.max()) + 1
        logger.info(f"reference {name}: {info[name]} clusters")

    # 319枚パイロットの旧候補patch
    a = cfg["anchors"]
    matched = pd.read_parquet(root / a["matched_patches"], columns=["slide_id", "x", "y", "cluster_k100"])
    where = locate(matched["slide_id"].astype(str).to_numpy(), matched["x"].to_numpy(), matched["y"].to_numpy())
    old_stats = pd.read_parquet(root / a["old_cluster_stats"])
    old_candidates = old_stats.loc[
        (old_stats.k == 100) & old_stats.control_absent & (old_stats.q_value < 0.05) & (old_stats.n_compounds >= 1),
        "cluster_id",
    ].to_numpy()
    np.savez(run_dir / "anchors.npz", cache_index=where, old_cluster=matched["cluster_k100"].to_numpy(),
             old_candidates=old_candidates)
    logger.info(f"anchors: {len(where)} patches, {len(old_candidates)} old candidates")

    rng = np.random.default_rng(cfg["seed"])
    stability_idx = np.sort(rng.choice(len(ctx.embeddings), size=cfg["stability_sample"], replace=False))
    np.save(run_dir / "stability_idx.npy", stability_idx)
    info.update(n_anchor_patches=int(len(where)), old_candidates=[int(c) for c in old_candidates])
    return info


# ── cluster ──────────────────────────────────────────────────────────────────


def run_cluster(ctx: Ctx, task: dict, run_dir: Path, logger: logging.Logger) -> dict:
    from lib.clustering import kmeans_lloyd, leiden_clusters, vote_labels

    cfg = ctx.config
    if task["kind"] == "ward":
        return run_ward(ctx, run_dir, logger)

    sub_idx, sub = ctx.subsample(task["subsample"])
    method, param, seed = task["method"], task["param"], int(task["init_seed"])
    started = time.time()
    meta = {}
    if method == "minibatch":
        from sklearn.cluster import MiniBatchKMeans

        mb = cfg["minibatch"]
        model = MiniBatchKMeans(n_clusters=int(param), random_state=seed, batch_size=mb["batch_size"],
                                n_init=mb["n_init"], max_iter=mb["max_iter"]).fit(sub)
        sub_labels = model.labels_
        meta.update(inertia=float(model.inertia_), n_iter=int(model.n_iter_), n_steps=int(model.n_steps_))
    elif method == "lloyd":
        ll = cfg["lloyd"]
        sub_labels, centers, n_iter = kmeans_lloyd(sub, int(param), seed, ctx.device,
                                                   max_iter=ll["max_iter"], tol=ll["tol"])
        meta.update(inertia=float(((sub - centers[sub_labels]) ** 2).sum()), n_iter=int(n_iter))
    elif method == "leiden":
        neighbors = np.load(ctx.done(f"prepare_{task['subsample']}") / "graph_neighbors.npy")
        sub_labels = leiden_clusters(neighbors, float(param), seed)
    else:
        raise SystemExit(f"unknown method: {method}")
    cluster_seconds = time.time() - started
    n_clusters = int(sub_labels.max()) + 1
    logger.info(f"{task['label']}: {n_clusters} clusters on {len(sub)} patches / {cluster_seconds:.0f}s {meta}")

    neighbor_idx = np.load(ctx.done(f"prepare_{task['subsample']}") / "knn_idx.npy", mmap_mode="r")
    labels = vote_labels(neighbor_idx, sub_labels, ctx.device).astype(np.int32)
    np.save(run_dir / "sub_labels.npy", sub_labels.astype(np.int32))
    np.save(run_dir / "labels.npy", labels)
    logger.info(f"{task['label']}: propagated to {len(labels)} patches / {time.time() - started:.0f}s")
    return dict(meta, n_clusters=n_clusters, cluster_seconds=round(cluster_seconds, 1),
                total_seconds=round(time.time() - started, 1))


def run_ward(ctx: Ctx, run_dir: Path, logger: logging.Logger) -> dict:
    """L0上の微小クラスタ（Lloyd）→ 重心のWard法 → 複数kで切る。

    重心だけでWardを組むので、微小クラスタの大きさ（点数）は結合の重みに入らない。
    """
    from scipy.cluster.hierarchy import fcluster, ward

    from lib.clustering import kmeans_lloyd, vote_labels

    w = ctx.config["ward"]
    _, sub = ctx.subsample(w["subsample"])
    started = time.time()
    micro, centers, n_iter = kmeans_lloyd(sub, int(w["micro_k"]), int(w["seed"]), ctx.device,
                                          max_iter=ctx.config["lloyd"]["max_iter"], tol=ctx.config["lloyd"]["tol"])
    logger.info(f"ward: micro k={w['micro_k']} / {n_iter} iter / {time.time() - started:.0f}s")
    tree = ward(centers.astype(np.float64))
    logger.info(f"ward: linkage / {time.time() - started:.0f}s")

    neighbor_idx = np.load(ctx.done(f"prepare_{w['subsample']}") / "knn_idx.npy", mmap_mode="r")
    micro_full = vote_labels(neighbor_idx, micro, ctx.device)
    np.save(run_dir / "micro_sub_labels.npy", micro.astype(np.int32))
    np.save(run_dir / "micro_labels.npy", micro_full.astype(np.int32))
    for c in w["cuts"]:
        mapping = fcluster(tree, t=c, criterion="maxclust") - 1
        np.save(run_dir / f"labels_k{c}.npy", mapping[micro_full].astype(np.int32))
        logger.info(f"ward cut k={c}: {int(mapping.max()) + 1} clusters")
    return {"micro_k": int(w["micro_k"]), "n_iter": int(n_iter), "total_seconds": round(time.time() - started, 1)}


# ── consensus ────────────────────────────────────────────────────────────────


def run_consensus(ctx: Ctx, task: dict, run_dir: Path, logger: logging.Logger, task_index: int) -> dict:
    from lib.clustering import _undirected_pairs, leiden_on_edges, vote_labels

    c = ctx.config["consensus"]
    prep = ctx.done(f"prepare_{c['subsample']}")
    pairs = _undirected_pairs(np.load(prep / "graph_neighbors.npy"))
    members = [r for r in grid_runs(ctx.config)
               if r["method"] in c["member_methods"] and r["subsample"] == c["subsample"]]
    same = np.zeros(len(pairs), dtype=np.float32)
    per_run = {}
    for r in members:
        sl = np.load(ctx.done(r["variant"]) / "sub_labels.npy")
        hit = sl[pairs[:, 0]] == sl[pairs[:, 1]]
        same += hit
        per_run[r["label"]] = hit
    weights = same / len(members)
    logger.info(f"consensus: {len(members)} runs, {len(pairs)} edges, weight>0: {(weights > 0).mean():.3f}")

    if task_index == 0:
        # 辺単位のPAC（手法×kごと）。同じ手法・kのseed間で、隣接patch対の共起割合が
        # 曖昧な帯（既定0.1〜0.9）に入る辺の割合。小さいほど安定に切れている。
        lo, hi = c["pac_band"]
        rows = []
        for (method, k), group in itertools.groupby(
            sorted(members, key=lambda r: (r["method"], r["param"])), key=lambda r: (r["method"], r["param"])
        ):
            group = list(group)
            frac = np.mean([per_run[r["label"]] for r in group], axis=0)
            rows.append({"method": method, "k": k, "n_runs": len(group),
                         "edge_pac": float(((frac > lo) & (frac < hi)).mean()),
                         "edge_cut_always": float((frac == 0).mean()), "edge_joined_always": float((frac == 1).mean())})
        pd.DataFrame(rows).to_csv(run_dir / "edge_pac.csv", index=False)
        logger.info("\n" + pd.DataFrame(rows).to_string(index=False))

    keep = weights > 0
    started = time.time()
    sub_labels = leiden_on_edges(int(pairs.max()) + 1, pairs[keep], weights[keep], float(task["param"]),
                                 int(task["init_seed"]))
    logger.info(f"{task['label']}: {int(sub_labels.max()) + 1} clusters / {time.time() - started:.0f}s")
    labels = vote_labels(np.load(prep / "knn_idx.npy", mmap_mode="r"), sub_labels, ctx.device).astype(np.int32)
    np.save(run_dir / "sub_labels.npy", sub_labels.astype(np.int32))
    np.save(run_dir / "labels.npy", labels)
    return {"n_members": len(members), "n_clusters": int(sub_labels.max()) + 1,
            "cluster_seconds": round(time.time() - started, 1)}


# ── evaluate ─────────────────────────────────────────────────────────────────


class _QuietLogger:
    def info(self, *a, **k): pass
    def warning(self, *a, **k): pass


def eval_manifest(ctx: Ctx) -> tuple[pd.DataFrame, np.ndarray]:
    clean = pd.read_parquet(ctx.root / ctx.config["eval_manifest_parquet"])
    clean["slide_id"] = clean["slide_id"].astype(str)
    clean = clean.sort_values("slide_id").reset_index(drop=True)
    rows = pd.Index(ctx.cache_slide_ids).get_indexer(clean["slide_id"])
    assert (rows >= 0).all(), "eval slides missing from cache"
    return clean, rows


def run_evaluate(ctx: Ctx, task: dict, run_dir: Path, logger: logging.Logger) -> dict:
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    from lib.cluster_analysis import (
        cluster_statistics, finding_associations, leave_one_compound_out_auroc, select_candidates,
    )

    cfg = ctx.config
    labels = np.load(ctx.done(task["variant"]) / task["file"])
    k = int(labels.max()) + 1
    counts = np.bincount(ctx.slide_index.astype(np.int64) * k + labels,
                         minlength=len(ctx.cache_slide_ids) * k).reshape(-1, k).astype(np.float64)
    clean, rows = eval_manifest(ctx)
    c = counts[rows]
    occ = c / c.sum(axis=1, keepdims=True)
    eps = cfg["presence_eps"]
    kwargs = dict(presence_eps=eps, min_treated_presence=cfg["min_treated_presence"])

    started = time.time()
    stats = cluster_statistics(occ, clean, logger=_QuietLogger(), **kwargs)
    candidates = select_candidates(stats, q_threshold=cfg["q_threshold"], min_compounds=cfg["min_compounds"])
    assoc = finding_associations(occ, clean, candidates, presence_eps=eps, max_finding_types=cfg["max_finding_types"])
    pooled, per_compound = leave_one_compound_out_auroc(
        occ, clean, q_threshold=cfg["q_threshold"], min_compounds=cfg["min_compounds"], logger=_QuietLogger(), **kwargs
    )
    logger.info(f"{task['label']}: k={k}, {len(candidates)} candidates, LOCO={pooled} / {time.time() - started:.0f}s")

    # 候補ごとの要約: 有意な所見のうち保有内率最大（同率はq小）を代表とする（0010と同じ規則）
    profile = pd.DataFrame({"cluster_id": candidates})
    profile["n_patches"] = counts[:, candidates].sum(axis=0).astype(np.int64) if len(candidates) else []
    carriers = occ[:, candidates] > eps
    profile["n_carrier_slides"] = carriers.sum(axis=0)
    treated = clean["dose_level"].ne("Control").to_numpy()
    shares = []
    for j in range(len(candidates)):
        comp = clean.loc[carriers[:, j] & treated, "compound_name"].value_counts(normalize=True)
        shares.append((comp.index[0], float(comp.iloc[0])) if len(comp) else (None, np.nan))
    profile["top_compound"] = [s[0] for s in shares]
    profile["top_compound_share"] = [s[1] for s in shares]
    if not assoc.empty:
        assoc["precision"] = assoc.n_slides_both / (assoc.n_slides_both + assoc.n_slides_cluster_only)
        sig = assoc[assoc.q_value < cfg["q_threshold"]].sort_values(["precision", "q_value"], ascending=[False, True])
        best = sig.groupby("cluster_id").head(1).set_index("cluster_id")
        profile = profile.join(best[["finding_type", "precision", "q_value"]], on="cluster_id")
        assoc.to_csv(run_dir / "associations.csv", index=False)
    else:
        profile[["finding_type", "precision", "q_value"]] = np.nan
    profile.to_csv(run_dir / "candidates.csv", index=False)
    per_compound.to_csv(run_dir / "loco_per_compound.csv", index=False)

    # 旧候補patch（319枚パイロット）が候補クラスタに入る割合
    prep = ctx.done("prepare_reference")
    anchors = np.load(prep / "anchors.npz")
    in_cand = np.isin(labels[anchors["cache_index"]], candidates)
    old = anchors["old_cluster"]
    anchor_rows = {f"old{int(o)}": float(in_cand[old == o].mean()) for o in anchors["old_candidates"]}
    for fam, ids in cfg["anchors"]["families"].items():
        anchor_rows[fam] = float(in_cand[np.isin(old, ids)].mean())

    # 内部指標（参考）: 全runで共通の固定patch標本上
    stab = np.load(prep / "stability_idx.npy")
    np.save(run_dir / "stability_labels.npy", labels[stab])
    x = np.ascontiguousarray(ctx.embeddings[stab])
    ls = labels[stab]
    internal = {
        "silhouette": float(silhouette_score(x, ls, sample_size=cfg["internal_index_sample"], random_state=cfg["seed"])),
        "calinski_harabasz": float(calinski_harabasz_score(x, ls)),
        "davies_bouldin": float(davies_bouldin_score(x, ls)),
    }

    sizes = np.bincount(labels, minlength=k)
    has_best = profile["precision"].notna()
    clean_patches = c.sum()
    result = {
        **{key: task.get(key) for key in ["label", "method", "subsample", "param_name", "param", "init_seed"]},
        "n_clusters": k,
        "n_empty_clusters": int((sizes == 0).sum()),
        "patches_per_cluster_median": float(np.median(sizes[sizes > 0])),
        "patches_per_cluster_p05": float(np.quantile(sizes[sizes > 0], 0.05)),
        "patches_per_cluster_p95": float(np.quantile(sizes[sizes > 0], 0.95)),
        "n_candidates": int(len(candidates)),
        "candidate_patch_fraction": float(c[:, candidates].sum() / clean_patches) if len(candidates) else 0.0,
        "n_candidates_no_sig_finding": int((~has_best).sum()),
        "precision_median": float(profile.loc[has_best, "precision"].median()) if has_best.any() else None,
        "n_precise": int(((profile["precision"] >= cfg["precision_threshold"])
                          & (profile["n_carrier_slides"] >= cfg["min_carrier_slides"])).sum()),
        "n_single_compound_90": int((profile["top_compound_share"] >= 0.9).sum()),
        "loco_pooled_auroc": pooled,
        "loco_macro_auroc": float(per_compound["auroc"].mean()) if "auroc" in per_compound else None,
        **{f"anchor_{key}": v for key, v in anchor_rows.items()},
        **internal,
        "eval_seconds": round(time.time() - started, 1),
    }
    return result


# ── summarize ────────────────────────────────────────────────────────────────


def stability_pairs(labelings: list[dict]) -> list[tuple[str, str, str]]:
    """(比較の種類, label_a, label_b)。seedだけ・部分集合・手法を変えた対を作る。"""
    grid = [r for r in labelings if r["kind"] == "grid"]
    by = lambda *keys: itertools.groupby(sorted(grid, key=lambda r: [str(r[k]) for k in keys]),
                                         key=lambda r: [str(r[k]) for k in keys])
    pairs = []
    for _, group in by("method", "subsample", "param"):
        for a, b in itertools.combinations(list(group), 2):
            pairs.append(("init_seed", a["label"], b["label"]))
    sub_group = [r for r in grid if r["method"] == "minibatch" and r["init_seed"] == 0
                 and r["subsample"] in {"S0", "S1", "S2", "S3", "S4"}]
    for _, group in itertools.groupby(sorted(sub_group, key=lambda r: r["param"]), key=lambda r: r["param"]):
        group = list(group)
        if len({r["subsample"] for r in group}) > 1:
            for a, b in itertools.combinations(group, 2):
                pairs.append(("subsample", a["label"], b["label"]))
    index = {r["label"]: r for r in grid}
    for r in grid:
        if r["method"] == "minibatch":
            twin = r["label"].replace("minibatch_", "lloyd_", 1)
            if twin in index:
                pairs.append(("minibatch_vs_lloyd", r["label"], twin))
    for r in grid:
        if r["method"] == "minibatch" and r["subsample"] == "S0" and r["param"] == 2000:
            pairs.append(("vs_adopted", "adopted_mb_k2000", r["label"]))
    return pairs


def compare_pair(ctx: Ctx, a: dict, b: dict) -> dict:
    from sklearn.metrics import adjusted_rand_score

    ea, eb = ctx.done(f"eval_{a['label']}"), ctx.done(f"eval_{b['label']}")
    ari = adjusted_rand_score(np.load(ea / "stability_labels.npy"), np.load(eb / "stability_labels.npy"))
    la = np.load(ctx.done(a["variant"]) / a["file"])
    lb = np.load(ctx.done(b["variant"]) / b["file"])
    ca = pd.read_csv(ea / "candidates.csv")["cluster_id"].to_numpy()
    cb = pd.read_csv(eb / "candidates.csv")["cluster_id"].to_numpy()
    ma, mb = np.isin(la, ca), np.isin(lb, cb)
    union = ma | mb
    jaccard = float((ma & mb).sum() / union.sum()) if union.any() else np.nan
    # 候補1個ずつの最良対応Jaccard（aの候補→bの候補）。どちらかの候補に入るpatchだけで数える。
    best = []
    if len(ca) and len(cb):
        xa, xb = la[union], lb[union]
        size_a = pd.Series(xa[np.isin(xa, ca)]).value_counts()
        size_b = pd.Series(xb[np.isin(xb, cb)]).value_counts()
        both = np.isin(xa, ca) & np.isin(xb, cb)
        inter = pd.DataFrame({"a": xa[both], "b": xb[both]}).value_counts()
        for cid in ca:
            if cid not in size_a.index:
                continue
            if cid in inter.index.get_level_values("a"):
                row = inter.loc[cid]
                jac = row / (size_a[cid] + size_b.reindex(row.index).to_numpy() - row)
                best.append(float(jac.max()))
            else:
                best.append(0.0)
    return {"ari": float(ari), "candidate_patch_jaccard": jaccard,
            "candidate_best_match_jaccard_median": float(np.median(best)) if best else np.nan,
            "candidate_best_match_ge_0.5": float(np.mean(np.array(best) >= 0.5)) if best else np.nan}


def run_summarize(ctx: Ctx, run_dir: Path, logger: logging.Logger) -> dict:
    # 失敗した評価があっても残りで集約し、欠けたものは明示して残す（黙って落とさない）。
    labelings, missing = [], []
    for r in build_tasks(ctx.config)["evaluate"]:
        ok = (ctx.out / f"eval_{r['label']}" / "result.json").exists()
        (labelings if ok else missing).append(r)
    (run_dir / "missing_evaluations.json").write_text(json.dumps([r["label"] for r in missing], indent=2))
    if missing:
        logger.warning(f"{len(missing)} evaluations missing: {[r['label'] for r in missing]}")
    rows = [json.loads((ctx.done(f"eval_{r['label']}") / "result.json").read_text()) for r in labelings]
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    logger.info("\n" + summary[["label", "n_clusters", "n_candidates", "precision_median", "n_precise",
                               "loco_pooled_auroc", "anchor_cholesterol_fatty"]].to_string(index=False))

    index = {r["label"]: r for r in labelings}
    pair_rows = []
    for kind, a, b in stability_pairs(labelings):
        if a not in index or b not in index:
            continue
        started = time.time()
        pair_rows.append({"kind": kind, "a": a, "b": b, **compare_pair(ctx, index[a], index[b])})
        logger.info(f"{kind} {a} vs {b}: {pair_rows[-1]} / {time.time() - started:.0f}s")
        pd.DataFrame(pair_rows).to_csv(run_dir / "stability_pairs.csv", index=False)
    pairs = pd.DataFrame(pair_rows)
    pairs["group"] = pairs["a"].str.replace(r"_s\d+$", "", regex=True)
    grouped = pairs.groupby(["kind", "group"]).agg(
        n_pairs=("ari", "size"), ari_mean=("ari", "mean"), ari_min=("ari", "min"),
        candidate_patch_jaccard_mean=("candidate_patch_jaccard", "mean"),
        candidate_best_match_jaccard_median=("candidate_best_match_jaccard_median", "mean"),
    ).reset_index()
    grouped.to_csv(run_dir / "stability_groups.csv", index=False)
    return {"n_labelings": len(rows), "n_pairs": len(pair_rows)}


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    args = parse_args()
    config = load_config(Path(__file__).parent, args.config)
    tasks = build_tasks(config)

    if args.stage == "count":
        print(json.dumps({stage: len(items) for stage, items in tasks.items()}))
        return

    items = tasks[args.stage]
    if os.environ.get("N_TASKS") and int(os.environ["N_TASKS"]) != len(items):
        raise SystemExit(f"N_TASKS={os.environ['N_TASKS']} but stage {args.stage} has {len(items)} tasks")
    if not 0 <= args.task < len(items):
        raise SystemExit(f"--task {args.task} out of range for stage {args.stage} ({len(items)} tasks)")
    task = items[args.task]
    variant = {
        "prepare": lambda t: f"prepare_{t['name']}",
        "cluster": lambda t: t["variant"],
        "consensus": lambda t: t["variant"],
        "evaluate": lambda t: f"eval_{t['label']}",
        "summarize": lambda t: "summary",
    }[args.stage](task)

    run_dir = get_run_dir(project_root, __file__, variant, output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant, stage=args.stage, task=task)
    logger.info(f"Starting: {exp_name} / {args.stage} #{args.task} / {variant}")

    ctx = Ctx(project_root, config, exp_name)
    logger.info(f"device: {ctx.device} / cache: {ctx.embeddings.shape}")
    if args.stage == "prepare":
        result = run_prepare(ctx, task, run_dir, logger)
    elif args.stage == "cluster":
        result = run_cluster(ctx, task, run_dir, logger)
    elif args.stage == "consensus":
        result = run_consensus(ctx, task, run_dir, logger, args.task)
    elif args.stage == "evaluate":
        result = run_evaluate(ctx, task, run_dir, logger)
    else:
        result = run_summarize(ctx, run_dir, logger)

    (run_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
