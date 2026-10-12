"""Plot saved 0003/0009 clustering on identical patches; do not refit clusters.

Usage: .venv/bin/python scripts/visualize_clustering_scale.py
The t-SNE layout includes sampled original and added slides, is fitted once,
and is reused across panels. Full correspondence plots use all matched patches.
"""

import argparse
import colorsys
import hashlib
import json
import mmap
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/0010_20260925_pathology_benchmark_subset"
TRACE = BASE / "scale_trace"
CACHE = ROOT / "outputs/0010_20260925_pathology_benchmark_subset/scale_trace"
BIG = ROOT / "outputs/0009_20260919_tier1_corpus_clustering"
OLD = ROOT / "outputs/0003_20260912_control_absent_cluster_mining/cluster_mining_ccl4"
SEED = 20261009
FAMILIES = {
    "CCl4 composite": [4, 5, 14, 52, 62, 67],
    "WY-14643 composite": [0, 21, 25, 55, 65, 77, 93],
    "Cholesterol fatty": [42, 53],
}
ORANGE, GREY = "#e57522", "#d8dfe5"


def save(fig, out, stem):
    fig.savefig(out / f"{stem}.png", dpi=180, facecolor="white")
    fig.savefig(out / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def candidates(k):
    s = pd.read_csv(TRACE / f"large_k{k}_stats.csv")
    return s.loc[s.control_absent & (s.q_value < .05) & (s.n_compounds >= 1), "cluster_id"].to_numpy()


def make_sample(out):
    """Match sampled original rows to cache by slide and exact coordinates."""
    print("Reading exact patch correspondence", flush=True)
    matched = pd.read_parquet(CACHE / "matched_patches.parquet")
    matched.slide_id = matched.slide_id.astype(str)
    assert not matched.duplicated(["slide_id", "x", "y"]).any()
    rng = np.random.default_rng(SEED)
    parts = []
    for sid, group in matched.groupby("slide_id", observed=True, sort=True):
        take = rng.choice(len(group), min(40, len(group)), replace=False)
        parts.append(group.iloc[take])
    original = pd.concat(parts, ignore_index=True)
    original["sample_origin"] = "Original 319 slides"
    print("Joining original sampled coordinates to feature cache", flush=True)

    emb_cache = BIG / "embedding_cache"
    ids = np.load(emb_cache / "slide_ids.npy", allow_pickle=False).astype(str)
    slide_index = np.load(emb_cache / "slide_index.npy", mmap_mode="r")
    xy = np.load(emb_cache / "xy.npy", mmap_mode="r")
    embeddings = np.load(emb_cache / "embeddings.npy", mmap_mode="r")
    # Sparse sampling of a 14 GB file should not prefetch unrelated pages on NFS.
    for array in [xy, embeddings]:
        array._mmap.madvise(mmap.MADV_RANDOM)
    starts = np.searchsorted(slide_index, np.arange(len(ids) + 1))
    positions = {sid: i for i, sid in enumerate(ids)}
    original["cache_index"] = -1
    for sid, group in original.groupby("slide_id", sort=True):
        i = positions[sid]
        a, b = starts[i:i + 2]
        # Cache coordinates are float32 integers in level-0 pixels.
        coords = np.asarray(xy[a:b], dtype=np.int64)
        index = pd.MultiIndex.from_arrays(coords.T)
        assert index.is_unique, f"Duplicate cache coordinate: {sid}"
        loc = index.get_indexer(pd.MultiIndex.from_frame(group[["x", "y"]]))
        assert (loc >= 0).all(), f"Missing sampled coordinate: {sid}"
        original.loc[group.index, "cache_index"] = a + loc
    assert (original.cache_index >= 0).all()

    old_ids = set(matched.slide_id)
    added = []
    for i, sid in enumerate(ids):
        if sid in old_ids:
            continue
        a, b = starts[i:i + 2]
        idx = a + rng.choice(b - a, min(8, b - a), replace=False)
        coordinates = np.asarray(xy[idx], dtype=np.int64)
        added.append(pd.DataFrame({"slide_id": sid, "x": coordinates[:, 0],
                                   "y": coordinates[:, 1], "cache_index": idx,
                                   "sample_origin": "Added 3,122 slides"}))
    sample = pd.concat([original, *added], ignore_index=True)
    assert not sample.duplicated(["slide_id", "x", "y"]).any()
    print(f"Loading cached PCA features for {len(sample):,} sampled patches", flush=True)
    # pread releases the GIL while waiting for NFS. A NumPy fancy-indexing
    # loop over mmap would incur one synchronous page fault after another.
    cache_indices = sample.cache_index.to_numpy()
    features = np.empty((len(sample), embeddings.shape[1]), dtype=embeddings.dtype)
    row_bytes = embeddings.shape[1] * embeddings.dtype.itemsize
    fd = os.open(emb_cache / "embeddings.npy", os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        def read_block(start):
            for target in range(start, min(start + 128, len(sample))):
                raw = os.pread(fd, row_bytes, embeddings.offset + int(cache_indices[target]) * row_bytes)
                if len(raw) != row_bytes:
                    raise ValueError("Incomplete cached feature row")
                features[target] = np.frombuffer(raw, dtype=embeddings.dtype)
        with ThreadPoolExecutor(max_workers=24) as pool:
            list(pool.map(read_block, range(0, len(sample), 128)))
    finally:
        os.close(fd)
    probe = np.random.default_rng(SEED).choice(len(sample), size=5, replace=False)
    assert np.array_equal(features[probe], embeddings[cache_indices[probe]]), "pread/cache mismatch"
    assert np.isfinite(features).all()
    print(f"Matched sample: {len(original):,} original + {len(sample) - len(original):,} added patches", flush=True)
    with threadpool_limits(limits=8):
        layout = TSNE(n_components=2, perplexity=40, init="pca", learning_rate="auto",
                      max_iter=1000, random_state=SEED, n_jobs=8, verbose=1).fit_transform(features)
    assert np.isfinite(layout).all()
    sample["tsne_x"], sample["tsne_y"] = layout.T
    sample.to_parquet(out / "sample_layout.parquet", index=False)
    # Agreement measures use every matched patch, not the visual subsample.
    metrics = []
    for k in [100, 1000, 2000]:
        metrics.append({"large_k": k, "matched_patches": len(matched),
                        "ARI": adjusted_rand_score(matched.cluster_k100, matched[f"large_k{k}"]),
                        "NMI": normalized_mutual_info_score(matched.cluster_k100, matched[f"large_k{k}"])})
    pd.DataFrame(metrics).to_csv(out / "full_patch_agreement.csv", index=False)
    print("Shared t-SNE and all-patch agreement saved", flush=True)
    return sample


def palette(k, offset):
    return np.asarray([colorsys.hsv_to_rgb((i * .61803398875 + offset) % 1,
                       .55 + .3 * ((i % 3) / 2), .65 + .25 * ((i % 5) / 4)) for i in range(k)])


def embedding_plot(sample, out, old_candidates):
    s = sample[sample.cluster_k100.notna()].copy()
    xy = s[["tsne_x", "tsne_y"]].to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    runs = [("cluster_k100", 100, "Before: 319 slides / k=100", old_candidates),
            ("large_k100", 100, "After: 3,441 slides / k=100", candidates(100)),
            ("large_k2000", 2000, "After: 3,441 slides / k=2,000", candidates(2000))]
    order = np.random.default_rng(SEED).permutation(len(s))
    for col, (key, k, title, selected) in enumerate(runs):
        labels = s[key].to_numpy(dtype=int)
        axes[0, col].scatter(*xy[order].T, c=palette(k, col * .17)[labels[order]], s=3,
                             alpha=.7, linewidths=0, rasterized=True)
        axes[0, col].set_title(title, fontsize=13, pad=12)
        mask = np.isin(labels, selected)
        axes[1, col].scatter(*xy[~mask].T, color=GREY, s=3, alpha=.45,
                             linewidths=0, rasterized=True)
        axes[1, col].scatter(*xy[mask].T, color=ORANGE, s=4, alpha=.85,
                             linewidths=0, rasterized=True)
        axes[1, col].set_title(f"Selected candidates: {len(selected):,} clusters in full run\n"
                               f"{mask.mean():.1%} of displayed original patches", fontsize=11)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("How the original patches are partitioned before and after scaling", fontsize=18, y=.97)
    fig.text(.5, .918, f"Same {len(s):,} patches from all 319 original slides; one shared t-SNE layout", ha="center", fontsize=11)
    fig.text(.5, .063, "Top: saved cluster IDs (colors are independent in each run).  Bottom: selected candidate patches in orange.", ha="center", fontsize=10)
    fig.text(.5, .035, "Descriptive comparison: PCA fit, training sample and assignment method also differ. 2D overlap does not measure cluster quality.", ha="center", fontsize=9, color="#4a5568")
    fig.subplots_adjust(top=.85, bottom=.11, hspace=.22, wspace=.07)
    save(fig, out, "clustering_before_after")


def context_plot(sample, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharex=True, sharey=True)
    old = sample.sample_origin.eq("Original 319 slides")
    xy = sample[["tsne_x", "tsne_y"]].to_numpy()
    axes[0].scatter(*xy[old].T, s=3, color="#2563a6", alpha=.6, linewidths=0, rasterized=True)
    axes[0].set_title("Before: 319 slides / 9 compounds", pad=15)
    axes[1].scatter(*xy[~old].T, s=3, color="#f0a24a", alpha=.55, linewidths=0, rasterized=True)
    axes[1].scatter(*xy[old].T, s=3, color="#2563a6", alpha=.5, linewidths=0, rasterized=True)
    axes[1].set_title("After: 3,441 slides / 44 compounds", pad=15)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("Corpus expansion in the same embedding view", fontsize=17, y=.97)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color="#2563a6", label="Original slides"),
                        Line2D([], [], marker="o", ls="", color="#f0a24a", label="Added slides")],
               loc="lower center", bbox_to_anchor=(.5, .07), ncol=2, frameon=False)
    fig.text(.5, .035, "Shared t-SNE of cached PCA64 features. Sample: 40 patches/original slide, 8/added slide; densities are not corpus proportions.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.84, bottom=.18, wspace=.07)
    save(fig, out, "corpus_expansion")


def trace_plot(out, old_candidates):
    ids = [cid for group in FAMILIES.values() for cid in group]
    ids += sorted(set(old_candidates) - set(ids))
    names = {cid: name for name, cids in FAMILIES.items() for cid in cids}
    y = np.arange(len(ids))
    fig, axes = plt.subplots(1, 3, figsize=(17, 9), sharey=True)
    colors = ["#234f83", "#427faf", "#70a3c7", "#9dc3dc", "#c6dcea", "#e4e7eb"]
    rows = []
    for ax, k in zip(axes[:2], [100, 2000]):
        matrix = np.load(CACHE / f"contingency_k{k}.npy")
        counts = matrix[ids]
        ranks = np.argsort(-counts, axis=1, kind="stable")
        ranked = np.take_along_axis(counts, ranks, axis=1)
        fractions = ranked / counts.sum(axis=1, keepdims=True)
        data = np.column_stack([fractions[:, :5], 1 - fractions[:, :5].sum(axis=1)])
        left = np.zeros(len(ids))
        for j in range(6):
            ax.barh(y, data[:, j], left=left, color=colors[j], height=.72,
                    label=f"Rank {j + 1}" if j < 5 else "Other destinations")
            left += data[:, j]
        ax.set_title(f"After: k={k:,}\nWhere each original cluster goes", fontsize=12)
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_xlabel("Fraction of original cluster patches")
        ax.grid(axis="x", alpha=.12)
        for i, cid in enumerate(ids):
            rows.append({"old_cluster": cid, "large_k": k, "family": names.get(cid, "Other old candidate"),
                         "patches": int(counts[i].sum()), "top1_destination": int(ranks[i, 0]),
                         "top1_fraction": fractions[i, 0], "top5_fraction": fractions[i, :5].sum(),
                         "selected_fraction": counts[i, candidates(k)].sum() / counts[i].sum()})
    axes[0].set_yticks(y, [f"{cid:02d}  {names.get(cid, 'Other old candidate')}" for cid in ids], fontsize=10)
    table = pd.DataFrame(rows)
    previous = pd.read_csv(TRACE / "old_cluster_retention.csv")
    check = table.merge(previous, left_on=["old_cluster", "large_k"], right_on=["old_cluster", "k"],
                        suffixes=("_new", "_previous"), validate="one_to_one")
    assert len(check) == len(table)
    for col in ["top1_fraction", "top5_fraction"]:
        assert np.allclose(check[f"{col}_new"], check[f"{col}_previous"], atol=1e-12)
    assert np.allclose(check.selected_fraction, check.fraction_in_selected_clusters, atol=1e-12)
    for k, offset, color in [(100, -.18, "#477eaa"), (2000, .18, ORANGE)]:
        values = table.loc[table.large_k.eq(k), "selected_fraction"].to_numpy()
        axes[2].barh(y + offset, values, height=.32, color=color, label=f"After k={k:,}")
    axes[2].set_title("Candidate selection\nHow many original patches remain selected", fontsize=12)
    axes[2].set_xlim(0, 1.04)
    axes[2].xaxis.set_major_formatter(PercentFormatter(1))
    axes[2].set_xlabel("Fraction retained in selected candidates")
    axes[2].legend(loc="upper center", bbox_to_anchor=(.5, -.10), ncol=2, frameon=False, fontsize=9)
    axes[0].invert_yaxis()
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=color) for color in colors],
               labels=[f"Rank {i + 1}" for i in range(5)] + ["Other destinations"],
               loc="lower center", bbox_to_anchor=(.53, .073), ncol=6, fontsize=9, frameon=False)
    fig.suptitle("Old candidate clusters split; some also drop out of selection", fontsize=18, y=.975)
    fig.text(.5, .923, "Exact correspondence of all 5,111,952 original patches; 19 original candidate clusters shown", ha="center", fontsize=11)
    fig.text(.5, .028, "Left/middle: destinations ranked separately within each row; matching shades do not mean matching cluster IDs. Family names are historical associations.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.2, right=.985, bottom=.20, top=.855, wspace=.19)
    save(fig, out, "original_cluster_splitting")
    table.to_csv(out / "original_cluster_splitting.csv", index=False)


def fatty_plot(out):
    matrix = np.load(CACHE / "contingency_k2000.npy")
    counts = matrix[[42, 53]].sum(axis=0)
    ids = [1356, 846, 703, 535]
    fraction = counts[ids] / counts.sum()
    original = pd.read_csv(TRACE / "large_k2000_on_original_299.csv").set_index("cluster_id")
    full = pd.read_csv(TRACE / "large_k2000_stats.csv").set_index("cluster_id")
    q0, q1 = original.loc[ids, "q_value"].to_numpy(), full.loc[ids, "q_value"].to_numpy()
    assert original.loc[ids, "selected"].all()
    assert full.loc[ids, "control_absent"].all() and (q1 >= .05).all()
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].bar(range(5), [*fraction, 1 - fraction.sum()], color=["#467eaa"] * 4 + [GREY])
    axes[0].set_xticks(range(5), [str(cid) for cid in ids] + ["All other\nclusters"])
    axes[0].set_ylim(0, .5)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel("Fraction of original 42/53 patches")
    axes[0].set_title(f"{fraction.sum():.1%} move to these four clusters\nAll four fail full-corpus candidate selection", fontsize=12)
    for i, f in enumerate([*fraction, 1 - fraction.sum()]):
        axes[0].text(i, f + .009, f"{f:.1%}", ha="center", fontsize=10)
    x = np.arange(4)
    axes[1].bar(x - .18, -np.log10(q0), width=.35, color="#467eaa", label="Original cohort (299 eligible slides)")
    axes[1].bar(x + .18, -np.log10(q1), width=.35, color=ORANGE, label="Full corpus (3,357 eligible slides)")
    axes[1].axhline(-np.log10(.05), color="#4a5568", ls="--", lw=1, label="q = 0.05 threshold")
    axes[1].set_xticks(x, [str(cid) for cid in ids])
    axes[1].set_ylabel("-log10(q); higher = stronger statistical evidence")
    axes[1].set_ylim(0, 3.9)
    axes[1].set_title("Same k=2,000 assignments; comparison scope changes\nControl-rarity criterion passes in both scopes", fontsize=12)
    axes[1].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.set_xlabel("Large-corpus cluster ID (k=2,000)")
    fig.suptitle("Cholesterol-associated fatty candidates: preserved grouping, lost selection", fontsize=15, y=.98)
    fig.text(.5, .035, "Original clusters 42/53: 46,638 patches. This is a historical association, not independently confirmed patch-level pathology.", ha="center", fontsize=9)
    fig.subplots_adjust(top=.78, bottom=.16, wspace=.26)
    save(fig, out, "fatty_candidate_loss")
    pd.DataFrame({"large_cluster": ids, "fraction_of_old_42_53": fraction,
                  "q_original_299": q0, "q_full_3357": q1}).to_csv(out / "fatty_candidate_loss.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=BASE / "scale_visualization_20261009")
    parser.add_argument("--reuse-layout", action="store_true", help="Regenerate figures from saved sample layout")
    parser.add_argument("--trace-only", action="store_true", help="Generate full-patch trace figures without t-SNE")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    stats = pd.read_parquet(OLD / "cluster_stats.parquet")
    old_candidates = stats.loc[stats.k.eq(100) & stats.control_absent &
                               stats.q_value.lt(.05) & stats.n_compounds.ge(1), "cluster_id"].to_numpy()
    assert len(old_candidates) == 19
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    if args.trace_only:
        trace_plot(out, old_candidates)
        fatty_plot(out)
        print("Full-patch trace figures saved", flush=True)
        return
    layout_path = out / "sample_layout.parquet"
    if args.reuse_layout:
        sample = pd.read_parquet(layout_path)
    else:
        if layout_path.exists():
            raise SystemExit("Layout already exists; use --reuse-layout or a new --out-dir")
        sample = make_sample(out)
    embedding_plot(sample, out, old_candidates)
    context_plot(sample, out)
    trace_plot(out, old_candidates)
    fatty_plot(out)
    provenance = {
        "seed": SEED, "original_slides": 319, "large_slides": 3441,
        "sample_original_patches": int(sample.cluster_k100.notna().sum()),
        "sample_added_patches": int(sample.cluster_k100.isna().sum()),
        "sampling": "Up to 40 patches per original slide and 8 per added slide; without replacement",
        "layout": "One t-SNE fit on shared cached L2-normalized UNI2-h -> PCA64 features; perplexity=40, max_iter=1000",
        "layout_basis": "PCA fitted for large corpus; used only as shared visualization, not to refit clusters",
        "source_old": str(OLD.relative_to(ROOT)), "source_large": str(BIG.relative_to(ROOT)),
        "matching": "Original labels from previously exact-matched slide_id,x,y; feature cache join rechecked by same keys",
        "full_patch_trace": 5111952,
        "selection": "control_q95 < 0.001, BH q < 0.05, min_compounds=1, min_treated_presence=0; tissue filter not applied",
        "evaluation": "Large-corpus selection uses 3,357 audited eligible slides; original-subset selection uses 299",
        "limitations": "Different k, PCA fit, training sample (500k vs 200k), direct vs kNN assignment; descriptive, not scale-only causal comparison. 2D layout does not establish pathology or cluster quality.",
        "source_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            CACHE / "matched_patches.parquet", *[CACHE / f"contingency_k{k}.npy" for k in [100, 1000, 2000]],
            TRACE / "large_k2000_on_original_299.csv", TRACE / "large_k2000_stats.csv", OLD / "cluster_stats.parquet"]},
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Saved figures and provenance to {out}", flush=True)


if __name__ == "__main__":
    main()
