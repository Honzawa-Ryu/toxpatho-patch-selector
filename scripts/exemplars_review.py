"""Build a small, auditable visual review set from exp0009's ranked k=2000 candidates.

Each sheet separates finding-positive treated tiles, finding-negative treated
tiles from the same experiment/timepoint, and finding-negative controls from
the same experiment/timepoint. The last panel is a baseline sample, not a
sample of the candidate cluster. No panel is filled from a different experiment.

Usage: python scripts/exemplars_review.py [--limit 1]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.reanalyze_cluster_thresholds import MANIFEST, PART_FOR_K, ROOT  # noqa: E402

K = 2000
N_PER_PANEL = 12
MAX_PER_SLIDE = 2
TILE = 256
COLS = 4
PANELS = ("positive_cluster", "negative_cluster", "matched_control")
OUT = ROOT / "exemplars_review_k2000"


def select_shortlist(index: pd.DataFrame, limit_per_finding: int = 3) -> pd.DataFrame:
    """Choose high-support candidates, preferring distinct leading compounds."""
    eligible = index[(index["保有内率"] >= 0.8) & (index["n_carrying"] >= 20)].copy()
    eligible["leading_compound"] = eligible["compounds"].str.split("(", regex=False).str[0].str.strip()
    eligible = eligible.sort_values(
        ["primary_finding", "保有内率", "n_carrying", "cluster_id"],
        ascending=[True, False, False, True],
    )
    chosen = []
    for _, group in eligible.groupby("primary_finding", sort=True):
        seen_compounds: set[str] = set()
        selected_ids: set[int] = set()
        for row in group.itertuples():
            if row.leading_compound not in seen_compounds:
                chosen.append(row.Index)
                selected_ids.add(int(row.cluster_id))
                seen_compounds.add(row.leading_compound)
            if len(selected_ids) == limit_per_finding:
                break
        if len(selected_ids) < limit_per_finding:
            for row in group.itertuples():
                if int(row.cluster_id) not in selected_ids:
                    chosen.append(row.Index)
                    selected_ids.add(int(row.cluster_id))
                if len(selected_ids) == limit_per_finding:
                    break
    return eligible.loc[chosen].sort_values(["primary_finding", "保有内率"], ascending=[True, False])


def sample_positions_by_slide(
    positions: np.ndarray, slide_index: np.ndarray, rng: np.random.Generator,
    n: int = N_PER_PANEL, max_per_slide: int = MAX_PER_SLIDE,
) -> np.ndarray:
    """Sample slides uniformly before patches, with a per-slide cap."""
    if not len(positions):
        return np.empty(0, dtype=np.int64)
    slides = slide_index[positions]
    order = np.argsort(slides, kind="stable")
    sorted_slides = slides[order]
    unique, starts, counts = np.unique(sorted_slides, return_index=True, return_counts=True)
    options = {
        int(s): rng.choice(positions[order[start:start + count]], size=min(max_per_slide, count), replace=False)
        for s, start, count in zip(unique, starts, counts)
    }
    chosen: list[int] = []
    for round_index in range(max_per_slide):
        for slide in rng.permutation(unique):
            values = options[int(slide)]
            if round_index < len(values):
                chosen.append(int(values[round_index]))
                if len(chosen) == n:
                    return np.asarray(chosen, dtype=np.int64)
    return np.asarray(chosen, dtype=np.int64)


def draw_sheet(tiles: dict[str, list[tuple[int, Image.Image]]], title: str, path: Path) -> None:
    header = 48
    panel_header = 28
    panel_height = panel_header + 3 * TILE
    sheet = Image.new("RGB", (COLS * TILE, header + len(PANELS) * panel_height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 5), title[:110], fill="black")
    draw.text((8, 23), "Negative-cluster tiles may be below the slide-carrying threshold; controls are outside the cluster.", fill="black")
    for panel_i, panel in enumerate(PANELS):
        y0 = header + panel_i * panel_height
        draw.rectangle((0, y0, COLS * TILE, y0 + panel_header), fill="#e8e8e8")
        draw.text((8, y0 + 7), f"{panel} ({len(tiles[panel])}/{N_PER_PANEL})", fill="black")
        for j, (tile_id, tile) in enumerate(tiles[panel]):
            x = (j % COLS) * TILE
            y = y0 + panel_header + (j // COLS) * TILE
            sheet.paste(tile, (x, y))
            draw.rectangle((x, y, x + 33, y + 18), fill="black")
            draw.text((x + 3, y + 2), str(tile_id), fill="white")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def tissue_fraction(tile: Image.Image) -> float:
    """Fraction excluding white areas connected to the tile edge."""
    pixels = np.asarray(tile.resize((128, 128)), dtype=np.float32).mean(axis=2)
    white = pixels >= 220
    components, n = ndimage.label(white)
    if not n:
        return 1.0
    border = np.unique(np.r_[components[0], components[-1], components[:, 0], components[:, -1]])
    border = border[border != 0]
    return float((~np.isin(components, border)).mean())


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None, help="Generate only the first N shortlisted clusters")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    OUT = args.out
    if OUT.exists():
        raise SystemExit("Review output already exists; use --out with a new directory to preserve images and judgments")

    ranked_path = ROOT / "exemplars_ranked_k2000/index.csv"
    ranked = pd.read_csv(ranked_path)
    quality = pd.read_csv(ROOT / "cluster_quality_metrics_k2000.csv")
    ranked = ranked.merge(quality[["cluster_id", "tissue_frac_cc"]], on="cluster_id", validate="one_to_one")
    ranked = ranked[ranked["tissue_frac_cc"] >= 0.85]
    shortlist = select_shortlist(ranked)
    if args.limit is not None:
        shortlist = shortlist.head(args.limit)
    print(f"selected {len(shortlist)} clusters across {shortlist.primary_finding.nunique()} findings", flush=True)

    manifest = pd.read_parquet(MANIFEST).sort_values("slide_id").reset_index(drop=True)
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    slide_ids = manifest["slide_id"].to_numpy()
    positions_of = {s: i for i, s in enumerate(slide_ids)}
    path = ROOT / PART_FOR_K[K] / "cluster_assignments.parquet"
    col = f"cluster_kmeans_k{K}"
    table = pq.read_table(path, columns=["slide_id", "x", "y", col])
    slide_col = table.column("slide_id").to_pandas()
    if isinstance(slide_col.dtype, pd.CategoricalDtype):
        category_to_position = np.array([positions_of[str(s)] for s in slide_col.cat.categories], dtype=np.int32)
        slide_index = category_to_position[slide_col.cat.codes.to_numpy()]
    else:
        slide_index = np.array([positions_of[str(s)] for s in slide_col], dtype=np.int32)
    labels = table.column(col).to_numpy().astype(np.int32)
    xs = table.column("x").to_numpy()
    ys = table.column("y").to_numpy()
    if len(slide_index) != len(labels) or np.any(slide_index[1:] < slide_index[:-1]):
        raise ValueError("assignments are not in manifest slide order")
    slide_starts = np.r_[0, np.flatnonzero(slide_index[1:] != slide_index[:-1]) + 1]
    slide_ends = np.r_[slide_starts[1:], len(slide_index)]
    if len(slide_starts) != len(manifest):
        raise ValueError("assignment slide count differs from manifest")

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tile_rows: list[dict] = []
    review_rows: list[dict] = []
    for number, (_, row) in enumerate(shortlist.iterrows(), 1):
        cid = int(row["cluster_id"])
        finding = str(row["primary_finding"])
        member_pos = np.flatnonzero(labels == cid)
        member_slides = slide_index[member_pos]
        has_target = manifest["finding_types"].map(lambda fs: finding in fs).to_numpy()
        treated = (manifest["dose_level"] != "Control").to_numpy()
        positive_pool = member_pos[has_target[member_slides] & treated[member_slides]]
        positive_pick = sample_positions_by_slide(positive_pool, slide_index, rng)
        positive_slides = np.unique(slide_index[positive_pick])
        matched_keys = {
            (manifest.iloc[int(s)].exp_id, manifest.iloc[int(s)].sacrifice_period)
            for s in positive_slides
        }
        same_stratum = np.array([
            (m.exp_id, m.sacrifice_period) in matched_keys for m in manifest.itertuples()
        ], dtype=bool)
        negative_pool = member_pos[
            (~has_target[member_slides]) & treated[member_slides] & same_stratum[member_slides]
        ]
        negative_counts = np.bincount(slide_index[negative_pool], minlength=len(manifest))
        negative_capacity = int(np.minimum(negative_counts, MAX_PER_SLIDE).sum())
        negative_pick = sample_positions_by_slide(negative_pool, slide_index, rng)

        control_slides = np.flatnonzero((~treated) & (~has_target) & same_stratum)
        control_ranges = [np.arange(slide_starts[s], slide_ends[s], dtype=np.int64) for s in control_slides]
        control_pool = np.concatenate(control_ranges) if control_ranges else np.empty(0, dtype=np.int64)
        control_pool = control_pool[labels[control_pool] != cid]
        # Oversample control candidates so edge/background tiles can be skipped.
        control_pick = sample_positions_by_slide(control_pool, slide_index, rng, n=48, max_per_slide=12)

        cluster_counts = np.bincount(member_slides, minlength=len(manifest))
        slide_counts = slide_ends - slide_starts
        cluster_occ = cluster_counts / slide_counts

        selections = dict(zip(PANELS, (positive_pick, negative_pick, control_pick)))
        sheet_tiles: dict[str, list[tuple[int, Image.Image]]] = {p: [] for p in PANELS}
        sheet_name = f"cluster_{cid:04d}.png"
        for panel, picks in selections.items():
            # Open each source slide only once; errors remain visible in the index.
            panel_slides, first_seen = np.unique(slide_index[picks], return_index=True)
            for slide_pos in panel_slides[np.argsort(first_seen)]:
                if len(sheet_tiles[panel]) >= N_PER_PANEL:
                    break
                slide = manifest.iloc[int(slide_pos)]
                source = picks[slide_index[picks] == slide_pos]
                accepted_from_slide = 0
                try:
                    feature_file = ROOT / "features_tier1_corpus" / f"{slide.slide_id}.h5"
                    with h5py.File(feature_file, "r") as h:
                        attrs = h["coords"].attrs
                        size = int(attrs.get("patch_size_level0", attrs.get("patch_size", 256)))
                    wsi = openslide.OpenSlide(slide.svs_path)
                    try:
                        for pos in source:
                            if len(sheet_tiles[panel]) >= N_PER_PANEL or accepted_from_slide >= MAX_PER_SLIDE:
                                break
                            tile = wsi.read_region((int(xs[pos]), int(ys[pos])), 0, (size, size)).convert("RGB")
                            if not tile.getbbox():
                                raise ValueError("empty tile")
                            if panel == "matched_control" and tissue_fraction(tile) < 0.85:
                                continue
                            tile_id = len(tile_rows) + 1
                            sheet_tiles[panel].append((tile_id, tile.resize((TILE, TILE))))
                            accepted_from_slide += 1
                            tile_rows.append({
                                "tile_id": tile_id, "cluster_id": cid, "primary_finding": finding,
                                "panel": panel, "sheet": sheet_name, "slide_id": slide.slide_id,
                                "compound_name": slide.compound_name, "dose_level": slide.dose_level,
                                "sacrifice_period": slide.sacrifice_period, "exp_id": slide.exp_id,
                                "has_target_finding": bool(has_target[int(slide_pos)]),
                                "in_candidate_cluster": bool(labels[pos] == cid),
                                "cluster_occupancy_on_slide": float(cluster_occ[int(slide_pos)]),
                                "slide_carries_cluster": bool(cluster_occ[int(slide_pos)] > 0.001),
                                "x": int(xs[pos]), "y": int(ys[pos]), "patch_size_level0": size,
                            })
                    finally:
                        wsi.close()
                except Exception as e:
                    print(f"cluster {cid} {panel} slide {slide.slide_id}: {e}", flush=True)
        draw_sheet(
            sheet_tiles,
            f"cluster {cid} | {finding} | P(finding | carrying slide)={row['保有内率']:.2f}",
            OUT / sheet_name,
        )
        review_rows.append({
            "cluster_id": cid, "primary_finding": finding, "sheet": sheet_name,
            "slide_precision": row["保有内率"], "n_carrying_slides": row["n_carrying"],
            "leading_compound": row["leading_compound"], "compound_shares": row["compounds"],
            "positive_tiles": len(sheet_tiles["positive_cluster"]),
            "negative_tiles": len(sheet_tiles["negative_cluster"]),
            "control_tiles": len(sheet_tiles["matched_control"]),
            "negative_eligible_slides": int(np.unique(slide_index[negative_pool]).size),
            "control_eligible_slides": int(len(control_slides)),
            "negative_shortfall": (
                "" if len(sheet_tiles["negative_cluster"]) == N_PER_PANEL else
                "no same-experiment/time cluster patches" if not len(negative_pool) else
                "fewer than 12 eligible tiles under the two-per-slide cap" if negative_capacity < N_PER_PANEL else
                "tile crop failures"
            ),
            "control_shortfall": (
                "" if len(sheet_tiles["matched_control"]) == N_PER_PANEL else
                "no same-experiment/time finding-negative controls" if not len(control_pool) else
                "insufficient tissue-rich control tiles or crop failures"
            ),
            "visual_finding": "", "artifact_suspected": "", "comments": "",
        })
        print(f"{number}/{len(shortlist)} cluster {cid}: " + ", ".join(
            f"{p}={len(sheet_tiles[p])}" for p in PANELS
        ), flush=True)

    pd.DataFrame(tile_rows).to_csv(OUT / "tiles.csv", index=False)
    review_path = OUT / "review.csv"
    if review_path.exists():
        previous = pd.read_csv(review_path, keep_default_na=False).set_index("cluster_id")
        for record in review_rows:
            cid = record["cluster_id"]
            if cid in previous.index:
                for field in ("visual_finding", "artifact_suspected", "comments"):
                    if field in previous:
                        record[field] = previous.loc[cid, field]
    pd.DataFrame(review_rows).to_csv(review_path, index=False)
    (OUT / "README.md").write_text(
        "# k=2000 visual review set\n\n"
        "Source: `exemplars_ranked_k2000/index.csv` and saved k=2000 assignments. "
        "Selection: tissue fraction >=0.85 (source index), slide-level "
        "P(finding | carrying slide) >=0.8, >=20 carrying slides, up to three "
        "clusters per finding with leading-compound diversity preferred. "
        f"Random seed: {args.seed}.\n\n"
        "Each sheet shows up to 12 tiles in each panel, with at most two from "
        "one slide. `positive_cluster` contains candidate-cluster tiles from "
        "treated finding-positive slides. `negative_cluster` contains the same "
        "cluster on treated finding-negative slides from the positive slides' "
        "experiments and timepoints. These may be below the 0.1% occupancy "
        "threshold used to count a slide as carrying the cluster. "
        "`matched_control` contains tissue-rich, non-cluster tiles from "
        "finding-negative controls in the same experiments and timepoints. "
        "Panels are not replenished from unrelated experiments.\n\n"
        "`tiles.csv` maps the number on every tile to its slide and coordinate. "
        "`review.csv` records panel shortages and has blank columns for "
        "`visual_finding` (yes/no/uncertain), `artifact_suspected` (yes/no), "
        "and `comments`. Filled review columns are preserved on rerun. "
        "The slide-level association is not a patch-level diagnosis.\n",
        encoding="utf-8",
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
