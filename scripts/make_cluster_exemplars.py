"""Crop exemplar patches for clusters found by 0003, straight from its assignments.

0003 の experiment.py も代表patchを書き出すが、あれは実行時に選んだ k・クラスタ分しか
出さない。こちらは `cluster_assignments.parquet`（全patchのslide_id・座標・クラスタID）
だけを読むので、埋め込みの再計算もクラスタリングのやり直しもせずに、後から任意の
k・任意のクラスタを見返せる。

重心に一番近いpatchではなく一様ランダムに抜くのは、クラスタが実際にどれだけ均質かを
見るため（重心近傍だけ見ると、どんなクラスタでも綺麗に見える）。

Usage:
    python scripts/make_cluster_exemplars.py --k 100 --clusters 45 87 20 --n 36
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = (
    PROJECT_ROOT / "outputs/0003_20260912_control_absent_cluster_mining/cluster_mining"
)
DEFAULT_FEATURES_DIR = (
    PROJECT_ROOT
    / "outputs/0001_20260907_extract_wsi_embeddings/trident_job/20x_256px_0px_overlap/features_uni_v2"
)


def patch_size_level0(features_dir: Path, slide_id: str) -> int:
    """Side length in level-0 pixels of one patch (256 at 20x native, 512 at 40x)."""
    with h5py.File(features_dir / f"{slide_id}.h5", "r") as h:
        attrs = h["coords"].attrs
        return int(attrs.get("patch_size_level0", attrs.get("patch_size", 256)))


def describe_cluster(members: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Which slides contribute to this cluster, and what those slides are."""
    counts = members.groupby("slide_id", observed=True).size().rename("n_patches")
    out = manifest.merge(counts, left_on="slide_id", right_index=True)
    out["patch_frac"] = out["n_patches"] / out["n_patches"].sum()
    cols = [
        "slide_id",
        "compound_name",
        "dose_level",
        "sacrifice_period",
        "n_patches",
        "max_grade",
        "finding_types",
    ]
    return out.sort_values("n_patches", ascending=False)[cols]


def contact_sheet(
    members: pd.DataFrame,
    manifest: pd.DataFrame,
    features_dir: Path,
    n: int,
    grid_cols: int,
    rng: np.random.Generator,
):
    import openslide
    from PIL import Image

    svs_paths = dict(zip(manifest["slide_id"], manifest["svs_path"]))
    take = members.iloc[rng.choice(len(members), size=min(n, len(members)), replace=False)]

    tiles = []
    for slide_id, grp in take.groupby("slide_id", observed=True):
        size = patch_size_level0(features_dir, slide_id)
        slide = openslide.OpenSlide(svs_paths[slide_id])
        for row in grp.itertuples():
            tile = slide.read_region((int(row.x), int(row.y)), 0, (size, size)).convert("RGB")
            tiles.append(tile.resize((256, 256)))
        slide.close()

    if not tiles:
        return None
    rows = (len(tiles) + grid_cols - 1) // grid_cols
    sheet = Image.new("RGB", (grid_cols * 256, rows * 256), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % grid_cols) * 256, (i // grid_cols) * 256))
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--clusters", type=int, nargs="+", required=True)
    parser.add_argument("--n", type=int, default=36)
    parser.add_argument("--grid-cols", type=int, default=6)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir or (args.run_dir / f"exemplars_k{args.k}")
    out_dir.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_parquet(args.run_dir / "cluster_assignments.parquet")
    manifest = pd.read_parquet(args.run_dir / "slide_manifest.parquet")
    manifest["slide_id"] = manifest["slide_id"].astype(str)

    column = f"cluster_k{args.k}"
    if column not in assignments.columns:
        raise SystemExit(
            f"{column} is not in the assignments (have: "
            f"{[c for c in assignments.columns if c.startswith('cluster_k')]})"
        )

    for cluster_id in args.clusters:
        members = assignments[assignments[column] == cluster_id]
        if members.empty:
            print(f"cluster {cluster_id}: no patches")
            continue

        desc = describe_cluster(members, manifest)
        print(f"\n=== k={args.k} cluster {cluster_id}: {len(members)} patches "
              f"/ {len(desc)} slides / {desc['compound_name'].nunique()} compounds ===")
        print(desc.head(12).to_string(index=False))
        print(f"化合物内訳: {desc['compound_name'].value_counts().to_dict()}")
        print(f"用量内訳:   {desc['dose_level'].value_counts().to_dict()}")

        sheet = contact_sheet(
            members, manifest, args.features_dir, args.n, args.grid_cols, rng
        )
        if sheet is not None:
            path = out_dir / f"cluster_{cluster_id:05d}.png"
            sheet.save(path)
            print(f"-> {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
