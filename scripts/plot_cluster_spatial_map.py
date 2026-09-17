"""Color-code a WSI thumbnail by which k=100 cluster each patch was assigned
to, so the spatial (zonal) pattern behind a cluster can be checked against
the actual tissue instead of just contact-sheet exemplars.

Built for the CCl4-vs-cholesterol `Degeneration, fatty` comparison in
experiments/0003_20260912_control_absent_cluster_mining/config_ccl4.yml:
does the fatty-degeneration cluster set for each compound actually sit where
its known lobular distribution says it should (CCl4 = centrilobular,
cholesterol = peripheral)?

Usage: python scripts/plot_cluster_spatial_map.py
"""

from pathlib import Path

import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = PROJECT_ROOT / "outputs/0003_20260912_control_absent_cluster_mining/cluster_mining_ccl4"
OUT_DIR = RUN_DIR / "spatial_maps"
PATCH_SIZE_LEVEL0 = 256  # config.yml: patch_size, same for 0001 and 0007
THUMB_MAX_SIDE = 1800

# tab-style qualitative colors, one per cluster, kept distinct from each other
CCL4_CLUSTERS = {4: "#c0392b", 5: "#e67e22", 14: "#d35400", 52: "#f39c12", 62: "#a04000", 67: "#cb4335"}
CHOL_CLUSTERS = {96: "#1f618d", 39: "#2980b9", 61: "#17a589", 53: "#5dade2", 42: "#148f77"}
# WY-14643: Degeneration, granular, eosinophilic + Hypertrophy
WY_CLUSTERS = {0: "#6c3483", 77: "#8e44ad", 55: "#a569bd", 25: "#5b2c6f"}

SLIDES = [
    # 高用量スライドはfatty系クラスタが組織のほぼ全域を覆ってしまい小葉内分布が
    # 見えないため、CCl4はより軽度な低用量スライドを選ぶ。cholesterolは
    # fatty陽性がHigh用量にしか記録されていないため、その中で被覆率が
    # 低めのものを選ぶ。
    {"slide_id": "26647", "label": "CCl4, 29 day, High (被覆率97%・重度)", "clusters": CCL4_CLUSTERS},
    {"slide_id": "26461", "label": "CCl4, 4 day, Low (被覆率26%・軽度)", "clusters": CCL4_CLUSTERS},
    {"slide_id": "66207", "label": "cholesterol+cholate, 29 day, High (被覆率81%)", "clusters": CHOL_CLUSTERS},
    {"slide_id": "66179", "label": "cholesterol+cholate, 15 day, High (被覆率56%)", "clusters": CHOL_CLUSTERS},
    # control / 別所見での確認用（同じ6クラスタ色分けで、control群やWY-14643の
    # Degeneration, granular, eosinophilic + Hypertrophy でどう見えるか）
    {"slide_id": "26594", "label": "CCl4 Control, 29 day (同clusterでの被覆率7.2%・最大値)", "clusters": CCL4_CLUSTERS},
    {"slide_id": "29647", "label": "WY-14643, 8 day, High (被覆率91%)", "clusters": WY_CLUSTERS},
    {"slide_id": "29514", "label": "WY-14643 Control, 15 day", "clusters": WY_CLUSTERS},
]


def _distinct_palette(n: int) -> dict[int, str]:
    """n visually-separated colors via evenly spaced hue, alternating value/
    saturation every third step so neighboring cluster ids don't blur together."""
    import colorsys

    palette = {}
    for i in range(n):
        hue = (i * 0.6180339887) % 1.0  # golden-ratio spacing, avoids near-repeats
        sat = 0.55 + 0.35 * ((i % 3) / 2)
        val = 0.55 + 0.35 * (((i + 1) % 3) / 2)
        r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
        palette[i] = "#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255))
    return palette


def render(slide_id: str, label: str, clusters: dict[int, str], *, full: bool = False) -> Path:
    manifest = pd.read_parquet(RUN_DIR.parent / "manifest_219_plus_ccl4.parquet")
    manifest["slide_id"] = manifest["slide_id"].astype(str)
    svs_path = manifest.loc[manifest["slide_id"] == slide_id, "svs_path"].iloc[0]

    assign = pd.read_parquet(RUN_DIR / "cluster_assignments.parquet")
    assign["slide_id"] = assign["slide_id"].astype(str)
    patches = assign.loc[assign["slide_id"] == slide_id, ["x", "y", "cluster_k100"]]

    slide = openslide.OpenSlide(svs_path)
    w0, h0 = slide.dimensions
    thumb = slide.get_thumbnail((THUMB_MAX_SIDE, THUMB_MAX_SIDE)).convert("L").convert("RGB")
    slide.close()
    scale = thumb.width / w0

    draw = ImageDraw.Draw(thumb, "RGBA")
    box = max(2, round(PATCH_SIZE_LEVEL0 * scale))
    n_highlighted = 0
    for x, y, c in patches.itertuples(index=False):
        color = clusters.get(int(c))
        if color is None:
            continue
        n_highlighted += 1
        px, py = x * scale, y * scale
        draw.rectangle([px, py, px + box, py + box], fill=color)

    # 全クラスタ版は100色の凡例を並べると図が読めなくなるので省略し、代わりに
    # 出現したクラスタ上位のみ別途テキストで出す（呼び出し側でprint）。
    legend_h = 0 if full else 26
    canvas = Image.new("RGB", (thumb.width, thumb.height + legend_h), "white")
    canvas.paste(thumb, (0, 0))
    if not full:
        ld = ImageDraw.Draw(canvas)
        lx = 6
        for cluster_id, color in clusters.items():
            ld.rectangle([lx, thumb.height + 6, lx + 14, thumb.height + 20], fill=color)
            ld.text((lx + 18, thumb.height + 6), str(cluster_id), fill="black")
            lx += 46

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_allclusters" if full else ""
    out_path = OUT_DIR / f"{slide_id}_cluster_map{suffix}.png"
    canvas.save(out_path)
    print(f"{slide_id} ({label}): {n_highlighted}/{len(patches)} patches highlighted -> {out_path}")
    if full:
        top = patches["cluster_k100"].value_counts().head(10)
        print("  top clusters by patch count:", top.to_dict())
    return out_path


FULL_PALETTE = _distinct_palette(100)

FULL_SLIDES = [
    {"slide_id": "26594", "label": "CCl4 Control, 29 day, k=100全クラスタ"},
]


def main() -> None:
    for s in SLIDES:
        render(s["slide_id"], s["label"], s["clusters"])
    for s in FULL_SLIDES:
        render(s["slide_id"], s["label"], FULL_PALETTE, full=True)


if __name__ == "__main__":
    main()
