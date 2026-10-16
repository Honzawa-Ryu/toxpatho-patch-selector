"""試験の塊ごとに、無所見Controlのpatchを無作為に並べる（同じ無所見でも見た目がどれだけ違うかの目視用）。

各塊から最大12試験を選び、各試験のControl slideを1枚、そのslideのpatchを1つ無作為に取る（座標は0016のsub2048）。
同じpatchを、元の画像（raw）と、0016と同じ参照へのMacenko正規化後（埋め込みに使った見た目）で2枚のシートに出す。
usage: .venv/bin/python scripts/block_patch_examples.py <exp_blocks.csv> [n_rows_per_block]
"""
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.stain import apply_macenko, fit_macenko, macenko_matrix  # noqa: E402

RUN = ROOT / "outputs/0015_20261012_ctrl_batch_structure/allctrl_macenko"
O16 = ROOT / "outputs/0016_20261013_tier1_macenko_embeddings"
SUB = O16 / "20x_256px_0px_overlap/features_uni_v2_macenko_sub2048"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
TILE, PER_ROW, HEAD = 192, 12, 26
SEED = 0

spec = importlib.util.spec_from_file_location("exp0005", ROOT / "experiments/0005_20260913_stain_normalized_embeddings/experiment.py")
exp0005 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp0005)


def main():
    blocks_csv = Path(sys.argv[1])
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    blocks = pd.read_csv(blocks_csv)
    slides = pd.read_csv(RUN / "slides.csv", dtype={"slide_id": str})
    ref = json.loads((O16 / "reference.json").read_text())
    he_ref, maxc_ref = np.asarray(ref["reference_he"]), np.asarray(ref["reference_max_c"])
    rng = np.random.default_rng(SEED)
    font = ImageFont.truetype(FONT, 12)
    order = blocks.block.value_counts().sort_index().index
    sheets = {k: Image.new("RGB", (PER_ROW * TILE, len(order) * n_rows * (TILE + HEAD) + 30), "white") for k in ["raw", "macenko"]}
    draws = {k: ImageDraw.Draw(v) for k, v in sheets.items()}
    rows = []
    for bi, b in enumerate(order):
        exps = blocks[blocks.block == b].exp_id.to_numpy()
        for r in range(n_rows):
            pick = rng.choice(exps, size=min(PER_ROW, len(exps)), replace=False)
            for j, e in enumerate(pick):
                sl = slides[slides.exp_id == e].sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
                with h5py.File(SUB / f"{sl.slide_id}.h5", "r") as h:
                    coords = h["coords"][:]
                    size = int(h["coords"].attrs.get("patch_size_level0", 256))
                xy = coords[rng.integers(len(coords))]
                w = openslide.OpenSlide(sl.svs_path)
                rgb = np.asarray(w.read_region((int(xy[0]), int(xy[1])), 0, (size, size)).convert("RGB"))
                w.close()
                fit = exp0005.sample_patches(sl.svs_path, coords, size, 40, np.random.default_rng([0, int(sl.slide_id)]))
                he, mc = fit_macenko(fit.reshape(-1, 3))
                norm = apply_macenko(rgb, macenko_matrix(he, mc, he_ref=he_ref, maxc_ref=maxc_ref))
                y = 30 + (bi * n_rows + r) * (TILE + HEAD)
                for k, img in [("raw", rgb), ("macenko", norm)]:
                    sheets[k].paste(Image.fromarray(img).resize((TILE, TILE)), (j * TILE, y + HEAD))
                    tag = f"[B{b}] " if j == 0 else ""
                    draws[k].text((j * TILE + 3, y + 2), f"{tag}e{e} {sl.compound_name[:14]}", fill="black", font=font)
                    draws[k].text((j * TILE + 3, y + 13), f"{sl.scanner} {sl.sacrifice_period}", fill="#555555", font=font)
                rows.append({"block": b, "exp_id": e, "slide_id": sl.slide_id, "x": int(xy[0]), "y": int(xy[1])})
            if r == 0:   # 塊の区切り線
                for d in draws.values():
                    d.line((0, y, PER_ROW * TILE, y), fill="#eb6834", width=3)
    out = blocks_csv.parent
    for k, sh in sheets.items():
        ImageDraw.Draw(sh).text((6, 6), f"No-finding Ctrl patches per block ({blocks_csv.parent.name}), {k}; "
                                f"{n_rows} rows per block, one random patch per experiment; blocks in order {list(order)}",
                                fill="black", font=ImageFont.truetype(FONT, 15))
        sh.save(out / f"block_patch_examples_{k}.jpg", quality=90)
    pd.DataFrame(rows).to_csv(out / "block_patch_examples.csv", index=False)
    print(out)


if __name__ == "__main__":
    main()
