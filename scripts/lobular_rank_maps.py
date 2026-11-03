"""0036 の小葉内の位置（回帰の slide 内の順位）を、脂肪変性の純プロトタイプと並べて slide 上に描く（2026-11-04）。
0036 の出力（arrays.npz、lobular_rank.npy、slides.csv）と 0035 の純プロトタイプ（fatty_figures/fatty_pure_prototypes.csv）を読むだけで、表現は作り直さない。

出力: outputs/0036_.../<版>/figures/F7_lobular_map_slide<番号>.jpg
  左 原画像、中 小葉内の位置（青 = 中心静脈側、橙 = 門脈域側。slide 内の順位）、右 脂肪変性の純プロトタイプ（橙 CCl4 側、青 cholesterol 食側）
usage: python scripts/lobular_rank_maps.py [版] [slide_id ...]   （コンテナ内、Slurm経由。CPU、数分）
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/0036_20261103_local_prototype_context" / (sys.argv[1] if len(sys.argv) > 1 else "v5")
R35 = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin/v1"
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"
THUMB = 2400


def main():
    import openslide
    from PIL import Image, ImageDraw, ImageFont
    a = np.load(RUN / "arrays.npz")
    qf, row, xy = a["qf"].astype(np.int64), a["row"].astype(np.int64), a["xy"]
    zl = np.load(RUN / "lobular_rank.npy")
    sl = pd.read_csv(RUN / "slides.csv", dtype={"slide_id": str})
    pure = pd.read_csv(R35 / "fatty_figures" / "fatty_pure_prototypes.csv")
    pure = pure[(pure.rep == "A128_sphering_lda") & (pure.method == "split_record")]
    pset = {cn: set(pure.prototype[pure.compound == cn]) for cn in ("CCl4", "cholesterol")}
    starts = np.r_[0, np.cumsum(np.bincount(row, minlength=len(sl)))]
    want = sys.argv[2:]
    if not want:
        for cn, g in (("CCl4", 3), ("CCl4", 2), ("cholesterol", 2), ("cholesterol", 4)):
            c_ = sl[(sl.cmpd == cn) & (sl.fatty_grade == g) & (sl.set == "fatty")]
            want += list(c_.slide_id[:1])
        want += list(sl[(sl.set == "fatty") & sl.is_ctrl & (sl.exp == 67)].slide_id[:1])
    cmap = matplotlib.colormaps["PuOr_r"]   # 0 = 紫（中心静脈側）、1 = 橙（門脈域側）。発散型、中央 0.5 は白
    col_pure = {"CCl4": (230, 159, 0), "cholesterol": (0, 114, 178)}
    fj, fjt = ImageFont.truetype(FONT_JP, 26), ImageFont.truetype(FONT_JP, 34)
    for sid in want:
        i = int(sl.index[sl.slide_id == sid][0])
        r = sl.iloc[i]
        a_, b_ = starts[i], starts[i + 1]
        w = openslide.OpenSlide(r.svs)
        th = w.get_thumbnail((THUMB, THUMB)).convert("RGB")
        scl = th.width / w.dimensions[0]
        w.close()
        x0, y0 = xy[a_:b_].min(0) * scl
        x1, y1 = (xy[a_:b_].max(0) + int(r.size0)) * scl
        box = (int(max(0, x0 - 10)), int(max(0, y0 - 10)), int(min(th.width, x1 + 10)), int(min(th.height, y1 + 10)))
        th = th.crop(box)
        bpx = max(2, round(int(r.size0) * scl))
        gray = th.convert("L").convert("RGBA")
        panels = [th]
        for mode in ("lobular", "pure"):
            ov = Image.new("RGBA", th.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            for (px, py), v, q in zip(xy[a_:b_], zl[a_:b_], qf[a_:b_]):
                if mode == "lobular":
                    if not np.isfinite(v):
                        continue
                    c = tuple(int(255 * u) for u in cmap(v)[:3])
                else:
                    c = next((col_pure[cn] for cn, ps in pset.items() if q in ps), None)
                    if c is None:
                        continue
                X0, Y0 = px * scl - box[0], py * scl - box[1]
                d.rectangle([X0, Y0, X0 + bpx, Y0 + bpx], fill=c + (225,))
            panels.append(Image.alpha_composite(gray, ov).convert("RGB"))
        hdr, ph = 120, 44
        canvas = Image.new("RGB", (th.width * 3 + 20, hdr + ph + th.height), "white")
        dd = ImageDraw.Draw(canvas)
        dd.text((8, 6), f"slide {r.slide_id}（{r.compound}、exp {r.exp}{'、対照群' if r.is_ctrl else ''}）　脂肪変性 grade {int(r.fatty_grade)} {r.fatty_topo if isinstance(r.fatty_topo, str) else ''}", fill="black", font=fjt)
        dd.text((8, 52), "中: 小葉内の位置の推定（Bayer の中心静脈・門脈域ポリゴンで学んだ回帰を、周囲の patch だけから。脂肪変性・空胞化のプロトタイプは周囲から除外）の slide 内の順位。", fill="#333333", font=fj)
        dd.text((8, 84), "紫 = 中心静脈側、白 = 中間、橙 = 門脈域側。右: 脂肪変性の純プロトタイプ（橙 = CCl4 側、青 = cholesterol 食側。0035 の目視で確かめた 10 個）。", fill="#333333", font=fj)
        for j, (pn, tt) in enumerate(zip(panels, ["原画像（H&E）", "小葉内の位置の推定", "脂肪変性の純プロトタイプ"])):
            canvas.paste(pn, (j * (th.width + 10), hdr + ph))
            dd.text((j * (th.width + 10) + 8, hdr + 6), tt, fill="black", font=fj)
        canvas.save(RUN / "figures" / f"F7_lobular_map_slide{sid}.jpg", quality=85)
        print(f"map {sid}", flush=True)


if __name__ == "__main__":
    main()
