"""0025の探索で選んだ隣接情報の入れ方（best_context）と、隣接情報なし（self）のプロトタイプを、画像シートとslideの地図にする。

対象: Bayerのポリゴンがある230枚の全patch（0025 cache_v1）。割り当ては search_v3 の assign_<版>.npy（k=30、繰り返し0回目）。
画像: 各プロトタイプで、土台の表現（OOFのcombo）での平均に近い順に、同じslideからは1枚だけ選んだ24枚。
  best_contextの表現には周りの情報も入っているが、画像の選び方は両方とも土台の表現でそろえる。
地図: 選んだslideで、元画像・self・best_contextを並べる。色は版ごとに独立（番号は版の間で対応しない）。
出力: outputs/prototype_gallery_search_20261026/（img/, maps/, prototypes_<版>.csv）

usage: python scripts/context_search_gallery.py
"""
import colorsys
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "outputs/0025_20261026_context_mixing_search"
OUT = ROOT / "outputs/prototype_gallery_search_20261026"
VERSIONS = ["self", "best_context"]
MAP_SLIDES = ["25661", "52750", "63530", "13607"]
TILES, THUMB = 24, 1400


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def palette(k):
    return [tuple(int(255 * c) for c in colorsys.hsv_to_rgb((i * 0.618) % 1, 0.65 + 0.3 * (i % 2), 0.95 - 0.25 * ((i // 2) % 2))) for i in range(k)]


def main():
    import openslide
    ex17 = load_module(ROOT / "experiments/0017_20261016_block_consensus_patterns/experiment.py", "exp0017")
    cfg20 = yaml.safe_load(open(ROOT / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    prio = cfg20["label_priority"]
    c = np.load(SRC / "cache_v1/cache.npz")
    lab, row, xy = c["lab"].astype(int), c["row"], c["xy"]
    lab = np.where(np.isin(lab, [k for k in np.unique(lab) if k >= 0 and (lab == k).sum() >= cfg20["min_label_patches"]]), lab, -1)
    slides = pd.read_csv(SRC / "cache_v1/slides.csv", dtype=str).rename(
        columns={"EXP_ID": "exp", "COMPOUND_NAME": "compound", "DOSE_LEVEL": "dose", "SACRIFICE_PERIOD": "period"})
    # cacheのsvs_pathは時点のフォルダから作った推定。実際の場所は0021と追加抽出の一覧にある（tggate-bench側にあるslideなど）
    known = {**pd.read_csv(ROOT / "outputs/0021_20261025_patch_prototypes/v3_maps/slides.csv", dtype=str).set_index("slide_id").svs_path.to_dict(),
             **pd.read_csv(ROOT / "outputs/target_slide_list/bayer_annotated_all_missing.csv", dtype=str).set_index("slide_id").svs_path.to_dict()}
    slides["svs_path"] = [known.get(s_, p_) for s_, p_ in zip(slides.slide_id, slides.svs_path)]
    size0 = dict(zip(slides.slide_id, c["size0"].tolist()))
    Z = np.load(SRC / "z_cache/Z_oof_rd16_rw0.5.npy")
    picks = json.loads((SRC / "search_v3/picks.json").read_text())
    is_ctrl = (slides.dose.to_numpy() == "Control")[row]
    exp = slides.exp.to_numpy()[row]
    cmp_ = slides.compound.to_numpy()[row]
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    (OUT / "maps").mkdir(exist_ok=True)
    pairs = c["pairs"]
    assigns = {}
    for v in VERSIONS:
        cl = np.load(SRC / f"search_v3/assign_{v}.npy").astype(int)
        assigns[v] = cl
        same = np.bincount(cl[pairs[:, 0]], weights=(cl[pairs[:, 0]] == cl[pairs[:, 1]]).astype(float), minlength=cl.max() + 1) / np.maximum(np.bincount(cl[pairs[:, 0]], minlength=cl.max() + 1), 1)
        recs = []
        for k in range(cl.max() + 1):
            mk = np.flatnonzero(cl == k)
            d = np.linalg.norm(Z[mk] - Z[mk].mean(0), axis=1)
            sel, seen = [], set()
            for p in mk[np.argsort(d)]:
                if row[p] not in seen:
                    sel.append(int(p))
                    seen.add(row[p])
                if len(sel) == TILES:
                    break
            tiles = ex17.read_tiles(sel, row, xy, slides, size0,
                                    lambda s_, p: (f"e{s_.exp} {str(s_.compound)[:13]} {str(s_.dose)[0]}", f"{s_.period} {prio[lab[p]] if lab[p] >= 0 else ''}"))
            ex17.draw_sheet("", [("", tiles)], OUT / "img" / f"{v}_{k:02d}.jpg")
            im = Image.open(OUT / "img" / f"{v}_{k:02d}.jpg")
            im.crop((0, 54, im.width, im.height)).save(OUT / "img" / f"{v}_{k:02d}.jpg", quality=82)
            lm = lab[mk] >= 0
            lc = pd.Series(lab[mk][lm]).value_counts()
            ec = pd.Series(exp[mk]).value_counts()
            recs.append({"cluster": k, "n": len(mk), "n_slides": len(np.unique(row[mk])), "n_exp": len(ec),
                         "top_exp_compound": cmp_[mk][exp[mk] == ec.index[0]][0], "top_exp_share": ec.iloc[0] / len(mk),
                         "ctrl_share": float(is_ctrl[mk].mean()), "n_labeled": int(lm.sum()),
                         "top_label": prio[lc.index[0]] if len(lc) else "", "top_label_purity": float(lc.iloc[0] / lm.sum()) if len(lc) else np.nan,
                         "neighbor_same": float(same[k])})
            print(v, k, flush=True)
        pd.DataFrame(recs).to_csv(OUT / f"prototypes_{v}.csv", index=False)
    # 地図
    for sid in MAP_SLIDES:
        si = int(np.flatnonzero(slides.slide_id.to_numpy() == sid)[0])
        r = slides.iloc[si]
        w = openslide.OpenSlide(r.svs_path)
        th = w.get_thumbnail((THUMB, THUMB)).convert("RGB")
        sc = th.width / w.dimensions[0]
        mk = np.flatnonzero(row == si)
        b = max(2, round(size0[sid] * sc))
        panels = [th]
        for v in VERSIONS:
            pal = palette(int(assigns[v].max()) + 1)
            ov = Image.new("RGBA", th.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(ov)
            for (x0, y0), k in zip(xy[mk], assigns[v][mk]):
                d.rectangle([x0 * sc, y0 * sc, x0 * sc + b, y0 * sc + b], fill=pal[k] + (230,))
            panels.append(Image.alpha_composite(th.convert("L").convert("RGBA"), ov).convert("RGB"))
        W = sum(p.width for p in panels) + 10 * (len(panels) - 1)
        canvas = Image.new("RGB", (W, th.height), "white")
        x = 0
        for p in panels:
            canvas.paste(p, (x, 0))
            x += p.width + 10
        canvas.save(OUT / "maps" / f"{sid}.jpg", quality=85)
        print("map", sid, r.compound, r.dose, flush=True)
    (OUT / "picks.json").write_text(json.dumps(picks))


if __name__ == "__main__":
    main()
