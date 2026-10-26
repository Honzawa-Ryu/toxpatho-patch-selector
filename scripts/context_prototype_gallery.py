"""0021 explore_v2 の、隣接patchの情報を入れたプロトタイプ（k=30、アノテーション132枚の全patch）を一覧のページにする。

比べる版（explore_v2の設定と同じ。表現Zはcontext_build_v2のキャッシュ＝combo）:
  self_only   : 自分だけ（隣接情報なし、132枚で作り直したもの）
  w05_h4      : 自分×0.5 ＋ 周りを4回伝播した平均×1.0
  ctx_only_h4 : 周りを4回伝播した平均だけ
各プロトタイプについて、重心（その版の表現での平均）に近い順に、同じslideからは1枚だけ選んだ24枚を切り出す。
出力: outputs/prototype_gallery_context_20261026/（img/、prototypes_<版>.csv、index.htmlはgallery作成の別処理）

usage: python scripts/context_prototype_gallery.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SRC = ROOT / "outputs/0021_20261025_patch_prototypes"
OUT = ROOT / "outputs/prototype_gallery_context_20261026"
VARIANTS = {"self_only": (1.0, 0.0), "w05_h4": (0.5, 1.0), "ctx_only_h4": (0.0, 1.0)}   # (自分の重み, 周りの重み)
HOPS, ALPHA, TILES = 4, 0.5, 24


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    ex21 = load_module(ROOT / "experiments/0021_20261025_patch_prototypes/experiment.py", "exp0021")
    ex17 = load_module(ROOT / "experiments/0017_20261016_block_consensus_patterns/experiment.py", "exp0017")
    cfg20 = yaml.safe_load(open(ROOT / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    prio = cfg20["label_priority"]
    c = np.load(SRC / "context_build_v2/context_cache.npz")
    Z, lab, row, xy, pairs = c["Z"], c["lab"], c["row"], c["xy"], c["pairs"]
    lab = np.where(np.isin(lab, [k for k in np.unique(lab) if k >= 0 and (lab == k).sum() >= cfg20["min_label_patches"]]), lab, -1)
    slides = pd.read_csv(SRC / "v3_maps/slides.csv", dtype={"slide_id": str})
    slides = slides[slides.source == "annotated"].reset_index(drop=True)
    size0 = dict(zip(slides.slide_id, c["size0"].tolist()))
    ctx = ex21.contextualize(Z, pairs, HOPS, ALPHA)
    is_ctrl = (slides.dose.to_numpy() == "Control")[row]
    exp = slides.exp.astype(str).to_numpy()[row]
    cmp_ = slides.compound.to_numpy()[row]
    (OUT / "img").mkdir(parents=True, exist_ok=True)
    for v, (sw, cw) in VARIANTS.items():
        cl = np.load(SRC / f"explore_v2/assign_{v}.npy")
        X = np.concatenate([Z * sw, ctx * cw], axis=1)
        src_cl = cl[pairs[:, 0]]
        same = np.bincount(src_cl, weights=(src_cl == cl[pairs[:, 1]]).astype(float), minlength=cl.max() + 1) / np.maximum(np.bincount(src_cl, minlength=cl.max() + 1), 1)
        recs = []
        for k in range(int(cl.max()) + 1):
            mk = np.flatnonzero(cl == k)
            d = np.linalg.norm(X[mk] - X[mk].mean(0), axis=1)
            picks, seen = [], set()
            for p in mk[np.argsort(d)]:
                if row[p] not in seen:
                    picks.append(int(p))
                    seen.add(row[p])
                if len(picks) == TILES:
                    break
            tiles = ex17.read_tiles(picks, row, xy, slides, size0,
                                    lambda s_, p: (f"e{s_.exp} {str(s_.compound)[:13]} {str(s_.dose)[0]}", f"{s_.period} {prio[lab[p]] if lab[p] >= 0 else ''}"))
            ex17.draw_sheet("", [("", tiles)], OUT / "img" / f"{v}_{k:02d}.jpg")
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


if __name__ == "__main__":
    main()
