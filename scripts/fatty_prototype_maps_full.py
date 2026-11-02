"""0035 の脂肪変性プロトタイプの slide 上の分布（小葉中心性か辺縁性か）を、slide の全 patch で描く（2026-11-02。2,048 patch の地図は疎で読めなかった）。

slide の全 patch を A128（Sphering128 ‖ LDA13）に写し、同じ slide の間引き 2,048 patch への 1 近傍で 0035 の割り当てを補間する（重心は保存していないため）。
  fatty_figures/map_<method>_<化合物>_slide<番号>.jpg : 左 原画像、中 純プロトタイプ（CCl4 側 橙、cholesterol 側 青）、右 その化合物側の上位 6 個を 1 つずつ色分け。題・説明・凡例（記録による名前）入り
対象 slide と純プロトタイプと名前は fatty_pure_prototypes.csv（scripts/fatty_prototype_sheets.py の出力）に合わせる。先にそちらを実行すること。
usage: python scripts/fatty_prototype_maps_full.py [版] [rep:method ...]  （コンテナ内、Slurm経由。CPU、10分程度）
"""
import importlib.util
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_RUN = sys.argv[1] if len(sys.argv) > 1 else "v1"
PAIRS = [a.split(":") for a in sys.argv[2:]] or [["A128_sphering_lda", "split_record"], ["A128_sphering_lda", "flat_k1000"]]
EXP = ROOT / "experiments/0035_20261102_finer_prototypes_split_rejoin"
RUN = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin" / _RUN
THUMB, D = 2400, 128
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"
METHOD_JP = {"split_record": "k=100 から記録で純度の低いクラスタを分けた 887 クラスタ", "flat_k1000": "k=1000 の k-means"}
GROUP_COL = {"CCl4": (230, 159, 0), "cholesterol": (0, 114, 178)}
PROTO_COL = [(230, 159, 0), (86, 180, 233), (0, 158, 115), (240, 228, 66), (0, 114, 178), (213, 94, 0)]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    import openslide
    import torch
    from PIL import Image, ImageDraw, ImageFont
    cfg = yaml.safe_load(open(EXP / "config.yml"))
    load_module(ROOT / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    ex23 = load_module(ROOT / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex26 = load_module(ROOT / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(ROOT / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    ex30 = load_module(ROOT / "experiments" / cfg["exp0030"] / "experiment.py", "exp0030")
    cfg20 = yaml.safe_load(open(ROOT / "experiments/0020_20261024_batch_robust_methods/config.yml"))
    pre_path = str(ROOT / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl")
    pre = pickle.loads(Path(pre_path).read_bytes())["pre"]
    OUTF = RUN / "fatty_figures"
    OUTF.mkdir(exist_ok=True)

    def h5_of(sid):
        for d_ in cfg["features_dirs"]:
            if (ROOT / d_ / f"{sid}.h5").exists():
                return str(ROOT / d_ / f"{sid}.h5")
        return ""

    # LDA（0035 と同じ: Bayer 230 枚、21 枚のラベルを除く）
    src = ROOT / "outputs" / cfg["exp0025"] / cfg["cache_from"]
    c = np.load(src / "cache.npz")
    K, laba, row_b = c["K"], c["lab"].astype(np.int64), c["row"]
    bayer = pd.read_csv(src / "slides.csv", dtype=str)
    keep = [k_ for k_ in np.unique(laba) if k_ >= 0 and (laba == k_).sum() >= cfg20["min_label_patches"]]
    lab = np.where(np.isin(laba, keep), np.searchsorted(keep, np.maximum(laba, 0)), -1)
    ex_ids = pd.read_csv(ROOT / cfg["lda_exclude_slides"], dtype=str).slide_id.tolist()
    lab = np.where(np.isin(bayer.slide_id.to_numpy()[row_b], ex_ids), -1, lab)
    tf = ex26.fit_combo(K, lab, cfg, cfg["seed"])
    L = len(keep) - 1
    nb = len(K)
    del K, c
    print("LDA fitted", flush=True)

    # 脂肪変性の slide、間引き patch（0035 と同じ順）、対照群（白色化用）
    tgt = ex26.target_slides({**cfg, "target_exps": cfg["fatty_exps"], "features_dir": cfg["features_dirs"][0]}, ROOT)
    tgt = tgt[~tgt.slide_id.isin(bayer.slide_id)].reset_index(drop=True)
    G, topo = ex26.fat_grades({**cfg, "fat_findings": ["Degeneration, fatty", "Vacuolization, cytoplasmic", "Deposit, lipid", "Degeneration, vacuolar"]}, ROOT, tgt)
    fat = G["Degeneration, fatty"].to_numpy()
    topo_fat = topo["Degeneration, fatty"].reindex(tgt.slide_id).fillna("").to_numpy() if "Degeneration, fatty" in topo else np.full(len(tgt), "")
    cmpd = np.where(tgt.compound.str.contains("tetrachloride", case=False), "CCl4", np.where(tgt.compound.str.contains("cholesterol", case=False), "cholesterol", "other"))
    with ProcessPoolExecutor(max_workers=8, initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts = list(ex.map(ex30._load_sub, tgt.h5, chunksize=4))
    off = np.cumsum([0] + [len(p_[0]) for p_ in parts])
    nf = int(off[-1])
    q = pd.read_csv(cfg["qc_catalog"], dtype=str)
    qc = q[q.EXP_ID.notna() & (q.EXP_ID != "") & (q.DOSE_LEVEL == "Control") & q.EXP_ID.astype(int).isin(cfg["fatty_exps"])].copy()
    qc["h5"] = [h5_of(x) for x in qc.slide_id]
    qc = qc[qc.h5 != ""]
    with ProcessPoolExecutor(max_workers=8, initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts_c = list(ex.map(ex27._load_ctrl, qc.h5, chunksize=4))
    Kc = np.concatenate(parts_c)[:, :D].astype(np.float32)
    exp_c = np.concatenate([np.full(len(p_), int(e)) for p_, e in zip(parts_c, qc.EXP_ID)])
    del parts_c
    print(f"fatty slides {len(tgt)}, controls {len(qc)}", flush=True)

    def rep_of(Kx, e):
        U = np.ascontiguousarray(Kx[:, :D]).astype(np.float32)
        Sw = ex27.sphering(U, np.full(len(U), e), Kc, exp_c) if (exp_c == e).any() else ex27.sphering(U, np.full(len(U), -999), Kc, np.full(len(Kc), -999))
        Lw = tf(Kx)[:, :L].astype(np.float32)
        return Sw, Lw

    pure_tab = pd.read_csv(RUN / "fatty_pure_prototypes.csv")
    fjt, fj = ImageFont.truetype(FONT_JP, 26), ImageFont.truetype(FONT_JP, 18)
    tgt["svs_ok"] = [Path(p).exists() for p in tgt.svs_path]
    for rep, method in PAIRS:
        cl_f = np.load(RUN / f"assign_{rep}_{method}.npy").astype(np.int64)[nb:nb + nf]
        pure = {cn: pure_tab[(pure_tab.rep == rep) & (pure_tab.method == method) & (pure_tab.compound == cn)].prototype.tolist() for cn in GROUP_COL}
        for cname in GROUP_COL:
            cand_s = np.where((cmpd == cname) & (fat > 0) & tgt.svs_ok.to_numpy())[0]
            cand_s = cand_s[np.argsort(-fat[cand_s])][:2]
            for i in cand_s:
                r = tgt.iloc[i]
                with h5py.File(r.h5, "r") as h:
                    f_ = h["features"][:].astype(np.float32)
                    xy = h["coords"][:]
                    s0 = int(h["coords"].attrs["patch_size_level0"])
                Kf = pre.transform(ex23.slide_rank(f_)).astype(np.float32)
                del f_
                Sw, Lw = rep_of(Kf, int(r.exp))
                Ks = parts[i][0]
                Sw_s, Lw_s = rep_of(Ks, int(r.exp))
                fac = float(Sw_s.std(0).mean() / Lw_s.std(0).mean())
                Xf = torch.from_numpy(np.concatenate([Sw, fac * Lw], 1))
                Xs = torch.from_numpy(np.concatenate([Sw_s, fac * Lw_s], 1))
                nn = torch.cat([torch.cdist(x, Xs).argmin(1) for x in Xf.split(4096)]).numpy()
                cl_full = cl_f[off[i] + nn]
                # 描画: 組織の範囲に切り出したサムネイル
                w = openslide.OpenSlide(r.svs_path)
                th = w.get_thumbnail((THUMB, THUMB)).convert("RGB")
                scl = th.width / w.dimensions[0]
                w.close()
                x0, y0 = xy.min(0) * scl
                x1, y1 = (xy.max(0) + s0) * scl
                box = (int(max(0, x0 - 10)), int(max(0, y0 - 10)), int(min(th.width, x1 + 10)), int(min(th.height, y1 + 10)))
                th = th.crop(box)
                b_ = max(2, round(s0 * scl))
                gray = th.convert("L").convert("RGBA")
                panels = [th]
                for mode in ("group", "proto"):
                    ov = Image.new("RGBA", th.size, (0, 0, 0, 0))
                    d = ImageDraw.Draw(ov)
                    for (px, py), j in zip(xy, cl_full):
                        col = None
                        if mode == "group":
                            for cn, cc in GROUP_COL.items():
                                if j in pure[cn]:
                                    col = cc
                        else:
                            if j in pure[cname]:
                                col = PROTO_COL[pure[cname].index(j) % len(PROTO_COL)]
                        if col is not None:
                            X0, Y0 = px * scl - box[0], py * scl - box[1]
                            d.rectangle([X0, Y0, X0 + b_, Y0 + b_], fill=col + (235,))
                    panels.append(Image.alpha_composite(gray, ov).convert("RGB"))
                n_in = {cn: int(np.isin(cl_full, pure[cn]).sum()) for cn in GROUP_COL}
                # 説明: 題、何の図か、パネルごとの見出し、右パネルの色の凡例（プロトタイプ番号と記録による名前）
                PH = 34
                lines = [(f"脂肪変性プロトタイプの slide 上の分布: slide {r.slide_id}（{r.compound}、{r.dose}、{r.period}、exp {r.exp}）", fjt),
                         (f"この slide の記録: 脂肪変性 grade {int(fat[i])}、部位 {topo_fat[i] or '-'}。slide の全 {len(cl_full):,} patch を、0035 の間引き patch への 1 近傍で、"
                          f"0035 のプロトタイプ（表現 A128、{METHOD_JP.get(method, method)}）に割り当てた。", fj),
                         ("色を塗った patch は、tiles_*.jpg に並べた「化合物内で脂肪変性ありの slide 由来が 90% 以上」の純プロトタイプに入ったもの。四角 1 つが 1 patch（約 128 µm 四方）。", fj)]
                hdr = 12 + sum(30 if f_ is fjt else 24 for _, f_ in lines) + 8
                leg = pure_tab[(pure_tab.rep == rep) & (pure_tab.method == method) & (pure_tab.compound == cname)].set_index("prototype")
                leg_lines = [f"p{p:04d}: {leg.record_name.get(p, '')}（{int(leg.n.get(p, 0)):,} patch）" for p in pure[cname][:len(PROTO_COL)]]
                foot = 10 + 24 * len(leg_lines)
                canvas = Image.new("RGB", (th.width * 3 + 20, hdr + PH + th.height + foot), "white")
                dd = ImageDraw.Draw(canvas)
                yy = 8
                for t_, f_ in lines:
                    dd.text((8, yy), t_, fill="black" if f_ is fjt else "#333333", font=f_)
                    yy += 30 if f_ is fjt else 24
                titles = ["原画像（H&E）",
                          f"純プロトタイプの位置: 橙 = CCl4 側（{n_in['CCl4']:,} patch）、青 = cholesterol 食側（{n_in['cholesterol']:,} patch）",
                          f"{cname} 側の純プロトタイプを 1 つずつ色分け（凡例は下）"]
                for j_, (pn, tt) in enumerate(zip(panels, titles)):
                    x_ = j_ * (th.width + 10)
                    dd.text((x_ + 6, hdr + 4), tt, fill="black", font=fj)
                    canvas.paste(pn, (x_, hdr + PH))
                x_ = 2 * (th.width + 10)
                for k_, (p, txt) in enumerate(zip(pure[cname][:len(PROTO_COL)], leg_lines)):
                    yb = hdr + PH + th.height + 8 + 24 * k_
                    dd.rectangle([x_ + 6, yb + 4, x_ + 22, yb + 20], fill=PROTO_COL[k_])
                    dd.text((x_ + 30, yb), txt, fill="black", font=fj)
                canvas.save(OUTF / f"map_{method}_{cname}_slide{r.slide_id}.jpg", quality=85)
                print(f"{rep} {method} {r.slide_id}: {len(cl_full):,} patches, in pure {n_in}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
