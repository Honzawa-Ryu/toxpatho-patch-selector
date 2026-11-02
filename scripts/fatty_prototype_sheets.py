"""0035 の脂肪変性プロトタイプを目で確かめる（2026-11-02ユーザー: 本当に脂肪変性か、CCl4（小葉中心性・大滴）と cholesterol 食（辺縁性・小滴）が分かれているか）。

0035 v1 の割り当て（assign_<rep>_<method>.npy）のうち脂肪変性の 3 試験（188枚、各 2,048 patch）の部分を使い、
  tiles/<rep>_<method>_<化合物>.jpg : 化合物内の純度（脂肪変性ありの slide 由来の割合）が 0.9 以上のプロトタイプを大きい順に、各 12 枚のタイル（別々の slide を優先）
  maps/<rep>_<method>_<slide>.jpg   : 脂肪変性の程度が高い slide（CCl4 2枚、cholesterol 2枚）のサムネイルに、CCl4 側の純プロトタイプ（橙）と cholesterol 側（青）の patch を重ねる
  fatty_pure_prototypes.csv          : 選んだプロトタイプの一覧（n、純度、程度別の内訳、Bayer での名前）
patch の座標は 0030 の _load_sub（slide ごとに決まる間引き）で読み直す。地図は 2,048 patch の位置だけなので疎。
usage: python scripts/fatty_prototype_sheets.py [版] [rep:method ...]  （コンテナ内、Slurm経由。既定は v1 と A128_sphering_lda:split_record A128_sphering_lda:flat_k1000）
"""
import importlib.util
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_RUN = sys.argv[1] if len(sys.argv) > 1 else "v1"
PAIRS = [a.split(":") for a in sys.argv[2:]] or [["A128_sphering_lda", "split_record"], ["A128_sphering_lda", "flat_k1000"]]
EXP = ROOT / "experiments/0035_20261102_finer_prototypes_split_rejoin"
RUN = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin" / _RUN
TILES_PER_PROTO, MAX_PROTOS, MIN_N, PURITY, THUMB = 12, 6, 200, 0.9, 1200
COL = {"CCl4": (230, 159, 0), "cholesterol": (0, 114, 178)}   # Okabe-Ito 橙・青


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    import openslide
    from PIL import Image, ImageDraw, ImageFont
    cfg = yaml.safe_load(open(EXP / "config.yml"))
    load_module(ROOT / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    load_module(ROOT / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex17 = load_module(ROOT / "experiments/0017_20261016_block_consensus_patterns/experiment.py", "exp0017")
    ex26 = load_module(ROOT / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(ROOT / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    ex30 = load_module(ROOT / "experiments" / cfg["exp0030"] / "experiment.py", "exp0030")
    pre_path = str(ROOT / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl")
    (RUN / "tiles").mkdir(exist_ok=True)
    (RUN / "maps").mkdir(exist_ok=True)

    # 脂肪変性の 3 試験の slide と patch（0035 と同じ順）。Bayer の patch 数 nb の後ろに並ぶ
    bayer = pd.read_csv(ROOT / "outputs" / cfg["exp0025"] / cfg["cache_from"] / "slides.csv", dtype=str)
    nb = len(np.load(ROOT / "outputs" / cfg["exp0025"] / cfg["cache_from"] / "cache.npz")["row"])
    tgt = ex26.target_slides({**cfg, "target_exps": cfg["fatty_exps"], "features_dir": cfg["features_dirs"][0]}, ROOT)
    tgt = tgt[~tgt.slide_id.isin(bayer.slide_id)].reset_index(drop=True)
    G, topo = ex26.fat_grades({**cfg, "fat_findings": ["Degeneration, fatty", "Vacuolization, cytoplasmic", "Deposit, lipid", "Degeneration, vacuolar"]}, ROOT, tgt)
    fat = G["Degeneration, fatty"].to_numpy()
    cmpd = np.where(tgt.compound.str.contains("tetrachloride", case=False), "CCl4", np.where(tgt.compound.str.contains("cholesterol", case=False), "cholesterol", "other"))
    with ProcessPoolExecutor(max_workers=8, initializer=ex27._init_ctrl, initargs=(pre_path, cfg["patches_per_slide"])) as ex:
        parts = list(ex.map(ex30._load_sub, tgt.h5, chunksize=4))
    row = np.concatenate([np.full(len(p_[0]), i) for i, p_ in enumerate(parts)])
    xy = np.concatenate([p_[1] for p_ in parts])
    size0 = {sid: p_[2] for sid, p_ in zip(tgt.slide_id, parts)}
    nf = len(row)
    del parts
    tgt["svs_ok"] = [Path(p).exists() for p in tgt.svs_path]
    print(f"fatty slides {len(tgt)} (svs found {int(tgt.svs_ok.sum())}), patches {nf:,}; fatty > 0: {int((fat > 0).sum())}", flush=True)
    topo_fat = topo["Degeneration, fatty"].reindex(tgt.slide_id).fillna("").to_numpy() if "Degeneration, fatty" in topo else np.full(len(tgt), "")
    sdf = tgt[["slide_id", "svs_path", "compound", "dose", "period", "exp"]].copy()
    lbl = lambda s_, q_: (f"{s_.slide_id} e{s_.exp} {s_.dose}", f"fatty grade {int(fat[row[q_]])} {str(topo_fat[row[q_]])[:12]}")
    rng = np.random.default_rng(0)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    rows_csv = []
    for rep, method in PAIRS:
        cl = np.load(RUN / f"assign_{rep}_{method}.npy").astype(np.int64)[nb:nb + nf]
        K = int(cl.max()) + 1
        pure = {}
        for cname in ("CCl4", "cholesterol"):
            m_ = cmpd[row] == cname
            n_c = np.bincount(cl[m_], minlength=K)
            n_cf = np.bincount(cl[m_ & (fat[row] > 0)], minlength=K)
            fr = n_cf / np.maximum(n_c, 1)
            cand = np.where((n_c >= MIN_N) & (fr >= PURITY))[0]
            cand = cand[np.argsort(-n_c[cand])][:MAX_PROTOS]
            pure[cname] = cand
            sheet = []
            for j in cand:
                idx = np.where((cl == j) & m_ & (fat[row] > 0) & tgt.svs_ok.to_numpy()[row])[0]
                picks, seen = [], set()
                for q in rng.permutation(idx):
                    if row[q] in seen:
                        continue
                    picks.append(int(q)); seen.add(row[q])
                    if len(picks) >= TILES_PER_PROTO:
                        break
                if len(picks) < TILES_PER_PROTO:
                    extra = [int(q) for q in rng.permutation(idx) if q not in picks][:TILES_PER_PROTO - len(picks)]
                    picks += extra
                gr = {g: int(((cl == j) & m_ & (fat[row] == g)).sum()) for g in (0, 1, 2, 3, 4)}
                other = int(((cl == j) & ~m_).sum())
                hdr = f"p{j:04d}  {cname}: {int(n_c[j]):,} patches, purity {fr[j]:.2f} (from fatty slides)   by grade 0/1/2/3/4: {gr[0]}/{gr[1]}/{gr[2]}/{gr[3]}/{gr[4]}   other compounds: {other:,}"
                sheet.append((hdr, ex17.read_tiles(picks, row, xy, sdf, size0, lbl)))
                rows_csv.append({"rep": rep, "method": method, "compound": cname, "prototype": int(j), "n": int(n_c[j]), "purity": float(fr[j]), **{f"n_grade{g}": v for g, v in gr.items()}, "n_other_compounds": other})
            if sheet:
                ex17.draw_sheet(f"{rep} {method}: {cname} prototypes with purity >= {PURITY} (n >= {MIN_N}), tiles from fatty-degeneration slides", sheet, RUN / "tiles" / f"{rep}_{method}_{cname}.jpg")
            print(f"{rep} {method} {cname}: {len(cand)} prototypes", flush=True)
        # 地図: 程度の高い slide（化合物ごとに 2 枚）。CCl4 側の純プロトタイプは橙、cholesterol 側は青
        for cname in ("CCl4", "cholesterol"):
            cand_s = np.where((cmpd == cname) & (fat > 0) & tgt.svs_ok.to_numpy())[0]
            cand_s = cand_s[np.argsort(-fat[cand_s])][:2]
            for i in cand_s:
                r = tgt.iloc[i]
                w = openslide.OpenSlide(r.svs_path)
                th = w.get_thumbnail((THUMB, THUMB)).convert("RGB")
                scl = th.width / w.dimensions[0]
                w.close()
                b_ = max(3, round(size0[r.slide_id] * scl))
                ov = Image.new("RGBA", th.size, (0, 0, 0, 0))
                d = ImageDraw.Draw(ov)
                m_s = row == i
                n_drawn = {}
                for c2, color in COL.items():
                    sel = m_s & np.isin(cl, pure[c2])
                    n_drawn[c2] = int(sel.sum())
                    for x0, y0 in xy[sel]:
                        d.rectangle([x0 * scl, y0 * scl, x0 * scl + b_, y0 * scl + b_], fill=color + (230,))
                canvas = Image.new("RGB", (th.width * 2 + 10, th.height + 48), "white")
                canvas.paste(th, (0, 48))
                canvas.paste(Image.alpha_composite(th.convert("L").convert("RGBA"), ov).convert("RGB"), (th.width + 10, 48))
                dd = ImageDraw.Draw(canvas)
                dd.text((4, 4), f"{r.slide_id}  {r.compound} {r.dose} {r.period} (exp {r.exp})  fatty grade {int(fat[i])} {topo_fat[i]}", fill="black", font=font)
                dd.text((4, 24), f"{rep} {method}: orange = CCl4-pure prototypes ({n_drawn['CCl4']} of {int(m_s.sum())} sampled patches), blue = cholesterol-pure prototypes ({n_drawn['cholesterol']})", fill="black", font=font)
                canvas.save(RUN / "maps" / f"{rep}_{method}_{cname}_{r.slide_id}.jpg", quality=85)
        print(f"{rep} {method}: maps done", flush=True)
    pd.DataFrame(rows_csv).to_csv(RUN / "fatty_pure_prototypes.csv", index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
