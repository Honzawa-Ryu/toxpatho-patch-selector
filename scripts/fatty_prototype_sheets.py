"""0035 の脂肪変性プロトタイプを目で確かめるタイルのシート（2026-11-02ユーザー: 本当に脂肪変性か、CCl4 と cholesterol 食が分かれているか。
名前を付け、図ごとに何の図か分かるように）。

0035 v1 の割り当て（assign_<rep>_<method>.npy）のうち脂肪変性の 3 試験（188枚、各 2,048 patch）の部分を使う。
名前は scripts/prototype_record_names.py の出力（prototype_names_<rep>_<method>.csv）を読む。先にそちらを実行すること。
出力（outputs/0035_.../<版>/fatty_figures/）:
  tiles_<method>_<化合物>.jpg : 化合物内の純度（脂肪変性ありの slide 由来の割合）0.9 以上のプロトタイプを大きい順に最大 6 個、各 12 枚のタイル
  fatty_pure_prototypes.csv   : 選んだプロトタイプの一覧（名前、n、純度、程度別の内訳）
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
OUT = RUN / "fatty_figures"
TILES_PER_PROTO, MAX_PROTOS, MIN_N, PURITY = 12, 6, 200, 0.9
FONT_JP = "/workspace/andre01/honzawa/.local/share/fonts/NotoSansCJKjp-Regular.otf"
TILE, COLS = 192, 8
METHOD_JP = {"split_record": "k=100 から、記録で純度の低いクラスタを 5 つに分ける操作を 2 回（887 クラスタ）", "flat_k1000": "k=1000 の k-means（コンセンサス）"}
COMP_JP = {"CCl4": "CCl4（exp 66・67、記録は小葉中心性）", "cholesterol": "1% cholesterol + 0.25% sodium cholate 食（exp 698、記録は辺縁性）"}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def wrap(text, font, width):
    """font で描いたときの幅が width を超えないように、文字単位で折り返す。"""
    lines, cur = [], ""
    for ch in text:
        if font.getlength(cur + ch) > width and cur:
            lines.append(cur)
            cur = ch.lstrip()
        else:
            cur += ch
    return lines + [cur] if cur else lines


def draw_sheet(banner, blocks, path):
    """banner: 先頭の説明（行のリスト、1 行目が題）。blocks: [(見出し 2 行, タイルのリスト)]。タイルは ex17.read_tiles の出力。長い行は折り返す。"""
    from PIL import Image, ImageDraw, ImageFont
    f_title, f_text, f_small = ImageFont.truetype(FONT_JP, 22), ImageFont.truetype(FONT_JP, 15), ImageFont.truetype(FONT_JP, 11)
    W = COLS * TILE
    LH = 21
    ban = [wrap(b, f_text, W - 16) for b in banner[1:]]
    hb = 42 + LH * sum(len(b) for b in ban) + 16
    heads = [(wrap(h1, f_text, W - 16), wrap(h2, f_text, W - 16)) for (h1, h2), _ in blocks]
    rows_h = [8 + LH * (len(a) + len(b)) + ((len(t) + COLS - 1) // COLS) * (TILE + 30) + 12 for (a, b), (_, t) in zip(heads, blocks)]
    sheet = Image.new("RGB", (W, hb + sum(rows_h)), "white")
    d = ImageDraw.Draw(sheet)
    d.text((8, 8), banner[0], fill="black", font=f_title)
    yy = 42
    for b in ban:
        for line in b:
            d.text((8, yy), line, fill="#333333", font=f_text)
            yy += LH
    y = hb
    for (a, b), (_, tiles) in zip(heads, blocks):
        hh = 8 + LH * (len(a) + len(b))
        d.rectangle([0, y, W, y + hh - 4], fill="#F2F2F2")
        for k_, line in enumerate(a):
            d.text((8, y + 2 + LH * k_), line, fill="#B34700", font=f_text)
        for k_, line in enumerate(b):
            d.text((8, y + 2 + LH * (len(a) + k_)), line, fill="#333333", font=f_text)
        y += hh
        for k, t in enumerate(tiles):
            x0, y0 = (k % COLS) * TILE, y + (k // COLS) * (TILE + 30)
            d.text((x0 + 3, y0 + 1), t["l1"], fill="black", font=f_small)
            d.text((x0 + 3, y0 + 15), t["l2"], fill="#555555", font=f_small)
            sheet.paste(t["image"], (x0, y0 + 30))
        y += ((len(tiles) + COLS - 1) // COLS) * (TILE + 30) + 12
    sheet.save(path, quality=88)


def main():
    cfg = yaml.safe_load(open(EXP / "config.yml"))
    load_module(ROOT / "experiments" / cfg["exp0021"] / "experiment.py", "exp0021")
    load_module(ROOT / "experiments/0023_20261025_layer_reality_check/experiment.py", "exp0023")
    ex17 = load_module(ROOT / "experiments/0017_20261016_block_consensus_patterns/experiment.py", "exp0017")
    ex26 = load_module(ROOT / "experiments" / cfg["exp0026"] / "experiment.py", "exp0026")
    ex27 = load_module(ROOT / "experiments" / cfg["exp0027"] / "experiment.py", "exp0027")
    ex30 = load_module(ROOT / "experiments" / cfg["exp0030"] / "experiment.py", "exp0030")
    pre_path = str(ROOT / "outputs" / cfg["exp0021"] / "v3_maps/models.pkl")
    OUT.mkdir(exist_ok=True)

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
    topo_fat = topo["Degeneration, fatty"].reindex(tgt.slide_id).fillna("").to_numpy() if "Degeneration, fatty" in topo else np.full(len(tgt), "")
    sdf = tgt[["slide_id", "svs_path", "compound", "dose", "period", "exp"]].copy()
    lbl = lambda s_, q_: (f"slide {s_.slide_id}  exp {s_.exp}  {s_.dose}", f"脂肪変性 grade {int(fat[row[q_]])} {str(topo_fat[row[q_]])[:14]}")
    rng = np.random.default_rng(0)
    rows_csv = []
    for rep, method in PAIRS:
        cl = np.load(RUN / f"assign_{rep}_{method}.npy").astype(np.int64)[nb:nb + nf]
        names = pd.read_csv(RUN / f"prototype_names_{rep}_{method}.csv").set_index("prototype")
        K = int(cl.max()) + 1
        for cname in ("CCl4", "cholesterol"):
            m_ = cmpd[row] == cname
            n_c = np.bincount(cl[m_], minlength=K)
            n_cf = np.bincount(cl[m_ & (fat[row] > 0)], minlength=K)
            fr = n_cf / np.maximum(n_c, 1)
            cand = np.where((n_c >= MIN_N) & (fr >= PURITY))[0]
            cand = cand[np.argsort(-n_c[cand])][:MAX_PROTOS]
            blocks = []
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
                    picks += [int(q) for q in rng.permutation(idx) if q not in picks][:TILES_PER_PROTO - len(picks)]
                gr = {g: int(((cl == j) & m_ & (fat[row] == g)).sum()) for g in (0, 1, 2, 3, 4)}
                other = int(((cl == j) & ~m_).sum())
                nm = names.loc[j]
                rec = nm.record_name if isinstance(nm.record_name, str) and nm.record_name else "なし"
                rec_s = f"{rec}（記録あり slide 由来 {nm.record_share:.0%}、基準の {nm.record_lift:.0f} 倍）" if rec != "なし" else "なし（基準に届かない）"
                bay = nm.bayer_name if isinstance(nm.bayer_name, str) and nm.bayer_name else "なし"
                top = nm.bayer_top_label if isinstance(nm.bayer_top_label, str) and nm.bayer_top_label else "-"
                h1 = f"プロトタイプ p{j:04d}　記録による名前: {rec_s}　Bayer ラベルによる名前: {bay}（最多ラベル {top}）"
                h2 = f"{cname} の patch {int(n_c[j]):,} 個、うち脂肪変性ありの slide 由来 {fr[j]:.0%}　程度 0/1/2/3/4 の内訳 {gr[0]}/{gr[1]}/{gr[2]}/{gr[3]}/{gr[4]}　他の化合物の patch {other:,} 個（全 3,737 枚中）"
                blocks.append(((h1, h2), ex17.read_tiles(picks, row, xy, sdf, size0, lbl)))
                rows_csv.append({"rep": rep, "method": method, "compound": cname, "prototype": int(j), "record_name": rec, "bayer_name": bay, "bayer_top_label": top,
                                 "n": int(n_c[j]), "purity": float(fr[j]), **{f"n_grade{g}": v for g, v in gr.items()}, "n_other_compounds": other})
            if blocks:
                banner = [f"脂肪変性のプロトタイプのタイル: {COMP_JP[cname]}",
                          f"何の図か: 0035 で作ったプロトタイプのうち、{cname} の slide の中で「脂肪変性の記録がある slide 由来の patch」が 90% 以上のもの（patch 200 個以上）を、patch の多い順に最大 {MAX_PROTOS} 個並べた。",
                          f"作り方: 表現 A128（slide 内の順位 → PCA → 試験ごとに対照群で白色化した 128 次元 ‖ Bayer のラベルで学んだ LDA 13 次元）。クラスタ: {METHOD_JP.get(method, method)}。",
                          f"タイル: 各プロトタイプから、脂肪変性の記録がある slide の patch を 12 枚（なるべく別の slide）。20 倍、1 枚 約 128 µm 四方。タイル上の 2 行は slide 番号・試験・用量群と、その slide の脂肪変性の程度・部位（記録）。",
                          "名前: 「記録による名前」は TG-GATEs の病理記録（slide 単位）で、その所見の slide に偏るプロトタイプの意味（patch そのものがその所見とは限らない）。「Bayer ラベルによる名前」はポリゴンの注釈で付けたもの（Bayer に脂肪変性のラベルはない）。"]
                draw_sheet(banner, blocks, OUT / f"tiles_{method}_{cname}.jpg")
            print(f"{rep} {method} {cname}: {len(cand)} prototypes", flush=True)
    pd.DataFrame(rows_csv).to_csv(OUT / "fatty_pure_prototypes.csv", index=False)
    pd.DataFrame(rows_csv).to_csv(RUN / "fatty_pure_prototypes.csv", index=False)   # maps_full が読む
    print("done", flush=True)


if __name__ == "__main__":
    main()
