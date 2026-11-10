"""セミナー用: 同じ「正常」（対照群）でも、試験ごとに patch の見た目が違う例（2026-11-10 ユーザー: スライド 2 枚から数枚ずつ並べるだけでよい。
もっと離れた見た目がよい → 脂肪変性の試験の対照群）。

2 枚: CCl4 試験（67、SS1598）と 1% コレステロール食の試験（698、SS1601）の対照群、どちらも 4 日目。候補 6 枚を並べて目視で選んだ
（67・698 の 4 日目の対照群 2 枚ずつと、scripts/seminar_prototype_mixing_figure.py の F2 の 2 枚）。
patch は 0016 の座標（全 patch）から無作為に、白い画素（RGB 平均 > 215）が 15% 未満のものだけ N_TILES 個。スキャンしたままの画像（染色正規化の前）。
出力: outputs/0035_20261102_finer_prototypes_split_rejoin/control_similarity/seminar/
usage: PYTHONPATH=.:scripts python scripts/seminar_ctrl_patch_appearance.py   （コンテナ内、Slurm経由。CPU、1 分程度）
"""
import h5py
import numpy as np
import pandas as pd

from seminar_prototype_mixing_figure import FONT_JP, FULL_DIR, ROOT, WSI_DIRS

OUT = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin/control_similarity/seminar"
QC = "/workspace/andre01/honzawa/02-playground/tggate-wsi-qc/outputs/qc/catalog/slide_quality.csv"
SLIDES = ["26433", "66065"]
SHORT = {"carbon tetrachloride": "CCl4（四塩化炭素）", "1% cholesterol + 0.25% sodium cholate": "コレステロール食"}
N_TILES, SEED, WHITE, MAX_WHITE = 6, 0, 215, 0.15


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import openslide
    from matplotlib import font_manager
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    q = pd.read_csv(QC, dtype=str).set_index("slide_id")
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "font.size": 10})
    fig, axes = plt.subplots(len(SLIDES), N_TILES + 1, figsize=(2.1 * (N_TILES + 1) + 0.6, 2.4 * len(SLIDES) + 0.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1.6] + [1] * N_TILES})
    for r, sid in enumerate(SLIDES):
        s = q.loc[sid]
        with h5py.File(ROOT / FULL_DIR / f"{sid}.h5", "r") as h:
            xy, ps = h["coords"][:], int(h["coords"].attrs.get("patch_size_level0", 256))
        o = openslide.OpenSlide(str([p_ for d in WSI_DIRS for p_ in d.glob(f"*/{sid}.svs")][0]))
        ax = axes[r, 0]
        ax.imshow(o.get_thumbnail((600, 600)))
        ax.set(xticks=[], yticks=[])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{SHORT.get(s.COMPOUND_NAME, s.COMPOUND_NAME)} 試験（{int(s.EXP_ID)}）の対照群\n{s.SACRIFICE_PERIOD}、スキャナ {s.scanscope_id}、slide {sid}", fontsize=9)
        n = 0
        for k in rng.permutation(len(xy)):
            im = np.asarray(o.read_region(tuple(int(v) for v in xy[k]), 0, (ps, ps)).convert("RGB"))
            if (im.mean(2) > WHITE).mean() >= MAX_WHITE:
                continue
            ax = axes[r, n + 1]
            ax.imshow(im)
            ax.set(xticks=[], yticks=[])
            n += 1
            if n == N_TILES:
                break
    fig.suptitle("同じ「正常」の肝臓（対照群）でも、試験が違うと patch の見た目が違う\n"
                 "右の 6 枚は、それぞれの slide の組織の patch を無作為に選んだもの（256 px・20 倍、スキャンしたまま）", fontsize=11)
    fig.savefig(OUT / "P1_ctrl_patch_appearance.png", dpi=180)
    print("done", flush=True)


if __name__ == "__main__":
    main()
