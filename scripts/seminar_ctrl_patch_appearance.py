"""セミナー用: 同じ「正常」（対照群）でも、試験ごとに patch の見た目が違う例（2026-11-10 ユーザー: スライド 2 枚から数枚ずつ並べるだけでよい）。

2 枚は scripts/seminar_prototype_mixing_figure.py の F2 と同じ（別の試験・別のスキャナ・同じ解剖時点）。
組織の種類をそろえるため、0030 v4 の割り当て（各 slide 2,048 patch）で「LDA あり」の表現の実質（parenchyma）と名前の付いた
プロトタイプに入った patch から、slide ごとに無作為に N_TILES 個。スキャンしたままの画像（染色正規化の前）。
出力: outputs/0035_20261102_finer_prototypes_split_rejoin/control_similarity/seminar/
usage: PYTHONPATH=.:scripts python scripts/seminar_ctrl_patch_appearance.py   （コンテナ内、Slurm経由。CPU、1 分程度）
"""
import json

import numpy as np
import pandas as pd

from seminar_prototype_mixing_figure import FONT_JP, ROOT, WSI_DIRS

RUN = ROOT / "outputs/0030_20261028_tggates_unseen_findings/v4"
OUT = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin/control_similarity/seminar"
NAMES = ROOT / "outputs/0029_20261028_external_finding_maps/v9/prototype_names_C_lda.csv"
SLIDES = ["46630", "52646"]
N_TILES, SEED = 6, 0


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import openslide
    from matplotlib import font_manager
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sl = pd.read_csv(RUN / "slides.csv", dtype={"slide_id": str, "EXP_ID": str})
    A, F = np.load(RUN / "assignments.npz"), np.load(RUN / "features_sub.npz")
    row, xy, cl = A["row"], F["xy"], A["cl_C_lda"].astype(int)
    size0 = json.loads(str(F["size0"]))
    names = pd.read_csv(NAMES).set_index("prototype").name
    paren = names[names == "parenchyma"].index.to_numpy()
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "font.size": 10})
    fig, axes = plt.subplots(len(SLIDES), N_TILES + 1, figsize=(2.1 * (N_TILES + 1) + 0.6, 2.4 * len(SLIDES) + 0.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1.6] + [1] * N_TILES})
    for r, sid in enumerate(SLIDES):
        i = int(np.where(sl.slide_id == sid)[0][0])
        s = sl.iloc[i]
        o = openslide.OpenSlide(str([p_ for d in WSI_DIRS for p_ in d.glob(f"*/{sid}.svs")][0]))
        ax = axes[r, 0]
        ax.imshow(o.get_thumbnail((600, 600)))
        ax.set(xticks=[], yticks=[])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{s.COMPOUND_NAME} 試験（{s.EXP_ID}）の対照群\n{s.SACRIFICE_PERIOD}、スキャナ {s.scanscope_id}、slide {sid}", fontsize=9)
        pick = rng.choice(np.where((row == i) & np.isin(cl, paren))[0], N_TILES, replace=False)
        ps = size0[sid]
        for j, k in enumerate(pick):
            ax = axes[r, j + 1]
            ax.imshow(np.asarray(o.read_region(tuple(int(v) for v in xy[k]), 0, (ps, ps)).convert("RGB")))
            ax.set(xticks=[], yticks=[])
    fig.suptitle("同じ「正常」の肝臓（対照群）でも、試験が違うと patch の見た目が違う\n"
                 "右の 6 枚は、それぞれの slide の肝実質の patch を無作為に選んだもの（256 px・20 倍、スキャンしたまま）", fontsize=11)
    fig.savefig(OUT / "P1_ctrl_patch_appearance.png", dpi=180)
    print("done", flush=True)


if __name__ == "__main__":
    main()
