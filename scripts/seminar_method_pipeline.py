"""セミナー用: プロトタイプの作り方の流れ図（2026-11-10 ユーザー: 「順位変換 → PCA → LDA」の箱を足した版を別のスライドで使う）。
中身は 0032 v1（セミナーの図で使ったプロトタイプ）の手順どおり。呼び名は「LDA なし」「LDA あり」（C・D とは書かない）。

画像: WSI は Bayer の slide 25661（allyl alcohol 高用量 8 日）、patch はその slide から無作為に 3 個（白い画素 15% 未満、スキャンしたまま）。
      プロトタイプの例は 0032 v1 の LDA ありの見本シート（重心に近い順・別々の slide）の 1 枚目。
出力: outputs/0032_20261029_bayer_only_prototypes/v1/seminar/M1_method_pipeline.{png,svg}
usage: PYTHONPATH=.:scripts python scripts/seminar_method_pipeline.py   （コンテナ内、Slurm経由。CPU、1 分程度）
"""
import h5py
import numpy as np

from seminar_prototype_mixing_figure import FONT_JP, FULL_DIR, ROOT, WSI_DIRS

RUN = ROOT / "outputs/0032_20261029_bayer_only_prototypes/v1"
OUT = RUN / "seminar"
SLIDE, N_TILES, SEED, WHITE, MAX_WHITE = "25661", 3, 0, 215, 0.15
PROTOS = [(13, "壊死"), (3, "太い血管"), (4, "門脈域"), (18, "核分裂像")]   # LDA ありの番号と名前（prototype_names_C_lda.csv）
NAVY, ARROW, LDA, NOLDA, BAYER = "#1A5E80", "#1A5E80", "#D55E00", "#777777", "#E69F00"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import openslide
    from matplotlib import font_manager
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from PIL import Image
    OUT.mkdir(exist_ok=True)
    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "font.size": 11, "svg.fonttype": "none"})

    with h5py.File(ROOT / FULL_DIR / f"{SLIDE}.h5", "r") as h:
        xy, ps = h["coords"][:], int(h["coords"].attrs.get("patch_size_level0", 256))
    o = openslide.OpenSlide(str([p_ for d in WSI_DIRS for p_ in d.glob(f"*/{SLIDE}.svs")][0]))
    thumb = np.asarray(o.get_thumbnail((700, 700)).convert("RGB"))
    tiles = []
    for k in np.random.default_rng(SEED).permutation(len(xy)):
        im = np.asarray(o.read_region(tuple(int(v) for v in xy[k]), 0, (ps, ps)).convert("RGB"))
        if (im.mean(2) > WHITE).mean() < MAX_WHITE:
            tiles.append(im)
        if len(tiles) == N_TILES:
            break
    protos = [np.asarray(Image.open(RUN / f"tree/lda_k100/img/p{j:03d}.jpg").convert("RGB"))[84:276, 0:192] for j, _ in PROTOS]

    fig = plt.figure(figsize=(13.33, 6.4))
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, 133.3), ylim=(0, 64))
    ax.axis("off")

    def box(x, y, w, h, text, fc="#F4F4F4", ec="#333333", tc="black", lw=1.2, fs=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec=ec, lw=lw))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, linespacing=1.4)

    def arrow(p, q, color=ARROW, ls="-", lw=1.8, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=color, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))

    def image(im, x, y, w, h, ec=None):
        ax.imshow(im, extent=(x, x + w, y, y + h), zorder=2)
        if ec:
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, ec=ec, lw=1.2, zorder=3))

    # 1 段目: WSI → patch → 染色正規化 → UNI2-h → 順位変換 → PCA
    ax.text(66.6, 63, "データ: TG-GATEs ラット肝、Bayer のアノテーション付き 230 枚（約 325 万 patch）。"
            "k-means はラベルを見ない。Bayer のラベルを使うのは LDA と名前付けだけ（点線）",
            ha="center", va="top", fontsize=9.5, color="#333333")
    y1, h1 = 38, 13
    th_w = 15 * thumb.shape[1] / thumb.shape[0]
    image(thumb, 3, y1 - 1, th_w, 15)
    ax.text(3 + th_w / 2, y1 + 15, "WSI（230 枚）", ha="center", va="bottom", fontsize=11)
    arrow((3 + th_w + 1, y1 + h1 / 2), (24.5, y1 + h1 / 2))
    for i, t in enumerate(tiles):
        image(t, 25.5 + i * 1.6, y1 + 5.5 - i * 3.0, 7, 7, ec="#333333")
    ax.text(30.5, y1 + 15, "patch\n20 倍・256 px", ha="center", va="bottom", fontsize=10, linespacing=1.3)
    xs = [40, 58.5, 77, 95.5, 114]
    w1 = 15.5
    texts = ["Macenko\n染色正規化", "UNI2-h\n1 patch 1,536 次元", "slide 内の\n順位変換", "PCA\n256 次元\n（ラベルなしで学習）"]
    arrow((36.5, y1 + h1 / 2), (xs[0], y1 + h1 / 2))
    for i, t in enumerate(texts):
        if i == 1:
            box(xs[i], y1, w1, h1, t, fc=NAVY, ec=NAVY, tc="white")
        else:
            box(xs[i], y1, w1, h1, t)
        if i < len(texts) - 1:
            arrow((xs[i] + w1, y1 + h1 / 2), (xs[i + 1], y1 + h1 / 2))
    ax.text(xs[2] + w1 / 2, y1 - 1.2, "次元ごとに slide 内の順位を\n正規分布の値に置き換える", ha="center", va="top", fontsize=8.5, color="#555555", linespacing=1.3)

    # 2 段目: 表現 2 種 → 合議 k-means → プロトタイプ 100 個 → 名前付け
    # PCA から 2 段目の左端へは、カギ形の線で戻す（他の要素を横切らない）
    yA, yB, h2, x2, w2 = 20, 8, 9, 10, 22
    yr, xr = 32.5, 5
    ax.plot([xs[3] + w1 / 2, xs[3] + w1 / 2, xr, xr], [y1, yr, yr, yB + h2 / 2], color=ARROW, lw=1.8, solid_joinstyle="round")
    arrow((xr, yA + h2 / 2), (x2, yA + h2 / 2))
    arrow((xr, yB + h2 / 2), (x2, yB + h2 / 2))
    box(x2, yA, w2, h2, "LDA なし\nPCA の上位 64 次元", ec=NOLDA, lw=2)
    box(x2, yB, w2, h2, "LDA あり\nLDA 13 次元", ec=LDA, lw=2.4)
    xk, wk = 41, 27
    box(xk, yB, wk, yA + h2 - yB, "合議 k-means（k = 100）\n\n80 万 patch を無作為に選んで\nk-means を 10 回\n→ 1,000 個の重心を\nもう一度 k-means で 100 個に", fs=10)
    arrow((x2 + w2, yA + h2 / 2), (xk, yA + h2 / 2))
    arrow((x2 + w2, yB + h2 / 2), (xk, yB + h2 / 2))
    ym = (yB + yA + h2) / 2
    xp = 75
    ax.text(xp + 15.5, 29, "プロトタイプ 100 個（表現ごと）", ha="center", va="bottom", fontsize=11)
    ax.text(xp + 15.5, 27.6, "325 万 patch を最も近い重心に割り当て", ha="center", va="top", fontsize=9, color="#555555")
    for i, (im, (_, name)) in enumerate(zip(protos, PROTOS)):
        image(im, xp + i * 7.9, ym - 3.6, 7.2, 7.2 * im.shape[0] / im.shape[1])
        ax.text(xp + i * 7.9 + 3.6, ym - 4.4, name, ha="center", va="top", fontsize=9)
    arrow((xk + wk, ym), (xp - 1, ym))
    xn, wn = 112, 19
    box(xn, yB, wn, yA + h2 - yB, "名前付け\n\nBayer のラベルが\n20% 以上、または\n全体の割合の 3 倍以上", fs=9.5)
    arrow((xp + 31.5, ym), (xn, ym))

    # Bayer のラベルを使う所（点線）: 2 段目の下に置き、LDA ありと名前付けへ
    bx, bw = 50, 40
    box(bx, 0.6, bw, 5, "Bayer のアノテーション（patch のラベル 14 種）", fc="#FFF4DD", ec=BAYER, lw=1.6, fs=9.5)
    arrow((bx, 3.1), (x2 + w2 / 2, yB), color=BAYER, ls="--", lw=1.6, rad=-0.25)
    arrow((bx + bw, 3.1), (xn + wn / 2, yB), color=BAYER, ls="--", lw=1.6, rad=0.25)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"M1_method_pipeline.{ext}", dpi=200, facecolor="white")
    print("done", flush=True)


if __name__ == "__main__":
    main()
