"""セミナー用: 同じ「正常」（対照群）で、同じプロトタイプ（組織の種類）に入った patch でも、試験のまとまりごとに見た目が違う例（2026-11-10 ユーザー: パッチのグルーピングの話）。

0030 v4 の対照群（857 枚、43 試験）。「LDA あり」の表現（0032 の重心）で、実質（parenchyma）と名前の付いたプロトタイプのうち、
選んだまとまりのすべてで割合が高いもの 1 個に入った patch だけを使う（組織の種類をそろえるため）。
行 = 対照群の見分けやすさで作った試験のまとまり（0035 control_similarity の B128 のまとまり。対照群の slide が多い順に、スキャナごとに 2 つまで）。
各行に、別々の対照群 slide（なるべく別の試験）から 1 patch ずつ。左 = スキャンしたまま、右 = Macenko 正規化後（モデルが見る画像。
参照は 0016 の reference.json、slide ごとの推定は 0016 と同じ 40 patch。ただし patch の選び方は 0016 と同じではないので推定値は近似）。
出力: outputs/0035_20261102_finer_prototypes_split_rejoin/control_similarity/seminar/
usage: PYTHONPATH=.:scripts python scripts/seminar_ctrl_patch_appearance.py   （コンテナ内、Slurm経由。CPU、数分）
"""
import json

import numpy as np
import pandas as pd

from seminar_prototype_mixing_figure import FONT_JP, ROOT, WSI_DIRS

RUN = ROOT / "outputs/0030_20261028_tggates_unseen_findings/v4"
CS = ROOT / "outputs/0035_20261102_finer_prototypes_split_rejoin/control_similarity"
OUT = CS / "seminar"
NAMES = ROOT / "outputs/0029_20261028_external_finding_maps/v9/prototype_names_C_lda.csv"
N_ROWS_PER_SCANNER, N_TILES, N_FIT, SEED = 2, 6, 40, 0


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import openslide
    from matplotlib import font_manager
    from lib.stain import apply_macenko, fit_macenko, macenko_matrix
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    sl = pd.read_csv(RUN / "slides.csv", dtype={"slide_id": str, "EXP_ID": str})
    A, F = np.load(RUN / "assignments.npz"), np.load(RUN / "features_sub.npz")
    row, xy, cl = A["row"], F["xy"], A["cl_C_lda"].astype(int)
    size0 = json.loads(str(F["size0"]))
    E = pd.read_csv(CS / "experiments.csv")
    E["exp_id"] = E.exp_id.astype(int).astype(str)
    sl["grp"] = sl.EXP_ID.astype(int).astype(str).map(E.set_index("exp_id").group_B128)
    ctrl = np.where(sl.is_ctrl.to_numpy())[0]
    # まとまり: 対照群の slide が多い順に、スキャナごとに N_ROWS_PER_SCANNER 個
    c_ = sl.iloc[ctrl].dropna(subset=["grp"])
    size = c_.groupby("grp").agg(n=("slide_id", "size"), n_exp=("EXP_ID", "nunique"), scanner=("scanscope_id", lambda s: s.mode()[0])).sort_values("n", ascending=False)
    groups = pd.concat([g.head(N_ROWS_PER_SCANNER) for _, g in size[size.n_exp >= 2].groupby("scanner")]).sort_values(["scanner", "n"], ascending=[True, False])
    print(groups, flush=True)
    # プロトタイプ: 実質の名前が付いたもののうち、選んだまとまりの対照群での割合の最小値が最大のもの
    names = pd.read_csv(NAMES).set_index("prototype").name
    cand = names[names == "parenchyma"].index.to_numpy()
    pm = np.isin(row, ctrl)
    gr = sl.grp.to_numpy()[row]
    share = np.array([[np.mean(cl[pm & (gr == g)] == p) for g in groups.index] for p in cand])
    proto = int(cand[np.argmax(share.min(1))])
    print("prototype", proto, "shares", dict(zip(groups.index, share[np.argmax(share.min(1))].round(3))), flush=True)
    js = json.loads((ROOT / "outputs/0016_20261013_tier1_macenko_embeddings/reference.json").read_text())
    he_ref, maxc_ref = np.array(js["reference_he"]), np.array(js["reference_max_c"])

    def svs_of(sid):
        return str([p_ for d in WSI_DIRS for p_ in d.glob(f"*/{sid}.svs")][0])
    tiles = {}
    for g in groups.index:
        sids = sl.index[(sl.grp == g) & sl.is_ctrl]
        # 試験ごとに 1 枚ずつ、足りなければ同じ試験から追加
        by_exp = [rng.permutation(list(s)) for _, s in sl.loc[sids].groupby("EXP_ID").groups.items()]
        order = [x for k in range(max(map(len, by_exp))) for x in (rng.permutation([b[k] for b in by_exp if k < len(b)]))]
        out = []
        for i in order:
            pi = np.where((row == i) & (cl == proto))[0]
            if not len(pi):
                continue
            sid = sl.slide_id.iloc[i]
            ps = size0[sid]
            o = openslide.OpenSlide(svs_of(sid))
            allp = np.where(row == i)[0]
            fit = rng.choice(allp, min(N_FIT, len(allp)), replace=False)
            sample = np.stack([np.asarray(o.read_region(tuple(int(v) for v in xy[k]), 0, (ps, ps)).convert("RGB")) for k in fit])
            he, mc = fit_macenko(sample.reshape(-1, 3))
            mat = macenko_matrix(he, mc, he_ref=he_ref, maxc_ref=maxc_ref)
            k = rng.choice(pi)
            rgb = np.asarray(o.read_region(tuple(int(v) for v in xy[k]), 0, (ps, ps)).convert("RGB"))
            out.append((sl.COMPOUND_NAME.iloc[i], sl.EXP_ID.iloc[i], sid, rgb, apply_macenko(rgb, mat)))
            if len(out) == N_TILES:
                break
        tiles[g] = out

    font_manager.fontManager.addfont(FONT_JP)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT_JP).get_name(), "font.size": 9})
    fig = plt.figure(figsize=(2.0 * N_TILES * 2 + 1.5, 2.25 * len(groups) + 1.2), layout="constrained")
    sf = fig.subfigures(1, 2, wspace=0.03)
    for side, (sub, title) in enumerate(zip(sf, ["スキャンしたまま", "Macenko 正規化後（モデルが見る画像）"])):
        axes = sub.subplots(len(groups), N_TILES, squeeze=False)
        for r, g in enumerate(groups.index):
            for j in range(N_TILES):
                ax = axes[r, j]
                ax.set(xticks=[], yticks=[])
                if j < len(tiles[g]):
                    comp, exp, sid, rgb, nrm = tiles[g][j]
                    ax.imshow(rgb if side == 0 else nrm)
                    ax.set_title(f"{comp[:14]}（{exp}）", fontsize=7, pad=2)
                if j == 0:
                    gi = groups.loc[g]
                    ax.set_ylabel(f"まとまり {int(g)}\n{gi.scanner}\n{int(gi.n_exp)} 試験", fontsize=9)
        sub.suptitle(title, fontsize=11)
    fig.suptitle(f"同じ「正常」（対照群）で、同じプロトタイプ（実質、LDA あり #{proto}）に入った patch。行 = 対照群で見分けられない試験のまとまり\n"
                 "各 patch は別々の対照群 slide から 1 個（256 px・20 倍）", fontsize=11)
    fig.savefig(OUT / "P1_ctrl_patch_appearance.png", dpi=170)
    print("done", flush=True)


if __name__ == "__main__":
    main()
