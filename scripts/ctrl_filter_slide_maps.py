"""0018の結果（1段目の除外と2段目のコア）を、slideの全patchに広げて縮小画像の上に色で描く。

1段目: 各patchを0017のPCA空間に写し、0017の60 runの重心で割り当てて、割り当て先クラスタのControl割合を平均する
       （0018と同じスコア。評価用patchなら0018の値と一致する）。閾値以下を「残す」。
2段目: 残したpatchだけ、残した評価用patchの近傍15点の多数決でコアを決める（lib.clustering.propagate_labels）。
色: コアごとの色、残したがコア未割当は濃い灰色、1段目で除いたpatchは塗らない。
対象: 投与群で10枚以上に記録された所見それぞれについて、所見ありslideの多い試験上位3つから1枚ずつ
      （最も高い用量・遅い時点で、残した評価用patchが最も多いslide）と、1つ目の試験の同じ時点のControl 1枚。

usage: python scripts/ctrl_filter_slide_maps.py [embedding] [method] [k] [0018 run] [ctrl_keep]
       既定: pca0017 consensus 20 block6_all（閾値は0018の値）。ctrl_keepを与えると、Controlの評価用patchが残る割合が
       その値になる閾値に厳しくして描く（2026-10-23、コアは0018のまま）
"""
import colorsys
import importlib.util
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.clustering import propagate_labels  # noqa: E402

EXP18 = ROOT / "experiments/0018_20261022_ctrl_filter_then_cores"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
THUMB = 1400
KNN = 15
MIN_SLIDES = 10
DOSE_ORDER = {"Low": 1, "Middle": 2, "High": 3}

spec = importlib.util.spec_from_file_location("exp0018", EXP18 / "experiment.py")
ex18 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex18)


def palette(n):
    out = {}
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((i * .61803398875) % 1, .75, .9 - .3 * (i % 2))
        out[i] = (int(r * 255), int(g * 255), int(b * 255), 210)
    return out


def pick_slides(slides, ev_tab):
    """所見ごとの対象slide。Returns: list of (finding, slide row)"""
    trt = slides[slides.dose_level != "Control"].assign(findings=lambda d: d.finding_types.apply(ex18.as_list))
    counts = trt.findings.explode().dropna().value_counts()
    kept_n = ev_tab[ev_tab.kept].groupby("slide_id").size()
    out = []
    for f in counts[counts >= MIN_SLIDES].index:
        has = trt[trt.findings.apply(lambda v: f in v)]
        picks = []
        for e in has.exp_id.value_counts().index[:3]:
            g = has[has.exp_id == e].assign(d=lambda d: d.dose_level.map(DOSE_ORDER), t=lambda d: d.sacrifice_period.str.split().str[0].astype(int),
                                             n=lambda d: d.slide_id.map(kept_n).fillna(0))
            picks.append(g.sort_values(["d", "t", "n"], ascending=False).iloc[0])
        out += [(f, p) for p in picks]
        first = picks[0]
        ctrl = slides[(slides.exp_id == first.exp_id) & (slides.dose_level == "Control") & (slides.sacrifice_period == first.sacrifice_period)]
        if len(ctrl):
            out.append((f, ctrl.iloc[0]))
    return out


def main():
    emb_name = sys.argv[1] if len(sys.argv) > 1 else "pca0017"
    method = sys.argv[2] if len(sys.argv) > 2 else "consensus"
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    run = sys.argv[4] if len(sys.argv) > 4 else "block6_all"   # 0018の出力（block6_clean_thr0.07など）
    RUN18 = ROOT / "outputs/0018_20261022_ctrl_filter_then_cores" / run
    res = json.loads((RUN18 / "results.json").read_text())
    variant = res.get("source_variant", run)
    variant2 = res.get("stage2_variant", variant)   # 2段目の埋め込みの0017 variant
    if emb_name != "pca0017":
        raise NotImplementedError("pca_refitのPCAは保存していないため、まずpca0017のみ")
    col = f"{method}_k{k}"
    cfg = {**ex18.load_config(EXP18), "source_variant": variant}
    ex17 = ex18.load_exp0017(ROOT, cfg["source_exp"])
    cfg17 = ex17.load_config(ROOT / "experiments" / cfg["source_exp"])
    cfg17.update(cfg17["variants"][cfg["source_variant"]])
    src = ROOT / "outputs" / cfg["source_exp"] / cfg["source_variant"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slides = ex17.select_slides(cfg17, ROOT)
    pca, cached = ex17.cached_pca(cfg17, cfg["source_variant"], ROOT, slides)
    means = cached["means"]   # center_per_slideのときだけ。0017と同じ、抜き出したpatchでのslideの平均
    _, ev, _, row, _ = ex17.cached_eval_embedding(cfg17, cfg["source_variant"], ROOT)
    if variant2 == variant:
        pca2, means2, ev2 = pca, means, ev
    else:   # 2段目は別の埋め込み（例: slideの平均を引いたもの）で、コアを近傍の多数決で付ける
        cfg17b = ex17.load_config(ROOT / "experiments" / cfg["source_exp"])
        cfg17b.update(cfg17b["variants"][variant2])
        pca2, cached2 = ex17.cached_pca(cfg17b, variant2, ROOT, ex17.select_slides(cfg17b, ROOT))
        means2 = cached2["means"]
        _, ev2, _, _, _ = ex17.cached_eval_embedding(cfg17b, variant2, ROOT)
    thr = json.loads((RUN18 / "results.json").read_text())["threshold"]
    ctrl_keep = float(sys.argv[5]) if len(sys.argv) > 5 else None

    # 1段目: 0017の各runの重心と、そのクラスタのControl割合（評価用patchで数えたもの）
    labels = np.load(src / "eval_labels.npy")
    runs = pd.read_csv(src / "runs.csv")
    cz = np.load(src / "centers.npz")
    is_ctrl = (slides.dose_level.to_numpy() == "Control")[row].astype(float)
    centers, cfrac = [], []
    for r, lab in zip(runs.run, labels):
        n = np.bincount(lab, minlength=len(cz[r]))
        centers.append(torch.from_numpy(cz[r]).to(device))
        cfrac.append(torch.from_numpy(np.bincount(lab, weights=is_ctrl, minlength=len(n)) / np.maximum(n, 1)).to(device))

    def score_of(x):
        xt = torch.from_numpy(x).to(device)
        return torch.stack([cf[torch.cdist(xt, c).argmin(1)] for c, cf in zip(centers, cfrac)]).mean(0).cpu().numpy()

    # 2段目の参照: 残した評価用patchとそのコア
    ev_tab = pd.read_parquet(RUN18 / "stage1_scores.parquet")
    if ctrl_keep is not None:
        thr = float(np.quantile(ev_tab.score[ev_tab.group == "Control"], ctrl_keep))
    chk = score_of(ev[:2000])
    if np.isclose(chk, ev_tab.score.to_numpy()[:2000], atol=1e-5).mean() < .99:   # 重心との距離が同点のpatchだけは揺れうる
        raise ValueError("re-computed stage-1 scores do not match 0018")
    ass = pd.read_parquet(RUN18 / "stage2" / emb_name / "assignments.parquet")
    ref, ref_core = ev2[ev_tab.kept.to_numpy()], ass[col].to_numpy()
    tab = pd.read_csv(RUN18 / "stage2" / emb_name / f"cores_{method}_k{k}.csv").set_index("core")
    share = pd.read_csv(RUN18 / "figures" / f"finding_share_{emb_name}_{col}.csv").set_index("core")
    colors = palette(int(ref_core.max()) + 1)

    def core_label(c):
        fs = share.loc[c].sort_values(ascending=False).head(2)
        return f"{tab.loc[c].top_exp[:16]} {tab.loc[c].top_exp_share:.0%} | " + ", ".join(f"{ex18.short(f, 22)} {v:.0%}" for f, v in fs.items())

    out_root = RUN18 / (f"slide_maps_{emb_name}_{col}" + (f"_ctrlkeep{ctrl_keep:g}" if ctrl_keep is not None else ""))
    font = ImageFont.truetype(FONT, 14)
    summary, done = [], {}
    for finding, sl in pick_slides(slides, ev_tab):
        out = out_root / re.sub(r"[^A-Za-z0-9]+", "_", finding).strip("_") / f"{sl.slide_id}_e{sl.exp_id}_{sl.dose_level}_{sl.sacrifice_period.replace(' ', '')}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        if sl.slide_id in done:   # 別の所見で描いたslideは使い回す
            out.write_bytes(done[sl.slide_id][0].read_bytes())
            summary.append({**done[sl.slide_id][1], "finding": finding, "file": str(out.relative_to(out_root))})
            continue
        with h5py.File(ex18.ex17_features(cfg17, ROOT, sl.slide_id), "r") as h:
            f = ex17.l2(h["features"][:].astype(np.float32))
            xy = h["coords"][:]
            size0 = int(h["coords"].attrs.get("patch_size_level0", 256))
        si = int(np.flatnonzero(slides.slide_id.to_numpy() == sl.slide_id)[0])
        emb = pca.transform(f - means[si] if means is not None else f).astype(np.float32)
        emb2 = emb if variant2 == variant else pca2.transform(f - means2[si] if means2 is not None else f).astype(np.float32)
        score = score_of(emb)
        kept = score <= thr
        core = np.full(len(emb), -2)              # -2 = 1段目で除外
        if kept.any():
            core[kept] = propagate_labels(emb2[kept], ref, ref_core, KNN, device)
        w = openslide.OpenSlide(sl.svs_path)
        th = w.get_thumbnail((THUMB, THUMB)).convert("RGB")
        scale = th.width / w.dimensions[0]
        w.close()
        over = Image.new("RGBA", th.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        box = max(2, round(size0 * scale))
        for (x, y), c in zip(xy, core):
            if c == -2:
                continue
            d.rectangle([x * scale, y * scale, x * scale + box, y * scale + box], fill=colors.get(int(c), (70, 70, 70, 200)))
        mapped = Image.alpha_composite(th.convert("L").convert("RGBA"), over).convert("RGB")
        counts = pd.Series(core).value_counts()
        present = [c for c in counts.index if c >= 0 and counts[c] / len(core) >= 0.005]
        canvas = Image.new("RGB", (th.width * 2 + 10, th.height + 22 * (len(present) + 3) + 40), "white")
        canvas.paste(th, (0, 30))
        canvas.paste(mapped, (th.width + 10, 30))
        cd = ImageDraw.Draw(canvas)
        fl = ex18.as_list(sl.finding_types)
        cd.text((6, 6), f"{sl.slide_id}  e{sl.exp_id} {sl.compound_name}  {sl.dose_level}  {sl.sacrifice_period}  {sl.scanner}  "
                        f"findings: {', '.join(fl) if fl else 'none recorded'}", fill="black", font=font)
        y0 = th.height + 36
        cd.text((6, y0), f"{len(core):,} patches{' (2,048-patch subset)' if len(core) <= 2048 else ''}, stage-1 threshold {thr:.3f}: excluded (unpainted) "
                         f"{counts.get(-2, 0) / len(core):.0%}, kept but no core (dark grey) {counts.get(-1, 0) / len(core):.0%}, "
                         f"in cores {sum(counts.get(c, 0) for c in present) / len(core):.0%} (k={k} {method} on {emb_name}); "
                         f"legend: core, % of this slide, top compound share, top findings share", fill="black", font=font)
        for i, c in enumerate(present):
            yy = y0 + 22 * (i + 1)
            cd.rectangle([6, yy + 2, 22, yy + 16], fill=colors[c][:3])
            cd.text((28, yy), f"core {c}: {counts[c] / len(core):.1%}   {core_label(c)}", fill="black", font=font)
        canvas.save(out, quality=88)
        summary.append({"finding": finding, "slide_id": sl.slide_id, "exp_id": sl.exp_id, "compound": sl.compound_name,
                        "dose": sl.dose_level, "period": sl.sacrifice_period, "scanner": sl.scanner, "n_patches": len(core),
                        "frac_kept": float(kept.mean()), "frac_in_cores": float((core >= 0).mean()),
                        "top_cores": "; ".join(f"{c} {counts[c] / len(core):.0%}" for c in present[:3]), "file": str(out.relative_to(out_root))})
        done[sl.slide_id] = (out, summary[-1])
        print(out.relative_to(out_root), f"kept {kept.mean():.2f}", flush=True)
    pd.DataFrame(summary).to_csv(out_root / "summary.csv", index=False)
    print(out_root)


if __name__ == "__main__":
    main()
