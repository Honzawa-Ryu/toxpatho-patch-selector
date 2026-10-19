"""0017のコアを、slideの全patchに広げて縮小画像の上に色で描く（scripts/plot_cluster_spatial_map.py の0017版）。

コアは評価用patch（1枚50）にしか付いていないので、各patchを0017と同じPCA空間に写し、
評価用patchの近傍15点の多数決でコア（または未割当）を決める（lib.clustering.propagate_labels）。
色: Controlのpatch割合が5%未満のコアははっきりした色、それ以外（背景）は薄い灰色、未割当は塗らない。
対象: 13試験それぞれから、所見ありのHigh用量slideを1枚（Controlの少ないコアの評価用patchが最も多いもの）とControl 1枚。

usage: .venv/bin/python scripts/consensus_core_slide_maps.py [slide_id ...]
"""
import colorsys
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.clustering import propagate_labels  # noqa: E402

EXP = ROOT / "experiments/0017_20261016_block_consensus_patterns"
RUN = ROOT / "outputs/0017_20261016_block_consensus_patterns/pilot"
VIS = sorted(EXP.glob("visualization_*"))[-1]
OUT = RUN / "slide_maps_leiden_tau0.8"
COL = "leiden_tau0.8"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
THUMB = 1400
KNN = 15

spec = importlib.util.spec_from_file_location("exp0017", EXP / "experiment.py")
exp0017 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp0017)


def pca_model(cfg, slides):
    """0017と同じ手順・seedでPCAを作り直し、保存する（2回目からは保存分を読む）。"""
    path = VIS / "pca.npz"
    if not path.exists():
        class Quiet:
            def info(self, *_):
                pass
        feats, rows, _ = exp0017.load_subsets(slides, ROOT / cfg["features_dir"], cfg, Quiet())
        is_ctrl = (slides.dose_level == "Control").to_numpy()
        c = np.flatnonzero(is_ctrl[rows["pca"]])
        t = np.random.default_rng(cfg["seed"]).choice(np.flatnonzero(~is_ctrl[rows["pca"]]), size=len(c), replace=False)
        p = PCA(n_components=cfg["pca_components"], svd_solver="randomized", random_state=cfg["seed"]).fit(feats["pca"][np.r_[c, t]])
        np.savez(path, components=p.components_, mean=p.mean_)
    z = np.load(path)
    return z["components"], z["mean"]


def palette(ids):
    out = {}
    for i, c in enumerate(ids):
        r, g, b = colorsys.hsv_to_rgb((i * .61803398875) % 1, .75, .85 - .25 * (i % 2))
        out[c] = (int(r * 255), int(g * 255), int(b * 255), 210)
    return out


def main():
    cfg = yaml.safe_load(open(EXP / "config.yml"))
    m = pd.read_parquet(ROOT / cfg["manifest_parquet"])
    m["slide_id"] = m.slide_id.astype(str)
    slides = m[m.exp_id.isin(cfg["exp_ids"])].sort_values(["exp_id", "dose", "sacrifice_period", "individual_id"]).reset_index(drop=True)
    comp, mean = pca_model(cfg, slides)

    # 参照: 評価用patch（空白除外後）の埋め込みとコア
    z = np.load(VIS / "eval_embedding.npz")
    ev_all, erow, exy = z["ev"], z["erow"], z["exy"]
    kept = pd.read_parquet(RUN / "eval_patches.parquet")
    key = pd.DataFrame({"slide_id": slides.slide_id.to_numpy()[erow], "x": exy[:, 0], "y": exy[:, 1], "i": np.arange(len(erow))})
    j = kept.merge(key, on=["slide_id", "x", "y"], how="left", validate="one_to_one")
    ref, ref_core = ev_all[j.i.to_numpy().astype(int)], j[COL].to_numpy()

    tab = pd.read_csv(RUN / f"cores_{COL}.csv")
    vivid = tab[tab.ctrl_patch_frac < 0.05].sort_values("core")
    colors = palette(list(vivid.core))
    label_of = {r.core: (r.top_lift_findings.split(";")[0] if isinstance(r.top_lift_findings, str) else "") for r in vivid.itertuples()}

    if len(sys.argv) > 1:
        targets = slides[slides.slide_id.isin(sys.argv[1:])]
    else:
        kept["vivid"] = kept[COL].isin(vivid.core)
        score = kept.groupby("slide_id").vivid.sum()
        picks = []
        for e, g in slides.groupby("exp_id"):
            hi = g[(g.dose_level == "High") & g.has_finding]
            if len(hi):
                picks.append(hi.assign(s=hi.slide_id.map(score).fillna(0)).sort_values("s", ascending=False).iloc[0])
            picks.append(g[g.dose_level == "Control"].sample(1, random_state=0).iloc[0])
        targets = pd.DataFrame(picks)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    font = ImageFont.truetype(FONT, 14)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = []
    for sl in targets.itertuples():
        with h5py.File(ROOT / cfg["features_dir"] / f"{sl.slide_id}.h5", "r") as h:
            f = exp0017.l2(h["features"][:].astype(np.float32))
            xy = h["coords"][:]
            size0 = int(h["coords"].attrs.get("patch_size_level0", 256))
        emb = ((f - mean) @ comp.T).astype(np.float32)
        lab = propagate_labels(emb, ref, ref_core, KNN, device)
        w = openslide.OpenSlide(sl.svs_path)
        th = w.get_thumbnail((THUMB, THUMB)).convert("RGB")
        scale = th.width / w.dimensions[0]
        w.close()
        gray = th.convert("L").convert("RGB")
        over = Image.new("RGBA", gray.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        box = max(2, round(size0 * scale))
        for (x, y), c in zip(xy, lab):
            if c < 0:
                continue
            fill = colors.get(int(c), (150, 160, 170, 120))
            d.rectangle([x * scale, y * scale, x * scale + box, y * scale + box], fill=fill)
        mapped = Image.alpha_composite(gray.convert("RGBA"), over).convert("RGB")
        counts = pd.Series(lab).value_counts()
        present = [c for c in counts.index if c in colors and counts[c] / len(lab) >= 0.005]
        legend_h = 22 * (len(present) + 3)
        canvas = Image.new("RGB", (th.width * 2 + 10, max(th.height, 0) + legend_h + 30), "white")
        canvas.paste(th, (0, 30))
        canvas.paste(mapped, (th.width + 10, 30))
        cd = ImageDraw.Draw(canvas)
        cd.text((6, 6), f"{sl.slide_id}  e{sl.exp_id} {sl.compound_name}  {sl.dose_level}  {sl.sacrifice_period}  "
                        f"findings: {', '.join(sl.finding_types) if len(sl.finding_types) else 'none recorded'}", fill="black", font=font)
        y0 = th.height + 36
        bg = int(sum(counts.get(c, 0) for c in counts.index if c >= 0 and c not in colors))
        cd.text((6, y0), f"{len(lab):,} patches: Control-poor cores {sum(counts.get(c, 0) for c in colors) / len(lab):.0%}, "
                         f"background cores (light grey) {bg / len(lab):.0%}, unassigned (unpainted) {counts.get(-1, 0) / len(lab):.0%}",
                fill="black", font=font)
        for i, c in enumerate(present):
            yy = y0 + 22 * (i + 1)
            cd.rectangle([6, yy + 2, 22, yy + 16], fill=colors[c][:3])
            cd.text((28, yy), f"core {c}: {counts[c] / len(lab):.1%}  {label_of.get(c, '')}", fill="black", font=font)
        out = OUT / f"{sl.slide_id}_e{sl.exp_id}_{sl.dose_level}.jpg"
        canvas.save(out, quality=88)
        summary.append({"slide_id": sl.slide_id, "exp_id": sl.exp_id, "compound": sl.compound_name, "dose": sl.dose_level,
                        "period": sl.sacrifice_period, "n_patches": len(lab),
                        "frac_control_poor_cores": sum(counts.get(c, 0) for c in colors) / len(lab),
                        "frac_unassigned": counts.get(-1, 0) / len(lab), "file": out.name})
        print(out.name, f"{summary[-1]['frac_control_poor_cores']:.2f}", flush=True)
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    print(OUT)


if __name__ == "__main__":
    main()
