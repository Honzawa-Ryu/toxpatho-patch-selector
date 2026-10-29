"""脂肪変性が出る試験（CCl4 66・67、cholesterol 698）で、所見ありの肝臓と対照群の肝臓のpatchを無作為に並べて見比べる（2026-10-29ユーザー）。

各試験で、対照群と、脂肪変性の程度（1 minimal … 4 severe）ごとに、別々のslideから2枚ずつ最大16枚。patchはh5の座標（組織のpatch）から無作為。
出力: outputs/fatty_vs_control_20261029/{exp}.jpg
usage: python scripts/fatty_vs_control_patches.py  （コンテナ内、Slurm経由）
"""
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/0026_20261027_fatty_degeneration_prototypes/tree_v3"
OUT = ROOT / "outputs/fatty_vs_control_20261029"
FEAT = [ROOT / "outputs/0016_20261013_tier1_macenko_embeddings/20x_256px_0px_overlap/features_uni_v2_macenko",
        ROOT / "outputs/0016_20261013_tier1_macenko_embeddings/20x_256px_0px_overlap/features_uni_v2_macenko_sub2048"]
GRADE = {0: "Control / no fatty degeneration", 1: "grade 1 (minimal)", 2: "grade 2 (slight)", 3: "grade 3 (moderate)", 4: "grade 4 (severe)"}


def main():
    OUT.mkdir(exist_ok=True)
    spec = importlib.util.spec_from_file_location("exp0017", ROOT / "experiments/0017_20261016_block_consensus_patterns/experiment.py")
    ex17 = importlib.util.module_from_spec(spec); sys.modules["exp0017"] = ex17; spec.loader.exec_module(ex17)
    sl = pd.read_csv(RUN / "slides.csv", dtype=str).merge(pd.read_csv(RUN / "slide_fat_grades.csv", dtype=str), on="slide_id")
    sl["fat"] = sl["Degeneration, fatty"].astype(float).astype(int)
    rng = np.random.default_rng(0)
    per_slide, n_slides = 2, 8
    for e in sorted(sl.exp.unique(), key=int):
        se = sl[sl.exp == e]
        rows, summary = [], []
        groups = [("ctrl", se[se.dose == "Control"])] + [(g, se[(se.dose != "Control") & (se.fat == g)]) for g in sorted(se.fat.unique()) if g > 0]
        for g, d in groups:
            if not len(d):
                continue
            chosen = d.sample(min(n_slides, len(d)), random_state=0)
            xy_all, row_all, sdf, size0 = [], [], [], {}
            for i, r in enumerate(chosen.itertuples()):
                h5 = next((p / f"{r.slide_id}.h5" for p in FEAT if (p / f"{r.slide_id}.h5").exists()), None)
                if h5 is None:
                    continue
                with h5py.File(h5, "r") as h:
                    xy = h["coords"][:]
                    size0[r.slide_id] = int(h["coords"].attrs["patch_size_level0"])
                pick = xy[rng.choice(len(xy), per_slide, replace=False)]
                xy_all.append(pick); row_all += [len(sdf)] * per_slide
                sdf.append({"slide_id": r.slide_id, "svs_path": r.svs_path, "dose": r.dose, "period": r.period, "fat": r.fat})
            if not sdf:
                continue
            xy_all, row_all, sdf = np.concatenate(xy_all), np.array(row_all), pd.DataFrame(sdf)
            title = ("Control" if g == "ctrl" else GRADE[g]) + f"  ({len(d)} slides in this group; {len(sdf)} shown, {per_slide} patches each)"
            lbl = lambda s_, q_: (f"{s_.slide_id} {s_.dose}", f"{s_.period}" + ("" if s_.fat == 0 else f"  fatty {s_.fat}"))
            rows.append((title, ex17.read_tiles(list(range(len(row_all))), row_all, xy_all, sdf, size0, lbl)))
            summary.append((g, len(d), len(sdf)))
        cmp = se.compound.iloc[0]
        ex17.draw_sheet(f"Experiment {e}: {cmp}. Random tissue patches, control vs fatty degeneration by recorded grade", rows, OUT / f"exp{e}.jpg")
        print(e, cmp, summary)


if __name__ == "__main__":
    main()
