import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

import openslide
from PIL import Image, ImageDraw, ImageFont
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_config(exp_dir: Path) -> dict:
    """Load config.yml from the experiment directory."""
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml")   # run_slurm.shが渡す。読むのは常に実験ディレクトリのconfig.yml
    return parser.parse_args()


FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
GRADES = ["minimal", "slight", "moderate", "severe"]
# ラベルの色（重ね描き・拡大画像の輪郭）。構造は寒色、病変は暖色
CLASS_COLORS = {
    "Parenchyma": (190, 190, 190), "Portal tract": (60, 120, 220), "Central vein": (40, 180, 200),
    "Large vessels": (100, 100, 230), "Connective tissue": (120, 90, 200),
    "Necrosis (area)": (230, 40, 40), "Necrosis (single cell)": (250, 120, 30), "Single cell necrosis/apoptosis": (240, 170, 0),
    "Mitosis": (220, 0, 200), "Inflammatory infiltrate": (40, 170, 60), "Extramedullary hematopoiesis": (130, 200, 40),
    "Dark hepatocytes": (120, 60, 20), "Hemorrhage": (150, 0, 40), "Neoplasia": (0, 0, 0),
    "Oval cell/bile duct proliferation": (0, 140, 120), "Other lesions": (90, 90, 90),
    "Staining, contamination, and out-of-focus artifact": (255, 230, 0),
}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.dpi": 140})


def short(name, n=28):
    return name.replace("Staining, contamination, and out-of-focus artifact", "Artifact (stain/contam./focus)")[:n]


def load_annotations(ann_dir: Path) -> pd.DataFrame:
    """GeoJSONのポリゴンを1行1つの表にする（面積・重心・外接矩形はlevel 0の画素）。"""
    from shapely.geometry import shape
    recs, geoms = [], []
    for f in sorted((ann_dir / "annotations/geojson").glob("*.svs.geojson")):
        sid = f.name.split(".")[0]
        for i, ft in enumerate(json.loads(f.read_text())["features"]):
            g = shape(ft["geometry"])
            if not g.is_valid:
                g = g.buffer(0)
            p = ft["properties"]
            x0, y0, x1, y1 = g.bounds
            recs.append({"slide_id": sid, "idx": i, "label": p["classification"]["name"], "author": p.get("author", ""),
                         "area_px": g.area, "cx": g.centroid.x, "cy": g.centroid.y, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
            geoms.append(g)
    t = pd.DataFrame(recs)
    t["geom"] = geoms
    return t


def slide_table(cfg, project_root, ann):
    sl = pd.DataFrame(json.loads((project_root / cfg["annotation_dir"] / "slides.json").read_text())["slides"])
    sl["slide_id"] = sl.filename.str.replace(".svs", "", regex=False)
    q = pd.read_csv(cfg["qc_catalog"], dtype={"slide_id": str, "EXP_ID": str, "GROUP_ID": str, "INDIVIDUAL_ID": str})
    sl = sl.merge(q[["slide_id", "EXP_ID", "GROUP_ID", "INDIVIDUAL_ID", "COMPOUND_NAME", "SINGLE_REPEAT_TYPE", "SACRIFICE_PERIOD",
                     "DOSE_LEVEL", "scanscope_id", "mpp", "tier"]], on="slide_id", how="left")

    def svs(sid):
        for d in cfg["wsi_dirs"]:
            p = Path(d) / f"{sid}.svs"
            if p.exists():
                return str(p)
        return None
    sl["svs_path"] = sl.slide_id.map(svs)
    # 病理所見の記録（個体で結びつける）
    p = pd.read_csv(project_root / cfg["pathology_csv"], dtype=str)
    p = p[p.ORGAN == "Liver"]
    key = ["EXP_ID", "GROUP_ID", "INDIVIDUAL_ID"]
    for d in (sl, p):
        for c in key:
            d[c + "_k"] = pd.to_numeric(d[c], errors="coerce")
    j = sl[["slide_id"] + [c + "_k" for c in key]].merge(p, on=[c + "_k" for c in key], how="inner")
    j["grade_ord"] = j.GRADE_TYPE.map({g: i + 1 for i, g in enumerate(GRADES)})
    j["spontaneous"] = j.SP_FLG.str.lower() == "true"
    return sl.drop(columns=[c + "_k" for c in key]), j


def tissue_mask(svs_path, cfg):
    w = openslide.OpenSlide(svs_path)
    lv = w.get_best_level_for_downsample(cfg["tissue_downsample"])
    ds = w.level_downsamples[lv]
    img = np.asarray(w.read_region((0, 0), lv, w.level_dimensions[lv]).convert("RGB"), dtype=np.float32)
    w.close()
    return img.mean(axis=2) < cfg["tissue_gray_max"], ds


def rasterize_polys(polys, shape_hw, ds):
    from PIL import Image as _I, ImageDraw as _D
    img = _I.new("L", (shape_hw[1], shape_hw[0]), 0)
    d = _D.Draw(img)
    for g in polys:
        for part in getattr(g, "geoms", [g]):
            pts = [(x / ds, y / ds) for x, y in part.exterior.coords]
            if len(pts) >= 3:
                d.polygon(pts, fill=1)
    return np.asarray(img, dtype=bool)


def overlay(sid, svs_path, polys, meta, cfg, out_path):
    w = openslide.OpenSlide(svs_path)
    th = w.get_thumbnail((cfg["thumb_size"], cfg["thumb_size"])).convert("RGB")
    s = th.width / w.dimensions[0]
    w.close()
    over = Image.new("RGBA", th.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    order = ["Parenchyma", "Large vessels", "Portal tract", "Central vein", "Connective tissue"]   # 構造を先に、病変を後に描く
    polys = sorted(polys, key=lambda lg: order.index(lg[0]) if lg[0] in order else len(order))
    present = []
    for lab, g in polys:
        c = CLASS_COLORS.get(lab, (90, 90, 90))
        fill = (*c, 40) if lab == "Parenchyma" else (*c, 110)
        for part in getattr(g, "geoms", [g]):
            pts = [(x * s, y * s) for x, y in part.exterior.coords]
            if len(pts) >= 3:
                d.polygon(pts, fill=fill, outline=(*c, 255))
        if lab not in present:
            present.append(lab)
    img = Image.alpha_composite(th.convert("RGBA"), over).convert("RGB")
    font = ImageFont.truetype(FONT, 14)
    cnt = pd.Series([lab for lab, _ in polys]).value_counts()
    leg_h = 20 * (len(present) + 1) + 10
    canvas = Image.new("RGB", (th.width * 2 + 10, th.height + leg_h + 30), "white")
    canvas.paste(th, (0, 30))
    canvas.paste(img, (th.width + 10, 30))
    cd = ImageDraw.Draw(canvas)
    cd.text((6, 6), meta, fill="black", font=font)
    for i, lab in enumerate(present):
        y = th.height + 36 + 20 * i
        cd.rectangle([6, y + 2, 20, y + 15], fill=CLASS_COLORS.get(lab, (90, 90, 90)))
        cd.text((26, y), f"{lab}: {cnt[lab]}", fill="black", font=font)
    canvas.save(out_path, quality=85)


def gallery_tile(row, svs_path, cfg):
    """アノテーション1つを切り出し、ポリゴンの輪郭を描いて返す。"""
    pad, side_max, T = cfg["gallery_pad"], cfg["gallery_max_side"], cfg["gallery_tile"]
    x0, y0, x1, y1 = row.x0 - pad, row.y0 - pad, row.x1 + pad, row.y1 + pad
    side = max(x1 - x0, y1 - y0, 192)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    w = openslide.OpenSlide(svs_path)
    lv = w.get_best_level_for_downsample(max(1.0, side / side_max)) if side > side_max else 0
    ds = w.level_downsamples[lv]
    img = w.read_region((max(x0, 0), max(y0, 0)), lv, (int(side / ds), int(side / ds))).convert("RGB")
    w.close()
    d = ImageDraw.Draw(img)
    c = CLASS_COLORS.get(row.label, (90, 90, 90))
    for part in getattr(row.geom, "geoms", [row.geom]):
        pts = [((x - max(x0, 0)) / ds, (y - max(y0, 0)) / ds) for x, y in part.exterior.coords]
        if len(pts) >= 3:
            d.line(pts + [pts[0]], fill=c, width=max(2, int(3 / ds) + 1))
    return img.resize((T, T)), side


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))
    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    parse_args()
    exp_name = os.environ["EXP_NAME"]
    cfg = load_config(Path(__file__).parent)
    run_dir = get_run_dir(project_root, __file__, "default", output_root=os.environ.get("OUTPUT_ROOT"))
    logger = setup_logger(run_dir, exp_name)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key="default", config=cfg)
    logger.info(f"Starting: {exp_name}")
    rng = np.random.default_rng(cfg["seed"])

    ann = load_annotations(project_root / cfg["annotation_dir"])
    sl, path = slide_table(cfg, project_root, ann)
    ann = ann.merge(sl[["slide_id", "COMPOUND_NAME", "DOSE_LEVEL", "SACRIFICE_PERIOD", "mpp"]], on="slide_id", how="left")
    ann["area_um2"] = ann.area_px * ann.mpp ** 2
    ann.drop(columns=["geom"]).to_parquet(run_dir / "annotations.parquet", index=False)
    local = sl[sl.svs_path.notna()]
    logger.info(f"{len(ann):,} polygons on {ann.slide_id.nunique()} slides; svs available for {len(local)} slides")

    # --- slideごとのまとめ ---
    cnt = ann.pivot_table(index="slide_id", columns="label", values="idx", aggfunc="count", fill_value=0).add_prefix("n:")
    area = ann.pivot_table(index="slide_id", columns="label", values="area_um2", aggfunc="sum", fill_value=0).add_prefix("um2:")
    auth = ann.groupby("slide_id").author.agg(lambda v: ";".join(sorted(set(v))))
    rec = path[~path.spontaneous].groupby("slide_id").apply(lambda d: "; ".join(f"{a} ({b})" for a, b in zip(d.FINDING_TYPE, d.GRADE_TYPE)))
    summ = sl.set_index("slide_id").join([cnt, area, auth.rename("authors"), rec.rename("recorded_findings")])

    # 組織の面積・実質の被覆・座標の確認（手元のsvsのみ）
    t0 = time.time()
    geo = []
    for r in local.itertuples():
        polys = list(zip(ann.label[ann.slide_id == r.slide_id], ann.geom[ann.slide_id == r.slide_id]))
        mask, ds = tissue_mask(r.svs_path, cfg)
        tissue_px = mask.sum() * ds * ds
        from shapely.ops import unary_union
        par = [g for lab, g in polys if lab == "Parenchyma"]
        les = [g for lab, g in polys if lab not in ("Large vessels", "Central vein", "Portal tract", "Parenchyma")]
        rec_ = {"slide_id": r.slide_id, "tissue_area_px": float(tissue_px),
                "parenchyma_frac_of_tissue": float(unary_union(par).area / tissue_px) if par and tissue_px > 0 else np.nan}
        allm = rasterize_polys([g for _, g in polys], mask.shape, ds)
        rec_["annot_in_tissue_frac"] = float((allm & mask).sum() / max(allm.sum(), 1))
        if les:
            lm = rasterize_polys(les, mask.shape, ds)
            rec_["lesion_in_tissue_frac"] = float((lm & mask).sum() / max(lm.sum(), 1))
        geo.append(rec_)
    geo = pd.DataFrame(geo).set_index("slide_id")
    summ = summ.join(geo)
    summ.to_csv(run_dir / "slide_summary.csv")
    logger.info(f"tissue/coverage computed for {len(geo)} slides; {time.time() - t0:.0f}s")

    out = run_dir / "figures"
    out.mkdir(exist_ok=True)

    # --- A. slide全体への重ね描き ---
    t0 = time.time()
    (out / "overlays").mkdir(exist_ok=True)
    for r in local.itertuples():
        polys = list(zip(ann.label[ann.slide_id == r.slide_id], ann.geom[ann.slide_id == r.slide_id]))
        meta = (f"{r.slide_id}  e{r.EXP_ID} {r.COMPOUND_NAME}  {r.DOSE_LEVEL}  {r.SACRIFICE_PERIOD} ({r.SINGLE_REPEAT_TYPE})  {r.scanscope_id} {r.tier}  "
                f"| recorded: {summ.recorded_findings.get(r.slide_id) if isinstance(summ.recorded_findings.get(r.slide_id), str) else 'none'}")
        fname = f"{str(r.COMPOUND_NAME).replace(' ', '_').replace('/', '_')[:30]}_{r.slide_id}.jpg"
        overlay(r.slide_id, r.svs_path, polys, meta[:220], cfg, out / "overlays" / fname)
    logger.info(f"overlays done; {time.time() - t0:.0f}s")

    # --- B. ラベルごとの拡大画像 ---
    t0 = time.time()
    font = ImageFont.truetype(FONT, 11)
    T = cfg["gallery_tile"]
    (out / "galleries").mkdir(exist_ok=True)
    loc = ann[ann.slide_id.isin(local.slide_id)].merge(local[["slide_id", "svs_path"]], on="slide_id")
    for lab, d in loc.groupby("label"):
        d = d.sample(frac=1, random_state=cfg["seed"])
        picks = d.drop_duplicates("slide_id").head(cfg["gallery_n"])
        if len(picks) < cfg["gallery_n"]:   # slideが足りなければ同じslideから追加
            picks = pd.concat([picks, d[~d.index.isin(picks.index)].head(cfg["gallery_n"] - len(picks))])
        cols = 6
        rows_ = int(np.ceil(len(picks) / cols))
        sheet = Image.new("RGB", (cols * T, 30 + rows_ * (T + 28)), "white")
        sd = ImageDraw.Draw(sheet)
        sd.text((6, 6), f"{lab}: {len(d):,} annotations on {d.slide_id.nunique()} local slides (random, one per slide); outline = polygon; "
                        f"label: slide, compound, dose, author, size", fill="black", font=ImageFont.truetype(FONT, 13))
        for i, r in enumerate(picks.itertuples()):
            img, side = gallery_tile(r, r.svs_path, cfg)
            x, y = (i % cols) * T, 30 + (i // cols) * (T + 28)
            sd.text((x + 3, y + 1), f"{r.slide_id} {str(r.COMPOUND_NAME)[:14]} {str(r.DOSE_LEVEL)[0]}", fill="black", font=font)
            sd.text((x + 3, y + 14), f"{r.author.replace('pathologist-', 'p')} | view {side * r.mpp:.0f} um", fill="#555555", font=font)
            sheet.paste(img, (x, y + 28))
        sheet.save(out / "galleries" / f"{lab.replace('/', '_').replace(' ', '_').replace(',', '')}.jpg", quality=88)
    logger.info(f"galleries done; {time.time() - t0:.0f}s")

    # --- C. 数の確認 ---
    labs = ann.groupby("label").agg(n=("idx", "size"), slides=("slide_id", "nunique"), area_mm2=("area_um2", lambda v: v.sum() / 1e6)).sort_values("slides", ascending=False)
    labs.to_csv(run_dir / "label_summary.csv")
    au = pd.crosstab(ann.label, ann.author).reindex(labs.index)
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5), gridspec_kw={"width_ratios": [1, 1.2]})
    ax = axes[0]
    y = np.arange(len(labs))
    ax.barh(y, labs.slides, color="#2a78d6")
    for i, r in enumerate(labs.itertuples()):
        ax.text(r.slides, i, f" {r.slides} slides, {r.n:,} polygons", va="center", fontsize=7.5)
    ax.set_yticks(y, [short(l, 34) for l in labs.index])
    ax.invert_yaxis()
    ax.set_xlim(0, 300)
    ax.set_title("slides and polygons per label (all 230 slides)")
    ax = axes[1]
    left = np.zeros(len(au))
    pal = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#9a6bd4", "#9aa3ad"]
    for c, col in zip(au.columns, pal):
        ax.barh(y, au[c] / au.sum(1), left=left, color=col, label=c)
        left += (au[c] / au.sum(1)).to_numpy()
    ax.set_yticks(y, [])
    ax.invert_yaxis()
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.set_title("share of polygons by author")
    fig.suptitle("C1. What is annotated, and by whom", fontsize=12)
    fig.savefig(out / "C1_labels_and_authors.png")
    plt.close(fig)

    # C2. 記録された所見と、描かれた量
    tr = path[~path.spontaneous]
    pairs = [("Necrosis", ["Necrosis (area)"], "area_mm2", "necrosis area (mm2)"),
             ("Necrosis", ["Necrosis (single cell)", "Single cell necrosis/apoptosis"], "count", "single-cell necrosis/apoptosis (count)"),
             ("Single cell necrosis", ["Necrosis (single cell)", "Single cell necrosis/apoptosis"], "count", "single-cell necrosis/apoptosis (count)"),
             ("Increased mitosis", ["Mitosis"], "count", "mitoses (count)"),
             ("Cellular infiltration", ["Inflammatory infiltrate"], "count", "inflammatory infiltrates (count)"),
             ("Hematopoiesis, extramedullary", ["Extramedullary hematopoiesis"], "count", "EMH (count)"),
             ("Proliferation, bile duct", ["Oval cell/bile duct proliferation"], "count", "oval cell/bile duct prolif. (count)")]
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.6 * len(pairs), 4.6), squeeze=False)
    conc = []
    for ax, (finding, labels_, kind, ylab) in zip(axes[0], pairs):
        a = ann[ann.label.isin(labels_)].groupby("slide_id")
        v = (a.area_um2.sum() / 1e6) if kind == "area_mm2" else a.size()
        v = v.reindex(sl.slide_id, fill_value=0)
        g = tr[tr.FINDING_TYPE == finding].groupby("slide_id").grade_ord.max().reindex(sl.slide_id).fillna(0)
        cats = ["none"] + GRADES
        data = [v[g == i].to_numpy() for i in range(5)]
        ax.boxplot(data, widths=.5, showfliers=False)
        for i, dd in enumerate(data):
            ax.scatter(i + 1 + rng.uniform(-.18, .18, len(dd)), dd, s=7, alpha=.5, color="#e34948", linewidths=0)
            ax.text(i + 1, 1.02, f"n={len(dd)}\n{(dd > 0).mean():.0%}>0" if len(dd) else "n=0", transform=ax.get_xaxis_transform(), ha="center", fontsize=6.5)
        ax.set_xticks(range(1, 6), ["none", "min", "sli", "mod", "sev"], fontsize=7.5)
        ax.set_title(f"record: {finding[:24]}", fontsize=8.5, pad=24)
        ax.set_ylabel(ylab, fontsize=8)
        rec_pos = g > 0
        conc.append({"recorded_finding": finding, "annotation": "+".join(labels_), "slides_recorded": int(rec_pos.sum()),
                     "recorded_with_annotation": float((v[rec_pos] > 0).mean()) if rec_pos.any() else np.nan,
                     "not_recorded_with_annotation": float((v[~rec_pos] > 0).mean())})
    fig.suptitle("C2. Annotated amount vs the TG-GATEs record of the same animal (x: recorded grade; '% >0' = slides with any annotation)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "C2_annotation_vs_record.png")
    plt.close(fig)
    pd.DataFrame(conc).to_csv(run_dir / "annotation_vs_record.csv", index=False)

    # C3. 実質の被覆と座標の確認
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, c, title in [(axes[0], "parenchyma_frac_of_tissue", "Parenchyma polygons / tissue area"),
                         (axes[1], "annot_in_tissue_frac", "share of all polygon area on tissue (coordinate check)"),
                         (axes[2], "lesion_in_tissue_frac", "share of lesion polygon area on tissue")]:
        v = summ[c].dropna()
        ax.hist(v, bins=30, color="#2a78d6")
        ax.set_title(f"{title}\nmedian {v.median():.2f} (n={len(v)})", fontsize=9.5)
    fig.suptitle("C3. Coverage and coordinate check (local svs only; tissue = thumbnail pixels darker than threshold)", fontsize=11)
    fig.savefig(out / "C3_coverage_and_alignment.png")
    plt.close(fig)

    results = {"n_slides": int(len(sl)), "n_local_svs": int(len(local)), "n_polygons": int(len(ann)),
               "median_parenchyma_frac": float(summ.parenchyma_frac_of_tissue.median()),
               "median_annot_in_tissue_frac": float(summ.annot_in_tissue_frac.median()), "annotation_vs_record": conc}
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
