"""0017のコアのうち条件に合うものを選んで、画像シートを作る（k-meansは回し直さない）。

既定はControlのpatch割合が5%未満のコア（コアマップの左端）。上位1試験のシェアが低い（試験をまたぐ）順に並べ、
1ファイル10コアずつに分けて出す。各コアは中心性（コア内の平均共所属率）の高い順に、slideが重ならないよう24 patch。
usage: .venv/bin/python scripts/consensus_core_sheets.py [max_ctrl_frac] [method] [tau]   既定: 0.05 leiden 0.8
"""
import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments/0017_20261016_block_consensus_patterns"
VARIANT = __import__("os").environ.get("CORE_VARIANT", "pilot")   # 0017のvariant（pilot / block3）
RUN = ROOT / "outputs/0017_20261016_block_consensus_patterns" / VARIANT
PER_FILE = 10

spec = importlib.util.spec_from_file_location("exp0017", EXP / "experiment.py")
exp0017 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp0017)


def main():
    max_ctrl = float(sys.argv[1]) if len(sys.argv) > 1 else 0.05
    method = sys.argv[2] if len(sys.argv) > 2 else "leiden"
    tau = sys.argv[3] if len(sys.argv) > 3 else "0.8"
    col = f"{method}_tau{tau}"
    # PER_K=200 なら、kごとのコンセンサス（scripts/consensus_per_k.py）のk=200のコアのうち、
    # 全粒度で入れ子になりControlが少ないもの（chainsの nested_all_levels）を対象にする
    per_k = __import__("os").environ.get("PER_K")
    cfg = yaml.safe_load(open(EXP / "config.yml"))
    cfg.update(cfg.get("variants", {}).get(VARIANT, {}))

    ev = pd.read_parquet(RUN / "eval_patches.parquet")
    edges = pd.read_parquet(RUN / "edges.parquet")
    a, b, rate = edges.a.to_numpy(), edges.b.to_numpy(), edges.rate.to_numpy()
    if per_k:
        col = f"core_k{per_k}"
        ev = ev[["slide_id", "x", "y"]].merge(pd.read_parquet(RUN / "per_k" / f"assignments_tau{tau}.parquet"),
                                              on=["slide_id", "x", "y"], how="left", validate="one_to_one")
        rate = edges[f"rate_k{per_k}"].to_numpy()
    core = ev[col].to_numpy()
    n = len(ev)
    same = (core[a] == core[b]) & (core[a] >= 0)
    c_sum = np.bincount(a[same], weights=rate[same], minlength=n) + np.bincount(b[same], weights=rate[same], minlength=n)
    c_n = np.bincount(a[same], minlength=n) + np.bincount(b[same], minlength=n)
    centrality = np.where(c_n > 0, c_sum / np.maximum(c_n, 1), 0.0)

    m = pd.read_parquet(ROOT / cfg["manifest_parquet"])
    m["slide_id"] = m.slide_id.astype(str)
    scan = pd.read_csv(ROOT / "experiments/0010_20260925_pathology_benchmark_subset/label_audit/slide_scanner_meta.csv",
                       dtype={"slide_id": str})[["slide_id", "aperio.ScanScope ID"]].rename(columns={"aperio.ScanScope ID": "scanner"})
    slides = m[m.slide_id.isin(ev.slide_id)].merge(scan, on="slide_id", how="left").reset_index(drop=True)
    row = ev.slide_id.map({s: i for i, s in enumerate(slides.slide_id)}).to_numpy()
    xy = ev[["x", "y"]].to_numpy()
    size0 = {}
    for sid in slides.slide_id:
        with h5py.File(ROOT / cfg["features_dir"] / f"{sid}.h5", "r") as h:
            size0[sid] = int(h["coords"].attrs.get("patch_size_level0", 256))

    if per_k:
        tab = pd.read_csv(RUN / "per_k" / f"cores_k{per_k}_tau{tau}.csv").rename(columns={"split_half_J": "split_half_jaccard_median"})
        ch = pd.read_csv(RUN / "per_k" / f"chains_k{per_k}_to_k50_tau{tau}.csv")
        keep = ch[ch.nested_all_levels & (ch.ctrl_frac < max_ctrl)][f"core_k{per_k}"]
        sel = tab[tab.core.isin(keep)].sort_values(["top_exp_share", "n_patches"], ascending=[True, False])
        out = RUN / "sheets" / f"per_k{per_k}_nested_ctrl_lt{max_ctrl:g}_tau{tau}"
    else:
        tab = pd.read_csv(RUN / f"cores_{col}.csv")
        sel = tab[tab.ctrl_patch_frac < max_ctrl].sort_values(["top_exp_share", "n_patches"], ascending=[True, False])
        out = RUN / "sheets" / f"selected_ctrl_lt{max_ctrl:g}_{col}"
    out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / "selected_cores.csv", index=False)
    for f0 in range(0, len(sel), PER_FILE):
        rows = []
        for r in sel.iloc[f0:f0 + PER_FILE].itertuples():
            members = np.flatnonzero(core == r.core)
            members = members[np.argsort(-centrality[members])]
            picks, seen = [], set()
            for p in members:
                if row[p] not in seen:
                    picks.append(p)
                    seen.add(row[p])
                if len(picks) == cfg["sheet_tiles"]:
                    break
            tiles = exp0017.read_tiles(picks, row, xy, slides, size0,
                                       lambda sl, p: (f"e{sl.exp_id} {sl.compound_name[:13]} {sl.dose_level[0]}",
                                                      f"{sl.sacrifice_period} {sl.scanner} {'F' if sl.has_finding else '-'}"))
            exps = slides.iloc[row[members]].compound_name.value_counts(normalize=True)
            header = (f"core {r.core}: {r.n_patches} patches / {r.n_slides} slides / {r.n_exp} exp | Ctrl {r.ctrl_patch_frac:.2f} | "
                      f"split-half J {r.split_half_jaccard_median:.2f} | {exp0017.lift_text(r)} | "
                      + ", ".join(f"{k[:12]} {v:.0%}" for k, v in exps.head(4).items()))
            rows.append((header, tiles))
        exp0017.draw_sheet(f"0017 {VARIANT} cores ({col}) with Control patch fraction < {max_ctrl:g}{' nested across k=50/100/200' if per_k else ''} (tau={tau}), sorted by top-experiment share "
                           f"(multi-experiment first). Label: exp, compound, dose C/L/M/H, time, scanner, F = slide has a recorded finding",
                           rows, out / f"cores_{f0 // PER_FILE + 1:02d}.jpg")
    print(out, len(sel))


if __name__ == "__main__":
    main()
