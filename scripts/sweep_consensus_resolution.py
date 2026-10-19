"""0017のLeiden解像度を振り、コア数・未割当率・半分ずつのrunでの一致度がどう変わるかを見る（k-meansは回し直さない）。

保存済みの eval_labels.npy（runごとの割り当て）と edges.parquet（kNN辺）から、0017本体と同じ cores_from_edges / best_jaccard を使う。
usage: .venv/bin/python scripts/sweep_consensus_resolution.py [variant] [tau] [res,res,...]   既定: pilot 0.8 0.05〜2.0
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXP = ROOT / "experiments/0017_20261016_block_consensus_patterns"
RESOLUTIONS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]
spec = importlib.util.spec_from_file_location("exp0017", EXP / "experiment.py")
exp0017 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp0017)


def main():
    variant = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    tau = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8
    resolutions = [float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else RESOLUTIONS
    tag = "" if len(sys.argv) <= 3 else "_" + sys.argv[3].replace(",", "-")
    run = ROOT / "outputs/0017_20261016_block_consensus_patterns" / variant
    cfg = yaml.safe_load(open(EXP / "config.yml"))
    labels = np.load(run / "eval_labels.npy")
    e = pd.read_parquet(run / "edges.parquet")
    pairs = e[["a", "b"]].to_numpy()
    rate = e.rate.to_numpy()
    ev = pd.read_parquet(run / "eval_patches.parquet")
    m = pd.read_parquet(ROOT / cfg["manifest_parquet"])
    m["slide_id"] = m.slide_id.astype(str)
    s = ev[["slide_id"]].merge(m[["slide_id", "exp_id", "dose_level"]], on="slide_id", how="left")
    row = pd.factorize(ev.slide_id)[0]
    is_ctrl = (s.dose_level == "Control").to_numpy()
    exp = s.exp_id.to_numpy()
    n = len(ev)

    rng = np.random.default_rng(cfg["seed"])
    splits = []
    for _ in range(cfg["n_split_repeats"]):
        p = rng.permutation(len(labels))
        splits.append([p[: len(p) // 2], p[len(p) // 2:]])
    half_rates = [[(labels[h][:, pairs[:, 0]] == labels[h][:, pairs[:, 1]]).mean(0) for h in sp] for sp in splits]

    recs, per_core = [], []
    for res in resolutions:
        c = dict(cfg, leiden_resolution=res)
        core = exp0017.cores_from_edges(n, pairs, rate, tau, "leiden", row, c)
        js = [exp0017.best_jaccard(core, exp0017.cores_from_edges(n, pairs, r, tau, "leiden", row, c)) for hr in half_rates for r in hr]
        js = np.median(np.stack(js), axis=0) if core.max() >= 0 else np.array([])
        sizes = np.bincount(core[core >= 0]) if core.max() >= 0 else np.array([])
        ctrl_frac = np.array([is_ctrl[core == k].mean() for k in range(core.max() + 1)])
        top_share = np.array([pd.Series(exp[core == k]).value_counts(normalize=True).iloc[0] for k in range(core.max() + 1)])
        recs.append({"resolution": res, "n_cores": int(core.max() + 1), "assigned_frac": float((core >= 0).mean()),
                     "median_core_patches": float(np.median(sizes)) if len(sizes) else 0,
                     "largest_core_frac": float(sizes.max() / n) if len(sizes) else 0,
                     "split_half_J_median": float(np.median(js)) if len(js) else np.nan,
                     "split_half_J_patch_weighted": float(np.average(js, weights=sizes)) if len(js) else np.nan,
                     "frac_cores_J_ge_0.7": float((js >= 0.7).mean()) if len(js) else np.nan,
                     "n_cores_ctrl_lt5pct": int((ctrl_frac < 0.05).sum()),
                     "J_median_ctrl_lt5pct": float(np.median(js[ctrl_frac < 0.05])) if (ctrl_frac < 0.05).any() else np.nan,
                     "J_median_ctrl_ge5pct": float(np.median(js[ctrl_frac >= 0.05])) if (ctrl_frac >= 0.05).any() else np.nan})
        per_core += [{"resolution": res, "core": k, "n_patches": int(sizes[k]), "ctrl_frac": float(ctrl_frac[k]),
                      "top_exp_share": float(top_share[k]), "split_half_J": float(js[k])} for k in range(core.max() + 1)]
        ev[f"leiden_tau{tau:g}_res{res:g}"] = core
        print(recs[-1], flush=True)
    out = run / "diagnostics"
    out.mkdir(exist_ok=True)
    pd.DataFrame(recs).to_csv(out / f"resolution_sweep_tau{tau:g}{tag}.csv", index=False)
    pd.DataFrame(per_core).to_csv(out / f"resolution_sweep_tau{tau:g}{tag}_cores.csv", index=False)
    ev[[c for c in ev.columns if c.startswith("slide_id") or c in ("x", "y") or "_res" in c]] \
        .to_parquet(out / f"resolution_sweep_tau{tau:g}{tag}_assignments.parquet", index=False)


if __name__ == "__main__":
    main()
