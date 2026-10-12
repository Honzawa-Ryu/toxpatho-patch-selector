"""Plot the existing 0013 finding/bin counts, without recomputing votes."""

import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, StrMethodFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/0013_20260929_stable_candidate_layer_review/vote_steps"
SOURCE = BASE / "vote_step_table.csv"
OUT = ROOT / "experiments/0013_20260929_stable_candidate_layer_review/vote_fraction_histograms_20261009"


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170, facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", facecolor="white")
    plt.close(fig)


def bars(ax, rows, normalized=False):
    n = rows.bin_n_patches.to_numpy(dtype=np.int64)
    values = n / n.sum() if normalized else n
    width = np.minimum(rows.bin_hi.to_numpy(), 1.0) - rows.bin_lo.to_numpy()
    ax.bar(rows.bin_lo, values, width=width, align="edge", color="#3979aa",
           edgecolor="white", linewidth=.7)
    ax.set_xlim(.025, 1.025)
    ax.set_xticks([.1, .3, .5, .7, .9, 1.0])
    ax.tick_params(labelsize=9)
    if normalized:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    else:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)


def facets(table, summary, normalized):
    fig, axes = plt.subplots(5, 4, figsize=(17, 16))
    for ax, finding in zip(axes.flat, summary.finding):
        rows = table[table.finding.eq(finding)].sort_values("bin_lo")
        bars(ax, rows, normalized)
        n = int(rows.bin_n_patches.sum())
        ax.set_title(textwrap.fill(finding, width=32), fontsize=11, loc="left", y=1.13, pad=0)
        ax.text(0, 1.015, f"N = {n:,} patches with V >= 0.05", transform=ax.transAxes,
                fontsize=9, color="#55616d")
        ax.tick_params(labelbottom=True)
    for ax in list(axes.flat)[len(summary):]:
        ax.set_visible(False)
    fig.suptitle("VoteFraction distributions by most-voted finding", fontsize=19, y=.982)
    label = "Fraction of displayed patches within each finding (common 0-100% scale)" if normalized else "Patch counts per bin (each panel has its own y scale)"
    fig.text(.5, .955, label + " | 110 runs | 17 reviewed findings", ha="center", fontsize=11)
    fig.supxlabel("VoteFraction = maximum votes for one finding / 110", y=.052, fontsize=12)
    fig.text(.5, .028, "Only V >= 0.05 shown. Findings with fewer than 100 patches at V >= 0.1 are absent. Last bin includes V = 1.", ha="center", fontsize=10)
    fig.text(.5, .012, "Fixed 1M-patch reference sample, eligible slides only. Finding names are run-vote associations, not patch-level diagnoses.", ha="center", fontsize=9)
    fig.subplots_adjust(left=.065, right=.985, top=.905, bottom=.10, hspace=.62, wspace=.29)
    save(fig, "vote_fraction_by_finding_percent" if normalized else "vote_fraction_by_finding_counts")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(SOURCE)
    assert not table.duplicated(["finding", "bin_lo", "bin_hi"]).any()
    assert (table.bin_n_patches >= 0).all()
    assert np.equal(table.bin_n_patches, np.floor(table.bin_n_patches)).all()
    assert table.bin_lo.min() == .05 and table.bin_hi.max() == 1.01
    records = []
    for finding, rows in table.groupby("finding", sort=True):
        rows = rows.sort_values("bin_lo")
        assert len(rows) == 10
        assert np.allclose(rows.bin_hi.to_numpy()[:-1], rows.bin_lo.to_numpy()[1:])
        # Independent saved cumulative counts must agree with plotted bin counts.
        assert np.array_equal(rows.bin_n_patches.to_numpy()[::-1].cumsum()[::-1],
                              rows.cum_n_patches.to_numpy())
        n = int(rows.bin_n_patches.sum())
        n50 = int(rows.loc[rows.bin_lo.ge(.5), "bin_n_patches"].sum())
        n90 = int(rows.loc[rows.bin_lo.ge(.9), "bin_n_patches"].sum())
        assert n > 0
        records.append({"finding": finding, "displayed_patches_v_ge_005": n,
                        "patches_v_ge_05": n50, "patches_v_ge_09": n90,
                        "fraction_v_ge_05_among_displayed": n50 / n,
                        "fraction_v_ge_09_among_displayed": n90 / n})
    summary = pd.DataFrame(records).sort_values("displayed_patches_v_ge_005", ascending=False)
    assert len(summary) == 17
    summary.to_csv(OUT / "finding_summary.csv", index=False)
    table.to_csv(OUT / "source_bin_counts.csv", index=False)
    pooled = table.groupby(["bin_lo", "bin_hi"], as_index=False).bin_n_patches.sum()
    assert pooled.bin_n_patches.sum() == summary.displayed_patches_v_ge_005.sum()
    pooled.to_csv(OUT / "pooled_bin_counts.csv", index=False)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10, 5.8))
    bars(ax, pooled)
    ax.set_ylabel("Patches per bin")
    ax.set_xlabel("VoteFraction = maximum votes for one finding / 110")
    ax.set_title("VoteFraction histogram — pooled 17 reviewed findings\n"
                 f"{int(pooled.bin_n_patches.sum()):,} patches with V >= 0.05", fontsize=15, pad=15)
    for row in pooled.itertuples():
        x = (row.bin_lo + min(row.bin_hi, 1.0)) / 2
        ax.text(x, row.bin_n_patches, f"{row.bin_n_patches:,}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, pooled.bin_n_patches.max() * 1.17)
    fig.text(.5, .028, "Eligible slides in the fixed 1M reference sample; excludes V < 0.05 and findings absent from the saved review table.", ha="center", fontsize=8.5)
    fig.text(.5, .007, "Bin widths differ: first 0.05, others 0.1. Heights show counts, not density. Last bin includes V = 1.", ha="center", fontsize=8.5)
    fig.subplots_adjust(left=.11, right=.98, top=.82, bottom=.18)
    save(fig, "vote_fraction_pooled")
    facets(table, summary, normalized=False)
    facets(table, summary, normalized=True)
    provenance = {"source": str(SOURCE.relative_to(ROOT)),
                  "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                  "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "n_runs": 110, "n_findings": 17, "n_displayed_patches": int(pooled.bin_n_patches.sum()),
                  "definition": "Maximum votes for one finding / 110; finding = argmax of finding-specific votes",
                  "scope": "Fixed 1M patch reference sample; audited eligible slides; V>=0.05; findings with >=100 patches at V>=0.1",
                  "endpoint": "Saved final bin [0.9,1.01) displayed as [0.9,1.0], including 1.0",
                  "checks": "Every finding's bin counts agree with saved cumulative counts; pooled and per-finding totals agree",
                  "limitations": "110 runs mix methods and k. VoteFraction is neither diagnostic confidence nor repeatability under one condition."}
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved 3 PNG/PDF figures to {OUT}", flush=True)


if __name__ == "__main__":
    main()
