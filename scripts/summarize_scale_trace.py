"""Summarize the saved patch trace and compare selection with fixed vocabulary."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.trace_319_clusters import vector_stats


def main():
    base = ROOT / 'experiments/0010_20260925_pathology_benchmark_subset'
    out = base / 'scale_trace'
    cache = ROOT / 'outputs/0010_20260925_pathology_benchmark_subset/scale_trace'
    old_dir = ROOT / 'outputs/0003_20260912_control_absent_cluster_mining/cluster_mining_ccl4'
    m = pd.read_parquet(ROOT / 'outputs/0009_20260919_tier1_corpus_clustering/manifest_tier1_corpus.parquet').sort_values('slide_id').reset_index(drop=True)
    a = pd.read_csv(base / 'label_audit/slide_label_audit.csv', dtype={'slide_id': str}).set_index('slide_id')
    valid = (a.loc[m.slide_id, 'label_evaluation_eligible'] & ~a.loc[m.slide_id, 'duplicate_animal_image']).to_numpy()
    occ = np.load(base / 'label_audit/occupancy_k2000.npy')[valid]
    m = m[valid].reset_index(drop=True)
    old = pd.read_parquet(old_dir / 'slide_manifest.parquet')
    mask = m.slide_id.isin(old.slide_id.astype(str)).to_numpy()
    q, absent, carrier, effects = vector_stats(occ[mask], m.loc[mask, 'dose_level'].eq('Control').to_numpy())
    original = pd.DataFrame(dict(cluster_id=np.arange(2000), q_value=q, control_absent=absent,
        treated_carrier=carrier, selected=(q < .05) & absent & carrier, rank_effect=effects))
    original.to_csv(out / 'large_k2000_on_original_299.csv', index=False)
    full = pd.read_csv(out / 'large_k2000_stats.csv')
    q_full, _, _, _ = vector_stats(occ, m.dose_level.eq('Control').to_numpy())
    assert np.allclose(q_full, full.q_value, atol=1e-12), 'Vectorized and scalar tests disagree'
    os = set(original.loc[original.selected, 'cluster_id'])
    fs = set(full.loc[full.control_absent & (full.q_value < .05) & (full.n_compounds >= 1), 'cluster_id'])
    profiles = pd.read_csv(out / 'candidate_profiles_k2000.csv')
    profiles['selected_on_original_299'] = profiles.cluster_id.isin(os)
    profiles.to_csv(out / 'candidate_profiles_with_original_selection.csv', index=False)
    selection = full[['cluster_id', 'q_value', 'control_absent']].merge(
        original[['cluster_id', 'q_value', 'control_absent', 'selected']], on='cluster_id', suffixes=('_full', '_original'))
    selection['selected_full'] = selection.cluster_id.isin(fs)
    selection.to_csv(out / 'selection_change_same_vocabulary.csv', index=False)
    matrix = np.load(cache / 'contingency_k2000.npy')
    families = []
    for name, ids in [('cholesterol_fatty', [42,53]), ('ccl4_composite', [4,5,14,52,62,67]),
                      ('wy14643_composite', [0,21,25,55,65,77,93])]:
        counts = matrix[ids].sum(axis=0)
        rank = np.argsort(-counts)
        fractions = counts[rank] / counts.sum()
        n90 = int(np.searchsorted(fractions.cumsum(), .9) + 1)
        families.append(dict(family=name, old_clusters=str(ids), patches=int(counts.sum()),
            top5_fraction=fractions[:5].sum(), n90=n90,
            purity_of_90pct_destinations=counts[rank[:n90]].sum()/matrix[:,rank[:n90]].sum(),
            selected_fraction=counts[list(fs)].sum()/counts.sum()))
    pd.DataFrame(families).to_csv(out / 'family_retention.csv', index=False)

    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ids = sorted(pd.read_csv(out / 'old_cluster_retention.csv').old_cluster.unique())
    for k, color in [(100,'#2b6cb0'), (1000,'#319795'), (2000,'#b7791f')]:
        counts = np.load(cache / f'contingency_k{k}.npy')[ids]
        fractions = -np.sort(-counts, axis=1) / counts.sum(axis=1, keepdims=True)
        curves = fractions.cumsum(axis=1)[:, :40]
        axes[0,0].plot(np.arange(1,41), np.median(curves,axis=0), color=color, label=f'Large corpus k={k}')
    axes[0,0].set(xlabel='Number of destination clusters', ylabel='Fraction of old patches recovered',
                  title='A  Original 19 clusters: median recovery', ylim=(0,1.02), xlim=(1,40))
    axes[0,0].legend(fontsize=8)
    categories = ['Selected in both', 'Added in full corpus', 'Lost in full corpus']
    values = [len(os&fs),len(fs-os),len(os-fs)]
    axes[0,1].bar(categories, values, color=['#319795','#b7791f','#718096'])
    for i, n in enumerate(values): axes[0,1].text(i,n+3,str(n),ha='center')
    axes[0,1].set(ylabel='Clusters', title='B  Same k=2000 vocabulary; selection changes', ylim=(0,max(values)*1.2))
    data = [profiles.loc[profiles.selected_on_original_299, 'best_precision'].dropna(),
            profiles.loc[~profiles.selected_on_original_299, 'best_precision'].dropna()]
    axes[1,0].boxplot(data, tick_labels=['Selected in both (114)', 'Added (216)'], showfliers=False)
    axes[1,0].set(title='C  Best finding association on full corpus', ylabel='P(recorded finding | carrying slide)', ylim=(0,1.05))
    d = pd.read_csv(out / 'slide_count_sensitivity.csv').groupby('slides').candidates.agg(['median','min','max'])
    axes[1,1].errorbar(d.index, d['median'], yerr=[d['median']-d['min'],d['max']-d['median']],
                      marker='o', color='#2b6cb0', capsize=5, label='Full-corpus mixture, 30 draws*')
    axes[1,1].scatter([int(mask.sum())],[len(os)], marker='*', s=160, color='#b7791f', label='Original 9-compound cohort')
    axes[1,1].set(title='D  Fixed k=2000; sample size and composition', xlabel='Slides', ylabel='Selected clusters')
    axes[1,1].legend(fontsize=8)
    fig.suptitle('Tracing 5,111,952 identical patches from 319 slides', fontsize=14)
    fig.text(.5,.012,'*Full-corpus point uses all 3,357 eligible slides. Association is slide-level and exploratory; not patch diagnostic accuracy.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.035,1,.95])
    fig.savefig(out / 'scale_trace_summary.png', dpi=170)
    fig.savefig(out / 'scale_trace_summary.pdf')
    print('Saved summaries and figure')


if __name__ == '__main__':
    main()
