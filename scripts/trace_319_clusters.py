"""Trace identical patches from the 319-slide vocabulary into tier1 clusters.

Also vary slide sample size with the k=2000 vocabulary fixed. This is a
descriptive sensitivity analysis, not independent lesion validation.
"""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.stats import mannwhitneyu, false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.cluster_analysis import cluster_statistics, select_candidates, finding_associations
from scripts.reanalyze_cluster_thresholds import _QuietLogger

OLD = ROOT / 'outputs/0003_20260912_control_absent_cluster_mining/cluster_mining_ccl4'
BIG = ROOT / 'outputs/0009_20260919_tier1_corpus_clustering'
BASE = ROOT / 'experiments/0010_20260925_pathology_benchmark_subset'
OUT = BASE / 'scale_trace'
CACHE = ROOT / 'outputs/0010_20260925_pathology_benchmark_subset/scale_trace'
PARTS = {100: 0, 1000: 1, 2000: 3}


def vector_stats(occ, control):
    c, t = occ[control], occ[~control]
    u, p = mannwhitneyu(t, c, axis=0, alternative='greater', method='asymptotic')
    constant = np.ptp(occ, axis=0) == 0
    p[constant] = 1
    q = false_discovery_control(p, method='bh')
    absent = np.quantile(c, .95, axis=0) < .001
    # min_compounds=1 reduces here to at least one treated carrier.
    carrier = (t > .001).any(axis=0)
    return q, absent, carrier, u / (len(t) * len(c))


def main():
    if OUT.exists() or CACHE.exists():
        raise SystemExit('Trace output already exists; preserve the prior run')
    OUT.mkdir()
    CACHE.mkdir(parents=True)
    m = pd.read_parquet(BIG / 'manifest_tier1_corpus.parquet').sort_values('slide_id').reset_index(drop=True)
    m.slide_id = m.slide_id.astype(str)
    audited = pd.read_csv(BASE / 'label_audit/slide_label_audit.csv', dtype={'slide_id': str}).set_index('slide_id')
    valid = (audited.loc[m.slide_id, 'label_evaluation_eligible'] & ~audited.loc[m.slide_id, 'duplicate_animal_image']).to_numpy()
    clean = m[valid].reset_index(drop=True)
    old = pd.read_parquet(OLD / 'cluster_assignments.parquet', columns=['slide_id', 'x', 'y', 'cluster_k100'])
    old.slide_id = old.slide_id.astype(str)
    assert not old.duplicated(['slide_id', 'x', 'y']).any()
    assert ((old.x >= 0) & (old.y >= 0) & (old.x < 2**32) & (old.y < 2**32)).all()
    old_labels = old.cluster_k100.to_numpy()
    old_codes = pd.Categorical(old.slide_id).codes
    old_slide_ids = np.array(pd.Categorical(old.slide_id).categories)
    old_packed = (old.x.to_numpy().astype(np.uint64) << np.uint64(32)) | old.y.to_numpy().astype(np.uint64)
    lookup = {}
    for sid, indices in old.groupby('slide_id', sort=False).indices.items():
        order = np.argsort(old_packed[indices])
        lookup[sid] = (old_packed[indices][order], indices[order])
    source_stats = pd.read_parquet(OLD / 'cluster_stats.parquet')
    old_candidates = source_stats.loc[(source_stats.k == 100) & source_stats.control_absent &
        (source_stats.q_value < .05) & (source_stats.n_compounds >= 1), 'cluster_id'].to_numpy()
    global_index = pd.Index(m.slide_id)
    retained, destinations, candidate_profiles, run_summary = [], [], [], []
    matched = old.copy()
    for k, part in PARTS.items():
        counts = np.zeros((len(m), k), dtype=np.int64)
        new_labels = np.full(len(old), -1, dtype=np.int32)
        file = BIG / f'kmeans_k_sweep_part{part}/cluster_assignments.parquet'
        col = f'cluster_kmeans_k{k}'
        for batch in pq.ParquetFile(file).iter_batches(batch_size=500000, columns=['slide_id', 'x', 'y', col]):
            frame = batch.to_pandas()
            frame.slide_id = frame.slide_id.astype(str)
            indices = global_index.get_indexer(frame.slide_id)
            labels = frame[col].to_numpy()
            assert (indices >= 0).all() and (labels >= 0).all() and (labels < k).all()
            counts += np.bincount(indices * k + labels, minlength=counts.size).reshape(counts.shape)
            for sid, group in frame[frame.slide_id.isin(lookup)].groupby('slide_id', sort=False):
                packed = (group.x.to_numpy().astype(np.uint64) << np.uint64(32)) | group.y.to_numpy().astype(np.uint64)
                keys, positions = lookup[sid]
                where = np.searchsorted(keys, packed)
                assert (where < len(keys)).all() and np.array_equal(keys[where], packed), f'Patch mismatch: {sid}'
                target = positions[where]
                assert (new_labels[target] == -1).all(), 'Duplicate destination patch'
                new_labels[target] = group[col].to_numpy()
        assert (new_labels >= 0).all(), 'Missing original patches'
        matched[f'large_k{k}'] = new_labels
        matrix = np.bincount(old_labels * k + new_labels, minlength=100*k).reshape(100, k)
        np.save(CACHE / f'contingency_k{k}.npy', matrix)
        occ = counts / counts.sum(axis=1, keepdims=True)
        stats = cluster_statistics(occ[valid], clean, presence_eps=.001, min_treated_presence=0., logger=_QuietLogger())
        candidate = select_candidates(stats, q_threshold=.05, min_compounds=1)
        is_candidate = np.isin(np.arange(k), candidate)
        stats.to_csv(OUT / f'large_k{k}_stats.csv', index=False)
        for cid in old_candidates:
            n = matrix[cid].sum()
            rank = np.argsort(-matrix[cid], kind='stable')
            fractions = matrix[cid, rank] / n
            n90 = int(np.searchsorted(np.cumsum(fractions), .9) + 1)
            # Patch count and slide-weighted views share the same destinations.
            mask = old_labels == cid
            denom = np.bincount(old_codes[mask], minlength=len(old_slide_ids))
            numerator = np.bincount(old_codes[mask & np.isin(new_labels, rank[:5])], minlength=len(old_slide_ids))
            retained.append(dict(k=k, old_cluster=int(cid), old_patches=int(n),
                top1_cluster=int(rank[0]), top1_fraction=float(fractions[0]),
                top3_fraction=float(fractions[:3].sum()), top5_fraction=float(fractions[:5].sum()),
                destinations_for_90pct=n90, effective_destinations=float(1 / np.sum(fractions**2)),
                fraction_in_selected_clusters=float(matrix[cid, is_candidate].sum()/n),
                slide_mean_top5_fraction=float((numerator[denom>0]/denom[denom>0]).mean())))
            for rank_no, dest in enumerate(rank[:max(10, n90)], 1):
                destinations.append(dict(k=k, old_cluster=int(cid), rank=rank_no, new_cluster=int(dest),
                    n_patches=int(matrix[cid,dest]), fraction_of_old=float(matrix[cid,dest]/n),
                    purity_on_original_319=float(matrix[cid,dest]/matrix[:,dest].sum()) if matrix[:,dest].sum() else 0,
                    fraction_of_full_new=float(matrix[cid,dest]/counts[:,dest].sum()),
                    selected=bool(is_candidate[dest])))
        run_summary.append(dict(k=k, all_patches=int(counts.sum()), mean_patches_per_cluster=float(counts.sum()/k),
                                candidates=len(candidate), mean_patches_per_candidate=float(counts[:,candidate].sum(axis=0).mean())))
        print(f'k={k}: matched {len(old):,} patches; {len(candidate)} candidates', flush=True)
        if k == 2000:
            association = finding_associations(occ[valid], clean, candidate, presence_eps=.001, max_finding_types=36)
            association['precision'] = association.n_slides_both / (association.n_slides_both + association.n_slides_cluster_only)
            association.to_csv(OUT / 'large_k2000_associations.csv', index=False)
            sig = association[association.q_value < .05].sort_values(['precision','q_value'], ascending=[False,True])
            best = sig.groupby('cluster_id').head(1).set_index('cluster_id')
            quality = pd.read_csv(BIG / 'cluster_quality_metrics_k2000.csv').set_index('cluster_id')
            control = clean.dose_level.eq('Control').to_numpy()
            _, _, _, effects = vector_stats(occ[valid], control)
            for cid in candidate:
                shared = int(matrix[:,cid].sum())
                former = int(matrix[old_candidates,cid].sum())
                carriers = occ[valid,cid] > .001
                compound = clean.loc[carriers,'compound_name'].value_counts()
                ctrl_table = clean.loc[control,['exp_id']].assign(present=carriers[control]).groupby('exp_id').present.agg(['mean','size'])
                eligible_ctrl = ctrl_table[ctrl_table['size'] >= 10]
                category = ('little_original_support' if shared < 100 else
                            'majority_from_old_candidates' if former/shared >= .5 else 'majority_from_other_old_clusters')
                b = best.loc[cid] if cid in best.index else None
                candidate_profiles.append(dict(cluster_id=int(cid), correspondence_group=category,
                    original_319_patches=shared, original_candidate_patches=former,
                    original_candidate_fraction=former/shared if shared else np.nan,
                    all_patches=int(counts[:,cid].sum()), carrying_slides=int(carriers.sum()),
                    dominant_compound=compound.index[0], dominant_compound_fraction=float(compound.iloc[0]/carriers.sum()),
                    best_finding=b.finding_type if b is not None else '', best_precision=float(b.precision) if b is not None else np.nan,
                    tissue_fraction=float(quality.loc[cid,'tissue_frac_cc']) if cid in quality.index else np.nan,
                    treatment_rank_effect=float(effects[cid]),
                    max_control_presence_in_experiment=float(eligible_ctrl['mean'].max()),
                    max_control_experiment=int(eligible_ctrl['mean'].idxmax()),
                    global_control_presence=float(carriers[control].mean())))
            sampling = []
            ctrl_idx, trt_idx = np.flatnonzero(control), np.flatnonzero(~control)
            rng = np.random.default_rng(20260925)
            for n in [299,1000,len(clean)]:
                repeats = 30 if n < len(clean) else 1
                for repeat in range(repeats):
                    nc = round(n * control.mean())
                    pick = np.r_[rng.choice(ctrl_idx,nc,replace=False),rng.choice(trt_idx,n-nc,replace=False)]
                    q, absent, carrier, _ = vector_stats(occ[valid][pick], control[pick])
                    selected = (q < .05) & absent & carrier
                    sampling.append(dict(slides=n,repeat=repeat,n_control=nc,n_treated=n-nc,
                        significant_clusters=int((q<.05).sum()),control_absent_clusters=int(absent.sum()),
                        candidates=int(selected.sum()),full_candidates_significant=int((q[candidate]<.05).sum()),
                        full_candidates_selected=int(selected[candidate].sum())))
            pd.DataFrame(sampling).to_csv(OUT / 'slide_count_sensitivity.csv', index=False)
    matched.slide_id = matched.slide_id.astype('category')
    matched.to_parquet(CACHE / 'matched_patches.parquet', index=False)
    pd.DataFrame(retained).to_csv(OUT / 'old_cluster_retention.csv', index=False)
    pd.DataFrame(destinations).to_csv(OUT / 'destinations.csv', index=False)
    pd.DataFrame(candidate_profiles).to_csv(OUT / 'candidate_profiles_k2000.csv', index=False)
    pd.DataFrame(run_summary).to_csv(OUT / 'run_summary.csv', index=False)
    (OUT / 'provenance.json').write_text(json.dumps(dict(
        source_old=str(OLD.relative_to(ROOT)), source_large=str(BIG.relative_to(ROOT)),
        matched_key=['slide_id','x','y'], matched_patches=len(old),
        evaluation_slides=len(clean), seed=20260925,
        sampling='uniform without replacement within control/treated, preserving full corpus ratio; 30 replicates',
        fixed_vocabulary=True, source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caution='k, PCA, training sample and assignment method differ; correspondence is descriptive, not causal',
        correspondence_rule='at least 100 shared patches and >=50% from the union of 19 old candidates',
        artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}), indent=2))
    print('Done', flush=True)


if __name__ == '__main__':
    main()
