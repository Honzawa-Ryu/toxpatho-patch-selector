"""Compare global and experiment/time-stratified candidate selection at fixed k."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, false_discovery_control

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.cluster_analysis import finding_associations

NAME = '0011_20260925_stratified_candidate_selection'
REPORT = ROOT / 'experiments' / NAME
OUT = ROOT / 'outputs' / NAME
PREVIOUS = ROOT / 'experiments/0010_20260925_pathology_benchmark_subset'
BIG = ROOT / 'outputs/0009_20260919_tier1_corpus_clustering'
ANCHORS = [1356, 846, 703, 535, 1625, 147, 697, 271, 801, 209, 95, 1425]


def stratified_compound(occ, manifest):
    """Weighted rank-sum over independent experiment/time strata, tie corrected.

    Weights are 1/(n+1). Return one one-sided asymptotic p-value per cluster.
    No test result or pathology annotation influences weights.
    """
    k = occ.shape[1]
    score, variance, usum = np.zeros(k), np.zeros(k), np.zeros(k)
    pairs = 0
    usable = np.zeros(len(manifest), dtype=bool)
    strata = 0
    for _, indices in manifest.groupby(['exp_id', 'sacrifice_period']).indices.items():
        ctrl = manifest.iloc[indices].dose_level.eq('Control').to_numpy()
        nc, nt = int(ctrl.sum()), int((~ctrl).sum())
        if min(nc, nt) < 2:
            continue
        usable[indices] = True
        values = occ[indices]
        n = nc + nt
        ranks = rankdata(values, axis=0, method='average')
        u = ranks[~ctrl].sum(axis=0) - nt*(nt+1)/2
        var = nt*nc/(n*(n-1)) * ((ranks-(n+1)/2)**2).sum(axis=0)
        weight = 1/(n+1)
        score += weight * (u - nt*nc/2)
        variance += weight**2 * var
        usum += u
        pairs += nt*nc
        strata += 1
    p = np.ones(k)
    positive_variance = variance > 0
    p[positive_variance] = norm.sf(score[positive_variance]/np.sqrt(variance[positive_variance]))
    control = manifest.dose_level.eq('Control').to_numpy() & usable
    treated = ~manifest.dose_level.eq('Control').to_numpy() & usable
    c, t = occ[control], occ[treated]
    return pd.DataFrame(dict(cluster_id=np.arange(k), p_value=p,
        rank_effect=usum/pairs, presence_difference=(t>.001).mean(axis=0)-(c>.001).mean(axis=0),
        mean_occupancy_difference=t.mean(axis=0)-c.mean(axis=0),
        treated_carriers=(t>.001).sum(axis=0), control_carriers=(c>.001).sum(axis=0),
        treated_slides=len(t), control_slides=len(c), control_q95=np.quantile(c,.95,axis=0),
        strata=strata))


def main():
    REPORT.mkdir(exist_ok=True)
    if OUT.exists():
        raise SystemExit('Output exists; preserve this experiment')
    OUT.mkdir()
    m = pd.read_parquet(BIG / 'manifest_tier1_corpus.parquet').sort_values('slide_id').reset_index(drop=True)
    audit = pd.read_csv(PREVIOUS / 'label_audit/slide_label_audit.csv', dtype={'slide_id':str}).set_index('slide_id')
    valid = (audit.loc[m.slide_id,'label_evaluation_eligible'] & ~audit.loc[m.slide_id,'duplicate_animal_image']).to_numpy()
    occ = np.load(PREVIOUS / 'label_audit/occupancy_k2000.npy')[valid]
    m = m[valid].reset_index(drop=True)
    m.to_parquet(OUT / 'manifest.parquet', index=False)
    results = []
    for compound, indices in m.groupby('compound_name').indices.items():
        frame = stratified_compound(occ[indices], m.iloc[indices].reset_index(drop=True))
        frame.insert(0,'compound_name',compound)
        results.append(frame)
    local = pd.concat(results, ignore_index=True)
    # One correction over every compound x cluster comparison, not per compound.
    local['q_value'] = false_discovery_control(local.p_value.to_numpy(), method='bh')
    local['supported'] = (local.q_value < .05) & (local.control_q95 < .001) & (local.treated_carriers > 0)
    local.to_parquet(OUT / 'compound_cluster_statistics.parquet', index=False)
    supporting = local[local.supported]
    # Effects order candidates; no post-hoc threshold on lesion extent is added.
    best = supporting.sort_values(['presence_difference','rank_effect','q_value'], ascending=[False,False,True]).groupby('cluster_id').head(1).set_index('cluster_id')
    global_stats = pd.read_csv(PREVIOUS / 'scale_trace/large_k2000_stats.csv').set_index('cluster_id')
    global_selected = global_stats.control_absent & (global_stats.q_value < .05) & (global_stats.n_compounds >= 1)
    union = sorted(set(global_stats.index[global_selected]) | set(best.index))
    associations = finding_associations(occ, m, np.arange(2000), presence_eps=.001, max_finding_types=36)
    associations['slide_precision'] = associations.n_slides_both / (associations.n_slides_both+associations.n_slides_cluster_only)
    associations.to_parquet(OUT / 'all_cluster_finding_associations.parquet', index=False)
    primary = associations[associations.q_value<.05].sort_values(['slide_precision','q_value'],ascending=[False,True]).groupby('cluster_id').head(1).set_index('cluster_id')
    rows = []
    for cid in union:
        b = best.loc[cid] if cid in best.index else None
        f = primary.loc[cid] if cid in primary.index else None
        carrying = occ[:,cid]>.001
        compound_counts = m.loc[carrying,'compound_name'].value_counts()
        rows.append(dict(cluster_id=int(cid), global_selected=bool(global_selected.loc[cid]),
            stratified_selected=cid in best.index,
            status='both' if global_selected.loc[cid] and cid in best.index else 'added' if cid in best.index else 'removed',
            supporting_compounds=int((supporting.cluster_id==cid).sum()),
            best_compound=b.compound_name if b is not None else '', local_q=b.q_value if b is not None else np.nan,
            presence_difference=b.presence_difference if b is not None else np.nan,
            rank_effect=b.rank_effect if b is not None else np.nan,
            treated_carriers=b.treated_carriers if b is not None else np.nan,
            control_carriers=b.control_carriers if b is not None else np.nan,
            treated_slides=b.treated_slides if b is not None else np.nan,
            control_slides=b.control_slides if b is not None else np.nan,
            carrying_slides=int(carrying.sum()), dominant_compound=compound_counts.index[0],
            dominant_compound_fraction=float(compound_counts.iloc[0]/carrying.sum()),
            best_finding=f.finding_type if f is not None else '',
            slide_precision=f.slide_precision if f is not None else np.nan,
            global_q=global_stats.loc[cid,'q_value'], anchor=cid in ANCHORS))
    catalog = pd.DataFrame(rows).sort_values(['stratified_selected','presence_difference','rank_effect'],ascending=[False,False,False])
    catalog.to_csv(REPORT / 'candidate_comparison.csv', index=False)
    local[local.cluster_id.isin(ANCHORS)].to_csv(REPORT / 'anchor_statistics.csv', index=False)
    settings = dict(k=2000, slides=len(m), compounds=m.compound_name.nunique(),
        strata=['experiment','timepoint'], comparison='all treated dose levels vs Control within stratum',
        statistic='sum w*(U-E[U]); w=1/(n+1); tie-corrected sum w^2 Var(U); normal upper tail',
        multiplicity='BH over all compound x cluster p-values', tests=len(local),
        q_threshold=.05, presence_eps=.001, local_control_q95_max=.001,
        effect_rule='rank by largest significant within-compound presence difference, then rank effect; no effect cutoff',
        image_tissue_threshold=.85, image_probe_patches=24, image_seed=20260925,
        anchors=ANCHORS, source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitation='post-hoc exploratory comparison; no unseen test cohort or facility adjustment')
    (REPORT / 'protocol.json').write_text(json.dumps(settings,indent=2)+'\n')
    print(catalog.groupby('status').size().to_string(),flush=True)
    print(catalog[catalog.anchor][['cluster_id','status','best_compound','local_q','presence_difference','rank_effect']].to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
