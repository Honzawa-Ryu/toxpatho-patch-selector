"""Isolate the effect of changing the Control-absence reference population."""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NAME='0011_20260925_stratified_candidate_selection'
REPORT=ROOT/'experiments'/NAME
OUT=ROOT/'outputs'/NAME


def main():
    local=pd.read_parquet(OUT/'compound_cluster_statistics.parquet')
    g=pd.read_csv(ROOT/'experiments/0010_20260925_pathology_benchmark_subset/scale_trace/large_k2000_stats.csv').set_index('cluster_id')
    alternative=local[(local.q_value<.05)&local.cluster_id.map(g.control_absent)&(local.treated_carriers>0)]
    best=alternative.sort_values(['presence_difference','rank_effect','q_value'],ascending=[False,False,True]).groupby('cluster_id').head(1).set_index('cluster_id')
    base=pd.read_csv(REPORT/'candidate_comparison.csv')
    assoc=pd.read_parquet(OUT/'all_cluster_finding_associations.parquet')
    primary=assoc[assoc.q_value<.05].sort_values(['slide_precision','q_value'],ascending=[False,True]).groupby('cluster_id').head(1).set_index('cluster_id')
    m=pd.read_parquet(OUT/'manifest.parquet')
    original=pd.read_parquet(ROOT/'outputs/0009_20260919_tier1_corpus_clustering/manifest_tier1_corpus.parquet').sort_values('slide_id').reset_index(drop=True)
    pos=pd.Index(original.slide_id).get_indexer(m.slide_id)
    occ=np.load(ROOT/'experiments/0010_20260925_pathology_benchmark_subset/label_audit/occupancy_k2000.npy')[pos]
    extra=[]
    for cid in sorted(set(best.index)-set(base.cluster_id)):
        b=best.loc[cid];f=primary.loc[cid] if cid in primary.index else None
        carry=occ[:,cid]>.001;comp=m.loc[carry,'compound_name'].value_counts()
        extra.append(dict(cluster_id=int(cid),global_selected=False,stratified_selected=False,status='alternative_only',
            supporting_compounds=int((alternative.cluster_id==cid).sum()),best_compound=b.compound_name,
            local_q=b.q_value,presence_difference=b.presence_difference,rank_effect=b.rank_effect,
            treated_carriers=b.treated_carriers,control_carriers=b.control_carriers,treated_slides=b.treated_slides,
            control_slides=b.control_slides,carrying_slides=int(carry.sum()),dominant_compound=comp.index[0],
            dominant_compound_fraction=comp.iloc[0]/carry.sum(),best_finding=f.finding_type if f is not None else '',
            slide_precision=f.slide_precision if f is not None else np.nan,global_q=g.loc[cid,'q_value'],anchor=False))
    combined=pd.concat([base,pd.DataFrame(extra)],ignore_index=True)
    combined['stratified_global_absence_selected']=combined.cluster_id.isin(best.index)
    combined.to_csv(REPORT/'candidate_comparison_variants.csv',index=False)
    matrix=np.load(ROOT/'outputs/0010_20260925_pathology_benchmark_subset/scale_trace/contingency_k2000.npy')
    old=pd.read_csv(REPORT/'old_patch_retention_before_tissue.csv')
    old['stratified_global_absence_selected']=[matrix[int(c),list(best.index)].sum()/matrix[int(c)].sum() for c in old.old_cluster]
    old.to_csv(REPORT/'old_patch_retention_variants.csv',index=False)
    print('alternative candidates',len(best),'union',len(combined),'extra images',len(extra))
    print(old.round(3).to_string(index=False))


if __name__=='__main__':main()
