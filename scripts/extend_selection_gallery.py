"""Append the global-Control-reference ablation, preserving existing samples."""
import shutil
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import build_selection_gallery as g


def main():
    previous = pd.read_csv(g.GALLERY/'catalog.csv')
    catalog = pd.read_csv(g.REPORT/'candidate_comparison_variants.csv')
    wanted = set(catalog.cluster_id)-set(previous.cluster_id)
    if not wanted:
        raise SystemExit('Already extended')
    eligible = set(pd.read_parquet(g.OUT/'manifest.parquet').slide_id.astype(str))
    rng = np.random.default_rng(20260926)
    reservoirs = {c:pd.DataFrame() for c in wanted}
    populations = dict.fromkeys(wanted,0)
    for batch in pq.ParquetFile(g.BIG/'kmeans_k_sweep_part3/cluster_assignments.parquet').iter_batches(batch_size=500000,columns=['slide_id','x','y','cluster_kmeans_k2000']):
        f=batch.to_pandas()
        f=f[f.cluster_kmeans_k2000.isin(wanted)&f.slide_id.isin(eligible)].copy()
        f['priority']=rng.random(len(f))
        for c,group in f.groupby('cluster_kmeans_k2000',sort=False):
            populations[c]+=len(group)
            reservoirs[c]=pd.concat([reservoirs[c],group.nsmallest(g.N,'priority')]).nsmallest(g.N,'priority')
    samples=pd.concat(reservoirs.values(),ignore_index=True).rename(columns={'cluster_kmeans_k2000':'cluster_id'})
    samples.to_parquet(g.OUT/'gallery_sample_coordinates_extension.parquet',index=False)
    quality=[]; tiles=[]; errors=[]
    with ProcessPoolExecutor(max_workers=6,initializer=g.init_worker) as pool:
        jobs=[(int(c),f.to_dict('records')) for c,f in samples.groupby('cluster_id')]
        for i,(q,t,e) in enumerate(pool.map(g.draw_cluster,jobs),1):
            quality.append(q);tiles.extend(t);errors.extend(e)
            if i%10==0 or i==len(jobs):print(f'{i}/{len(jobs)} appended',flush=True)
    extra=pd.DataFrame(quality)
    extra['sample_population_patches']=extra.cluster_id.map(populations)
    cols=['cluster_id','quality_n','tissue_fraction','sample_slides','sample_compounds','sample_population_patches']
    catalog=catalog.merge(pd.concat([previous[cols],extra],ignore_index=True),on='cluster_id',validate='one_to_one')
    catalog['tissue_pass']=(catalog.quality_n==g.N)&(catalog.tissue_fraction>=.85)
    # Provide explicitly separate B2 evidence for candidates absent from B1.
    local=pd.read_parquet(g.OUT/'compound_cluster_statistics.parquet')
    best=local[local.q_value<.05].sort_values(['presence_difference','rank_effect','q_value'],ascending=[False,False,True]).groupby('cluster_id').head(1).set_index('cluster_id')
    catalog['b2_best_compound']=catalog.cluster_id.map(best.compound_name).where(catalog.stratified_global_absence_selected,'')
    for column in ['q_value','presence_difference','treated_carriers','control_carriers','treated_slides','control_slides']:
        target='b2_local_q' if column=='q_value' else 'b2_'+column
        catalog[target]=catalog.cluster_id.map(best[column]).where(catalog.stratified_global_absence_selected)
    catalog['b2_supporting_compounds']=catalog.cluster_id.map(local[local.q_value<.05].groupby('cluster_id').size()).where(catalog.stratified_global_absence_selected,0)
    shutil.copy2(g.GALLERY/'catalog.csv',g.GALLERY/'catalog_primary.csv')
    catalog.to_csv(g.GALLERY/'catalog.csv',index=False)
    catalog.to_csv(g.REPORT/'candidate_comparison_all_variants_with_quality.csv',index=False)
    for filename,frame in [('tiles.csv',pd.DataFrame(tiles)),('crop_errors.csv',pd.DataFrame(errors,columns=['cluster_id','slide_id','error']))]:
        pd.concat([pd.read_csv(g.GALLERY/filename),frame],ignore_index=True).to_csv(g.GALLERY/filename,index=False)
    review=pd.read_csv(g.GALLERY/'review_template.csv')
    new=catalog[catalog.cluster_id.isin(wanted)][['cluster_id','status','anchor']]
    pd.concat([review,new],ignore_index=True).to_csv(g.GALLERY/'review_template.csv',index=False)
    rows=[]; retention=[]
    matrix=np.load(g.ROOT/'outputs/0010_20260925_pathology_benchmark_subset/scale_trace/contingency_k2000.npy')
    old=pd.read_csv(g.REPORT/'old_patch_retention_variants.csv').old_cluster
    for method,col in [('A_global','global_selected'),('B1_local_control','stratified_selected'),('B2_global_control','stratified_global_absence_selected')]:
        for tissue in [False,True]:
            take=catalog[catalog[col]&(catalog.tissue_pass if tissue else True)]
            rows.append(dict(method=method,tissue_filter=tissue,candidates=len(take),median_best_slide_precision=take.slide_precision.median(),recovered_fatty_anchors=int(take.cluster_id.isin([1356,846,703,535]).sum()),retained_background_anchors=int(take.cluster_id.isin([95,1425]).sum())))
            for c in old:
                retention.append(dict(method=method,tissue_filter=tissue,old_cluster=c,patch_retention=matrix[c,take.cluster_id.tolist()].sum()/matrix[c].sum()))
    pd.DataFrame(rows).to_csv(g.REPORT/'ablation_summary_all_variants.csv',index=False)
    pd.DataFrame(retention).to_csv(g.REPORT/'old_patch_retention_with_tissue.csv',index=False)
    g.build_html(catalog)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
