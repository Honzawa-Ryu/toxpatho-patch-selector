"""Audit metadata coverage and rerun label selection on fixed k=2000 clusters."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lib.cluster_analysis import cluster_statistics, select_candidates, leave_one_compound_out_auroc
from scripts.reanalyze_cluster_thresholds import MANIFEST, PART_FOR_K, ROOT as SOURCE, _QuietLogger


def main():
    out = ROOT / 'experiments/0010_20260925_pathology_benchmark_subset/label_audit'
    out.mkdir(exist_ok=True)
    m = pd.read_parquet(MANIFEST).sort_values('slide_id').reset_index(drop=True)
    m['slide_id'] = m.slide_id.astype(str)
    keys = ['exp_id', 'group_id', 'individual_id']
    individual = pd.read_csv(ROOT / 'data/corrected/open_tggates_individual.csv')
    individual = individual.rename(columns={k.upper(): k for k in keys})
    coverage = m[keys].merge(individual[keys].drop_duplicates(), on=keys, how='left', indicator=True)
    known = coverage['_merge'].eq('both').to_numpy() & m.dose_level.isin(['Control', 'Low', 'Middle', 'High']).to_numpy()
    unique = ~m.duplicated(keys).to_numpy()
    audit = m.drop(columns=['finding_types', 'finding_sites']).copy()
    audit['individual_metadata_matched'] = coverage['_merge'].eq('both').to_numpy()
    audit['label_evaluation_eligible'] = known
    audit['duplicate_animal_image'] = ~unique
    audit['pathology_label_status'] = np.where(known, np.where(m.has_finding, 'recorded_finding', 'no_finding_recorded'), 'unknown')
    audit.to_csv(out / 'slide_label_audit.csv', index=False)
    audit[~known].to_csv(out / 'unknown_slides.csv', index=False)
    clean = m[known & unique].copy()
    clean['pathology_label_status'] = np.where(clean.has_finding, 'recorded_finding', 'no_finding_recorded')
    clean.to_parquet(out / 'manifest_known_unique.parquet', index=False)
    print(f'known={known.sum()} known_unique={(known & unique).sum()} unknown={(~known).sum()}', flush=True)

    # Bounded-memory accumulation. Keep the same cluster vocabulary in every arm.
    path = SOURCE / PART_FOR_K[2000] / 'cluster_assignments.parquet'
    occ = np.zeros((len(m), 2000), dtype=np.float64)
    index = pd.Index(m.slide_id)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=500000, columns=['slide_id', 'cluster_kmeans_k2000']):
        d = batch.to_pandas()
        si = index.get_indexer(d.slide_id.astype(str))
        assert (si >= 0).all()
        ci = d.cluster_kmeans_k2000.to_numpy()
        occ += np.bincount(si * 2000 + ci, minlength=occ.size).reshape(occ.shape)
    assert (occ.sum(axis=1) > 0).all()
    occ /= occ.sum(axis=1, keepdims=True)
    np.save(out / 'occupancy_k2000.npy', occ)
    rows, candidates = [], {}
    legacy_reference = lambda frame: frame.dose_level.eq('Control').to_numpy()
    for name, mask in [('legacy_all', np.ones(len(m), dtype=bool)), ('known_labels', known), ('known_unique_animals', known & unique)]:
        sub = m[mask].reset_index(drop=True)
        kwargs = dict(presence_eps=0.001, min_treated_presence=0.0, logger=_QuietLogger(), reference_selector=legacy_reference)
        stats = cluster_statistics(occ[mask], sub, **kwargs)
        selected = select_candidates(stats, q_threshold=0.05, min_compounds=1)
        score, per = leave_one_compound_out_auroc(occ[mask], sub, q_threshold=0.05, min_compounds=1, **kwargs)
        per.to_csv(out / f'{name}_loco.csv', index=False)
        stats.to_csv(out / f'{name}_cluster_stats.csv', index=False)
        candidates[name] = [int(c) for c in selected]
        rows.append(dict(arm=name, slides=len(sub), candidates=len(selected), pooled_any_finding_auroc=score,
                         macro_any_finding_auroc=per.auroc.mean(), evaluable_compounds=int(per.auroc.notna().sum())))
        pd.DataFrame(rows).to_csv(out / 'sensitivity.csv', index=False)
        print(rows[-1], flush=True)
    (out / 'candidates.json').write_text(json.dumps(candidates, indent=2))
    (out / 'provenance.json').write_text(json.dumps(dict(
        fixed_clustering=True, k=2000, pathology_verified=False,
        sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [MANIFEST, ROOT / 'data/corrected/open_tggates_individual.csv', Path(__file__)]}), indent=2))


if __name__ == '__main__':
    main()
