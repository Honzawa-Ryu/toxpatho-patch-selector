"""Compare exact dose/time matches and freeze an exploratory Necrosis subset."""
from pathlib import Path
import json
import hashlib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'experiments/0010_20260925_pathology_benchmark_subset'
FINDINGS = ['Necrosis', 'Degeneration, fatty', 'Degeneration, granular, eosinophilic']


def main():
    out = BASE / 'snapshot_v2'
    if out.exists():
        raise SystemExit('Snapshot exists; do not overwrite')
    m = pd.read_parquet(BASE / 'label_audit/manifest_known_unique.parquet')
    p = pd.read_csv(ROOT / 'data/corrected/open_tggates_pathology.csv')
    keys = ['exp_id', 'group_id', 'individual_id']
    p = p.rename(columns={k.upper(): k for k in keys})
    a = m.merge(p[p.ORGAN == 'Liver'], on=keys)
    comparisons, options = [], {}
    for finding in FINDINGS:
        for topo in sorted(a.loc[a.FINDING_TYPE == finding, 'TOPOGRAPHY_TYPE'].dropna().unique()):
            pos_ids = set(a.loc[(a.FINDING_TYPE == finding) & (a.TOPOGRAPHY_TYPE == topo), 'slide_id'])
            any_ids = set(a.loc[a.FINDING_TYPE == finding, 'slide_id'])
            for compound, group in m.groupby('compound_name'):
                count = 0
                for (exp, time, dose), g in group[group.dose_level != 'Control'].groupby(['exp_id', 'sacrifice_period', 'dose_level']):
                    pos = g[g.slide_id.isin(pos_ids)].sort_values('slide_id')
                    neg = g[~g.slide_id.isin(any_ids)].sort_values('slide_id')
                    count += min(len(pos), len(neg))
                    if finding == 'Necrosis' and topo == 'Hepatocyte':
                        for _, pr in pos.iterrows():
                            if neg.empty:
                                break
                            # Prefer the closest set of other source findings, without using clusters.
                            fs = set(pr.finding_types) - {finding}
                            distances = neg.finding_types.map(lambda x: len(fs.symmetric_difference(set(x))))
                            nr = neg.loc[distances.idxmin()]
                            neg = neg.drop(nr.name)
                            options.setdefault(compound, []).append((pr, nr, int(distances.min())))
                comparisons.append(dict(finding=finding, topography=topo, compound=compound, same_dose_time_pairs=count))
    selected = []
    for compound, pairs in sorted(options.items()):
        if len(pairs) < 3:
            continue
        # Prefer closer cofinding profiles, spread timepoints, use slide IDs as ties.
        ranked = sorted(pairs, key=lambda x: (x[2], x[0].sacrifice_period, x[0].slide_id))
        used_controls, chosen = set(), 0
        for pr, nr, distance in ranked:
            controls = m[(m.compound_name == compound) & (m.exp_id == pr.exp_id) &
                         (m.sacrifice_period == pr.sacrifice_period) & (m.dose_level == 'Control') &
                         ~m.finding_types.map(lambda f: 'Necrosis' in f) & ~m.slide_id.isin(used_controls)].sort_values('slide_id')
            if controls.empty:
                continue
            cr = controls.iloc[0]
            used_controls.add(cr.slide_id)
            chosen += 1
            for role, row in [('positive', pr), ('treated_negative', nr), ('control', cr)]:
                record = row.to_dict()
                target = a[(a.slide_id == row.slide_id) & (a.FINDING_TYPE == 'Necrosis') & (a.TOPOGRAPHY_TYPE == 'Hepatocyte')]
                record.update(role=role, match_id=f'{compound}:{chosen}', facility_id='unknown',
                              target_finding='Necrosis', target_topography='Hepatocyte',
                              target_grades='|'.join(sorted(set(target.GRADE_TYPE.dropna()))),
                              target_spontaneous_flags='|'.join(sorted(set(target.SP_FLG.astype(str)))),
                              cofinding_distance=distance, label_status='source_annotation_unreviewed',
                              holdout_group=compound)
                for col in ['finding_types', 'finding_sites']:
                    record[col] = json.dumps(list(record[col]))
                record['svs_path'] = str(Path(record['svs_path']).relative_to(ROOT))
                selected.append(record)
            if chosen == 3:
                break
        assert chosen == 3, 'Not enough independent controls'
    s = pd.DataFrame(selected)
    assert s.slide_id.is_unique and not s.duplicated(keys).any()
    assert s.groupby(['compound_name', 'role']).size().eq(3).all()
    for _, g in s.groupby('match_id'):
        assert len(g) == 3 and g.exp_id.nunique() == g.sacrifice_period.nunique() == 1
        assert g[g.role != 'control'].dose_level.nunique() == 1
    out.mkdir()
    pd.DataFrame(comparisons).to_csv(out / 'candidate_matching_capacity.csv', index=False)
    s.to_csv(out / 'slides.csv', index=False)
    for role in ['positive', 'treated_negative', 'control']:
        (out / f'{role}_slide_ids.txt').write_text('\n'.join(s.loc[s.role == role, 'slide_id']) + '\n')
    s.groupby(['compound_name', 'role', 'dose_level', 'target_grades', 'target_spontaneous_flags'], dropna=False).size().rename('slides').to_csv(out / 'distribution.csv')
    (out / 'provenance.json').write_text(json.dumps(dict(
        target='Necrosis @ Hepatocyte (including source-annotated spontaneous lesions)',
        purpose='exploratory morphology review, not a validated benchmark',
        facility_verified=False, labels_visually_verified=False,
        positive_negative_matching=['compound', 'experiment', 'timepoint', 'dose_level'],
        compound_cap=3, cofindings='greedy minimum symmetric difference',
        sources={str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest() for f in
                 [Path(__file__), BASE / 'label_audit/manifest_known_unique.parquet', ROOT / 'data/corrected/open_tggates_pathology.csv']},
        artifacts={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir()}), indent=2))
    print(s.groupby(['compound_name','role']).size().to_string())


if __name__ == '__main__':
    main()
