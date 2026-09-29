"""Make label-hidden, random tissue-patch sheets for every v2 slide.

These samples screen image quality; absence of a focal lesion in six random
patches is not evidence that a slide is negative. No cluster is used.
"""
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
import pandas as pd
import openslide
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiments/0010_20260925_pathology_benchmark_subset/snapshot_v2'
OUT = ROOT / 'outputs/0010_20260925_pathology_benchmark_subset/review_v2'


def main():
    if OUT.exists():
        raise SystemExit('Review exists; preserve it')
    OUT.mkdir(parents=True)
    slides = pd.read_csv(SOURCE / 'slides.csv', dtype={'slide_id': str}).sample(frac=1, random_state=20260925)
    rng = np.random.default_rng(20260925)
    rows, tiles = [], []
    for number, row in enumerate(slides.itertuples(), 1):
        rid = f'R{number:03d}'
        file = ROOT / 'outputs/0009_20260919_tier1_corpus_clustering/features_tier1_corpus' / f'{row.slide_id}.h5'
        with h5py.File(file, 'r') as h:
            coords = h['coords'][:]
            attrs = h['coords'].attrs
            if 'patch_size_level0' not in attrs:
                raise ValueError(f'Missing physical crop size in {file}')
            size = int(attrs['patch_size_level0'])
        wsi = openslide.OpenSlide(str(ROOT / row.svs_path))
        sheet = Image.new('RGB', (768, 540), 'white')
        draw = ImageDraw.Draw(sheet)
        draw.text((6, 5), f'{rid} | random tissue patches | source labels hidden', fill='black')
        n = 0
        for idx in rng.permutation(len(coords))[:300]:
            x, y = map(int, coords[idx])
            tile = wsi.read_region((x, y), 0, (size, size)).convert('RGB').resize((256, 256))
            white = np.asarray(tile).mean(axis=2) >= 220
            labels, _ = ndimage.label(white)
            border = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
            fraction = 1 - np.isin(labels, border[border != 0]).mean()
            if fraction < 0.85:
                continue
            sheet.paste(tile, ((n % 3) * 256, 28 + (n // 3) * 256))
            tiles.append(dict(review_id=rid, slide_id=row.slide_id, tile=n + 1, x=x, y=y,
                              patch_size_level0=size, tissue_fraction=fraction))
            n += 1
            if n == 6:
                break
        wsi.close()
        path = OUT / f'{rid}.png'
        sheet.save(path)
        rows.append(dict(review_id=rid, slide_id=row.slide_id, role=row.role, compound=row.compound_name,
                         match_id=row.match_id, sheet=str(path.relative_to(ROOT)), n_tiles=n,
                         image_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        if number % 9 == 0:
            print(f'{number}/{len(slides)}', flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'unblinding_key.csv', index=False)
    pd.DataFrame(tiles).to_csv(OUT / 'tiles.csv', index=False)
    review = pd.DataFrame(rows)[['review_id', 'sheet', 'n_tiles', 'image_sha256']]
    for col in ['reviewer', 'image_quality', 'suspected_morphology', 'finding_confirmed', 'comments']:
        review[col] = ''
    review.to_csv(OUT / 'review.csv', index=False)
    # Nine sheets per page at native tile resolution; no source label in pages.
    for page in range((len(rows) + 8) // 9):
        canvas = Image.new('RGB', (2304, 1620), 'white')
        for j, row in enumerate(rows[page * 9:(page + 1) * 9]):
            canvas.paste(Image.open(ROOT / row['sheet']), ((j % 3) * 768, (j // 3) * 540))
        canvas.save(OUT / f'page_{page+1}.jpg', quality=95)
    (OUT / 'protocol.json').write_text(json.dumps(dict(seed=20260925, patches_per_slide=6,
        tissue_threshold=0.85, candidate_draw_cap=300, random_patch_screening=True,
        blind_scope='source labels hidden on sheets; not an independent clinical diagnosis',
        limitation='random crops cannot establish absence of focal necrosis'), indent=2))


if __name__ == '__main__':
    main()
