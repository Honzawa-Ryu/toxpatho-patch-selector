"""Draw unbiased cluster samples and an offline gallery for selection comparison."""
from pathlib import Path
import json
import html
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import h5py
import openslide
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
NAME = '0011_20260925_stratified_candidate_selection'
REPORT = ROOT / 'experiments' / NAME
OUT = ROOT / 'outputs' / NAME
GALLERY = OUT / 'gallery'
BIG = ROOT / 'outputs/0009_20260919_tier1_corpus_clustering'
N = 24
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def tissue_fraction(tile):
    white = np.asarray(tile.resize((128,128)),dtype=float).mean(axis=2) >= 220
    components, _ = ndimage.label(white)
    border = np.unique(np.r_[components[0],components[-1],components[:,0],components[:,-1]])
    return float(1-np.isin(components,border[border!=0]).mean())


def init_worker():
    global MANIFEST
    MANIFEST = pd.read_parquet(OUT / 'manifest.parquet').set_index('slide_id')


def draw_cluster(task):
    cid, sample = task
    sheet = Image.new('RGB',(1536,54+4*282),'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT,14)
    draw.text((8,8),f'Cluster {cid} | 24 random cluster patches | no pathology-label or tissue filtering',fill='black',font=font)
    records, errors = [], []
    paths = {}
    try:
        for j, rec in enumerate(sample):
            sid = str(rec['slide_id'])
            row = MANIFEST.loc[sid]
            try:
                if sid not in paths:
                    with h5py.File(BIG/'features_tier1_corpus'/f'{sid}.h5','r') as h:
                        size = int(h['coords'].attrs['patch_size_level0'])
                    paths[sid] = (openslide.OpenSlide(str(row.svs_path)),size)
                wsi, size = paths[sid]
                tile = wsi.read_region((int(rec['x']),int(rec['y'])),0,(size,size)).convert('RGB')
                tf = tissue_fraction(tile)
                x,y = (j%6)*256,54+(j//6)*282
                sheet.paste(tile.resize((256,256)),(x,y+26))
                draw.text((x+2,y+2),f'{sid} {row.dose_level[:1]} {row.compound_name[:19]}',fill='black',font=font)
                records.append(dict(cluster_id=cid,tile=j+1,slide_id=sid,x=int(rec['x']),y=int(rec['y']),
                    patch_size_level0=size,compound_name=row.compound_name,dose_level=row.dose_level,
                    exp_id=int(row.exp_id),sacrifice_period=row.sacrifice_period,tissue_fraction=tf))
            except Exception as exc:
                errors.append(dict(cluster_id=cid,slide_id=sid,error=str(exc)))
    finally:
        for wsi,_ in paths.values(): wsi.close()
    sheet.save(GALLERY/'sheets'/f'cluster_{cid:04d}.jpg',quality=92)
    sheet.resize((512,394)).save(GALLERY/'thumbs'/f'cluster_{cid:04d}.jpg',quality=85)
    quality = dict(cluster_id=cid,quality_n=len(records),tissue_fraction=np.mean([r['tissue_fraction'] for r in records]) if records else np.nan,
                   sample_slides=len(set(r['slide_id'] for r in records)),sample_compounds=len(set(r['compound_name'] for r in records)))
    return quality,records,errors


def build_html(catalog):
    records=[]
    for row in catalog.to_dict('records'):
        records.append({k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()})
    page = '''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>クラスタ候補の比較 — 実験0011</title><style>
body{font-family:system-ui,sans-serif;margin:24px;color:#17212b;background:#f4f6f8}h1{font-size:24px}header{position:sticky;top:0;background:#f4f6f8;padding:10px 0;z-index:1;border-bottom:1px solid #ccc}select,input,button{font:inherit;padding:7px;margin:4px}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px;margin-top:16px}.card{background:white;border:1px solid #ddd;border-radius:8px;padding:12px}.card img{width:100%;height:auto}.meta{font-size:13px;line-height:1.7}.badge{background:#e6edf5;padding:3px 7px;border-radius:4px}.muted{color:#526270}a{color:#145ca0}.warn{color:#9b3b00}summary{cursor:pointer}textarea{width:96%;height:45px}
</style><h1>候補選定を画像で比較する</h1>
<p>同じ k=2000。全体比較と実験・時点内比較の候補__COUNT__個を表示します。各画像はクラスタ全体から無作為に抽出した24パッチです。</p>
<p class="muted">表示される所見名・対応率はスライドラベルとの関連で、パッチの診断ではありません。組織割合はこの24パッチの平均。効果量はB2選択時にはB2、それ以外は原則B1（B2のみ追加された候補はB2）の代表化合物で表示。クリックで拡大します。</p>
<header><label>候補 <select id="status"><option value="all">全__COUNT__候補</option><option value="anchors" selected>まず確認する12候補</option><option value="added">B1で追加</option><option value="removed">B1で除外</option><option value="both">A・B1共通</option><option value="new">B1：Control条件も変更</option><option value="old">A：従来の全体比較</option><option value="alternative">B2：Control条件は従来どおり</option><option value="loss">取りこぼし確認</option></select></label>
<label>組織割合 <select id="tissue"><option value="all">フィルタなし</option><option value="pass">0.85以上</option><option value="fail">0.85未満・未測定</option></select></label>
<input id="search" placeholder="ID・化合物・所見で検索"><label>並び <select id="sort"><option value="effect">保有率差が大きい順</option><option value="id">クラスタID順</option><option value="tissue">組織割合が低い順</option><option value="precision">所見との対応率順</option></select></label>
<div id="count"></div></header><div id="grid"></div><button id="more">さらに40件表示</button>
<p><a href="catalog.csv">全候補のCSV</a> · <a href="tiles.csv">各パッチのスライド・座標</a> · <a href="review_template.csv">判定記入用CSV</a></p>
<script>const DATA=__DATA__;let limit=40;const names={added:'追加',removed:'除外',both:'A・B1共通',alternative_only:'B2のみ追加'};
function esc(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function pct(x){return x==null?'未測定':(100*x).toFixed(1)+'%'}
function render(){const st=document.querySelector('#status').value,tf=document.querySelector('#tissue').value,q=document.querySelector('#search').value.toLowerCase(),sort=document.querySelector('#sort').value;
let rows=DATA.map(r=>st==='alternative'?{...r,best_compound:r.b2_best_compound,local_q:r.b2_local_q,presence_difference:r.b2_presence_difference,supporting_compounds:r.b2_supporting_compounds,treated_carriers:r.b2_treated_carriers,control_carriers:r.b2_control_carriers,treated_slides:r.b2_treated_slides,control_slides:r.b2_control_slides}:r).filter(r=>(st==='all'||st==='anchors'&&r.anchor||st==='new'&&r.stratified_selected||st==='old'&&r.global_selected||st==='alternative'&&r.stratified_global_absence_selected||st==='loss'&&[539,931,585,45,1682].includes(r.cluster_id)||r.status===st)&&(tf==='all'||tf==='pass'&&r.tissue_pass||tf==='fail'&&!r.tissue_pass)&&[r.cluster_id,r.best_compound,r.dominant_compound,r.best_finding].join(' ').toLowerCase().includes(q));
rows.sort((a,b)=>sort==='id'?a.cluster_id-b.cluster_id:sort==='tissue'?(a.tissue_fraction??-1)-(b.tissue_fraction??-1):sort==='precision'?(b.slide_precision??-1)-(a.slide_precision??-1):(b.presence_difference??-1)-(a.presence_difference??-1));
document.querySelector('#count').textContent=`${rows.length}件中 ${Math.min(limit,rows.length)}件表示`;
document.querySelector('#grid').innerHTML=rows.slice(0,limit).map(r=>{let id=String(r.cluster_id).padStart(4,'0');return `<article class="card"><b>Cluster ${r.cluster_id}</b> <span class="badge">${names[r.status]}</span>${r.anchor?' ★':''}<a href="sheets/cluster_${id}.jpg" target="_blank"><img loading="lazy" src="thumbs/cluster_${id}.jpg" alt="cluster ${r.cluster_id}の24パッチ"></a><div class="meta"><b class="${r.tissue_pass?'':'warn'}">組織割合 ${pct(r.tissue_fraction)}</b> · ${r.sample_slides}スライド/${r.sample_compounds}化合物から抽出<br>B2支持化合物：${esc(r.b2_best_compound||'なし')}<br>表示中の比較の代表化合物：${esc(r.best_compound||'新方式では支持なし')}<br>保有率差 ${pct(r.presence_difference)} · 支持化合物数 ${r.supporting_compounds}<br>A ${r.global_selected?'○':'—'} / B1 ${r.stratified_selected?'○':'—'} / B2 ${r.stratified_global_absence_selected?'○':'—'}<br>投与保有 ${r.treated_carriers??'—'}/${r.treated_slides??'—'}、Control保有 ${r.control_carriers??'—'}/${r.control_slides??'—'}<br>関連所見：${esc(r.best_finding||'有意な対応なし')}<br>所見との保有内率 ${pct(r.slide_precision)} · 保有スライド ${r.carrying_slides}</div></article>`}).join('');document.querySelector('#more').style.display=rows.length>limit?'block':'none';}
for(const e of document.querySelectorAll('select,input'))e.addEventListener('input',()=>{limit=40;render()});document.querySelector('#more').onclick=()=>{limit+=40;render()};render();</script></html>'''
    (GALLERY/'index.html').write_text(page.replace('__COUNT__',str(len(catalog))).replace('__DATA__',json.dumps(records,ensure_ascii=False).replace('</','<\\/')),encoding='utf-8')


def main():
    if GALLERY.exists(): raise SystemExit('Gallery exists; preserve review images')
    (GALLERY/'sheets').mkdir(parents=True)
    (GALLERY/'thumbs').mkdir()
    catalog = pd.read_csv(REPORT/'candidate_comparison.csv')
    wanted = set(catalog.cluster_id)
    manifest = pd.read_parquet(OUT/'manifest.parquet')
    eligible = set(manifest.slide_id.astype(str))
    rng = np.random.default_rng(20260925)
    reservoirs = {cid:pd.DataFrame() for cid in wanted}
    populations = {cid:0 for cid in wanted}
    path = BIG/'kmeans_k_sweep_part3/cluster_assignments.parquet'
    for batch in pq.ParquetFile(path).iter_batches(batch_size=500000,columns=['slide_id','x','y','cluster_kmeans_k2000']):
        frame = batch.to_pandas()
        frame = frame[frame.cluster_kmeans_k2000.isin(wanted) & frame.slide_id.isin(eligible)].copy()
        frame['priority'] = rng.random(len(frame))
        for cid,group in frame.groupby('cluster_kmeans_k2000',sort=False):
            populations[cid] += len(group)
            reservoirs[cid] = pd.concat([reservoirs[cid],group.nsmallest(N,'priority')],ignore_index=True).nsmallest(N,'priority')
    samples = pd.concat(list(reservoirs.values()),ignore_index=True).rename(columns={'cluster_kmeans_k2000':'cluster_id'})
    samples.to_parquet(OUT/'gallery_sample_coordinates.parquet',index=False)
    jobs = [(int(cid),g.to_dict('records')) for cid,g in samples.groupby('cluster_id')]
    qualities, tiles, errors = [], [], []
    with ProcessPoolExecutor(max_workers=6,initializer=init_worker) as pool:
        for i,(quality,records,failed) in enumerate(pool.map(draw_cluster,jobs),1):
            qualities.append(quality);tiles.extend(records);errors.extend(failed)
            if i%25==0 or i==len(jobs):print(f'{i}/{len(jobs)} sheets',flush=True)
    catalog = catalog.merge(pd.DataFrame(qualities),on='cluster_id',validate='one_to_one')
    catalog['sample_population_patches'] = catalog.cluster_id.map(populations)
    catalog['tissue_pass'] = (catalog.quality_n==N)&(catalog.tissue_fraction>=.85)
    catalog.to_csv(GALLERY/'catalog.csv',index=False)
    catalog.to_csv(REPORT/'candidate_comparison_with_quality.csv',index=False)
    pd.DataFrame(tiles).to_csv(GALLERY/'tiles.csv',index=False)
    pd.DataFrame(errors,columns=['cluster_id','slide_id','error']).to_csv(GALLERY/'crop_errors.csv',index=False)
    review=catalog[['cluster_id','status','anchor']].copy()
    for col in ['reviewer','morphology_present','artifact','keep','comments']:review[col]=''
    review.to_csv(GALLERY/'review_template.csv',index=False)
    rows=[]
    for method,column in [('global','global_selected'),('stratified','stratified_selected')]:
        for tissue in [False,True]:
            take=catalog[catalog[column] & (catalog.tissue_pass if tissue else True)]
            rows.append(dict(method=method,tissue_filter=tissue,candidates=len(take),
                median_best_slide_precision=take.slide_precision.median(),
                tissue_below_085=int((take.tissue_fraction<.85).sum()),
                recovered_fatty_anchors=int(take.cluster_id.isin([1356,846,703,535]).sum()),
                retained_morphology_anchors=int(take.cluster_id.isin([1625,147,697,271,801,209]).sum()),
                retained_background_anchors=int(take.cluster_id.isin([95,1425]).sum())))
    pd.DataFrame(rows).to_csv(REPORT/'ablation_summary.csv',index=False)
    build_html(catalog)
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
