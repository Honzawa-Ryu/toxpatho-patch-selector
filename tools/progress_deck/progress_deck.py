"""実験ごとの report.yaml から進捗スライド（Claude の Slides アーティファクト用ファイル）を作る。

    python3 tools/progress_deck/progress_deck.py build            # 目録 + スライド生成（make progress_deck）
    python3 tools/progress_deck/progress_deck.py register FILE    # アップロード済み図の ID を記録（各行 "<画像ファイル名> <blob id>"）
    python3 tools/progress_deck/progress_deck.py set-url URL      # スライドの URL を記録
    python3 tools/progress_deck/progress_deck.py mark-published   # 公開後に呼ぶ。今回の生成内容を「公開済み」にする

入力: experiments/*/metadata.yaml, experiments/*/report.yaml, docs/progress_deck/extras/*.yaml（実験ディレクトリ外の項目）,
      docs/progress_deck/config.yaml（title, goal）
出力: docs/progress_deck/inventory.json, docs/progress_deck/build/（Git 管理外）
状態: docs/progress_deck/state.json（URL, 図の blob ID, 公開済みスライドのハッシュ。Git 管理）
"""
import datetime, hashlib, html, json, re, shutil, sys
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PD = ROOT / "docs" / "progress_deck"
BUILD = PD / "build"
STATE = PD / "state.json"
FIG_EXT = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
IMG_EXT = {".png", ".jpg", ".jpeg"}  # スライドに載せられる形式（svg/pdf はノートに列挙のみ）

BG, DARK, MUTED, ACC, ACC2, LINE = "#F7F6F2", "#1C2430", "#4A5563", "#2F6DB5", "#C8642A", "#D9D6CC"
FONT = "'Noto Sans JP', Arial, sans-serif"
e = lambda s: html.escape(str(s), quote=True)
def clip(s, n):
    s = str(s or "")
    return s if len(s) <= n else s[: n - 1] + "…"


def load_yaml(p):
    return (yaml.safe_load(p.read_text()) or {}) if p.exists() else {}


def load_state():
    return json.loads(STATE.read_text()) if STATE.exists() else {"url": None, "blobs": {}, "published": {}}


def save_state(st):
    PD.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1) + "\n")


# ---------------------------------------------------------------- inventory
def count_figures(exp_name):
    d = ROOT / "outputs" / exp_name
    return sum(1 for p in d.rglob("*") if p.suffix.lower() in FIG_EXT) if d.is_dir() else 0


def entry(eid, date, name, rep):
    filled = any(rep.get(k) for k in ("purpose", "method", "results", "figures"))
    return {
        "id": eid, "date": date, "name": name,
        "title": rep.get("title") or name,
        "status": rep.get("status") or ("" if filled else "未記入"),
        "purpose": rep.get("purpose") or "", "method": rep.get("method") or "",
        "results": rep.get("results") or [], "figures": rep.get("figures") or [],
        "chart": rep.get("chart"), "notes": rep.get("notes") or "",
        "filled": filled, "n_figures_total": count_figures(name),
    }


def build_inventory():
    items = []
    for d in sorted((ROOT / "experiments").glob("[0-9][0-9][0-9][0-9]_*")):
        if not d.is_dir():
            continue
        # ID と日付はディレクトリ名から取る（metadata.yaml の exp_id: 0012 は YAML で8進数として読まれるため使わない）
        m = re.match(r"(\d{4})_(\d{4})(\d\d)(\d\d)", d.name)
        date = f"{m.group(2)}-{m.group(3)}-{m.group(4)}" if m else ""
        items.append(entry(d.name[:4], date, d.name, load_yaml(d / "report.yaml")))
    for p in sorted((PD / "extras").glob("*.yaml")):
        rep = load_yaml(p)
        items.append(entry(str(rep.get("id") or p.stem), str(rep.get("date") or ""), p.stem, rep))
    items.sort(key=lambda x: (x["date"], x["id"]))
    cfg = load_yaml(PD / "config.yaml")
    inv = {"title": cfg.get("title") or ROOT.name, "goal": cfg.get("goal") or "", "repo": ROOT.name, "experiments": items}
    PD.mkdir(parents=True, exist_ok=True)
    (PD / "inventory.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1) + "\n")
    return inv


# ---------------------------------------------------------------- images
def prepare_image(path, st):
    """縮小して build/imgs/<sha1>.<ext> に置く。戻り値: (画像ファイル名, blob url or None)"""
    src = ROOT / path
    if not src.exists() or src.suffix.lower() not in IMG_EXT:
        return None, None
    h = hashlib.sha1(src.read_bytes()).hexdigest()[:16]
    if h in st["blobs"]:
        return h, "/_blob/" + st["blobs"][h]
    out_dir = BUILD / "imgs"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not any(out_dir.glob(h + ".*")):
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(src)
        im.load()
        im = im.convert("RGBA") if im.mode in ("P", "LA", "RGBA") else im.convert("RGB")
        s = 1600 / max(im.size)
        if s < 1:
            im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        out = out_dir / f"{h}.png"
        im.save(out, optimize=True)
        if out.stat().st_size > 900_000:
            out.unlink()
            im.convert("RGB").save(out_dir / f"{h}.jpg", quality=85)
    return h, None


# ---------------------------------------------------------------- slides
def chart_html(c, width):
    lab_w, bar_max = 300, width - 300 - 110 - 32 - 64
    vals_all = [v for _, x in c["rows"] for v in (x if isinstance(x, list) else [x])]
    vmax = max(1.0, max(vals_all, default=1.0))
    o = [f'<div style="display:flex;flex-direction:column;gap:20px;width:{width}px;background:#FBFBF8;border:1px solid {LINE};border-radius:12px;padding:32px">',
         f'<p style="font-size:26px;font-weight:700">{e(c.get("title", ""))}</p>']
    ser = c.get("series") or []
    if len(ser) >= 2:
        o.append(f'<div style="display:flex;gap:24px;align-items:center"><div style="width:28px;height:20px;background:{ACC}"></div><p style="font-size:24px">{e(ser[0])}</p>'
                 f'<div style="width:28px;height:20px;background:{ACC2}"></div><p style="font-size:24px">{e(ser[1])}</p></div>')
    for lab, x in c["rows"]:
        xs = x if isinstance(x, list) else [x]
        o.append(f'<div style="display:flex;align-items:center;gap:16px"><p style="width:{lab_w}px;font-size:24px;text-align:right">{e(lab)}</p><div style="display:flex;flex-direction:column;gap:4px">')
        for i, v in enumerate(xs[:2]):
            o.append(f'<div style="display:flex;align-items:center;gap:12px"><div style="width:{max(4, int(bar_max * v / vmax))}px;height:{28 if len(xs) == 1 else 22}px;background:{(ACC, ACC2)[i]}"></div>'
                     f'<p style="font-size:24px;font-variant-numeric:tabular-nums">{v:.3f}</p></div>')
        o.append("</div></div>")
    return "".join(o + ["</div>"])


def status_color(s):
    return ACC2 if re.search(r"実行中|running|未|途中|失敗|中止|部分|要再", s) else MUTED


def fig_block(figs, rw):
    o = []
    if len(figs) == 1:
        g, src = figs[0]
        o.append(f'<img src="{src}" alt="{e(clip(g.get("caption"), 200))}" style="width:{rw}px;height:620px;object-fit:contain;background:#FFFFFF;border:1px solid {LINE}">')
        o.append(f'<p style="font-size:24px;color:{MUTED};line-height:1.35">{e(clip(g.get("caption"), 110))}</p>')
    elif len(figs) == 2:
        for k, (g, src) in enumerate(figs):
            o.append(f'<img src="{src}" alt="{e(clip(g.get("caption"), 200))}" style="width:{rw}px;height:290px;object-fit:contain;background:#FFFFFF;border:1px solid {LINE}">')
            o.append(f'<p style="font-size:24px;color:{MUTED};line-height:1.3">{"ab"[k]}. {e(clip(g.get("caption"), 70))}</p>')
    else:
        cw = (rw - 16) // 2
        o.append('<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">')
        o += [f'<img src="{src}" alt="{e(clip(g.get("caption"), 200))}" style="width:{cw}px;height:300px;object-fit:contain;background:#FFFFFF;border:1px solid {LINE}">' for g, src in figs]
        o.append("</div>")
        cap = "／".join(f'{"abc"[k]}. {clip(g.get("caption"), 40)}' for k, (g, _) in enumerate(figs))
        o.append(f'<p style="font-size:24px;color:{MUTED};line-height:1.3">{e(cap)}</p>')
    return o


def exp_slide(sid, ex, idx, total, st, missing):
    figs = []
    for g in [g for g in ex["figures"] if g.get("representative")][:3]:
        name, src = prepare_image(g["path"], st)
        if src:
            figs.append((g, src))
        elif name:
            missing.add(name)
    has_right = bool(figs) or bool(ex["chart"])
    lw = 700 if has_right else 1664
    rs, nres = ex["results"], (6 if has_right else 10)
    nf = ex["n_figures_total"] or len(ex["figures"])
    o = [f'<section id="{sid}" style="background:{BG};color:{DARK};font-family:{FONT};padding:88px 112px 136px;display:flex;flex-direction:column;gap:28px">',
         f'<div style="display:flex;flex-direction:column;gap:8px"><p style="font-size:24px;color:{ACC};font-weight:700">{e(ex["id"])}　|　{e(ex["date"])}</p>'
         f'<h2 style="font-size:48px;font-weight:700;line-height:1.2">{e(clip(ex["title"], 70))}</h2></div>',
         '<div style="display:flex;gap:56px;flex:1">', f'<div style="width:{lw}px;display:flex;flex-direction:column;gap:18px">']
    if ex["filled"]:
        o.append(f'<div data-text style="font-size:24px;line-height:1.45"><p><b>目的</b>　{e(clip(ex["purpose"], 160 if has_right else 300))}</p></div>')
        o.append(f'<div data-text style="font-size:24px;line-height:1.45;color:{MUTED}"><p><b>方法</b>　{e(clip(ex["method"], 200 if has_right else 400))}</p></div>')
        if rs:
            o.append('<p style="font-size:24px;font-weight:700">結果</p><ul style="font-size:24px;line-height:1.4">'
                     + "".join(f"<li>{e(clip(r, 120 if has_right else 220))}</li>" for r in rs[:nres]) + "</ul>")
            if len(rs) > nres:
                o.append(f'<p style="font-size:24px;color:{MUTED}">ほか {len(rs) - nres} 件はノート参照</p>')
    else:
        o.append(f'<p style="font-size:32px;color:{ACC2}">report.yaml 未記入</p><p style="font-size:24px;color:{MUTED}">experiments/{e(ex["name"])}/report.yaml を埋めると、ここに目的・方法・結果が入る</p>')
    o.append("</div>")
    if has_right:
        rw = 1664 - lw - 56
        o.append(f'<div style="width:{rw}px;display:flex;flex-direction:column;gap:12px">')
        o += [chart_html(ex["chart"], rw)] if ex["chart"] else fig_block(figs, rw)
        o.append("</div>")
    o.append("</div>")
    o.append(f'<p style="position:absolute;left:112px;bottom:56px;width:1696px;font-size:24px;color:{MUTED}">'
             f'<span style="color:{status_color(ex["status"])}">状態：{e(clip(ex["status"] or "—", 40))}</span>　|　{"図 %d 枚" % nf if nf else "図なし"}　|　{idx}/{total}</p>')
    n = [f"{ex['id']} {ex['date']} {ex['name']} 状態:{ex['status']}", f"目的: {ex['purpose']}", f"方法: {ex['method']}", "結果:"]
    n += [f"- {r}" for r in rs]
    if ex["chart"]:
        n.append("※棒グラフは記録された数値から作ったもの（元の図なし）。")
    n.append(f"図（outputs 内 {nf} 枚、report.yaml に列挙 {len(ex['figures'])} 枚）:")
    n += [f"- {'★' if g.get('representative') else ''}{g.get('caption', '')} :: {g.get('path')}" for g in ex["figures"]]
    if ex["notes"]:
        n.append(f"メモ: {ex['notes']}")
    note = "\n".join(n)
    o.append(f"<aside>{e(note if len(note) <= 3900 else note[:3880] + chr(10) + '…（以下省略）')}</aside></section>")
    return "\n".join(o)


def cover_slide(inv, n):
    return f'''<section id="cover" style="background:{DARK};color:#F2F0EA;font-family:{FONT};padding:128px;display:flex;flex-direction:column;justify-content:center;gap:40px">
<p style="font-size:32px;color:#8FB5E3;font-weight:700">{e(inv["repo"])}</p>
<h1 style="font-size:96px;font-weight:700;line-height:1.15">{e(inv["title"])}</h1>
<p style="font-size:32px;line-height:1.5;color:#C9CED6;width:1500px">{e(clip(inv["goal"], 200))}</p>
<p style="font-size:28px;color:#C9CED6">実験 {n} 件　|　更新 {datetime.date.today().isoformat()}</p>
</section>'''


def overview_slides(items):
    """一覧表。14 行ずつ分割。"""
    out = []
    for k in range(0, max(len(items), 1), 14):
        rows = "".join(f'<tr><td>{e(x["id"])}</td><td>{e(x["date"])}</td><td>{e(clip(x["title"], 46))}</td><td>{e(clip(x["status"], 10))}</td></tr>' for x in items[k:k + 14])
        sid = f"overview-{k // 14 + 1}"
        out.append((sid, f'''<section id="{sid}" style="background:{BG};color:{DARK};font-family:{FONT};padding:88px 128px 112px;display:flex;flex-direction:column;gap:28px">
<h2 style="font-size:48px;font-weight:700">実験一覧（{k + 1}〜{min(k + 14, len(items))}）</h2>
<table style="font-size:24px"><tr><th style="width:10%">ID</th><th style="width:16%">日付</th><th style="width:60%">内容</th><th style="width:14%">状態</th></tr>{rows}</table>
</section>'''))
    return out


def build():
    inv = build_inventory()
    st = load_state()
    items, missing, slides = inv["experiments"], set(), {}
    slides["cover"] = cover_slide(inv, len(items))
    order, sections = ["cover"], {"s00": {"description": "表紙と一覧", "start": "cover"}}
    for sid, h in overview_slides(items):
        slides[sid] = h
        order.append(sid)
    month = None
    for i, ex in enumerate(items, 1):
        sid = "exp-" + re.sub(r"[^A-Za-z0-9_-]", "-", ex["id"])[:60]
        slides[sid] = exp_slide(sid, ex, i, len(items), st, missing)
        order.append(sid)
        if ex["date"][:7] != month:
            month = ex["date"][:7]
            sections[f"m{month or 'nodate'}"] = {"description": f"{month or '日付なし'} の実験", "start": sid}
    if BUILD.exists():
        shutil.rmtree(BUILD / "project", ignore_errors=True)
    (BUILD / "project" / "slides").mkdir(parents=True, exist_ok=True)
    created = st.setdefault("createdOnFiles", {"v": 1, "at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    deck = {"v": 4, "createdOnFiles": created, "lists": "css", "title": inv["title"], "cover": "cover", "order": order,
            "sections": sections, "faces": {"noto-sans-jp": {"family": "Noto Sans JP", "href": "https://fonts.googleapis.com/css2?family=Noto+Sans+JP:wght@400;700&display=swap"}},
            "designSystems": []}
    files = {"project/deck.json": json.dumps(deck, ensure_ascii=False, indent=1)}
    files.update({f"project/slides/{s}.html": h for s, h in slides.items()})
    pending, changed = {}, {}
    for rel, text in files.items():
        (BUILD / rel).write_text(text)
        h = hashlib.sha1(text.encode()).hexdigest()
        pending[rel] = h
        if st["published"].get(rel) != h:
            changed[rel] = rel
    removed = {rel: None for rel in st["published"] if rel not in files}
    (BUILD / "publish_files.json").write_text(json.dumps({**changed, **removed}, ensure_ascii=False, indent=0))
    (BUILD / "pending_hashes.json").write_text(json.dumps(pending))
    save_state(st)
    up = sorted(str(p) for n in missing for p in (BUILD / "imgs").glob(n + ".*"))
    (BUILD / "upload_list.json").write_text(json.dumps(up, indent=0))
    unfilled = [x["id"] for x in items if not x["filled"]]
    print(f"実験 {len(items)} 件（未記入 {len(unfilled)}: {', '.join(unfilled[:20])}{' …' if len(unfilled) > 20 else ''}）")
    print(f"スライド {len(order)} 枚 / 変更 {len(changed)} / 削除 {len(removed)} / 未アップロード図 {len(up)}")
    print(f"URL: {st['url'] or '未作成'}")
    print(f"build: {BUILD}")


def main():
    cmd, args = (sys.argv[1] if len(sys.argv) > 1 else "build"), sys.argv[2:]
    st = load_state()
    if cmd == "build":
        build()
    elif cmd == "register":
        for line in Path(args[0]).read_text().splitlines():
            if line.strip():
                name, bid = line.split()[:2]
                st["blobs"][Path(name).name.split(".")[0]] = bid.replace("/_blob/", "")
        save_state(st)
        print(f"登録済み図 {len(st['blobs'])} 件")
    elif cmd == "set-url":
        st["url"] = args[0]
        save_state(st)
    elif cmd == "mark-published":
        st["published"] = json.loads((BUILD / "pending_hashes.json").read_text())
        save_state(st)
        print("公開済みとして記録")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
