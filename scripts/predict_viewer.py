#!/usr/bin/env python3
"""
predict_viewer.py — генерирует красивую HTML-страницу для просмотра предсказаний модели.

Использование:
  python predict_viewer.py artefacts/predictions_val_set_2500.csv
  python predict_viewer.py artefacts/predictions_val_set_2500.csv --cutoff 128 -o viewer.html
  python predict_viewer.py artefacts/predictions_50.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

DEFAULT_TOKENIZER = "model_cache/Qwen--Qwen3-Embedding-0.6B"
PAGE_SIZE = 20


def parse_args():
    p = argparse.ArgumentParser(description="Генератор HTML-просмотрщика предсказаний")
    p.add_argument("csv", help="Путь к CSV с предсказаниями")
    p.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    p.add_argument("--cutoff",     type=int, default=64,  help="Начальный cutoff токенов (default: 64)")
    p.add_argument("--max-cutoff", type=int, default=256, help="Максимум слайдера (default: 256)")
    p.add_argument("--max-samples", type=int, default=None, help="Ограничить кол-во примеров")
    p.add_argument("-o", "--output", default="predictions_viewer.html")
    return p.parse_args()


def f1_score(pred: str, target: str) -> float:
    pred_toks   = pred.lower().split()
    target_toks = target.lower().split()
    if not pred_toks or not target_toks:
        return 0.0
    common = set(pred_toks) & set(target_toks)
    if not common:
        return 0.0
    prec = len(common) / len(pred_toks)
    rec  = len(common) / len(target_toks)
    return round(2 * prec * rec / (prec + rec), 3)


def process(df: pd.DataFrame, tokenizer, max_cutoff: int) -> list:
    items = []
    has_answer = "answer" in df.columns

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Токенизация"):
        task = str(row["task"])

        if task == "narrative":
            src_ids  = tokenizer.encode(str(row["source_text"]), add_special_tokens=False)
            pred_ids = tokenizer.encode(str(row["prediction"]),  add_special_tokens=False)
            n = min(max_cutoff, len(src_ids), len(pred_ids))
            items.append({
                "task":        "narrative",
                "id":          str(row.get("id", "")),
                "src_preview": str(row["source_text"])[:800],
                "pred_tokens": [tokenizer.decode([t]) for t in pred_ids[:max_cutoff]],
                "matches":     [bool(src_ids[i] == pred_ids[i]) for i in range(n)],
                "src_total":   len(src_ids),
                "pred_total":  len(pred_ids),
            })
        else:  # qa
            ans  = str(row["answer"]).strip() if has_answer and pd.notna(row.get("answer")) else ""
            pred = str(row.get("prediction", "")).strip()
            items.append({
                "task":        "qa",
                "id":          str(row.get("id", "")),
                "source_text": str(row["source_text"]),
                "question":    str(row.get("question", "")),
                "answer":      ans,
                "prediction":  pred,
                "exact":       int(ans.lower() == pred.lower()) if ans else -1,
                "f1":          f1_score(pred, ans) if ans else -1,
            })
    return items


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)
    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42).reset_index(drop=True)

    print(f"Примеров: {len(df)} | Задачи: {df['task'].value_counts().to_dict()}")
    print(f"Загружаю токенайзер: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    items = process(df, tok, args.max_cutoff)

    data_json = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    html = build_html(data_json, args.cutoff, args.max_cutoff)

    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"Сохранено → {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def build_html(data_json: str, init_cutoff: int, max_cutoff: int) -> str:
    return (
        HTML_HEAD
        + f"\n<script>const DATA={data_json};"
        + f"const MAX_CUTOFF={max_cutoff};"
        + f"const PAGE_SIZE={PAGE_SIZE};"
        + f"let gCutoff={init_cutoff};</script>\n"
        + HTML_BODY
    )


# ─────────────────────────────────────────────────────────────────────────────
HTML_HEAD = """<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Просмотр предсказаний модели</title>
<style>
:root {
  --bg:        #f0f2f5;
  --card:      #ffffff;
  --border:    #e2e8f0;
  --text:      #1a202c;
  --muted:     #718096;
  --nar:       #4C72B0;
  --nar-light: #90afd4;
  --qa:        #DD8452;
  --qa-light:  #edaa7e;
  --match-bg:  #c6f6d5;
  --match-fg:  #22543d;
  --miss-bg:   #fed7d7;
  --miss-fg:   #742a2a;
  --beyond-fg: #a0aec0;
  --hdr:       #1e2a3a;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 14px; }

/* ── Header ── */
.hdr {
  position: sticky; top: 0; z-index: 100;
  background: var(--hdr); color: #fff;
  padding: 12px 20px; box-shadow: 0 2px 8px rgba(0,0,0,.35);
}
.hdr-top { display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
.hdr h1 { font-size:17px; font-weight:700; letter-spacing:.3px; flex:1; min-width:200px; }
.controls { display:flex; align-items:center; gap:16px; flex-wrap:wrap; }

/* Tabs */
.tabs { display:flex; gap:4px; }
.tab {
  padding:5px 14px; border:1px solid rgba(255,255,255,.3);
  border-radius:20px; background:transparent; color:#cbd5e0;
  cursor:pointer; font-size:13px; transition:.15s;
}
.tab:hover { background:rgba(255,255,255,.1); color:#fff; }
.tab.active { background:#fff; color:var(--hdr); font-weight:600; border-color:#fff; }

/* Slider */
.slider-wrap { display:flex; align-items:center; gap:8px; font-size:13px; }
.slider-wrap label { white-space:nowrap; color:#a0aec0; }
.slider-wrap strong { color:#fff; min-width:28px; display:inline-block; }
input[type=range] { width:120px; accent-color:#63b3ed; cursor:pointer; }

/* Search */
.search-wrap input {
  padding:5px 10px; border-radius:6px; border:none;
  background:rgba(255,255,255,.12); color:#fff; font-size:13px;
  outline:none; width:200px;
}
.search-wrap input::placeholder { color:#a0aec0; }
.search-wrap input:focus { background:rgba(255,255,255,.2); }

/* Stats bar */
.stats-bar { font-size:12px; color:#718096; padding:5px 20px; background:#fff; border-bottom:1px solid var(--border); }

/* Metrics bar */
.metrics-bar {
  border-top:1px solid rgba(255,255,255,.1); padding:8px 20px;
  display:flex; gap:20px; flex-wrap:wrap; align-items:center;
}
.mg { display:flex; align-items:center; gap:12px; }
.mg-label { font-size:10px; font-weight:800; text-transform:uppercase; letter-spacing:.8px; }
.mi { display:flex; flex-direction:column; gap:2px; }
.mi-label { font-size:10px; color:#a0aec0; }
.mi-row { display:flex; align-items:center; gap:7px; }
.mi-val { font-size:16px; font-weight:800; line-height:1; }
.mbar { width:72px; height:5px; background:rgba(255,255,255,.15); border-radius:3px; overflow:hidden; }
.mbar-fill { height:100%; border-radius:3px; transition:width .35s ease; }
.msep { width:1px; height:28px; background:rgba(255,255,255,.15); margin:0 4px; }

/* ── Content ── */
.content { max-width:1100px; margin:0 auto; padding:20px; }

/* ── Card ── */
.card {
  background:var(--card); border-radius:10px;
  border:1px solid var(--border); margin-bottom:16px;
  box-shadow:0 1px 4px rgba(0,0,0,.06);
  overflow:hidden;
}
.card-hdr {
  display:flex; align-items:center; gap:10px; flex-wrap:wrap;
  padding:10px 16px; border-bottom:1px solid var(--border);
  background:#f7fafc;
}
.badge {
  font-size:11px; font-weight:700; padding:3px 10px; border-radius:12px;
  text-transform:uppercase; letter-spacing:.5px;
}
.badge-nar  { background:var(--nar); color:#fff; }
.badge-qa   { background:var(--qa);  color:#fff; }
.card-id    { font-size:12px; color:var(--muted); font-family:monospace; }
.card-hdr-right { margin-left:auto; display:flex; align-items:center; gap:8px; }

/* Match badge (narrative) */
.mbadge {
  font-size:12px; font-weight:600; padding:3px 10px; border-radius:20px;
}
.mbadge-good { background:#c6f6d5; color:#22543d; }
.mbadge-ok   { background:#fefcbf; color:#744210; }
.mbadge-bad  { background:#fed7d7; color:#742a2a; }

/* QA score badge */
.qbadge { font-size:12px; font-weight:600; padding:3px 10px; border-radius:20px; }
.qbadge-exact { background:#c6f6d5; color:#22543d; }
.qbadge-good  { background:#bee3f8; color:#2a4365; }
.qbadge-ok    { background:#fefcbf; color:#744210; }
.qbadge-bad   { background:#fed7d7; color:#742a2a; }

/* ── Section ── */
.section { padding:12px 16px; border-bottom:1px solid var(--border); }
.section:last-child { border-bottom:none; }
.sec-label {
  font-size:11px; font-weight:700; text-transform:uppercase;
  letter-spacing:.6px; color:var(--muted); margin-bottom:6px;
  display:flex; align-items:center; gap:8px;
}
.len-pill {
  font-size:10px; font-weight:500; padding:1px 7px; border-radius:10px;
  background:var(--bg); color:var(--muted); font-family:monospace; text-transform:none; letter-spacing:0;
}

/* ── Token text ── */
.tok-text {
  font-family: "SFMono-Regular", Consolas, monospace; font-size:13px;
  line-height:1.9; word-break:break-word;
}
.tok-text span { border-radius:3px; padding:1px 0; }
.tok-m  { background:var(--match-bg); color:var(--match-fg); }
.tok-x  { background:var(--miss-bg);  color:var(--miss-fg);  }
.tok-b  { color:var(--beyond-fg); }

/* Legend */
.legend {
  display:flex; gap:12px; flex-wrap:wrap;
  font-size:11px; color:var(--muted);
  padding:6px 16px 10px; align-items:center;
}
.legend-item { display:flex; align-items:center; gap:4px; }
.legend-swatch { width:10px; height:10px; border-radius:2px; display:inline-block; }

/* Cutoff marker */
.cutoff-marker {
  display:inline-block; border-left:2px dashed #718096;
  margin:0 2px; height:1em; vertical-align:middle; opacity:.6;
}

/* ── Source preview (narrative) ── */
.src-preview {
  font-size:13px; line-height:1.7; color:#4a5568;
  max-height:80px; overflow:hidden; transition:max-height .3s ease;
  cursor:pointer;
}
.src-preview.expanded { max-height:500px; }
.expand-btn {
  font-size:11px; color:var(--nar); cursor:pointer;
  text-decoration:underline; margin-top:4px; display:inline-block;
}

/* ── QA layout ── */
.qa-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
@media(max-width:640px) { .qa-grid { grid-template-columns:1fr; } }
.qa-box {
  border:1px solid var(--border); border-radius:8px; padding:10px 12px;
  font-size:13px; line-height:1.6;
}
.qa-box.exact-match { border-color:#68d391; background:#f0fff4; }
.qa-box.good-match  { border-color:#90cdf4; background:#ebf8ff; }
.qa-box.bad-match   { border-color:#fc8181; background:#fff5f5; }
.qa-box-label { font-size:10px; font-weight:700; text-transform:uppercase; letter-spacing:.5px; color:var(--muted); margin-bottom:4px; }

.question-box {
  background:#fffbeb; border:1px solid #f6e05e; border-radius:8px;
  padding:10px 14px; font-size:14px; font-weight:500; color:#744210;
  line-height:1.5;
}
mark.ans-hl {
  background:#fefcbf; color:#744210; border-radius:2px;
  padding:1px 2px; font-weight:600;
}

/* ── Pagination ── */
.pagination {
  display:flex; justify-content:center; align-items:center;
  gap:8px; padding:24px 0 40px;
}
.pg-btn {
  padding:7px 18px; border-radius:8px; border:1px solid var(--border);
  background:var(--card); cursor:pointer; font-size:13px; color:var(--text);
  transition:.15s;
}
.pg-btn:hover:not(:disabled) { background:var(--nar); color:#fff; border-color:var(--nar); }
.pg-btn:disabled { opacity:.4; cursor:not-allowed; }
.pg-info { font-size:13px; color:var(--muted); padding:0 8px; }

/* Empty */
.empty { text-align:center; padding:60px; color:var(--muted); font-size:15px; }
</style>
</head>"""

HTML_BODY = """
<body>
<div class="hdr">
  <div class="hdr-top">
    <h1>Просмотр предсказаний модели</h1>
    <div class="controls">
      <div class="tabs" id="tabs">
        <button class="tab active" data-f="all">Все</button>
        <button class="tab" data-f="narrative">Narrative</button>
        <button class="tab" data-f="qa">QA</button>
      </div>
      <div class="slider-wrap" id="slider-wrap">
        <label>Cutoff:</label>
        <input type="range" id="slider" min="8" step="8">
        <strong id="cutoff-lbl"></strong> <span style="color:#a0aec0;font-size:12px">токенов</span>
      </div>
      <div class="search-wrap">
        <input type="text" id="search" placeholder="🔍 Поиск...">
      </div>
    </div>
  </div>
  <div class="metrics-bar" id="metrics-bar"></div>
</div>
<div class="stats-bar" id="stats-bar"></div>
<div class="content">
  <div id="cards"></div>
  <div class="pagination" id="pg"></div>
</div>

<script>
// ── helpers ────────────────────────────────────────────────────────────────
function esc(s){ return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;"); }

function highlightAnswer(src, ans){
  if(!ans) return esc(src);
  const lo = src.toLowerCase(), al = ans.toLowerCase();
  const i = lo.indexOf(al);
  if(i===-1) return esc(src);
  return esc(src.slice(0,i))+'<mark class="ans-hl">'+esc(src.slice(i,i+ans.length))+'</mark>'+esc(src.slice(i+ans.length));
}

// ── state ──────────────────────────────────────────────────────────────────
let gFilter = "all", gPage = 0, gQuery = "";
let gFiltered = DATA;

function applyFilter(){
  const q = gQuery.toLowerCase();
  gFiltered = DATA.filter(d => {
    if(gFilter !== "all" && d.task !== gFilter) return false;
    if(!q) return true;
    const haystack = d.task==="narrative"
      ? (d.src_preview||"")
      : (d.source_text||"")+" "+(d.question||"")+" "+(d.answer||"")+" "+(d.prediction||"");
    return haystack.toLowerCase().includes(q);
  });
  gPage = 0;
  render();
}

// ── Metrics ────────────────────────────────────────────────────────────────
function computeMetrics(){
  const narItems = gFiltered.filter(d=>d.task==="narrative");
  const qaItems  = gFiltered.filter(d=>d.task==="qa");
  let html = "";

  if(narItems.length){
    let totMatched=0, totCmp=0;
    for(const d of narItems){
      const n = Math.min(gCutoff, d.matches.length);
      totMatched += d.matches.slice(0,n).filter(Boolean).length;
      totCmp += n;
    }
    const avg = totCmp ? totMatched/totCmp*100 : 0;
    const col = avg>=70?"#68d391":avg>=40?"#f6e05e":"#fc8181";
    html += `<div class="mg">
      <span class="mg-label" style="color:var(--nar-light)">Narrative</span>
      <div class="mi">
        <span class="mi-label">Avg Match @ ${gCutoff} tok</span>
        <div class="mi-row">
          <div class="mbar"><div class="mbar-fill" style="width:${avg.toFixed(0)}%;background:${col}"></div></div>
          <span class="mi-val" style="color:${col}">${avg.toFixed(1)}%</span>
        </div>
      </div>
      <span style="font-size:11px;color:#718096">${narItems.length} прим.</span>
    </div>`;
  }

  if(narItems.length && qaItems.length) html += '<div class="msep"></div>';

  if(qaItems.length){
    const valid = qaItems.filter(d=>d.exact!==-1);
    const em = valid.length ? valid.filter(d=>d.exact).length/valid.length*100 : -1;
    const f1 = valid.length ? valid.reduce((s,d)=>s+d.f1,0)/valid.length*100 : -1;
    const emCol = em>=50?"#68d391":em>=25?"#f6e05e":"#fc8181";
    const f1Col = f1>=60?"#68d391":f1>=30?"#f6e05e":"#fc8181";
    const emStr = em>=0?em.toFixed(1)+"%" :"—";
    const f1Str = f1>=0?f1.toFixed(1)+"%" :"—";
    const emW   = em>=0?em.toFixed(0):0;
    const f1W   = f1>=0?f1.toFixed(0):0;
    html += `<div class="mg">
      <span class="mg-label" style="color:var(--qa-light)">QA</span>
      <div class="mi">
        <span class="mi-label">Exact Match</span>
        <div class="mi-row">
          <div class="mbar"><div class="mbar-fill" style="width:${emW}%;background:${emCol}"></div></div>
          <span class="mi-val" style="color:${emCol}">${emStr}</span>
        </div>
      </div>
      <div class="mi">
        <span class="mi-label">F1 Score</span>
        <div class="mi-row">
          <div class="mbar"><div class="mbar-fill" style="width:${f1W}%;background:${f1Col}"></div></div>
          <span class="mi-val" style="color:${f1Col}">${f1Str}</span>
        </div>
      </div>
      <span style="font-size:11px;color:#718096">${qaItems.length} прим.</span>
    </div>`;
  }

  document.getElementById("metrics-bar").innerHTML = html||'<span style="color:#718096;font-size:12px">Нет данных</span>';
}

// ── render ─────────────────────────────────────────────────────────────────
function render(){
  const total  = gFiltered.length;
  const pages  = Math.max(1, Math.ceil(total / PAGE_SIZE));
  if(gPage >= pages) gPage = pages - 1;
  const start  = gPage * PAGE_SIZE;
  const slice  = gFiltered.slice(start, start + PAGE_SIZE);

  // metrics bar (cutoff-sensitive)
  computeMetrics();

  // stats bar
  const narN = gFiltered.filter(d=>d.task==="narrative").length;
  const qaN  = gFiltered.filter(d=>d.task==="qa").length;
  document.getElementById("stats-bar").textContent =
    `Показано: ${total} примеров  |  Narrative: ${narN}  |  QA: ${qaN}  |  Страница ${gPage+1} из ${pages}`;

  // slider visibility
  const hasNar = gFilter !== "qa";
  document.getElementById("slider-wrap").style.display = hasNar ? "flex" : "none";

  // cards
  const html = slice.length ? slice.map(renderCard).join("") : '<div class="empty">Ничего не найдено</div>';
  document.getElementById("cards").innerHTML = html;

  // expand buttons
  document.querySelectorAll(".expand-btn").forEach(btn=>{
    btn.addEventListener("click", ()=>{
      const prev = btn.previousElementSibling;
      const expanded = prev.classList.toggle("expanded");
      btn.textContent = expanded ? "свернуть ▲" : "развернуть ▼";
    });
  });

  // pagination
  renderPagination(pages);
}

function renderCard(d){
  return d.task === "narrative" ? renderNarrative(d) : renderQA(d);
}

// ── Narrative ──────────────────────────────────────────────────────────────
function renderNarrative(d){
  const cutoff   = gCutoff;
  const cmpLen   = Math.min(cutoff, d.matches.length);
  const matched  = d.matches.slice(0, cmpLen).filter(Boolean).length;
  const pct      = cmpLen ? (matched/cmpLen*100).toFixed(1) : "—";
  const mbCls    = pct>=80?"mbadge-good":pct>=50?"mbadge-ok":"mbadge-bad";

  // Build token spans for prediction
  let tokHtml = "";
  for(let i=0; i<d.pred_tokens.length; i++){
    const t = esc(d.pred_tokens[i]);
    if(i === cutoff){
      tokHtml += '<span class="cutoff-marker" title="граница cutoff"></span>';
    }
    if(i >= cutoff){
      tokHtml += `<span class="tok-b">${t}</span>`;
    } else if(i < d.matches.length){
      tokHtml += `<span class="${d.matches[i]?'tok-m':'tok-x'}">${t}</span>`;
    } else {
      tokHtml += `<span class="tok-x">${t}</span>`;
    }
  }
  if(d.pred_total > MAX_CUTOFF){
    tokHtml += `<span class="tok-b"> …(ещё ${d.pred_total-MAX_CUTOFF} токенов)</span>`;
  }

  return `
<div class="card">
  <div class="card-hdr">
    <span class="badge badge-nar">Narrative</span>
    <span class="card-id">${esc(d.id)}</span>
    <div class="card-hdr-right">
      <span class="mbadge ${mbCls}">${matched}/${cmpLen} совпало (${pct}%)</span>
      <span class="len-pill">src ${d.src_total} tok</span>
      <span class="len-pill">pred ${d.pred_total} tok</span>
    </div>
  </div>
  <div class="section">
    <div class="sec-label">Исходный текст (цель)</div>
    <div class="src-preview">${esc(d.src_preview)}${d.src_preview.length>=800?"…":""}</div>
    <span class="expand-btn">развернуть ▼</span>
  </div>
  <div class="section">
    <div class="sec-label">Предсказание модели</div>
    <div class="tok-text">${tokHtml}</div>
    <div class="legend">
      <span class="legend-item"><span class="legend-swatch" style="background:var(--match-bg)"></span>совпадает</span>
      <span class="legend-item"><span class="legend-swatch" style="background:var(--miss-bg)"></span>не совпадает</span>
      <span class="legend-item"><span class="legend-swatch" style="background:#e2e8f0"></span>за пределами cutoff</span>
    </div>
  </div>
</div>`;
}

// ── QA ─────────────────────────────────────────────────────────────────────
function renderQA(d){
  const hasGT = d.exact !== -1;
  let scoreBadge = "";
  let ansCls = "";
  if(hasGT){
    const f1pct = (d.f1*100).toFixed(0);
    if(d.exact){
      scoreBadge = `<span class="qbadge qbadge-exact">Точное совпадение</span>`;
      ansCls = "exact-match";
    } else if(d.f1 >= 0.7){
      scoreBadge = `<span class="qbadge qbadge-good">F1 = ${f1pct}%</span>`;
      ansCls = "good-match";
    } else if(d.f1 >= 0.3){
      scoreBadge = `<span class="qbadge qbadge-ok">F1 = ${f1pct}%</span>`;
      ansCls = "bad-match";
    } else {
      scoreBadge = `<span class="qbadge qbadge-bad">F1 = ${f1pct}%</span>`;
      ansCls = "bad-match";
    }
  }

  const ctxHtml = highlightAnswer(d.source_text, d.answer);

  const ansRow = hasGT ? `
  <div class="section">
    <div class="qa-grid">
      <div>
        <div class="qa-box-label">Правильный ответ</div>
        <div class="qa-box">${esc(d.answer||"—")}</div>
      </div>
      <div>
        <div class="qa-box-label">Предсказание модели</div>
        <div class="qa-box ${ansCls}">${esc(d.prediction||"—")}</div>
      </div>
    </div>
  </div>` : `
  <div class="section">
    <div class="sec-label">Предсказание модели</div>
    <div class="qa-box">${esc(d.prediction||"—")}</div>
  </div>`;

  return `
<div class="card">
  <div class="card-hdr">
    <span class="badge badge-qa">QA</span>
    <span class="card-id">${esc(d.id)}</span>
    <div class="card-hdr-right">${scoreBadge}</div>
  </div>
  <div class="section">
    <div class="sec-label">Контекст <span class="len-pill">source_text</span></div>
    <div class="src-preview">${ctxHtml}</div>
    <span class="expand-btn">развернуть ▼</span>
  </div>
  <div class="section">
    <div class="sec-label">Вопрос</div>
    <div class="question-box">${esc(d.question)}</div>
  </div>
  ${ansRow}
</div>`;
}

// ── Pagination ─────────────────────────────────────────────────────────────
function renderPagination(pages){
  let html = "";
  html += `<button class="pg-btn" id="pg-prev" ${gPage===0?"disabled":""}>← Назад</button>`;
  html += `<span class="pg-info">Страница ${gPage+1} / ${pages}</span>`;
  html += `<button class="pg-btn" id="pg-next" ${gPage>=pages-1?"disabled":""}>Вперёд →</button>`;
  document.getElementById("pg").innerHTML = html;
  document.getElementById("pg-prev")?.addEventListener("click",()=>{ gPage--; render(); window.scrollTo(0,0); });
  document.getElementById("pg-next")?.addEventListener("click",()=>{ gPage++; render(); window.scrollTo(0,0); });
}

// ── Events ─────────────────────────────────────────────────────────────────
const slider = document.getElementById("slider");
const lbl    = document.getElementById("cutoff-lbl");
slider.max   = MAX_CUTOFF;
slider.value = gCutoff;
lbl.textContent = gCutoff;

slider.addEventListener("input", ()=>{
  gCutoff = +slider.value;
  lbl.textContent = gCutoff;
  render();
});

document.querySelectorAll(".tab").forEach(tab=>{
  tab.addEventListener("click",()=>{
    document.querySelectorAll(".tab").forEach(t=>t.classList.remove("active"));
    tab.classList.add("active");
    gFilter = tab.dataset.f;
    applyFilter();
  });
});

let searchTimer;
document.getElementById("search").addEventListener("input", e=>{
  clearTimeout(searchTimer);
  searchTimer = setTimeout(()=>{ gQuery = e.target.value; applyFilter(); }, 300);
});

// ── Init ───────────────────────────────────────────────────────────────────
applyFilter();
</script>
</body>
</html>"""


if __name__ == "__main__":
    main()
