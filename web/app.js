/* 美台股看盤（每日快照版）— iOS 介面
   四大導覽區域：自選股 / 預測 / 檢討 / 設定（網址 #/watch、#/predict、#/review、#/settings）
   這台裝置的「自選股清單」與「預測模型」存在本機 localStorage，不同裝置可以各自不同。 */
"use strict";

const LS = {
  get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* 私密模式等情況忽略 */ } },
};
const MODEL = { champion: "最佳模型", rule: "規則模型", calib: "校正模型", ml: "機器學習" };
const TABS = { watch: "自選股", predict: "預測", review: "檢討", settings: "設定" };
const ICON = {
  watch: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><circle cx="4.5" cy="6" r="1.2" fill="currentColor"/><circle cx="4.5" cy="12" r="1.2" fill="currentColor"/><circle cx="4.5" cy="18" r="1.2" fill="currentColor"/><path d="M9 6h11M9 12h11M9 18h11"/></svg>',
  predict: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 19h18"/><path d="M4 15l5-5 4 3 7-7"/><path d="M15 6h5v5"/></svg>',
  review: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="4" width="14" height="17" rx="2.5"/><path d="M9 4.5h6v2.5H9z"/><path d="M8.5 13.5l2.3 2.3 4.7-4.8"/></svg>',
  settings: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><circle cx="12" cy="12" r="3.2"/><path d="M12 2.8v2.6M12 18.6v2.6M2.8 12h2.6M18.6 12h2.6M5.5 5.5l1.8 1.8M16.7 16.7l1.8 1.8M5.5 18.5l1.8-1.8M16.7 7.3l1.8-1.8"/></svg>',
};

let SUM = null;
const CACHE = {};
const UI = { market: "全部", view: "走勢", range: "6月", rvModel: "最佳", rvSrc: "回溯驗證" };
let wl = LS.get("wl", "");
let model = LS.get("model", "champion");

// ---------------------------------------------------------------- 小工具
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 2) => (v === null || v === undefined || isNaN(v)) ? "—" : Number(v).toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const sgn = (v, d = 2) => (v === null || v === undefined || isNaN(v)) ? "—" : (v > 0 ? "+" : "") + num(v, d);
const pct = (v) => (v === null || v === undefined || isNaN(v)) ? "—" : Math.round(v * 100) + "%";
const col = (v) => (!v || isNaN(v)) ? "var(--label2)" : v > 0 ? "var(--red)" : "var(--green)";
// 剛用「＋」加入／移除、雲端還沒更新完的股票，先記在這台裝置（pending）
function pendingAll() { try { return JSON.parse(LS.get("pending", "{}")) || {}; } catch (e) { return {}; } }
function savePending(p) { LS.set("pending", JSON.stringify(p)); }
function pendingOf(list) { const p = pendingAll()[list] || {}; return { add: p.add || [], remove: p.remove || [], t: p.t || 0 }; }
function markPending(list, add = [], remove = []) {
  const all = pendingAll();
  const p = pendingOf(list);
  p.add = [...new Set([...p.add.filter((c) => !remove.includes(c)), ...add])];
  p.remove = [...new Set([...p.remove.filter((c) => !add.includes(c)), ...remove])];
  p.t = Date.now();
  all[list] = p; savePending(all);
}
function cleanPending() {             // 雲端更新後已反映的就清掉；超過 1 天的也清掉
  const all = pendingAll();
  for (const [list, p] of Object.entries(all)) {
    const base = SUM.lists[list] || [];
    p.add = (p.add || []).filter((c) => !(base.includes(c) && SUM.stocks[c]));
    p.remove = (p.remove || []).filter((c) => base.includes(c));
    if ((!p.add.length && !p.remove.length) || Date.now() - (p.t || 0) > 864e5) delete all[list];
  }
  savePending(all);
}
const codes = () => {
  const p = pendingOf(wl);
  const cs = (SUM.lists[wl] || []).filter((c) => !p.remove.includes(c));
  p.add.forEach((c) => { if (!cs.includes(c)) cs.push(c); });
  return cs;
};
const isPending = (c) => pendingOf(wl).add.includes(c) && !(SUM.lists[wl] || []).includes(c);
const nameOf = (c) => (SUM.stocks[c] || {}).name || c;
const isTW = (c) => /^\d{4,6}[A-Z]?$/.test(c);

function nav(title, sub, back, large = true, right = "") {
  const b = back ? `<a class="back" href="${back[1]}"><b>‹</b>${esc(back[0])}</a>` : "";
  const s = sub ? `<div class="sub">${esc(sub)}</div>` : "";
  const h = large ? "h1" : "h2";
  return `<div class="nav">${right}${b}${s}<${h}>${esc(title)}</${h}></div>`;
}
const PLUS = '<button class="act" data-act="add" aria-label="新增股票"><svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg></button>';
const section = (header, cells, footer) =>
  (header ? `<div class="sec-h">${esc(header)}</div>` : "") + `<div class="grp">${cells}</div>` + (footer ? `<div class="sec-f">${footer}</div>` : "");
function cell(k, v = "", { sub, link, vcolor, chev, attrs } = {}) {
  const s = sub ? `<small>${sub}</small>` : "";
  const vc = vcolor ? ` style="color:${vcolor}"` : "";
  const ch = (chev ?? !!link) ? '<span class="chev">›</span>' : "";
  if (link) return `<a class="cell" href="${link}"><div class="k">${k}${s}</div><div class="v"${vc}>${v}</div>${ch}</a>`;
  if (attrs) return `<button class="cell" ${attrs}><div class="k">${k}${s}</div><div class="v"${vc}>${v}</div>${ch}</button>`;
  return `<div class="cell"><div class="k">${k}${s}</div><div class="v"${vc}>${v}</div>${ch}</div>`;
}
const seg = (name, opts, val) =>
  `<div class="seg" data-seg="${name}">${opts.map((o) => `<button class="${o === val ? "on" : ""}" data-v="${esc(o)}">${esc(o)}</button>`).join("")}</div>`;
function pill(p) {
  if (p === null || p === undefined || isNaN(p)) return '<span class="pill fl">—</span>';
  return `<span class="pill ${p > 0 ? "up" : p < 0 ? "dn" : "fl"}">${sgn(p)}%</span>`;
}
function spark(arr) {
  if (!arr || arr.length < 2) return "";
  const lo = Math.min(...arr), hi = Math.max(...arr), r = (hi - lo) || 1;
  const pts = arr.map((y, i) => `${(2 + i * 60 / (arr.length - 1)).toFixed(1)},${(26 - (y - lo) / r * 22).toFixed(1)}`).join(" ");
  const c = arr[arr.length - 1] >= arr[0] ? "var(--red)" : "var(--green)";
  return `<svg class="spark" width="64" height="28" viewBox="0 0 64 28"><polyline points="${pts}" fill="none" stroke="${c}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/></svg>`;
}
function trendTag(f, h) {
  if (!f || f["p" + h] == null) return "";
  const d = f["dir" + h] || 0;
  const [cls, a, w] = d > 0 ? ["up", "▲", "偏多"] : d < 0 ? ["dn", "▼", "偏空"] : ["", "●", "盤整"];
  return `<span class="tag ${cls}">${h}日 ${a}${w} ${sgn((f["p" + h] / f.close - 1) * 100, 1)}%</span>`;
}
const fmtVol = (v, tw) => tw ? num(v / 1000, 0) + " 張" : v >= 1e9 ? num(v / 1e9) + "B" : v >= 1e6 ? num(v / 1e6) + "M" : num(v, 0);

async function stock(c) {
  if (!CACHE[c]) {
    const r = await fetch(`data/${encodeURIComponent(c)}.json?v=${encodeURIComponent(SUM.updated)}`);
    if (!r.ok) throw new Error("no data");
    CACHE[c] = await r.json();
  }
  return CACHE[c];
}
function plot(id, fig) {
  const el = document.getElementById(id);
  if (!el || !fig || !window.Plotly) return;
  Plotly.newPlot(el, fig.data, fig.layout, { displayModeBar: false, responsive: true, scrollZoom: false });
}

// ---------------------------------------------------------------- 自選股
function pageWatch() {
  const cs = codes();
  let h = nav("自選股", `${wl}・${cs.length} 檔・更新於 ${SUM.updated}`, null, true, PLUS);
  if (!cs.length) return h + `<div class="empty">這組清單是空的。<br>點右上角「＋」加入股票。</div>`;
  h += seg("market", ["全部", "台股", "美股"], UI.market);
  const show = cs.filter((c) => UI.market === "全部" || (UI.market === "台股") === isTW(c));
  const rows = show.map((c) => {
    const s = SUM.stocks[c] || {};
    if (isPending(c) || !SUM.stocks[c]) return cell(`<span class="sym">${esc(c)}</span>`, "⏳ 更新中",
      { sub: "已加入，雲端下載資料與訓練模型約需 10～15 分鐘", chev: false });
    if (s.nodata) return cell(`<span class="sym">${esc(c)}</span>`, "尚無資料", { sub: esc(s.name || "") + "（代號可能有誤）", chev: false });
    const f = (s.forecasts || {})[model];
    const tags = f ? trendTag(f, 5) + trendTag(f, 20) : '<span class="tag">尚無預測</span>';
    return `<a class="cell" href="#/watch/${encodeURIComponent(c)}" style="min-height:64px"><div class="k"><div class="sym">${esc(c)}</div>
      <div class="nm">${esc(s.name)}</div><div style="margin-top:3px">${tags}</div></div>${spark(s.spark)}
      <div><div class="px">${num(s.close)}</div>${pill(s.pct)}</div></a>`;
  });
  return h + section(null, rows.join(""), `點股票看明細・趨勢標籤為「${MODEL[model]}」的 5／20 日預測・紅漲綠跌`);
}

async function pageStock(c) {
  const d = await stock(c);
  let h = nav(d.name, `${c}・${d.tw ? "台股" : "美股"}`, ["自選股", "#/watch"], false);
  h += `<div class="hero"><span class="p">${num(d.close)}</span><span class="c" style="color:${col(d.chg)}">${sgn(d.chg)}（${sgn(d.pct)}%）</span>
        <div class="note" style="padding:0">資料日期 ${esc(d.date)}</div></div>`;
  h += seg("view", ["走勢", "籌碼", "高低點"], UI.view);
  const after = [];
  if (UI.view === "走勢") {
    h += seg("range", ["3月", "6月", "1年"], UI.range);
    h += `<div class="chart" id="ch1"></div>`;
    after.push(() => plot("ch1", d.figs[{ "3月": "k3", "6月": "k6", "1年": "k12" }[UI.range]]));
    const st = d.stats;
    h += section("今日數據", [
      cell("成交量", fmtVol(st.vol, d.tw), { sub: `量比 ${num(st.vr)}（今日 ÷ 前 5 日均量）` }),
      cell("MA5 乖離", sgn(st.bias5) + "%", { vcolor: col(st.bias5) }),
      cell("MA20 乖離", sgn(st.bias20) + "%", { vcolor: col(st.bias20) }),
      cell("MA60 乖離", sgn(st.bias60) + "%", { vcolor: col(st.bias60) }),
    ].join(""));
  } else if (UI.view === "籌碼") {
    const ch = d.chips || {};
    if (!d.tw) h += `<div class="empty">三大法人與融資融券為台股資料，美股沒有。</div>`;
    else {
      if (ch.inst) {
        h += section(`三大法人（${ch.inst_date}）`, ch.inst.map((x) => cell(x.g, sgn(x.v, 0) + " 張", {
          vcolor: col(x.v), sub: (x.streak ? `連${x.streak > 0 ? "買" : "賣"} ${Math.abs(x.streak)} 日・` : "") + `5 日 ${sgn(x.s5, 0)}`,
        })).join(""));
        h += `<div class="chart" id="ch2"></div>`;
        after.push(() => plot("ch2", d.figs.inst));
      }
      if (ch.margin) {
        const m = ch.margin;
        h += section("融資融券", [
          cell("融資餘額", num(m["融資餘額"], 0) + " 張", { sub: `增減 ${sgn(m["融資增減"], 0)}` }),
          cell("融券餘額", num(m["融券餘額"], 0) + " 張", { sub: `增減 ${sgn(m["融券增減"], 0)}` }),
          cell("券資比", m["券資比%"] == null ? "—" : num(m["券資比%"]) + "%"),
          cell("融資使用率", m["融資使用率%"] == null ? "—" : num(m["融資使用率%"]) + "%"),
        ].join(""));
      }
      if (!ch.inst && !ch.margin) h += `<div class="empty">尚無籌碼資料（ETF 或剛上市個股可能沒有）。</div>`;
    }
  } else {
    const a = d.hl.all, y = d.hl["52w"];
    h += section("高低點", [
      cell("歷史最高", num(a.high), { sub: `${a.high_date}・距今 ${sgn(a.from_high, 1)}%` }),
      cell("歷史最低", num(a.low), { sub: `${a.low_date}・距今 ${sgn(a.from_low, 1)}%` }),
      cell("52 週最高", num(y.high), { sub: `${y.high_date}・距今 ${sgn(y.from_high, 1)}%` }),
      cell("52 週最低", num(y.low), { sub: `${y.low_date}・距今 ${sgn(y.from_low, 1)}%` }),
    ].join(""), `資料起點 ${esc(d.hl.first)}`);
    h += `<div class="chart" id="ch3"></div>`;
    after.push(() => plot("ch3", d.figs.hist));
  }
  h += section(null, cell("查看 5／20 日預測", "", { link: `#/predict/${encodeURIComponent(c)}` }) +
    cell("查看預測檢討", "", { link: `#/review/${encodeURIComponent(c)}` }));
  if (codes().includes(c)) {
    h += section(null, cell(`<span style="color:var(--red)">從「${esc(wl)}」移除</span>`, "", { attrs: `data-remove="${esc(c)}"`, chev: false }));
  }
  return [h, after];
}

// ---------------------------------------------------------------- 預測
function pagePredict() {
  let h = nav("預測", `${MODEL[model]}・未來 5／20 個交易日`);
  const rows = codes().map((c) => {
    const s = SUM.stocks[c] || {};
    const f = (s.forecasts || {})[model];
    if (!f) return cell(`<span class="sym">${esc(c)}</span>`, "尚無預測", { sub: esc(s.name || "") });
    const champ = model === "champion" ? [5, 20].map((h_) => `<span class="tag">${h_}日：${MODEL[(s.champion || {})[h_]] || "—"}</span>`).join("") : "";
    const right = [5, 20].map((h_) => `<div style="font-size:15px;color:${f["dir" + h_] ? col(f["p" + h_] - f.close) : "var(--label2)"}">
        ${h_}日 ${num(f["p" + h_])}（${sgn((f["p" + h_] / f.close - 1) * 100, 1)}%）</div>`).join("");
    return `<a class="cell" href="#/predict/${encodeURIComponent(c)}" style="min-height:64px"><div class="k"><div class="sym">${esc(c)}</div>
      <div class="nm">${esc(s.name)}</div><div>${champ}</div></div><div class="v">${right}</div><span class="chev">›</span></a>`;
  });
  return h + section(null, rows.join(""), "預估價為最後收盤後第 5、第 20 個交易日（已跳過休市日）的收盤價。紅＝看漲、綠＝看跌、灰＝盤整。");
}

async function pagePredictStock(c) {
  const d = await stock(c);
  const f = d.forecasts[model];
  let h = nav(d.name, `${c}・${MODEL[model]}`, ["預測", "#/predict"], false);
  if (!f) return [h + `<div class="empty">尚無預測。</div>`, []];
  const cards = [5, 20].map((h_) => {
    const p = f["p" + h_], dir = f["dir" + h_] || 0;
    const word = dir > 0 ? "▲ 偏多" : dir < 0 ? "▼ 偏空" : "● 盤整";
    const mname = MODEL[f["model" + h_] || model] || "";
    return `<div class="kc"><div class="k">${h_} 日後・${d.dates[h_]}</div><div class="v" style="color:${col(dir)}">${num(p)}</div>
      <div class="s">${word}　${sgn((p / f.close - 1) * 100)}%</div><div class="s">區間 ${num(f["lo" + h_], 1)} ～ ${num(f["hi" + h_], 1)}</div>
      <div style="margin-top:6px"><span class="tag">${esc(mname)}</span></div></div>`;
  });
  h += `<div class="kg">${cards.join("")}</div><div class="chart" id="pt"></div>
        <div class="note">粗虛線＝5 日預估、細點線＝20 日預估，色帶＝可能區間；紅／綠長虛線＝支撐／壓力。</div>`;
  const after = [() => plot("pt", d.figs["trend_" + model] || d.figs.trend_rule)];
  const cells = ["rule", "calib", "ml"].filter((m) => d.forecasts[m]).map((m) => {
    const fm = d.forecasts[m];
    const best = [5, 20].filter((h_) => d.states[h_].champion === m).map((h_) => `<span class="tag best">${h_}日最佳</span>`).join("");
    const hit = [5, 20].map((h_) => pct(((d.states[h_].metrics || {})[m] || {}).ok5)).join("／");
    return cell(`${MODEL[m]} ${best}`, `命中 ${hit}`, { sub: [5, 20].map((h_) => `${h_}日 ${num(fm["p" + h_])}（${sgn((fm["p" + h_] / fm.close - 1) * 100, 1)}%）`).join("　") });
  });
  h += section("三種模型", cells.join(""), "命中＝最近 250 個交易日（樣本外）落差 ≤ 5% 的比例，依序為 5 日／20 日。每天更新後重新比較。");
  const imp = d.states["5"].importance || [];
  if (imp.length) {
    const mx = Math.max(...imp.map((i) => i.w)) || 1;
    h += section("機器學習最看重的條件（5 日）", imp.map((i) => cell(`${esc(i.name)}<div class="bar"><i style="width:${Math.round(i.w / mx * 100)}%"></i></div>`,
      i.sign > 0 ? "越高越看漲" : i.sign < 0 ? "越高越看跌" : "非線性")).join(""),
      `用 ${num(d.states["5"].n_train || 0, 0)} 個交易日訓練；方向取自線性模型係數，僅供解讀參考。`);
  }
  if (d.signals.length) {
    h += section("技術型態訊號（規則模型）", d.signals.map((s) => cell(`${s.dir > 0 ? "▲" : s.dir < 0 ? "▼" : "●"} ${esc(s.name)}`, sgn(s.w, 1),
      { sub: `${esc(s.kind)}・${esc(s.note)}`, vcolor: col(s.dir) })).join(""));
  }
  h += `<div class="note">⚠️ 預測為統計推估，不是保證，也不構成投資建議。</div>`;
  return [h, after];
}

// ---------------------------------------------------------------- 檢討
async function pageReview(c) {
  const cs = codes();
  let h = nav("檢討", "預測成績、失敗原因與個股化模型");
  if (!cs.length) return [h + `<div class="empty">清單是空的。</div>`, []];
  c = cs.includes(c) ? c : cs[0];
  h += `<select class="picker" id="pick">${cs.map((x) => `<option value="${esc(x)}" ${x === c ? "selected" : ""}>${esc(x)}　${esc(nameOf(x))}</option>`).join("")}</select>`;
  h += seg("rvModel", ["最佳", "規則", "校正", "ML"], UI.rvModel) + seg("rvSrc", ["回溯驗證", "實際紀錄"], UI.rvSrc);
  let d;
  try { d = await stock(c); } catch (e) { return [h + `<div class="empty">尚無資料。</div>`, []]; }
  const m = { "最佳": "champion", "規則": "rule", "校正": "calib", "ML": "ml" }[UI.rvModel];
  const rv = (d.review || {})[m] || {};
  const after = [];
  const list = UI.rvSrc === "實際紀錄" ? rv.live : rv.summary;
  if (!list || !list.length) {
    h += `<div class="empty">${UI.rvSrc === "實際紀錄" ? "還沒有實際紀錄。每天自動更新時會記錄當天的預測，5 個交易日後出現第一筆成績。" : "這個模型的樣本不足。"}</div>`;
  } else {
    const cards = list.map((s) => {
      const diff = s["落差≤5%"] - s["對照：股價不動"];
      const cc = diff > 0.03 ? "var(--blue)" : diff < -0.03 ? "var(--orange)" : "var(--label)";
      return `<div class="kc"><div class="k">${s["週期"]}・落差 ≤ 5%</div><div class="v" style="color:${cc}">${pct(s["落差≤5%"])}</div>
        <div class="s">股價不動 ${pct(s["對照：股價不動"])}・${s["已驗證"]} 筆</div></div>
        <div class="kc"><div class="k">${s["週期"]}・區間／方向</div><div class="v">${pct(s["落在預測區間"])}</div>
        <div class="s">方向命中 ${pct(s["方向命中"])}（${s["方向判斷次數"]} 次）</div></div>`;
    });
    h += `<div class="kg">${cards.join("")}</div><div class="note">藍色＝優於「股價不動」，橘色＝不如。落差＝實際收盤 ÷ 預估價 − 1。</div>`;
    if (m === "champion" && UI.rvSrc === "回溯驗證" && d.figs.review) {
      h += `<div class="chart" id="rvf"></div>`;
      after.push(() => plot("rvf", d.figs.review));
    }
  }
  const champ = (d.review || {}).champion || {};
  if (champ.causes && champ.causes.length) {
    const tot = champ.causes.reduce((a, x) => a + (x["5 日"] || 0) + (x["20 日"] || 0), 0) || 1;
    h += section("失敗原因（最佳模型，落差 > 5%）", champ.causes.map((x) => cell(
      `${esc(x.cause)}<div class="bar"><i style="width:${Math.round(((x["5 日"] || 0) + (x["20 日"] || 0)) / tot * 100)}%;background:var(--orange)"></i></div>`,
      ["5 日", "20 日"].filter((p) => p in x).map((p) => `${p} ${x[p]}`).join("｜"))).join(""),
      "同一事件常影響連續數天的預測，筆數以預測日計。");
    if (champ.fails) {
      h += `<details class="more"><summary>最近失敗明細</summary><div class="in">${champ.fails.map((x) =>
        `<div style="margin:8px 0"><b>${x.date}・${x.h}</b>　落差 ${sgn(x.err * 100, 1)}%・${esc(x.cause)}<div class="note" style="padding:0">${esc(x.detail)}</div></div>`).join("")}</div></details>`;
    }
  }
  const cells = [];
  for (const h_ of [5, 20]) {
    const ms = d.states[h_].metrics || {};
    for (const mm of ["rule", "calib", "ml"]) {
      const v = ms[mm] || {};
      if (!v.n) continue;
      const best = d.states[h_].champion === mm ? '<span class="tag best">最佳</span>' : "";
      cells.push(cell(`${h_}日・${MODEL[mm]} ${best}`, pct(v.ok5), { sub: `區間 ${pct(v.band)}・方向 ${pct(v.dir)}・IC ${sgn(v.ic, 3)}` }));
    }
  }
  if (cells.length) h += section("模型比較（近 250 日樣本外）", cells.join(""), "每天自動更新後，三個模型都會重新訓練並重新比較。");
  if (d.review && d.review.report_html) {
    h += `<details class="more"><summary>📝 初報告</summary><div class="in report">${d.review.report_html}</div></details>`;
    h += `<button class="btn" id="dl">下載完整初報告（HTML）</button>`;
  }
  after.push(() => {
    const p = document.getElementById("pick");
    if (p) p.onchange = () => { location.hash = "#/review/" + encodeURIComponent(p.value); };
    const b = document.getElementById("dl");
    if (b) b.onclick = () => {
      const blob = new Blob([d.review.report_full], { type: "text/html" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `${c}_預測檢討_${SUM.updated.slice(0, 10)}.html`;
      document.body.appendChild(a); a.click(); a.remove();
    };
  });
  return [h, after];
}

// ---------------------------------------------------------------- 設定
function pageSettings() {
  let h = nav("設定");
  h += section("這台裝置的自選股清單", Object.keys(SUM.lists).map((n) =>
    cell(esc(n), n === wl ? '<span class="check">✓</span>' : "", { sub: `${SUM.lists[n].length} 檔`, attrs: `data-wl="${esc(n)}"`, chev: false })).join(""),
    "每台手機、平板、電腦各自記住自己的選擇（存在這台裝置，不會互相影響）。");
  h += section("預測模型", Object.keys(MODEL).map((m) =>
    cell(MODEL[m], m === model ? '<span class="check">✓</span>' : "", { attrs: `data-model="${m}"`, chev: false,
      sub: m === "champion" ? "每檔 5 日、20 日各自採用最近 250 日最準的模型" : "" })).join(""));
  const repo = SUM.repo;
  const gh = repo ? `https://github.com/${repo}` : "";
  h += section("資料更新", [
    cell("最後更新", esc(SUM.updated)),
    cell("自動更新時間", "", { sub: "週一～五 16:30（台股收盤、法人公布後）<br>週二～六 06:30（美股收盤後）" }),
    gh ? cell("立即更新", "", { link: `${gh}/actions/workflows/daily.yml`, sub: "開啟 GitHub，按「Run workflow」" }) : "",
  ].join(""), "自動更新由 GitHub 雲端執行，不需要你的電腦開著。");
  h += section("自選股管理", [
    cell("GitHub 連線", GH.token() ? '<span class="check">已設定 ✓</span>' : "未設定",
      { attrs: 'data-act="token"', chev: true, sub: "設定後即可用自選股右上角「＋」新增、在個股頁移除" }),
    gh ? cell("在 GitHub 直接編輯清單", "", { link: `${gh}/edit/main/config.json`, sub: "可建立／重新命名多組清單" }) : "",
  ].join(""), "權杖只存在這台裝置的瀏覽器，每台裝置第一次使用時設定一次。");
  h += section("關於", [cell("版本", "每日快照版・iOS 介面"), cell("介面規範", "依 Apple 人機介面指南"), cell("資料來源", "Yahoo Finance、FinMind")].join(""),
    "⚠️ 所有預測為統計推估，不構成投資建議。");
  return h;
}

// ---------------------------------------------------------------- GitHub 連線（修改 config.json）
const GH = {
  token: () => LS.get("gh_token", ""),
  headers() {
    return { Authorization: `Bearer ${this.token()}`, Accept: "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28" };
  },
  url: () => `https://api.github.com/repos/${SUM.repo}/contents/config.json`,
};
const b64dec = (s) => new TextDecoder().decode(Uint8Array.from(atob(s.replace(/\s/g, "")), (c) => c.charCodeAt(0)));
const b64enc = (s) => { const b = new TextEncoder().encode(s); let o = ""; b.forEach((x) => { o += String.fromCharCode(x); }); return btoa(o); };
function ghError(status) {
  return ({ 401: "權杖無效或已過期，請到「設定 → GitHub 連線」重新設定。",
            403: "權杖沒有寫入權限：建立權杖時 Contents 要選「Read and write」。",
            404: "找不到儲存庫：權杖要授權給這個儲存庫（Only select repositories）。" })[status]
    || `GitHub 回應錯誤（${status}），請稍後再試。`;
}
async function ghGet() {
  const r = await fetch(GH.url() + "?ref=main&t=" + Date.now(), { headers: GH.headers(), cache: "no-store" });
  if (!r.ok) throw new Error(ghError(r.status));
  const j = await r.json();
  const text = b64dec(j.content).replace(/^\uFEFF/, "").replace(/,\s*([}\]])/g, "$1");   // 容忍多餘逗號、BOM
  try { return { sha: j.sha, cfg: JSON.parse(text) }; }
  catch (e) { throw new Error("GitHub 上的 config.json 格式有誤（可能少了引號或括號），請到 GitHub 檢查。"); }
}
async function ghEdit(mutate) {          // 讀取 → 修改 → 寫回；遇到同時修改的衝突自動重試一次
  for (let i = 0; i < 2; i++) {
    const { sha, cfg } = await ghGet();
    cfg.watchlists = cfg.watchlists || {};
    const msg = mutate(cfg);
    for (const k of Object.keys(cfg.watchlists)) cfg.watchlists[k] = [...new Set(cfg.watchlists[k])];   // 去除重複
    const r = await fetch(GH.url(), { method: "PUT", headers: GH.headers(),
      body: JSON.stringify({ message: msg, content: b64enc(JSON.stringify(cfg, null, 2) + "\n"), sha, branch: "main" }) });
    if (r.ok) return;
    if (r.status !== 409 && r.status !== 422) throw new Error(ghError(r.status));
  }
  throw new Error("清單剛好被同時修改，請再試一次。");
}
function parseCodes(t) {
  return [...new Set(t.replace(/，/g, ",").split(/[,\s]+/).map((x) => x.trim().toUpperCase().replace(/\.TWO?$/, "")).filter(Boolean))];
}

function openSheet(inner) {
  closeSheet(true);
  const ov = document.createElement("div");
  ov.className = "ov";
  ov.innerHTML = `<div class="sheet" role="dialog"><div class="grab"></div>${inner}</div>`;
  ov.addEventListener("click", (e) => { if (e.target === ov) closeSheet(); });
  document.body.appendChild(ov);
  requestAnimationFrame(() => ov.classList.add("show"));
  return ov;
}
function closeSheet(now = false) {
  document.querySelectorAll(".ov").forEach((o) => { o.classList.remove("show"); now ? o.remove() : setTimeout(() => o.remove(), 260); });
}
function toast(t) {
  const el = document.createElement("div");
  el.className = "toast"; el.textContent = t;
  document.body.appendChild(el);
  setTimeout(() => el.classList.add("show"), 10);
  setTimeout(() => { el.classList.remove("show"); setTimeout(() => el.remove(), 300); }, 3200);
}
const setMsg = (ov, t, cls = "") => { const m = ov.querySelector(".msg"); m.className = "msg " + cls; m.textContent = t; };

function openAdd() {
  if (!SUM.repo) { toast("這個網頁沒有連結 GitHub 儲存庫"); return; }
  if (!GH.token()) { openToken(true); return; }
  const lists = Object.keys(SUM.lists);
  const ov = openSheet(`<div class="sheet-bar"><button data-x>取消</button><b>新增股票</b><button class="done" data-ok>加入</button></div>
    <input class="field" id="addcode" placeholder="台股 2330、美股 NVDA，可用逗號一次多檔" autocomplete="off" autocapitalize="characters" enterkeyhint="done">
    ${lists.length > 1 ? `<div class="sec-h" style="margin-left:4px">加入到</div><select class="picker" id="addlist">${lists.map((n) =>
      `<option ${n === wl ? "selected" : ""}>${esc(n)}</option>`).join("")}</select>` : ""}
    <div class="msg">加入後，雲端會下載資料並訓練這檔的模型，約 10～15 分鐘後出現在清單。</div>`);
  const inp = ov.querySelector("#addcode");
  setTimeout(() => inp.focus(), 300);
  const go = async () => {
    const cs = parseCodes(inp.value);
    if (!cs.length) { setMsg(ov, "請輸入股票代號。", "err"); return; }
    const bad = cs.filter((c) => !/^(\d{4,6}[A-Z]?|[A-Z][A-Z0-9.\-]{0,9})$/.test(c));
    if (bad.length) { setMsg(ov, `代號格式不對：${bad.join("、")}`, "err"); return; }
    const list = (ov.querySelector("#addlist") || {}).value || wl;
    const btn = ov.querySelector("[data-ok]");
    btn.disabled = true; btn.textContent = "儲存中…"; setMsg(ov, "正在寫入 GitHub…");
    try {
      let added = [];
      await ghEdit((cfg) => {
        const L = cfg.watchlists[list] = cfg.watchlists[list] || [];
        added = cs.filter((c) => !L.includes(c));
        L.push(...added);
        return `新增 ${cs.join(", ")} 到「${list}」`;
      });
      markPending(list, added);
      closeSheet();
      const dup = cs.filter((c) => !added.includes(c));
      toast(added.length ? `已加入 ${added.join("、")}，約 10～15 分鐘後出現資料` + (dup.length ? `（${dup.join("、")} 已在清單中）` : "")
                         : "這些股票已經在清單裡了");
      render(true);
    } catch (e) {
      btn.disabled = false; btn.textContent = "加入"; setMsg(ov, e.message || "連線失敗，請檢查網路。", "err");
    }
  };
  ov.querySelector("[data-ok]").onclick = go;
  ov.querySelector("[data-x]").onclick = () => closeSheet();
  inp.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
}

function confirmRemove(c) {
  const ov = openSheet(`<div class="sheet-bar"><span></span><b>移除股票</b><span></span></div>
    <div class="msg" style="text-align:center">要從「${esc(wl)}」移除 ${esc(c)} ${esc(nameOf(c))} 嗎？</div>
    <button class="btn" data-ok style="color:var(--red)">移除</button><button class="btn" data-x>取消</button>`);
  ov.querySelector("[data-x]").onclick = () => closeSheet();
  ov.querySelector("[data-ok]").onclick = async () => {
    if (!GH.token()) { closeSheet(true); openToken(true); return; }
    const btn = ov.querySelector("[data-ok]");
    btn.disabled = true; btn.textContent = "移除中…";
    try {
      await ghEdit((cfg) => {
        cfg.watchlists[wl] = (cfg.watchlists[wl] || []).filter((x) => x !== c);
        return `從「${wl}」移除 ${c}`;
      });
      markPending(wl, [], [c]);
      closeSheet(); toast(`已移除 ${c}`);
      location.hash = "#/watch";
    } catch (e) { btn.disabled = false; btn.textContent = "移除"; setMsg(ov, e.message || "連線失敗", "err"); }
  };
}

function openToken(fromAdd = false) {
  const has = !!GH.token();
  const ov = openSheet(`<div class="sheet-bar"><button data-x>取消</button><b>GitHub 連線</b><button class="done" data-ok>儲存</button></div>
    ${fromAdd ? '<div class="msg">要用「＋」新增股票，這台裝置需要先設定一次 GitHub 權杖。</div>' : ""}
    <div class="grp" style="font-size:15px">
      <div class="cell"><div class="k">1. 到 GitHub 建立權杖<small><a href="https://github.com/settings/personal-access-tokens/new" target="_blank" rel="noopener">開啟建立權杖頁面 ↗</a></small></div></div>
      <div class="cell"><div class="k">2. Repository access 選 <b>Only select repositories</b><small>選擇 ${esc((SUM.repo || "").split("/")[1] || "這個儲存庫")}</small></div></div>
      <div class="cell"><div class="k">3. Permissions → Repository permissions<small><b>Contents</b> 設為 <b>Read and write</b>，其他不用動</small></div></div>
      <div class="cell"><div class="k">4. 按 Generate token，複製後貼在下面</div></div>
    </div>
    <input class="field" id="tok" type="password" placeholder="github_pat_…" autocomplete="off" style="margin-top:10px">
    <div class="msg">權杖只存在這台裝置，而且只能修改這個儲存庫的檔案。</div>
    ${has ? '<button class="btn" data-clear style="color:var(--red)">移除這台裝置的連線</button>' : ""}`);
  ov.querySelector("[data-x]").onclick = () => closeSheet();
  const clr = ov.querySelector("[data-clear]");
  if (clr) clr.onclick = () => { LS.set("gh_token", ""); closeSheet(); toast("已移除連線"); render(true); };
  ov.querySelector("[data-ok]").onclick = async () => {
    const t = ov.querySelector("#tok").value.trim();
    if (!t) { setMsg(ov, "請貼上權杖。", "err"); return; }
    const old = GH.token();
    LS.set("gh_token", t); setMsg(ov, "驗證中…");
    try {
      await ghGet();
      setMsg(ov, "連線成功 ✓", "ok");
      setTimeout(() => { closeSheet(); render(true); if (fromAdd) setTimeout(openAdd, 300); }, 600);
    } catch (e) { LS.set("gh_token", old); setMsg(ov, e.message || "驗證失敗", "err"); }
  };
}

// ---------------------------------------------------------------- 路由
function tabbar(active) {
  document.getElementById("tabbar").innerHTML = `<div class="in">${Object.entries(TABS).map(([k, v]) =>
    `<a class="${k === active ? "on" : ""}" href="#/${k}">${ICON[k]}<span>${v}</span></a>`).join("")}</div>`;
}
let renderSeq = 0;
async function render(keepScroll = false) {
  const seq = ++renderSeq;
  const [tab0, code] = location.hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
  const tab = TABS[tab0] ? tab0 : "watch";
  tabbar(tab);
  const y = window.scrollY;
  let out;
  try {
    if (tab === "watch") out = code ? await pageStock(code) : pageWatch();
    else if (tab === "predict") out = code ? await pagePredictStock(code) : pagePredict();
    else if (tab === "review") out = await pageReview(code);
    else out = pageSettings();
  } catch (e) {
    out = nav(TABS[tab]) + `<div class="empty">讀取資料失敗，請稍後重新整理。</div>`;
  }
  if (seq !== renderSeq) return;
  const [html, after] = Array.isArray(out) ? out : [out, []];
  document.getElementById("app").innerHTML = html;
  after.forEach((fn) => fn());
  window.scrollTo(0, keepScroll ? y : 0);
}

document.addEventListener("click", (e) => {
  const b = e.target.closest(".seg button");
  if (b) { UI[b.parentElement.dataset.seg] = b.dataset.v; render(true); return; }
  const w = e.target.closest("[data-wl]");
  if (w) { wl = w.dataset.wl; LS.set("wl", wl); render(true); return; }
  const m = e.target.closest("[data-model]");
  if (m) { model = m.dataset.model; LS.set("model", model); render(true); return; }
  const a = e.target.closest("[data-act]");
  if (a && a.dataset.act === "add") { openAdd(); return; }
  if (a && a.dataset.act === "token") { openToken(); return; }
  const r = e.target.closest("[data-remove]");
  if (r) confirmRemove(r.dataset.remove);
});
window.addEventListener("hashchange", () => render(false));

(async function init() {
  try {
    SUM = await fetch("data/summary.json?t=" + Date.now()).then((r) => r.json());
  } catch (e) {
    document.getElementById("app").innerHTML = '<div class="empty">尚未產生資料。請到 GitHub 的 Actions 執行一次「每日更新看盤」。</div>';
    return;
  }
  if (!SUM.lists[wl]) wl = Object.keys(SUM.lists)[0] || "";
  cleanPending();
  if (!MODEL[model]) model = "champion";
  render();
})();
