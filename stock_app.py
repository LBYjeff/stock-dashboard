"""美台股看盤 — iOS 風格版（遵循 Apple Human Interface Guidelines 的結構與規範）

四大導覽區域（底部標籤列）：自選股 ／ 預測 ／ 檢討 ／ 設定
基礎視圖元件：大標題導覽列、分組列表（inset grouped）、列表儲存格、分段控制、卡片、標籤
設計規範：系統字體與字級、系統色、深色模式、44pt 最小觸控範圍、安全區域

執行：streamlit run stock_app.py --server.address 0.0.0.0
網址參數：?tab=watch|predict|review|settings &code=2330 &wl=清單名稱（每台裝置可加入主畫面記住自己的清單）
"""
from __future__ import annotations

import datetime as dt
import html
import io
import json
import zipfile
from urllib.parse import quote, urlencode

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data as D
import evaluate as EV
import pipeline as P
import store as S
import trend as T
import trend_chart as TC

# ---------------------------------------------------------------- iOS 系統色（圖表用固定值）
RED, GREEN, BLUE, ORANGE, PURPLE, GRAY = "#FF3B30", "#34C759", "#007AFF", "#FF9500", "#AF52DE", "#8E8E93"
UP, DOWN = RED, GREEN                         # 台股慣例：紅漲綠跌
TC.UP, TC.DOWN = UP, DOWN
FONT = "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'PingFang TC', 'Helvetica Neue', 'Microsoft JhengHei', sans-serif"
TABS = {"watch": "自選股", "predict": "預測", "review": "檢討", "settings": "設定"}
PLOT_CFG = {"displayModeBar": False, "responsive": True}

st.set_page_config(page_title="美台股看盤", page_icon="📈", layout="centered", initial_sidebar_state="collapsed")

CSS = """
<style>
:root{
  --bg:#F2F2F7; --card:#FFFFFF; --card2:#F2F2F7; --label:#000000; --label2:rgba(60,60,67,.6);
  --label3:rgba(60,60,67,.3); --sep:rgba(60,60,67,.29); --fill:rgba(120,120,128,.12);
  --blue:#007AFF; --red:#FF3B30; --green:#34C759; --orange:#FF9500; --gray:#8E8E93;
  --bar:rgba(249,249,249,.94);
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#000000; --card:#1C1C1E; --card2:#2C2C2E; --label:#FFFFFF; --label2:rgba(235,235,245,.6);
    --label3:rgba(235,235,245,.3); --sep:rgba(84,84,88,.65); --fill:rgba(118,118,128,.24);
    --blue:#0A84FF; --red:#FF453A; --green:#30D158; --orange:#FF9F0A; --gray:#8E8E93;
    --bar:rgba(22,22,24,.94);
  }
}
html, body, [data-testid="stApp"], [data-testid="stAppViewContainer"], [data-testid="stMain"]{
  background:var(--bg)!important; color:var(--label);
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","PingFang TC","Helvetica Neue","Microsoft JhengHei",sans-serif!important;
  -webkit-font-smoothing:antialiased;
}
header[data-testid="stHeader"], [data-testid="stToolbar"], footer, [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"], [data-testid="stDecoration"],
#MainMenu{display:none!important}
.block-container{max-width:760px!important;padding:12px 16px calc(96px + env(safe-area-inset-bottom))!important}
[data-testid="stVerticalBlock"]{gap:.55rem}
[data-testid="stMarkdownContainer"]{margin-bottom:0!important}
[data-testid="stForm"]{border:none!important;background:var(--card);border-radius:10px;padding:12px 14px!important}
.stMarkdown p{margin-bottom:0}
a{color:var(--blue);text-decoration:none}

/* ---- 導覽列 ---- */
.nav{padding:6px 4px 10px}
.nav a, .nav a:hover{text-decoration:none!important}
.nav .back{font-size:17px;color:var(--blue);display:inline-flex;align-items:center;min-height:44px}
.nav .back b{font-size:26px;font-weight:400;margin-right:4px;line-height:1}
.nav .sub{font-size:13px;color:var(--label2);margin-bottom:2px}
.nav h1{font-size:34px!important;font-weight:700!important;letter-spacing:.37px;margin:0!important;padding:0!important;
        line-height:41px!important;color:var(--label)!important}
.nav h2{font-size:28px!important;font-weight:700!important;margin:0!important;padding:0!important;color:var(--label)!important}

/* ---- 分組列表 ---- */
.sec-h{font-size:13px;color:var(--label2);text-transform:uppercase;letter-spacing:-.08px;margin:18px 16px 6px}
.sec-f{font-size:13px;color:var(--label2);margin:6px 16px 4px;line-height:1.4}
.grp{background:var(--card);border-radius:10px;overflow:hidden}
.cell{display:flex;align-items:center;min-height:44px;padding:11px 16px;position:relative;color:var(--label)!important;
      text-decoration:none!important;font-size:17px;gap:10px}
.cell + .cell::before{content:"";position:absolute;top:0;left:16px;right:0;border-top:.5px solid var(--sep)}
a.cell:active{background:var(--fill)}
.cell .k{flex:1;min-width:0}
.cell .k small{display:block;font-size:13px;color:var(--label2);margin-top:1px}
.cell .v{color:var(--label2);text-align:right;font-variant-numeric:tabular-nums}
.cell .chev{color:var(--label3);font-size:22px;line-height:1;margin-left:2px}
.sym{font-size:17px;font-weight:600}
.nm{font-size:13px;color:var(--label2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.px{font-size:17px;font-weight:500;text-align:right;font-variant-numeric:tabular-nums}
.pill{display:inline-block;min-width:76px;text-align:right;padding:4px 7px;border-radius:6px;color:#fff;font-size:15px;
      font-weight:600;font-variant-numeric:tabular-nums;margin-top:3px}
.pill.up{background:var(--red)} .pill.dn{background:var(--green)} .pill.fl{background:var(--gray)}
.tag{display:inline-block;font-size:12px;padding:2px 6px;border-radius:5px;background:var(--fill);color:var(--label2);
     margin-right:4px;white-space:nowrap}
.tag.best{background:var(--blue);color:#fff}
.tag.up{color:var(--red)} .tag.dn{color:var(--green)}
.spark{flex:0 0 64px}

/* ---- 卡片 ---- */
.kg{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.kc{background:var(--card);border-radius:12px;padding:12px 14px}
.kc .k{font-size:13px;color:var(--label2)}
.kc .v{font-size:22px;font-weight:700;margin-top:2px;font-variant-numeric:tabular-nums}
.kc .s{font-size:13px;color:var(--label2);margin-top:2px}
.hero{padding:2px 4px 6px}
.hero .p{font-size:34px;font-weight:700;font-variant-numeric:tabular-nums}
.hero .c{font-size:17px;font-weight:600}
.bar{height:6px;border-radius:3px;background:var(--fill);overflow:hidden;margin-top:6px}
.bar i{display:block;height:100%;background:var(--blue);border-radius:3px}
.note{font-size:13px;color:var(--label2);padding:0 16px;line-height:1.45}
.report{background:var(--card);border-radius:10px;padding:14px 16px;font-size:15px;line-height:1.55}
.report h2{font-size:17px!important;font-weight:600!important;margin:14px 0 6px!important;padding:0!important}
.empty{text-align:center;color:var(--label2);padding:36px 16px;font-size:15px}

/* ---- 標籤列（底部四大導覽區域）---- */
.tabbar{position:fixed;left:0;right:0;bottom:0;z-index:999990;display:flex;justify-content:center;
  background:var(--bar);backdrop-filter:saturate(180%) blur(20px);-webkit-backdrop-filter:saturate(180%) blur(20px);
  border-top:.5px solid var(--sep);padding-bottom:env(safe-area-inset-bottom)}
.tabbar .in{display:flex;width:100%;max-width:760px}
.tabbar a{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;height:50px;
  color:var(--gray)!important;font-size:10px;font-weight:500;text-decoration:none!important;gap:2px}
.tabbar a.on{color:var(--blue)!important}
.tabbar svg{width:26px;height:26px}

/* ---- Streamlit 元件套用 iOS 外觀 ---- */
.stButton button, .stDownloadButton button, .stFormSubmitButton button{
  border-radius:10px!important;min-height:44px;font-size:17px!important;font-weight:600;border:none!important;
  background:var(--card)!important;color:var(--blue)!important;box-shadow:none!important}
.stButton button[kind="primary"], .stFormSubmitButton button[kind="primary"]{background:var(--blue)!important;color:#fff!important}
.stButton button p, .stDownloadButton button p{font-size:17px!important}
[data-testid="stTextInput"] input, [data-baseweb="select"] > div, [data-testid="stNumberInput"] input{
  border-radius:10px!important;min-height:44px}
/* 分段控制（Segmented Control）：撐滿整行、選中項為浮起的白色塊 */
[data-testid="stElementContainer"]:has(> [data-testid="stButtonGroup"]){width:100%!important}
[data-testid="stButtonGroup"]{display:block!important;width:100%!important}
[data-testid="stButtonGroup"] [role="radiogroup"]{display:flex!important;width:100%!important;max-width:none!important;background:var(--fill);
  border-radius:9px;padding:2px;gap:2px!important;flex-wrap:nowrap!important}
[data-testid="stButtonGroup"] button{flex:1 1 0;min-width:0;border:none!important;background:transparent!important;
  border-radius:7px!important;min-height:32px;padding:3px 6px!important;box-shadow:none!important}
[data-testid="stButtonGroup"] button p{font-size:13px!important;font-weight:500;color:var(--label)!important;white-space:nowrap}
[data-testid="stButtonGroup"] button[aria-checked="true"]{background:var(--card)!important;
  box-shadow:0 3px 8px rgba(0,0,0,.12),0 3px 1px rgba(0,0,0,.04)!important}
[data-testid="stButtonGroup"] button[aria-checked="true"] p{font-weight:600}
[data-testid="stExpander"] details{border:none!important;background:var(--card);border-radius:10px}
[data-testid="stExpander"] summary{min-height:44px;font-size:17px}
[data-testid="stDataFrame"]{border-radius:10px;overflow:hidden}
[data-testid="stStatusWidget"]{display:none}
.stPlotlyChart{background:var(--card);border-radius:12px;padding:6px 2px;touch-action:pan-y}
[data-testid="stAlert"]{border-radius:10px}
label p{font-size:13px!important;color:var(--label2)!important}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

ICON = {
    "watch": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round">'
             '<circle cx="4.5" cy="6" r="1.2" fill="currentColor"/><circle cx="4.5" cy="12" r="1.2" fill="currentColor"/>'
             '<circle cx="4.5" cy="18" r="1.2" fill="currentColor"/><path d="M9 6h11M9 12h11M9 18h11"/></svg>',
    "predict": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" '
               'stroke-linejoin="round"><path d="M3 19h18"/><path d="M4 15l5-5 4 3 7-7"/><path d="M15 6h5v5"/></svg>',
    "review": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" '
              'stroke-linejoin="round"><rect x="5" y="4" width="14" height="17" rx="2.5"/><path d="M9 4.5h6v2.5H9z"/>'
              '<path d="M8.5 13.5l2.3 2.3 4.7-4.8"/></svg>',
    "settings": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round">'
                '<circle cx="12" cy="12" r="3.2"/><path d="M12 2.8v2.6M12 18.6v2.6M2.8 12h2.6M18.6 12h2.6M5.5 5.5l1.8 1.8'
                'M16.7 16.7l1.8 1.8M5.5 18.5l1.8-1.8M16.7 7.3l1.8-1.8"/></svg>',
}

# ================================================================ 狀態（網址參數）
cfg = S.load_cfg()
qp = st.query_params
tab = qp.get("tab", "watch")
tab = tab if tab in TABS else "watch"
lists = cfg["watchlists"]
wl = qp.get("wl")
wl = wl if wl in lists else next(iter(lists))
codes = lists[wl]
code = qp.get("code")
TOKEN = cfg.get("finmind_token", "")


def href(**kw) -> str:
    p = {"tab": tab, "wl": wl}
    p.update(kw)
    return "?" + urlencode({k: v for k, v in p.items() if v is not None})


def esc(s) -> str:
    return html.escape(str(s))


def md(x: str):
    st.markdown(x, unsafe_allow_html=True)


# ================================================================ 資料讀取（快取，以最後更新時間為版本）
def stamp(c):
    return S.get_meta(c).get("last_update") or ""


@st.cache_data(show_spinner=False, max_entries=200)
def _px(c, _s):
    return S.load_prices(c)


@st.cache_data(show_spinner=False, max_entries=200)
def _fc(c, _s):
    return S.load_forecasts(c)


@st.cache_data(show_spinner=False, max_entries=200)
def _states(c, _s):
    return {h: S.load_state(c, h) for h in EV.HS}


@st.cache_data(show_spinner=False, max_entries=60)
def _tr(c, _s, wkey):
    w5, w20 = json.loads(wkey)
    return T.analyze(_px(c, _s), w5=w5, w20=w20)


@st.cache_data(show_spinner=False, max_entries=40)
def _recs(c, _s):
    px = _px(c, _s)
    base = S.load_rule_rec(c, px)
    if base.empty:
        return {}
    base = base.iloc[-500:]
    stt = _states(c, _s)
    cal = {h: stt[h].get("calib") for h in EV.HS}
    for h in EV.HS:
        cal[f"sd{h}"] = stt[h].get("resid_sd", 1.5)
    oos = {h: S.load_ml_oos(c, h) for h in EV.HS}
    recs = {m: P.model_rec(base, m, cal, oos) for m in P.MODELS}
    recs["champion"] = P.champion_rec(recs, {h: stt[h].get("champion", "rule") for h in EV.HS})
    return recs


def px_of(c):
    return _px(c, stamp(c))


def name_of(c):
    return S.get_meta(c).get("name") or c


def shown_model() -> str:
    return {"auto": "champion"}.get(cfg.get("model_mode", "auto"), cfg.get("model_mode"))


def run_update(todo: list[str], label: str = "更新資料與模型"):
    with st.status(f"{label}（{len(todo)} 檔）…", expanded=True) as box:
        bar = st.progress(0.0)
        res = P.update_many(todo, TOKEN, progress=lambda i, n, c: bar.progress(i / n, text=f"{c}（{i + 1}/{n}）"),
                            log=lambda m: None)
        bar.progress(1.0, text="完成")
        bad = [r["code"] for r in res if not r.get("ok")]
        box.update(label=f"{label}完成" + (f"，{len(bad)} 檔失敗：{', '.join(bad)}" if bad else ""),
                   state="error" if bad else "complete", expanded=bool(bad))
    st.cache_data.clear()


# 每次開啟：新加入或超過 6 小時未更新的股票自動更新
if "auto_checked" not in st.session_state:
    st.session_state.auto_checked = True
    missing = [c for c in codes if not S.get_meta(c).get("last_update")]
    stale = [c for c in codes if cfg.get("auto_update", True) and P.needs_update(c)]
    todo = list(dict.fromkeys(missing + stale))
    if todo:
        run_update(todo, "首次建立資料庫" if missing else "每日自動更新")


# ================================================================ 共用元件
def nav(title, sub=None, back=None, large=True):
    b = f'<a class="back" href="{back[1]}" target="_self"><b>‹</b>{esc(back[0])}</a>' if back else ""
    s = f'<div class="sub">{esc(sub)}</div>' if sub else ""
    md(f'<div class="nav">{b}{s}<{"h1" if large else "h2"}>{esc(title)}</{"h1" if large else "h2"}></div>')


def section(header=None, footer=None, cells=""):
    h = f'<div class="sec-h">{esc(header)}</div>' if header else ""
    f = f'<div class="sec-f">{footer}</div>' if footer else ""
    md(f'{h}<div class="grp">{cells}</div>{f}')


def cell(k, v="", sub=None, link=None, vcolor=None, chev=None):
    s = f"<small>{sub}</small>" if sub else ""
    vc = f' style="color:{vcolor}"' if vcolor else ""
    ch = '<span class="chev">›</span>' if (chev if chev is not None else bool(link)) else ""
    tag = f'a href="{link}" target="_self"' if link else "div"
    end = "a" if link else "div"
    return f'<{tag} class="cell"><div class="k">{k}{s}</div><div class="v"{vc}>{v}</div>{ch}</{end}>'


def pill(pct):
    if pct is None or pd.isna(pct):
        return '<span class="pill fl">—</span>'
    cls = "up" if pct > 0 else "dn" if pct < 0 else "fl"
    return f'<span class="pill {cls}">{pct:+.2f}%</span>'


def color_of(v):
    if v is None or pd.isna(v) or v == 0:
        return "var(--label2)"
    return "var(--red)" if v > 0 else "var(--green)"


def spark(series: pd.Series) -> str:
    s = series.dropna().iloc[-30:]
    if len(s) < 2:
        return ""
    y = s.values
    lo, hi = y.min(), y.max()
    yy = 26 - (y - lo) / ((hi - lo) or 1) * 22
    xx = np.linspace(2, 62, len(y))
    pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in zip(xx, yy))
    col = "var(--red)" if y[-1] >= y[0] else "var(--green)"
    return (f'<svg class="spark" width="64" height="28" viewBox="0 0 64 28"><polyline points="{pts}" fill="none" '
            f'stroke="{col}" stroke-width="1.6" stroke-linejoin="round" stroke-linecap="round"/></svg>')


def quote_of(px):
    if px is None or len(px) < 2:
        return None, None, None
    c, p = float(px["Close"].iloc[-1]), float(px["Close"].iloc[-2])
    return c, c - p, (c / p - 1) * 100


def fmt_vol(v, tw):
    if v is None or pd.isna(v):
        return "-"
    if tw:
        return f"{v / 1000:,.0f} 張"
    for u, d in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= d:
            return f"{v / d:,.2f}{u}"
    return f"{v:,.0f}"


def trend_tag(f, h):
    if not f:
        return ""
    d = f.get(f"dir{h}", 0)
    cls, arrow, word = ("up", "▲", "偏多") if d > 0 else ("dn", "▼", "偏空") if d < 0 else ("", "●", "盤整")
    pct = (f[f"p{h}"] / f["close"] - 1) * 100 if f.get(f"p{h}") else 0
    return f'<span class="tag {cls}">{h}日 {arrow}{word} {pct:+.1f}%</span>'


def ios_layout(fig, h, legend=True):
    fig.update_layout(height=h, margin=dict(l=6, r=6, t=26, b=6), hovermode="x unified", dragmode=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=legend,
                      legend=dict(orientation="h", y=1.07, x=0, font=dict(size=11)),
                      font=dict(family=FONT, size=11, color="#8E8E93"))
    fig.update_xaxes(showgrid=False, fixedrange=True, tickfont=dict(size=10))
    fig.update_yaxes(gridcolor="rgba(142,142,147,.18)", zeroline=False, fixedrange=True, tickfont=dict(size=10))
    fig.update_annotations(font_size=11, x=0, xanchor="left")
    return fig


def kline_fig(df, tw):
    x = df.index.strftime("%Y-%m-%d")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.18, 0.22],
                        subplot_titles=("", "成交量", "乖離率 %"))
    fig.add_trace(go.Candlestick(x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K",
                                 increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
                                 decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN), showlegend=False), 1, 1)
    for m, c in (("MA5", ORANGE), ("MA20", BLUE), ("MA60", PURPLE)):
        fig.add_trace(go.Scatter(x=x, y=df[m], name=m, line=dict(color=c, width=1.2)), 1, 1)
    vol = df["Volume"] / 1000 if tw else df["Volume"]
    fig.add_trace(go.Bar(x=x, y=vol, marker_color=[UP if c >= o else DOWN for o, c in zip(df["Open"], df["Close"])],
                         marker_line_width=0, showlegend=False, name="量"), 2, 1)
    for n, c in ((5, ORANGE), (20, BLUE), (60, PURPLE)):
        fig.add_trace(go.Scatter(x=x, y=df[f"BIAS{n}"], line=dict(color=c, width=1.1), showlegend=False,
                                 name=f"乖離{n}"), 3, 1)
    fig.add_hline(y=0, line=dict(color="rgba(142,142,147,.6)", width=1, dash="dot"), row=3, col=1)
    ios_layout(fig, 520)
    fig.update_xaxes(type="category", nticks=5, rangeslider_visible=False)
    return fig


def tabbar():
    items = "".join(
        f'<a class="{"on" if k == tab else ""}" href="?{urlencode({"tab": k, "wl": wl})}" target="_self">'
        f'{ICON[k]}<span>{v}</span></a>' for k, v in TABS.items())
    md(f'<nav class="tabbar"><div class="in">{items}</div></nav>')


def path_df(close, p, lo, hi, dates, h):
    t = np.arange(1, h + 1)
    mid = close + (p - close) * t / h
    half = (hi - lo) / 2 * np.sqrt(t / h)
    return pd.DataFrame({"mid": mid, "hi": mid + half, "lo": mid - half}, index=dates[:h])


# ================================================================ 自選股
def page_watch():
    if code and code in codes:
        return page_stock(code)
    lu = S.last_update(codes)
    nav("自選股", f"{wl}・{len(codes)} 檔" + (f"・更新於 {lu[5:16].replace('T', ' ')}" if lu else ""))
    if not codes:
        md(f'<div class="empty">清單是空的<br><a href="{href(tab="settings")}" target="_self">到「設定」加入股票</a></div>')
        return
    mk = st.segmented_control("市場", ["全部", "台股", "美股"], default="全部", label_visibility="collapsed",
                              key="mk") or "全部"
    show = [c for c in codes if mk == "全部" or (mk == "台股") == D.is_tw(c)]
    model = shown_model()
    rows = []
    for c in show:
        px = px_of(c)
        cl, ch, pct = quote_of(px)
        f = _fc(c, stamp(c)).get(model) or {}
        tags = trend_tag(f, 5) + trend_tag(f, 20) if f else '<span class="tag">尚無預測</span>'
        price = f"{cl:,.2f}" if cl else "—"
        rows.append(
            f'<a class="cell" href="{href(code=c)}" target="_self" style="min-height:64px">'
            f'<div class="k"><div class="sym">{esc(c)}</div><div class="nm">{esc(name_of(c))}</div>'
            f'<div style="margin-top:4px">{tags}</div></div>'
            f'{spark(px["Close"]) if cl else ""}'
            f'<div><div class="px">{price}</div>{pill(pct)}</div></a>')
    section(None, f"點股票看明細・趨勢標籤為「{P.MODEL_NAME.get(model, model)}」的 5／20 日預測・紅漲綠跌",
            "".join(rows))


def page_stock(c):
    px = px_of(c)
    tw = D.is_tw(c)
    nav(name_of(c), f"{c}・{'台股' if tw else '美股'}", back=("自選股", href(code=None)), large=False)
    if px.empty:
        md('<div class="empty">尚無資料，請到「設定」按「立即更新」。</div>')
        return
    cl, ch, pct = quote_of(px)
    md(f'<div class="hero"><span class="p">{cl:,.2f}</span>　<span class="c" style="color:{color_of(ch)}">'
       f'{ch:+,.2f}（{pct:+.2f}%）</span><div class="note" style="padding:0">資料日期 {px.index[-1]:%Y-%m-%d}</div></div>')
    view = st.segmented_control("檢視", ["走勢", "籌碼", "高低點"], default="走勢", label_visibility="collapsed",
                                key=f"sv_{c}") or "走勢"
    ind = D.add_indicators(px)
    last = ind.iloc[-1]
    if view == "走勢":
        rng = st.segmented_control("區間", ["3月", "6月", "1年"], default="6月", label_visibility="collapsed",
                                   key=f"rg_{c}") or "6月"
        n = {"3月": 63, "6月": 126, "1年": 250}[rng]
        st.plotly_chart(kline_fig(ind.iloc[-n:], tw), width="stretch", config=PLOT_CFG)
        vr = last["Volume"] / ind["Volume"].iloc[-6:-1].mean()
        section("今日數據", None, "".join([
            cell("成交量", fmt_vol(last["Volume"], tw), sub=f"量比 {vr:.2f}（今日 ÷ 前 5 日均量）"),
            cell("MA5 乖離", f"{last['BIAS5']:+.2f}%", vcolor=color_of(last["BIAS5"])),
            cell("MA20 乖離", f"{last['BIAS20']:+.2f}%", vcolor=color_of(last["BIAS20"])),
            cell("MA60 乖離", f"{last['BIAS60']:+.2f}%", vcolor=color_of(last["BIAS60"])),
        ]))
    elif view == "籌碼":
        if not tw:
            md('<div class="empty">三大法人與融資融券為台股資料，美股沒有。</div>')
        else:
            inst, mg = S.load_inst(c), S.load_margin(c)
            if not inst.empty:
                cells = []
                for g in ("外資", "投信", "自營商", "合計"):
                    v = inst[g].iloc[-1]
                    sk = D.streak(inst[g])
                    sub = (f"連{'買' if sk > 0 else '賣'} {abs(sk)} 日・" if sk else "") + f"5 日 {inst[g].iloc[-5:].sum():+,.0f}"
                    cells.append(cell(g, f"{v:+,.0f} 張", sub=sub, vcolor=color_of(v)))
                section(f"三大法人（{inst.index[-1]:%m/%d}）", None, "".join(cells))
                ii = inst.iloc[-60:]
                fig = go.Figure()
                for g, col in (("外資", BLUE), ("投信", ORANGE), ("自營商", PURPLE)):
                    fig.add_trace(go.Bar(x=ii.index.strftime("%m/%d"), y=ii[g], name=g, marker_color=col,
                                         marker_line_width=0))
                fig.update_layout(barmode="relative")
                ios_layout(fig, 300)
                fig.update_xaxes(type="category", nticks=5)
                st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
            if not mg.empty:
                lm = mg.iloc[-1]
                section("融資融券", None, "".join([
                    cell("融資餘額", f"{lm['融資餘額']:,.0f} 張", sub=f"增減 {lm['融資增減']:+,.0f}"),
                    cell("融券餘額", f"{lm['融券餘額']:,.0f} 張", sub=f"增減 {lm['融券增減']:+,.0f}"),
                    cell("券資比", f"{lm['券資比%']:.2f}%" if pd.notna(lm["券資比%"]) else "-"),
                    cell("融資使用率", f"{lm['融資使用率%']:.2f}%" if pd.notna(lm["融資使用率%"]) else "-"),
                ]))
            if inst.empty and mg.empty:
                md('<div class="empty">尚無籌碼資料（ETF 或剛上市個股可能沒有）。</div>')
    else:
        hl = D.high_low_stats(px)
        a, y = hl["all"], hl["52w"]
        section("高低點", f"資料起點 {hl['first_date']:%Y-%m-%d}", "".join([
            cell("歷史最高", f"{a['high']:,.2f}", sub=f"{a['high_date']:%Y-%m-%d}・距今 {a['from_high']:+.1f}%"),
            cell("歷史最低", f"{a['low']:,.2f}", sub=f"{a['low_date']:%Y-%m-%d}・距今 {a['from_low']:+.1f}%"),
            cell("52 週最高", f"{y['high']:,.2f}", sub=f"{y['high_date']:%Y-%m-%d}・距今 {y['from_high']:+.1f}%"),
            cell("52 週最低", f"{y['low']:,.2f}", sub=f"{y['low_date']:%Y-%m-%d}・距今 {y['from_low']:+.1f}%"),
        ]))
        fig = go.Figure(go.Scatter(x=px.index, y=px["Close"], line=dict(color=BLUE, width=1.3), name="收盤"))
        for d, v, col in ((a["high_date"], a["high"], UP), (a["low_date"], a["low"], DOWN)):
            fig.add_trace(go.Scatter(x=[d], y=[v], mode="markers", marker=dict(size=9, color=col), showlegend=False))
        ios_layout(fig, 280, legend=False)
        fig.update_yaxes(type="log")
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
    section(None, None, cell("查看 5／20 日預測", "", link=href(tab="predict", code=c)) +
            cell("查看預測檢討", "", link=href(tab="review", code=c)))


# ================================================================ 預測
def page_predict():
    model = shown_model()
    if code and code in codes:
        return page_predict_stock(code, model)
    nav("預測", f"{P.MODEL_NAME.get(model, model)}・未來 5／20 個交易日")
    rows = []
    for c in codes:
        f = _fc(c, stamp(c)).get(model)
        stt = _states(c, stamp(c))
        if not f:
            rows.append(cell(f'<span class="sym">{esc(c)}</span>', "尚無預測", sub=esc(name_of(c)), link=href(code=c)))
            continue
        champ = " ".join(f'<span class="tag">{h}日：{P.MODEL_NAME.get(stt[h].get("champion"), "-")}</span>'
                         for h in EV.HS) if model == "champion" else ""
        right = "".join(
            f'<div style="font-size:15px;color:{color_of(f["p" + str(h)] - f["close"]) if f.get("dir" + str(h)) else "var(--label2)"}">'
            f'{h}日 {f["p" + str(h)]:,.2f}（{(f["p" + str(h)] / f["close"] - 1) * 100:+.1f}%）</div>' for h in EV.HS)
        rows.append(f'<a class="cell" href="{href(code=c)}" target="_self" style="min-height:64px">'
                    f'<div class="k"><div class="sym">{esc(c)}</div><div class="nm">{esc(name_of(c))}</div>'
                    f'<div style="margin-top:4px">{champ}</div></div><div class="v">{right}</div>'
                    f'<span class="chev">›</span></a>')
    section(None, "預估價為預測日收盤後第 5、第 20 個交易日（已跳過休市日）的收盤價。顏色：紅＝看漲、綠＝看跌、灰＝盤整。",
            "".join(rows))


def page_predict_stock(c, model):
    px = px_of(c)
    stt = _states(c, stamp(c))
    fcs = _fc(c, stamp(c))
    nav(name_of(c), f"{c}・{P.MODEL_NAME.get(model, model)}", back=("預測", href(code=None)), large=False)
    f = fcs.get(model)
    if px.empty or not f:
        md('<div class="empty">尚無預測，請到「設定」按「立即更新」。</div>')
        return
    dates = D.next_sessions(px.attrs.get("cal"), px.index[-1], 20)
    cards = []
    for h in EV.HS:
        p, lo, hi = f[f"p{h}"], f[f"lo{h}"], f[f"hi{h}"]
        pc = (p / f["close"] - 1) * 100
        d = f.get(f"dir{h}", 0)
        word = "▲ 偏多" if d > 0 else "▼ 偏空" if d < 0 else "● 盤整"
        mname = P.MODEL_NAME.get(f.get(f"model{h}", model), "")
        cards.append(f'<div class="kc"><div class="k">{h} 日後・{dates[h - 1]:%m/%d}</div>'
                     f'<div class="v" style="color:{color_of(d)}">{p:,.2f}</div>'
                     f'<div class="s">{word}　{pc:+.2f}%</div><div class="s">區間 {lo:,.1f} ～ {hi:,.1f}</div>'
                     f'<div style="margin-top:6px"><span class="tag">{esc(mname)}</span></div></div>')
    md(f'<div class="kg">{"".join(cards)}</div>')

    # 圖：規則型態線 ＋ 選用模型的預估路徑
    w5, w20 = EV.weights_for(c, True)
    tr = _tr(c, stamp(c), json.dumps([w5, w20]))
    if tr:
        tr2 = dict(tr)
        tr2["proj"] = path_df(f["close"], f["p5"], f["lo5"], f["hi5"], dates, 5)
        tr2["proj20"] = path_df(f["close"], f["p20"], f["lo20"], f["hi20"], dates, 20)
        tr2["score"] = 2.5 * f.get("dir5", 0)
        tr2["score20"] = 2.5 * f.get("dir20", 0)
        ind = D.add_indicators(px)
        fig = TC.trend_fig(ind, tr2, D.is_tw(c), bars=60, mobile=True)
        fig.update_layout(font=dict(family=FONT, color="#8E8E93"))
        st.plotly_chart(fig, width="stretch", config=PLOT_CFG)
        md('<div class="note">粗虛線＝5 日預估、細點線＝20 日預估，色帶＝可能區間；紅／綠長虛線＝支撐／壓力。</div>')

    # 三模型比較
    cells = []
    for m in P.MODELS:
        fm = fcs.get(m)
        if not fm:
            continue
        badges = "".join(f'<span class="tag best">{h}日最佳</span>' for h in EV.HS if stt[h].get("champion") == m)
        sub = "　".join(
            f"{h}日 {fm['p' + str(h)]:,.2f}（{(fm['p' + str(h)] / fm['close'] - 1) * 100:+.1f}%）" for h in EV.HS)
        hit = "／".join(EV.pct((stt[h].get("metrics") or {}).get(m, {}).get("ok5")) for h in EV.HS)
        cells.append(cell(f"{P.MODEL_NAME[m]} {badges}", f"命中 {hit}", sub=sub))
    section("三種模型", "命中＝最近 250 個交易日（樣本外）落差 ≤ 5% 的比例，依序為 5 日／20 日。"
                     "每天更新後重新比較，「最佳」可能改變。", "".join(cells))

    # 機器學習看什麼
    imp = stt[5].get("importance") or []
    if imp:
        mx = max(i["w"] for i in imp) or 1
        cells = "".join(
            cell(f'{esc(i["name"])}<div class="bar"><i style="width:{i["w"] / mx * 100:.0f}%"></i></div>',
                 "越高越看漲" if i["sign"] > 0 else "越高越看跌" if i["sign"] < 0 else "非線性") for i in imp[:6])
        section("機器學習最看重的條件（5 日）", f"這檔股票用 {stt[5].get('n_train', 0):,} 個交易日訓練。"
                "方向取自線性模型係數，僅供解讀參考。", cells)

    if tr and tr["signals"]:
        cells = "".join(cell(f'{TC.ARROW[s["dir"]]} {esc(s["name"])}', f'{s["w"]:+.1f}',
                             sub=f'{esc(s["kind"])}・{esc(s["note"])}', vcolor=color_of(s["dir"])) for s in tr["signals"])
        section("技術型態訊號（規則模型）", None, cells)
    md('<div class="note">⚠️ 預測為統計推估，不是保證，也不構成投資建議。</div>')


# ================================================================ 檢討
OKC, BADC = "#007AFF", "#FF9500"


def review_fig(R):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("5 日預測落差", "20 日預測落差"))
    for row, h in ((1, 5), (2, 20)):
        d = R[R[f"act{h}"].notna()]
        if d.empty:
            continue
        ok = d[f"ok{h}"] == 1
        for mask, nm, col in ((ok, "成功 ≤5%", OKC), (~ok, "失敗 >5%", BADC)):
            fig.add_trace(go.Scatter(x=d.index[mask], y=d[f"err{h}"][mask] * 100, mode="markers", name=nm,
                                     marker=dict(color=col, size=4), showlegend=row == 1,
                                     hovertemplate="%{x|%Y-%m-%d} 落差 %{y:+.1f}%<extra></extra>"), row, 1)
        for yv in (5, -5):
            fig.add_hline(y=yv, line=dict(color="rgba(142,142,147,.7)", dash="dash", width=1), row=row, col=1)
    ios_layout(fig, 400)
    fig.update_layout(hovermode="closest", margin=dict(l=6, r=6, t=48, b=6),
                      legend=dict(orientation="h", y=1.16, x=1, xanchor="right"))
    fig.update_yaxes(ticksuffix="%")
    return fig


def page_review():
    nav("檢討", "預測成績、失敗原因與個股化模型")
    if not codes:
        md('<div class="empty">清單是空的。</div>')
        return
    c = code if code in codes else codes[0]
    sel = st.selectbox("股票", codes, index=codes.index(c), format_func=lambda x: f"{x}　{name_of(x)}",
                       label_visibility="collapsed")
    if sel != c:
        st.query_params.update({"tab": "review", "wl": wl, "code": sel})
        st.rerun()
    mopts = {"最佳": "champion", "規則": "rule", "校正": "calib", "ML": "ml"}
    mlab = st.segmented_control("模型", list(mopts), default="最佳", label_visibility="collapsed", key="rv_m") or "最佳"
    src = st.segmented_control("來源", ["回溯驗證", "實際紀錄"], default="回溯驗證", label_visibility="collapsed",
                               key="rv_s") or "回溯驗證"
    m = mopts[mlab]
    px = px_of(c)
    recs = _recs(c, stamp(c))
    stt = _states(c, stamp(c))
    if not recs:
        md('<div class="empty">尚無資料，請到「設定」按「立即更新」。</div>')
        return
    R = S.load_live(c, m, px) if src == "實際紀錄" else recs.get(m, pd.DataFrame())
    if R is None or R.empty:
        md('<div class="empty">' + ("還沒有實際紀錄。每天開啟時會自動記錄當天的預測，5 個交易日後出現第一筆成績。"
                                     if src == "實際紀錄" else "這個模型的樣本不足。") + "</div>")
    else:
        summ = EV.summary(R)
        cards = []
        for _, s in summ.iterrows():
            diff = s["落差≤5%"] - s["對照：股價不動"]
            col = "var(--blue)" if diff > 0.03 else "var(--orange)" if diff < -0.03 else "var(--label)"
            cards.append(f'<div class="kc"><div class="k">{s["週期"]}・落差 ≤ 5%</div>'
                         f'<div class="v" style="color:{col}">{EV.pct(s["落差≤5%"])}</div>'
                         f'<div class="s">股價不動 {EV.pct(s["對照：股價不動"])}・{s["已驗證"]} 筆</div></div>')
            cards.append(f'<div class="kc"><div class="k">{s["週期"]}・區間／方向</div>'
                         f'<div class="v">{EV.pct(s["落在預測區間"])}</div>'
                         f'<div class="s">方向命中 {EV.pct(s["方向命中"])}（{s["方向判斷次數"]} 次）</div></div>')
        md(f'<div class="kg">{"".join(cards)}</div>')
        md('<div class="note">藍色＝優於「股價不動」，橘色＝不如。落差＝實際收盤 ÷ 預估價 − 1。</div>')
        st.plotly_chart(review_fig(R), width="stretch", config=PLOT_CFG)

        F = EV.failures(R, px, px_of(D.market_symbol(c)), S.load_divs(c), S.load_inst(c))
        if not F.empty:
            ct = EV.cause_table(F)
            tot = max(len(F), 1)
            cells = "".join(cell(
                f'{esc(k)}<div class="bar"><i style="width:{ct.loc[k, [x for x in ct.columns if "佔比" not in x]].sum() / tot * 100:.0f}%;'
                f'background:var(--orange)"></i></div>',
                "｜".join(f"{p} {int(ct.loc[k, p])}" for p in ("5 日", "20 日") if p in ct)) for k in ct.index)
            section("失敗原因（落差 > 5%）", "同一事件常影響連續數天的預測，筆數以預測日計。", cells)
            with st.expander("最近失敗明細"):
                for _, r in F.sort_values("預測日", ascending=False).head(20).iterrows():
                    md(f'**{r["預測日"]:%m/%d}・{r["週期"]}**　落差 {r["落差"]:+.1%}・{esc(r["主要原因"])}<br>'
                       f'<span class="note" style="padding:0">{esc(r["細節"])}</span>')

    # 模型比較
    cells = []
    for h in EV.HS:
        ms = stt[h].get("metrics") or {}
        for mm in P.MODELS:
            v = ms.get(mm, {})
            if not v.get("n"):
                continue
            best = '<span class="tag best">最佳</span>' if stt[h].get("champion") == mm else ""
            cells.append(cell(f"{h}日・{P.MODEL_NAME[mm]} {best}", EV.pct(v["ok5"]),
                              sub=f"區間 {EV.pct(v['band'])}・方向 {EV.pct(v.get('dir'))}・IC {v['ic']:+.3f}"))
    if cells:
        section("模型比較（近 250 日樣本外）", "每天更新後三個模型都會重新訓練並重新比較。", "".join(cells))

    # 初報告
    base = recs.get("rule", pd.DataFrame())
    fails = EV.failures(base, px, px_of(D.market_symbol(c)), S.load_divs(c), S.load_inst(c)) if not base.empty else pd.DataFrame()
    cal = {h: stt[h].get("calib") for h in EV.HS}
    text = EV.report_text(c, name_of(c), EV.summary(base), fails, cal)
    text = text.replace("## 四、下一步", P.model_section(stt) + "## 五、下一步")
    with st.expander("📝 初報告", expanded=False):
        md('<div class="report">' + _md_to_html(text.split("\n", 2)[2]) + "</div>")
    tables = {"個股化預測邏輯（權重校正）": EV.calib_table(cal),
              "每日預測明細（規則模型，最近 60 日）": EV.daily_table(base, 60).assign(
                  **{k: (lambda col: (lambda x: x[col].map(lambda v: "-" if pd.isna(v) else f"{v:+.1%}")))(k)
                     for k in ("5日落差", "20日落差")})}
    st.download_button("下載初報告（HTML）", EV.report_html(text, tables).encode("utf-8"),
                       file_name=f"{c}_預測檢討_{dt.date.today():%Y%m%d}.html", mime="text/html", width="stretch")
    with st.expander("權重校正明細"):
        st.dataframe(EV.calib_table(cal), hide_index=True, width="stretch")


def _md_to_html(t: str) -> str:
    out = []
    for line in t.splitlines():
        e = EV._bold(html.escape(line))
        if line.startswith("## "):
            out.append(f"<h2>{EV._bold(html.escape(line[3:]))}</h2>")
        elif line.startswith("### "):
            out.append(f"<h2>{EV._bold(html.escape(line[4:]))}</h2>")
        elif line.strip().startswith("- "):
            out.append(f"<div style='margin:3px 0 3px {12 if line.startswith('  ') else 0}px'>• "
                       f"{EV._bold(html.escape(line.strip()[2:]))}</div>")
        elif line.startswith("> "):
            out.append(f"<div class='note' style='padding:4px 0'>{EV._bold(html.escape(line[2:]))}</div>")
        elif line.strip():
            out.append(f"<div>{e}</div>")
    return "".join(out)


# ================================================================ 設定
def page_settings():
    global cfg
    nav("設定")
    # ---- 自選股清單
    md('<div class="sec-h">這台裝置的自選股清單</div>')
    names = list(lists)
    pick = st.selectbox("清單", names, index=names.index(wl), label_visibility="collapsed")
    if pick != wl:
        st.query_params.update({"tab": "settings", "wl": pick})
        st.rerun()
    host = ""
    try:
        host = st.context.headers.get("Host", "")
    except Exception:
        pass
    link = f"http://{host}/?wl={quote(wl)}" if host else f"/?wl={quote(wl)}"
    shown = f"http://{host}/?wl={wl}" if host else f"/?wl={wl}"
    md(f'<div class="sec-f">每台裝置可以用不同清單：用那台裝置打開 <a href="{link}" target="_self"><b>{esc(shown)}</b></a>，'
       f'再「加入主畫面」，以後從主畫面開啟就固定是「{esc(wl)}」。所有資料都存在執行程式的這台電腦（market.db）。</div>')

    with st.form("add", clear_on_submit=True):
        new = st.text_input("加入股票", placeholder="台股 2330、美股 NVDA，可用逗號一次多檔")
        if st.form_submit_button("加入到「" + wl + "」", type="primary", width="stretch") and new.strip():
            add = [D.normalize(x) for x in new.replace("，", ",").split(",") if x.strip()]
            add = [x for x in add if x not in lists[wl]]
            lists[wl] += add
            S.save_cfg(cfg)
            if add:
                run_update(add, "下載新股票資料並訓練模型")
            st.rerun()
    keep = st.multiselect("目前清單（點 × 移除）", lists[wl], default=lists[wl])
    if keep != lists[wl]:
        lists[wl] = keep
        S.save_cfg(cfg)
        st.rerun()
    with st.expander("管理清單（新增／重新命名／刪除）"):
        nn = st.text_input("新清單名稱", key="nn")
        if st.button("建立新清單", width="stretch") and nn.strip() and nn.strip() not in lists:
            lists[nn.strip()] = []
            S.save_cfg(cfg)
            st.query_params.update({"tab": "settings", "wl": nn.strip()})
            st.rerun()
        rn = st.text_input("把目前清單改名為", key="rn")
        if st.button("重新命名", width="stretch") and rn.strip() and rn.strip() not in lists:
            cfg["watchlists"] = {(rn.strip() if k == wl else k): v for k, v in lists.items()}
            S.save_cfg(cfg)
            st.query_params.update({"tab": "settings", "wl": rn.strip()})
            st.rerun()
        if len(lists) > 1:
            ok = st.checkbox(f"確認刪除「{wl}」")
            if st.button("刪除目前清單", width="stretch", disabled=not ok):
                lists.pop(wl)
                S.save_cfg(cfg)
                st.query_params.update({"tab": "settings", "wl": next(iter(lists))})
                st.rerun()

    # ---- 資料與模型
    lu = S.last_update(codes)
    md('<div class="sec-h">資料與模型更新</div>')
    md(f'<div class="grp">{cell("最後更新", (lu or "尚未更新")[:16].replace("T", " "))}'
       f'{cell("資料庫大小", f"{S.db_size() / 1e6:,.1f} MB")}</div>')
    c1, c2 = st.columns(2)
    if c1.button("立即更新", type="primary", width="stretch"):
        run_update(codes)
        st.rerun()
    if c2.button("全部重新訓練", width="stretch"):
        run_update(S.all_codes(cfg), "所有清單重新訓練")
        st.rerun()
    au = st.toggle("開啟時自動更新（超過 6 小時未更新）", value=cfg.get("auto_update", True))
    if au != cfg.get("auto_update", True):
        cfg["auto_update"] = au
        S.save_cfg(cfg)
    md('<div class="sec-f">每天更新會：補抓新資料 → 規則模型補算 → 重新校正權重 → 機器學習重新訓練 → '
       '三模型比較選出最佳 → 記錄今日預測。第一次建立每檔約 10～30 秒。</div>')

    md('<div class="sec-h">預測模型</div>')
    modes = {"自動選最佳": "auto", "規則": "rule", "校正": "calib", "機器學習": "ml"}
    inv = {v: k for k, v in modes.items()}
    mm = st.segmented_control("模型", list(modes), default=inv.get(cfg.get("model_mode", "auto"), "自動選最佳"),
                              label_visibility="collapsed", key="mmode") or "自動選最佳"
    if modes[mm] != cfg.get("model_mode", "auto"):
        cfg["model_mode"] = modes[mm]
        S.save_cfg(cfg)
        st.rerun()
    md('<div class="sec-f">「自動選最佳」：每檔股票的 5 日、20 日各自採用最近 250 個交易日最準的模型。</div>')

    md('<div class="sec-h">資料來源</div>')
    tok = st.text_input("FinMind Token（選填）", value=cfg.get("finmind_token", ""), type="password")
    if tok != cfg.get("finmind_token", ""):
        cfg["finmind_token"] = tok.strip()
        S.save_cfg(cfg)
    md('<div class="sec-f">報價：Yahoo Finance。台股法人、融資券：FinMind（未登入每小時約 300 次查詢）。</div>')

    md('<div class="sec-h">備份與搬移</div>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (S.DB, S.CONFIG, EV.CALIB_FILE):
            if p.exists():
                z.write(p, p.name)
    st.download_button("下載備份（資料庫＋清單）", buf.getvalue(), file_name=f"看盤備份_{dt.date.today():%Y%m%d}.zip",
                       mime="application/zip", width="stretch")
    up = st.file_uploader("還原備份（.zip）", type=["zip"])
    if up is not None and st.button("用這份備份覆蓋目前資料", width="stretch"):
        with zipfile.ZipFile(up) as z:
            names = set(z.namelist())
            if "market.db" not in names:
                st.error("這不是看盤備份檔（找不到 market.db）。")
            else:
                for p in (S.DB, S.CONFIG, EV.CALIB_FILE):
                    if p.name in names:
                        p.write_bytes(z.read(p.name))
                S._inited = False
                st.cache_data.clear()
                st.success("已還原，重新整理後生效。")
    md('<div class="sec-f">換電腦時：在舊電腦下載備份，新電腦解壓縮程式後到這裡還原。</div>')

    md('<div class="sec-h">關於</div>')
    md(f'<div class="grp">{cell("版本", "iOS 介面版・第二階段（機器學習）")}'
       f'{cell("介面規範", "依 Apple 人機介面指南")}'
       f'{cell("休市日", "依台股／美股交易所日曆剔除")}</div>')
    md('<div class="sec-f">⚠️ 所有預測為統計推估，不構成投資建議。</div>')


# ================================================================ 主程式
{"watch": page_watch, "predict": page_predict, "review": page_review, "settings": page_settings}[tab]()
tabbar()
