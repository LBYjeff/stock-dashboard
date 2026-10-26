"""美台股看盤 — 手機版
執行： streamlit run mobile.py --server.address 0.0.0.0
與桌面版共用 data.py 與 config.json（自選股清單同步）。
"""
from __future__ import annotations

import datetime as dt
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data as D
import trend as T
import trend_chart as TC
import evaluate as EV
import predlog as PL

APP_DIR = Path(__file__).parent
CONFIG = APP_DIR / "config.json"
DEFAULT_CFG = {"watchlist": ["2330", "2454", "0050", "NVDA", "TSM", "AAPL"], "finmind_token": ""}

UP, DOWN, FLAT = "#e5484d", "#2f9e66", "#8b8d98"
INST_COLOR = {"外資": "#3e7bfa", "投信": "#f08c00", "自營商": "#9b59d0"}
MA_COLOR = {"MA5": "#f5a524", "MA20": "#3e7bfa", "MA60": "#9b59d0"}
RANGES = {"3月": 63, "6月": 126, "1年": 250}
PLOT_CFG = {"displayModeBar": False, "scrollZoom": False, "responsive": True}

st.set_page_config(page_title="美台股看盤", page_icon="📈", layout="centered",
                   initial_sidebar_state="collapsed")
st.markdown("""
<style>
  .block-container {padding: 1rem .75rem 3rem; max-width: 560px;}
  header[data-testid="stHeader"] {height: 0; background: transparent;}
  h1 {font-size: 1.35rem !important; padding: 0 0 .3rem !important;}
  .hd {display:flex; justify-content:space-between; align-items:flex-end; margin:.2rem 0 .6rem;}
  .hd .nm {font-size:1.15rem; font-weight:700; line-height:1.2;}
  .hd .sub {font-size:.75rem; opacity:.6;}
  .hd .px {font-size:1.6rem; font-weight:700; text-align:right; line-height:1.1;}
  .hd .chg {font-size:.9rem; font-weight:600; text-align:right;}
  .grid {display:grid; grid-template-columns:1fr 1fr; gap:.45rem; margin:.3rem 0 .6rem;}
  .card {background:rgba(128,128,128,.10); border-radius:10px; padding:.5rem .65rem;}
  .card .k {font-size:.72rem; opacity:.65;}
  .card .v {font-size:1.05rem; font-weight:700; margin-top:.1rem;}
  .card .s {font-size:.72rem; opacity:.75; margin-top:.05rem;}
  [data-testid="stTabs"] button p {font-size:.9rem;}
  .stPlotlyChart {touch-action: pan-y;}
</style>
""", unsafe_allow_html=True)


# ================================================================ 設定 / 快取
def load_cfg():
    try:
        return {**DEFAULT_CFG, **json.loads(CONFIG.read_text(encoding="utf-8"))}
    except Exception:
        return dict(DEFAULT_CFG)


def save_cfg(cfg):
    try:
        CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


if "cfg" not in st.session_state:
    st.session_state.cfg = load_cfg()
cfg = st.session_state.cfg
TOKEN = cfg.get("finmind_token") or os.environ.get("FINMIND_TOKEN", "")


@st.cache_data(ttl=86400, show_spinner=False)
def c_tw_info(token):
    df = D.tw_stock_info(token)
    return {} if df.empty else {r.stock_id: (r.stock_name, r.type) for r in df.itertuples()}


@st.cache_data(ttl=86400, show_spinner=False)
def c_us_name(code):
    return D.us_name(code)


@st.cache_data(ttl=600, show_spinner=False)
def c_watch(codes, token):
    info = c_tw_info(token)

    def one(code):
        mtype = info.get(code, (None, None))[1] if D.is_tw(code) else None
        return code, D.price_history(code, "1y", mtype, token)

    with ThreadPoolExecutor(max_workers=8) as ex:
        return dict(ex.map(one, codes))


@st.cache_data(ttl=600, show_spinner=False)
def c_hist(code, period, token):
    mtype = c_tw_info(token).get(code, (None, None))[1] if D.is_tw(code) else None
    return D.price_history(code, period, mtype, token)


@st.cache_data(ttl=1800, show_spinner=False)
def c_inst(code, token):
    return D.institutional(code, 150, token)


@st.cache_data(ttl=1800, show_spinner=False)
def c_trend(code, token, wkey):
    full = c_hist(code, "max", token)
    w5, w20 = json.loads(wkey)
    return T.analyze(full, w5=w5, w20=w20), T.backtest(full)


@st.cache_data(ttl=21600, show_spinner=False)
def c_review(code, token):
    full = c_hist(code, "max", token)
    if full.empty or len(full) < 160:
        return None
    mtype = c_tw_info(token).get(code, (None, None))[1] if D.is_tw(code) else None
    mkt = c_hist(D.market_symbol(code), "max", token)
    divs = D.dividends(code, mtype)
    inst = D.institutional(code, 800, token) if D.is_tw(code) else pd.DataFrame()
    rec = EV.simulate(full, 500)
    fails = EV.failures(rec, full, mkt, divs, inst)
    cal = {h: EV.calibrate(rec, h) for h in EV.HS}
    EV.save_calib(code, cal[5], cal[20])
    return {"rec": rec, "fails": fails, "cal": cal, "mkt": mkt, "divs": divs, "inst": inst}


def wkey_of(code):
    return json.dumps(EV.weights_for(code, st.session_state.get("use_cal", True)), sort_keys=True)


@st.cache_data(ttl=1800, show_spinner=False)
def c_margin(code, token):
    return D.margin(code, 200, token)


def name_of(code):
    if D.is_tw(code):
        return c_tw_info(TOKEN).get(code, (code, None))[0]
    return c_us_name(code)


# ================================================================ 小工具
def color_of(v):
    if v is None or pd.isna(v) or v == 0:
        return FLAT
    return UP if v > 0 else DOWN


def fmt_vol(v, tw):
    if v is None or pd.isna(v):
        return "-"
    if tw:
        return f"{v:,.0f} 張"
    for unit, d in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(v) >= d:
            return f"{v / d:,.2f}{unit}"
    return f"{v:,.0f}"


def cards(items):
    """items: [(標題, 數值, 副標, 數值顏色或None)] → 兩欄卡片。"""
    html = "".join(
        f"<div class='card'><div class='k'>{k}</div>"
        f"<div class='v' style='color:{c or 'inherit'}'>{v}</div>"
        f"<div class='s'>{s}</div></div>" for k, v, s, c in items)
    st.markdown(f"<div class='grid'>{html}</div>", unsafe_allow_html=True)


def mobile_layout(fig, h):
    fig.update_layout(height=h, margin=dict(l=4, r=4, t=26, b=4), hovermode="x unified",
                      dragmode=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.06, x=0, font=dict(size=10)),
                      font=dict(size=10, family="PingFang TC, Microsoft JhengHei, Noto Sans TC, sans-serif"))
    fig.update_xaxes(showgrid=False, type="category", nticks=4, fixedrange=True,
                     rangeslider_visible=False, tickfont=dict(size=9))
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False, fixedrange=True,
                     tickfont=dict(size=9))
    fig.update_annotations(font_size=10, x=0, xanchor="left")
    return fig


# ================================================================ 圖表
def kline_fig(df, tw):
    x = df.index.strftime("%Y-%m-%d")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.58, 0.2, 0.22], subplot_titles=("", "成交量", "乖離率 %"))
    fig.add_trace(go.Candlestick(
        x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K",
        increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN), showlegend=False), 1, 1)
    for m, c in MA_COLOR.items():
        fig.add_trace(go.Scatter(x=x, y=df[m], name=m, line=dict(color=c, width=1)), 1, 1)
    hi_i, lo_i = df["High"].values.argmax(), df["Low"].values.argmin()
    for i, val, pos in ((hi_i, df["High"].iloc[hi_i], "top center"), (lo_i, df["Low"].iloc[lo_i], "bottom center")):
        fig.add_trace(go.Scatter(x=[x[i]], y=[val], mode="markers+text", text=[f"{val:,.2f}"],
                                 textposition=pos, marker=dict(size=5, color="#888"), textfont=dict(size=9),
                                 showlegend=False, hoverinfo="skip"), 1, 1)
    vol = df["Volume"] / 1000 if tw else df["Volume"]
    vcol = [UP if c >= o else DOWN for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=x, y=vol, marker_color=vcol, marker_line_width=0, showlegend=False, name="量"), 2, 1)
    for n, c in ((5, "#f5a524"), (20, "#3e7bfa"), (60, "#9b59d0")):
        fig.add_trace(go.Scatter(x=x, y=df[f"BIAS{n}"], name=f"乖離{n}", line=dict(color=c, width=1),
                                 showlegend=False), 3, 1)
    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,.5)", width=1, dash="dot"), row=3, col=1)
    return mobile_layout(fig, 540)


def inst_fig(inst, px):
    x = inst.index.strftime("%Y-%m-%d")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for g in ("外資", "投信", "自營商"):
        fig.add_trace(go.Bar(x=x, y=inst[g], name=g, marker_color=INST_COLOR[g], marker_line_width=0),
                      secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=px["Close"].reindex(inst.index), name="股價",
                             line=dict(color="#888", width=1.4)), secondary_y=True)
    fig.update_layout(barmode="relative")
    mobile_layout(fig, 340)
    fig.update_yaxes(showgrid=False, secondary_y=True)
    return fig


def margin_fig(m):
    x = m.index.strftime("%Y-%m-%d")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.65, 0.35],
                        specs=[[{"secondary_y": True}], [{}]], subplot_titles=("", "券資比 %"))
    fig.add_trace(go.Scatter(x=x, y=m["融資餘額"], name="融資", line=dict(color=UP, width=1.5)), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=m["融券餘額"], name="融券(右)", line=dict(color=DOWN, width=1.5)),
                  1, 1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=m["券資比%"], line=dict(color="#9b59d0", width=1.2), fill="tozeroy",
                             fillcolor="rgba(155,89,208,.12)", showlegend=False, name="券資比"), 2, 1)
    mobile_layout(fig, 380)
    fig.update_yaxes(showgrid=False, row=1, col=1, secondary_y=True)
    return fig


def history_fig(full, hl):
    fig = go.Figure(go.Scatter(x=full.index, y=full["Close"], line=dict(color="#3e7bfa", width=1.2), name="收盤"))
    a = hl["all"]
    for d, v, col in ((a["high_date"], a["high"], UP), (a["low_date"], a["low"], DOWN)):
        fig.add_trace(go.Scatter(x=[d], y=[v], mode="markers", marker=dict(size=8, color=col), showlegend=False))
    mobile_layout(fig, 260)
    fig.update_xaxes(type="date", nticks=5)
    fig.update_yaxes(type="log")
    fig.update_layout(showlegend=False)
    return fig


# ================================================================ 自選股
st.title("📈 美台股看盤")
codes = cfg["watchlist"]

if not codes:
    st.info("清單是空的，從最下方「管理自選股」新增。")
else:
    with st.spinner("抓取報價…"):
        frames = c_watch(tuple(codes), TOKEN)

    has_tw, has_us = any(D.is_tw(c) for c in codes), any(not D.is_tw(c) for c in codes)
    opts = ["全部"] + (["台股"] if has_tw else []) + (["美股"] if has_us else [])
    mkt = st.segmented_control("市場", opts, default="全部", label_visibility="collapsed") or "全部"
    shown = [c for c in codes if mkt == "全部" or (mkt == "台股") == D.is_tw(c)]

    rows = []
    for c in shown:
        q = D.quote_row(c, frames.get(c))
        w5, w20 = json.loads(wkey_of(c))
        tr_ = T.analyze(frames.get(c), w5=w5, w20=w20)
        if tr_ is not None:
            PL.log(c, frames[c].index[-1], tr_, calibrated=bool(w5 or w20))
        short = lambda days: " ".join(TC.badge(tr_, days).split(" ")[:2]).replace("盤整觀望", "盤整").replace("強烈", "強")
        rows.append({"股票": f"{name_of(c)}  {c}" if D.is_tw(c) else c,
                     "收盤": q.get("收盤"), "漲跌%": q.get("漲跌%"),
                     "5日": short(5), "20日": short(20)})
    wl = pd.DataFrame(rows)
    paint = lambda v: f"color:{color_of(v)};font-weight:600" if pd.notna(v) else ""
    sty = (wl.style.map(paint, subset=["漲跌%"]).map(TC.badge_color, subset=["5日", "20日"])
           .format({"收盤": "{:,.2f}", "漲跌%": "{:+.2f}%"}, na_rep="-"))
    ev = st.dataframe(sty, hide_index=True, width="stretch", on_select="rerun",
                      selection_mode="single-row", key=f"wl_{mkt}",
                      column_config={"股票": st.column_config.TextColumn(width="medium")})
    sel_rows = list(ev.selection.rows) if ev and ev.selection else []
    if sel_rows and sel_rows != st.session_state.get(f"_last_{mkt}"):
        st.session_state.selected = shown[sel_rows[0]]
    st.session_state[f"_last_{mkt}"] = sel_rows
    st.caption("點一列看明細 ↓")

    # ============================================================ 個股明細
    if st.session_state.get("selected") not in codes:
        st.session_state.selected = codes[0]
    sel = st.selectbox("個股", codes, index=codes.index(st.session_state.selected),
                       format_func=lambda c: f"{c}　{name_of(c)}", label_visibility="collapsed")
    st.session_state.selected = sel
    tw = D.is_tw(sel)

    with st.spinner("載入中…"):
        full = c_hist(sel, "max", TOKEN)

    if full.empty:
        st.error(f"抓不到 {sel} 的資料，請確認代號或稍後再試。")
    else:
        ind = D.add_indicators(full)
        last, prev = ind.iloc[-1], ind.iloc[-2]
        chg = last["Close"] - prev["Close"]
        pct = chg / prev["Close"] * 100
        hl = D.high_low_stats(full)
        col = color_of(chg)
        st.markdown(
            f"<div class='hd'><div><div class='nm'>{name_of(sel)}</div>"
            f"<div class='sub'>{sel} · {'台股' if tw else '美股'} · {ind.index[-1]:%Y-%m-%d}</div></div>"
            f"<div><div class='px' style='color:{col}'>{last['Close']:,.2f}</div>"
            f"<div class='chg' style='color:{col}'>{chg:+,.2f} ({pct:+.2f}%)</div></div></div>",
            unsafe_allow_html=True)

        vol_ratio = last["Volume"] / ind["Volume"].iloc[-6:-1].mean()
        cards([
            ("成交量", fmt_vol(last["Volume"] / 1000 if tw else last["Volume"], tw), f"量比 {vol_ratio:.2f}", None),
            ("MA20 乖離", f"{last['BIAS20']:+.2f}%", f"MA5 {last['BIAS5']:+.2f}%｜MA60 {last['BIAS60']:+.2f}%",
             color_of(last["BIAS20"])),
            ("52 週高 / 低", f"{hl['52w']['high']:,.2f}", f"低 {hl['52w']['low']:,.2f}｜距高 {hl['52w']['from_high']:+.1f}%",
             None),
            ("歷史高", f"{hl['all']['high']:,.2f}", f"{hl['all']['high_date']:%Y-%m-%d}｜距 {hl['all']['from_high']:+.1f}%",
             None),
        ])

        rng = st.segmented_control("區間", list(RANGES), default="6月", label_visibility="collapsed") or "6月"
        n = RANGES[rng]
        tabs = st.tabs(["K線", "趨勢", "檢討", "法人", "融資券", "高低點"])

        with tabs[0]:
            st.plotly_chart(kline_fig(ind.iloc[-n:], tw), width="stretch", config=PLOT_CFG)
            st.caption("MA5 橘 · MA20 藍 · MA60 紫；灰點為區間高低點")

        with tabs[1]:
            tr, bt = c_trend(sel, TOKEN, wkey_of(sel))
            if not tr:
                st.warning("資料不足，無法分析趨勢。")
            else:
                items = []
                for days, key, sc, lab in ((5, "proj", tr["score"], tr["label"]),
                                           (20, "proj20", tr["score20"], tr["label20"])):
                    p = tr[key]; end = p["mid"].iloc[-1]
                    items.append((f"未來 {days} 日（{p.index[-1]:%m/%d}）", f"{TC.badge(tr, days).split(' ')[0]} {lab}",
                                  f"分數 {sc:+.1f}", TC.score_color(sc)))
                    items.append((f"{days} 日預估價", f"{end:,.2f}",
                                  f"{(end / tr['close'] - 1) * 100:+.2f}%｜{p['lo'].iloc[-1]:,.1f}~{p['hi'].iloc[-1]:,.1f}",
                                  None))
                items += [("支撐", f"{tr['support']:,.2f}" if tr["support"] else "-", "水平線（碰觸 ≥2 次）", UP),
                          ("壓力", f"{tr['resistance']:,.2f}" if tr["resistance"] else "-", "水平線（碰觸 ≥2 次）", DOWN)]
                cards(items)
                st.plotly_chart(TC.trend_fig(ind, tr, tw, bars=50, mobile=True), width="stretch", config=PLOT_CFG)
                st.caption("粗虛線＝5 日預估、細點線＝20 日預估，色帶＝可能區間；紅/綠長虛線＝支撐/壓力")
                if tr["signals"]:
                    items = "".join(
                        f"<div style='padding:.35rem 0;border-bottom:1px solid rgba(128,128,128,.15)'>"
                        f"<span style='color:{color_of(s['dir'])};font-weight:700'>{TC.ARROW[s['dir']]} {s['name']}</span>"
                        f"<span style='float:right;color:{color_of(s['w'])};font-weight:600'>{s['w']:+.1f}</span>"
                        f"<div style='font-size:.75rem;opacity:.7'>{s['kind']}・{s['note']}</div></div>"
                        for s in tr["signals"])
                    st.markdown(items, unsafe_allow_html=True)
                if tr["targets"]:
                    st.caption("型態目標價：" + "、".join(f"{n} {v:,.2f}" for n, v in tr["targets"]))
                if bt:
                    pct = lambda v: f"{v * 100:.0f}%" if v is not None else "-"
                    items = []
                    for days in (5, 20):
                        s_ = bt.get(days)
                        if s_:
                            items += [(f"{days}日・偏多 {s_['bull_n']} 次", pct(s_["bull_hit"]),
                                       f"對照任意持有 {pct(s_['base_up'])}", None),
                                      (f"{days}日・偏空 {s_['bear_n']} 次", pct(s_["bear_hit"]),
                                       f"{days} 日後下跌比例", None)]
                    cards(items)
                    st.caption(f"過去命中率（近 {bt[5]['days']} 日）：偏多命中率要高於「任意持有」才有參考價值。")
                st.caption("⚠️ 技術分析規則的傾向，不是保證，不構成投資建議。")

        with tabs[2]:
            with st.spinner("回溯模擬 500 日、分析失敗原因…"):
                rv = c_review(sel, TOKEN)
            if rv is None:
                st.warning("歷史資料不足，無法檢討。")
            else:
                rec, fails, cal = rv["rec"], rv["fails"], rv["cal"]
                live = PL.load(sel, full)
                src = st.segmented_control("來源", ["回溯模擬", "實際紀錄"], default="回溯模擬",
                                           key=f"m_src_{sel}", label_visibility="collapsed") or "回溯模擬"
                R = live if src == "實際紀錄" else rec
                F = (EV.failures(live, full, rv["mkt"], rv["divs"], rv["inst"])
                     if src == "實際紀錄" and not live.empty else fails)
                if R.empty:
                    st.info("還沒有實際紀錄，每天打開會自動記錄，5 個交易日後出現第一筆成績。")
                else:
                    items = []
                    for _, s_ in EV.summary(R).iterrows():
                        diff = s_["落差≤5%"] - s_["對照：股價不動"]
                        items.append((f"{s_['週期']}・落差≤5%", EV.pct(s_["落差≤5%"]),
                                      f"股價不動 {EV.pct(s_['對照：股價不動'])}｜{s_['已驗證']} 筆",
                                      "#3e7bfa" if diff > 0.03 else "#f08c00" if diff < -0.03 else None))
                        items.append((f"{s_['週期']}・區間/方向", EV.pct(s_["落在預測區間"]),
                                      f"方向命中 {EV.pct(s_['方向命中'])}", None))
                    cards(items)
                    if not F.empty:
                        st.markdown("**失敗主要原因**")
                        ct = EV.cause_table(F)
                        rows_ = "".join(
                            f"<div style='display:flex;justify-content:space-between;padding:.25rem 0;"
                            f"border-bottom:1px solid rgba(128,128,128,.15)'><span>{k}</span>"
                            f"<span style='opacity:.8'>" +
                            "｜".join(f"{per} {int(ct.loc[k, per])}" for per in ("5 日", "20 日") if per in ct) +
                            "</span></div>" for k in ct.index)
                        st.markdown(rows_, unsafe_allow_html=True)
                        with st.expander("最近失敗明細"):
                            for _, f_ in F.sort_values("預測日", ascending=False).head(15).iterrows():
                                st.markdown(f"**{f_['預測日']:%m/%d}・{f_['週期']}** 落差 {f_['落差']:+.1%}・"
                                            f"{f_['主要原因']}<br><span style='font-size:.75rem;opacity:.7'>"
                                            f"{f_['細節']}</span>", unsafe_allow_html=True)
                md = EV.report_text(sel, name_of(sel), EV.summary(rec), fails, cal, live)
                with st.expander("📝 初報告", expanded=False):
                    st.markdown(md.split("\n", 2)[2] if md.count("\n") >= 2 else md)
                adopted = [f"{h}日" for h in EV.HS if cal.get(h) and cal[h]["adopted"]]
                st.caption("個股校正：" + ("、".join(adopted) + " 已通過驗證" if adopted else "未通過驗證，使用預設規則") +
                           ("（已套用）" if st.session_state.get("use_cal", True) and adopted else ""))
                st.download_button("⬇️ 下載初報告", EV.report_html(md, {"個股化預測邏輯": EV.calib_table(cal)}).encode("utf-8"),
                                   file_name=f"{sel}_預測檢討_{dt.date.today():%Y%m%d}.html", mime="text/html",
                                   width="stretch")

        with tabs[3]:
            inst = c_inst(sel, TOKEN) if tw else pd.DataFrame()
            if not tw:
                st.info("三大法人為台股資料，美股沒有。")
            elif inst.empty:
                st.warning("抓不到法人資料，稍後再試。")
            else:
                items = []
                for g in ("外資", "投信", "自營商", "合計"):
                    v, stk = inst[g].iloc[-1], D.streak(inst[g])
                    tag = f"連{'買' if stk > 0 else '賣'}{abs(stk)}日｜" if stk else ""
                    items.append((f"{g} {inst.index[-1]:%m/%d}", f"{v:+,.0f} 張",
                                  f"{tag}5日 {inst[g].iloc[-5:].sum():+,.0f}", color_of(v)))
                cards(items)
                st.plotly_chart(inst_fig(inst.iloc[-min(n, len(inst)):], full), width="stretch", config=PLOT_CFG)

        with tabs[4]:
            mg = c_margin(sel, TOKEN).dropna(subset=["融資餘額"]) if tw else pd.DataFrame()
            if not tw:
                st.info("融資融券為台股資料，美股沒有。")
            elif mg.empty:
                st.warning("抓不到融資融券資料。")
            else:
                lm = mg.iloc[-1]
                cards([
                    ("融資餘額", f"{lm['融資餘額']:,.0f} 張", f"增減 {lm['融資增減']:+,.0f}", None),
                    ("融券餘額", f"{lm['融券餘額']:,.0f} 張", f"增減 {lm['融券增減']:+,.0f}", None),
                    ("券資比", f"{lm['券資比%']:.2f}%" if pd.notna(lm["券資比%"]) else "-", "", None),
                    ("融資使用率", f"{lm['融資使用率%']:.2f}%" if pd.notna(lm["融資使用率%"]) else "-", "", None),
                ])
                st.plotly_chart(margin_fig(mg.iloc[-min(n, len(mg)):]), width="stretch", config=PLOT_CFG)

        with tabs[5]:
            a, y = hl["all"], hl["52w"]
            cards([
                ("歷史最高", f"{a['high']:,.2f}", f"{a['high_date']:%Y-%m-%d}｜距 {a['from_high']:+.1f}%", UP),
                ("歷史最低", f"{a['low']:,.2f}", f"{a['low_date']:%Y-%m-%d}｜距 {a['from_low']:+.1f}%", DOWN),
                ("52 週高", f"{y['high']:,.2f}", f"{y['high_date']:%Y-%m-%d}｜距 {y['from_high']:+.1f}%", UP),
                ("52 週低", f"{y['low']:,.2f}", f"{y['low_date']:%Y-%m-%d}｜距 {y['from_low']:+.1f}%", DOWN),
            ])
            st.plotly_chart(history_fig(full, hl), width="stretch", config=PLOT_CFG)
            st.caption(f"資料起點 {hl['first_date']:%Y-%m-%d}，對數座標。")

# ================================================================ 管理
st.divider()
with st.expander("⚙️ 管理自選股"):
    with st.form("add", clear_on_submit=True):
        new = st.text_input("新增代號", placeholder="2330, NVDA")
        if st.form_submit_button("加入", width="stretch") and new.strip():
            add = [D.normalize(x) for x in new.replace("，", ",").split(",") if x.strip()]
            cfg["watchlist"] += [c for c in add if c not in cfg["watchlist"]]
            save_cfg(cfg)
            st.rerun()
    keep = st.multiselect("目前清單（點 × 移除）", cfg["watchlist"], default=cfg["watchlist"])
    if keep != cfg["watchlist"]:
        cfg["watchlist"] = keep
        save_cfg(cfg)
        st.rerun()
    st.toggle("套用個股校正權重", value=True, key="use_cal")
    if st.button("🔄 重新抓取資料", width="stretch"):
        st.cache_data.clear()
        st.rerun()
st.caption("資料：Yahoo Finance、FinMind｜日線、收盤後更新｜紅漲綠跌")
