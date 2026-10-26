"""美台股看盤 Dashboard
執行： streamlit run app.py
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

UP, DOWN, FLAT = "#e5484d", "#2f9e66", "#8b8d98"          # 台股慣例：紅漲綠跌
INST_COLOR = {"外資": "#3e7bfa", "投信": "#f08c00", "自營商": "#9b59d0"}
MA_COLOR = {"MA5": "#f5a524", "MA20": "#3e7bfa", "MA60": "#9b59d0"}
RANGES = {"3個月": 63, "6個月": 126, "1年": 250, "2年": 500}

st.set_page_config(page_title="美台股看盤", page_icon="📈", layout="wide")
st.markdown("""
<style>
  .block-container {padding-top: 1.6rem; padding-bottom: 2rem;}
  [data-testid="stMetricValue"] {font-size: 1.35rem;}
  [data-testid="stMetricLabel"] p {font-size: .82rem; opacity: .75;}
  h3 {margin-bottom: .2rem;}
</style>
""", unsafe_allow_html=True)


# ================================================================ 設定檔
def load_cfg() -> dict:
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        return {**DEFAULT_CFG, **cfg}
    except Exception:
        return dict(DEFAULT_CFG)


def save_cfg(cfg: dict) -> None:
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


if "cfg" not in st.session_state:
    st.session_state.cfg = load_cfg()
cfg = st.session_state.cfg
TOKEN = cfg.get("finmind_token") or os.environ.get("FINMIND_TOKEN", "")


# ================================================================ 快取資料
@st.cache_data(ttl=86400, show_spinner=False)
def c_tw_info(token):
    df = D.tw_stock_info(token)
    if df.empty:
        return {}
    return {r.stock_id: (r.stock_name, r.type) for r in df.itertuples()}


@st.cache_data(ttl=86400, show_spinner=False)
def c_us_name(code):
    return D.us_name(code)


@st.cache_data(ttl=600, show_spinner=False)
def c_watch(codes: tuple, token: str):
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
    """回溯模擬 + 失敗原因 + 個股權重校正（並寫入 calib.json）。"""
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


def name_of(code: str) -> str:
    if D.is_tw(code):
        return c_tw_info(TOKEN).get(code, (code, None))[0]
    return c_us_name(code)


# ================================================================ 側邊欄
with st.sidebar:
    st.header("📋 自選股")
    with st.form("add", clear_on_submit=True):
        new = st.text_input("新增代號", placeholder="台股 2330 / 美股 NVDA，可用逗號一次多檔")
        if st.form_submit_button("加入", width="stretch") and new.strip():
            added = [D.normalize(x) for x in new.replace("，", ",").split(",") if x.strip()]
            cfg["watchlist"] += [c for c in added if c not in cfg["watchlist"]]
            save_cfg(cfg)
            st.rerun()

    keep = st.multiselect("目前清單（點 × 移除）", cfg["watchlist"], default=cfg["watchlist"])
    if keep != cfg["watchlist"]:
        cfg["watchlist"] = keep
        save_cfg(cfg)
        st.rerun()

    st.divider()
    rng_label = st.radio("K 線區間", list(RANGES), index=1, horizontal=True)
    log_scale = st.toggle("歷史走勢用對數座標", value=True)
    st.toggle("套用個股校正權重", value=True, key="use_cal",
              help="開啟後，通過驗證的個股會使用自己校正過的訊號權重與幅度係數（見「預測檢討」分頁）。")

    if st.button("🔄 重新抓取資料", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    with st.expander("FinMind Token（選填）"):
        st.caption("不填也能用（每小時約 300 次查詢）。到 finmindtrade.com 免費註冊可提高上限。")
        tok = st.text_input("Token", value=cfg.get("finmind_token", ""), type="password")
        if tok != cfg.get("finmind_token", ""):
            cfg["finmind_token"] = tok.strip()
            save_cfg(cfg)
            st.cache_data.clear()
            st.rerun()

    st.caption("資料來源：Yahoo Finance（報價/K線）、FinMind（台股法人/融資券）。"
               "日線資料，收盤後更新；盤中報價可能有延遲。顏色採台股慣例：紅漲綠跌。")


# ================================================================ 共用小工具
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


def watch_table(codes, frames, key):
    rows = []
    for c in codes:
        r = D.quote_row(c, frames.get(c))
        r["名稱"] = name_of(c)
        w5, w20 = json.loads(wkey_of(c))
        tr_ = T.analyze(frames.get(c), w5=w5, w20=w20)
        if tr_ is not None:
            PL.log(c, frames[c].index[-1], tr_, calibrated=bool(w5 or w20))
        r["5日趨勢"] = TC.badge(tr_, 5)
        r["20日趨勢"] = TC.badge(tr_, 20)
        rows.append(r)
    if not rows:
        st.info("清單是空的，從左側新增。")
        return None
    df = pd.DataFrame(rows)
    tw = D.is_tw(codes[0])
    cols = ["代號", "名稱", "收盤", "漲跌", "漲跌%", "5日趨勢", "20日趨勢", "成交量", "量比", "MA20乖離%", "距52週高%", "日期"]
    df = df.reindex(columns=cols)

    def paint(v):
        return f"color: {color_of(v)}; font-weight: 600" if pd.notna(v) else ""

    sty = (df.style
           .map(paint, subset=["漲跌", "漲跌%", "MA20乖離%"])
           .map(TC.badge_color, subset=["5日趨勢", "20日趨勢"])
           .format({"收盤": "{:,.2f}", "漲跌": "{:+,.2f}", "漲跌%": "{:+.2f}%",
                    "成交量": (lambda v: fmt_vol(v, tw)), "量比": "{:.2f}",
                    "MA20乖離%": "{:+.2f}%", "距52週高%": "{:+.2f}%"}, na_rep="-"))
    ev = st.dataframe(sty, hide_index=True, width="stretch",
                      on_select="rerun", selection_mode="single-row", key=key)
    rows_sel = list(ev.selection.rows) if ev and ev.selection else []
    last_key = f"_last_{key}"
    if rows_sel and rows_sel != st.session_state.get(last_key):
        st.session_state[last_key] = rows_sel
        return codes[rows_sel[0]]
    st.session_state[last_key] = rows_sel
    return None


# ================================================================ 圖表
def base_layout(fig, h):
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=30, b=8), hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11)),
                      font=dict(family="Microsoft JhengHei, PingFang TC, Noto Sans TC, sans-serif"))
    fig.update_xaxes(showgrid=False, type="category", nticks=10, rangeslider_visible=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False)
    return fig


def kline_fig(df, tw):
    x = df.index.strftime("%Y-%m-%d")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                        row_heights=[0.58, 0.2, 0.22],
                        subplot_titles=("", "成交量" + ("（張）" if tw else ""), "乖離率 %"))
    fig.add_trace(go.Candlestick(
        x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K線",
        increasing=dict(line=dict(color=UP), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN), fillcolor=DOWN), showlegend=False), 1, 1)
    for m, c in MA_COLOR.items():
        fig.add_trace(go.Scatter(x=x, y=df[m], name=m, line=dict(color=c, width=1.3)), 1, 1)

    # 區間高低點
    hi_i, lo_i = df["High"].values.argmax(), df["Low"].values.argmin()
    for i, val, lab, pos in ((hi_i, df["High"].iloc[hi_i], "區間高", "top center"),
                             (lo_i, df["Low"].iloc[lo_i], "區間低", "bottom center")):
        fig.add_trace(go.Scatter(x=[x[i]], y=[val], mode="markers+text", text=[f"{lab} {val:,.2f}"],
                                 textposition=pos, marker=dict(size=7, color="#888", symbol="diamond"),
                                 textfont=dict(size=11), showlegend=False, hoverinfo="skip"), 1, 1)

    vol = df["Volume"] / 1000 if tw else df["Volume"]
    vcol = [UP if c >= o else DOWN for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(go.Bar(x=x, y=vol, marker_color=vcol, name="成交量", showlegend=False,
                         marker_line_width=0), 2, 1)
    vma = df["VOL_MA5"] / 1000 if tw else df["VOL_MA5"]
    fig.add_trace(go.Scatter(x=x, y=vma, name="均量5", line=dict(color="#f5a524", width=1)), 2, 1)

    for n, c in ((5, "#f5a524"), (20, "#3e7bfa"), (60, "#9b59d0")):
        fig.add_trace(go.Scatter(x=x, y=df[f"BIAS{n}"], name=f"乖離{n}", line=dict(color=c, width=1.2),
                                 showlegend=False), 3, 1)
    fig.add_hline(y=0, line=dict(color="rgba(128,128,128,.5)", width=1, dash="dot"), row=3, col=1)
    base_layout(fig, 720)
    fig.update_annotations(font_size=11, x=0, xanchor="left")
    return fig


def inst_fig(inst, px):
    x = inst.index.strftime("%Y-%m-%d")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for g in ("外資", "投信", "自營商"):
        fig.add_trace(go.Bar(x=x, y=inst[g], name=g, marker_color=INST_COLOR[g], marker_line_width=0),
                      secondary_y=False)
    if px is not None and not px.empty:
        p = px["Close"].reindex(inst.index)
        fig.add_trace(go.Scatter(x=x, y=p, name="收盤價", line=dict(color="#888", width=1.6)),
                      secondary_y=True)
    fig.update_layout(barmode="relative")
    base_layout(fig, 420)
    fig.update_yaxes(title_text="買賣超（張）", secondary_y=False)
    fig.update_yaxes(title_text="股價", secondary_y=True, showgrid=False)
    return fig


def margin_fig(m):
    x = m.index.strftime("%Y-%m-%d")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.68, 0.32],
                        specs=[[{"secondary_y": True}], [{}]], subplot_titles=("", "券資比 %"))
    fig.add_trace(go.Scatter(x=x, y=m["融資餘額"], name="融資餘額", line=dict(color=UP, width=1.8)),
                  1, 1, secondary_y=False)
    fig.add_trace(go.Scatter(x=x, y=m["融券餘額"], name="融券餘額（右軸）", line=dict(color=DOWN, width=1.8)),
                  1, 1, secondary_y=True)
    fig.add_trace(go.Scatter(x=x, y=m["券資比%"], name="券資比", line=dict(color="#9b59d0", width=1.4),
                             fill="tozeroy", fillcolor="rgba(155,89,208,.12)", showlegend=False), 2, 1)
    base_layout(fig, 480)
    fig.update_yaxes(title_text="融資（張）", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="融券（張）", row=1, col=1, secondary_y=True, showgrid=False)
    fig.update_annotations(font_size=11, x=0, xanchor="left")
    return fig


def history_fig(full, hl, log):
    fig = go.Figure(go.Scatter(x=full.index, y=full["Close"], name="收盤", line=dict(color="#3e7bfa", width=1.3)))
    a = hl["all"]
    for d, v, lab, col, pos in ((a["high_date"], a["high"], "歷史高", UP, "top left"),
                                (a["low_date"], a["low"], "歷史低", DOWN, "bottom right")):
        fig.add_trace(go.Scatter(x=[d], y=[v], mode="markers+text", text=[f"{lab} {v:,.2f}<br>{d:%Y-%m-%d}"],
                                 textposition=pos, marker=dict(size=9, color=col), showlegend=False))
    fig.update_layout(height=380, margin=dict(l=8, r=8, t=20, b=8), hovermode="x unified",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
    fig.update_yaxes(type="log" if log else "linear", gridcolor="rgba(128,128,128,.15)")
    fig.update_xaxes(showgrid=False)
    return fig


# ================================================================ 主畫面
st.title("📈 美台股看盤")
codes = cfg["watchlist"]
tw_codes = [c for c in codes if D.is_tw(c)]
us_codes = [c for c in codes if not D.is_tw(c)]

with st.spinner("抓取自選股報價…"):
    frames = c_watch(tuple(codes), TOKEN) if codes else {}

if "selected" not in st.session_state or st.session_state.selected not in codes:
    st.session_state.selected = codes[0] if codes else None

c1, c2 = st.columns(2, gap="large")
with c1:
    st.subheader("🇹🇼 台股")
    pick = watch_table(tw_codes, frames, "tw_tbl") if tw_codes else st.caption("尚無台股")
    if isinstance(pick, str):
        st.session_state.selected = pick
with c2:
    st.subheader("🇺🇸 美股")
    pick = watch_table(us_codes, frames, "us_tbl") if us_codes else st.caption("尚無美股")
    if isinstance(pick, str):
        st.session_state.selected = pick
st.caption("點表格任一列，可在下方看個股明細。量比 = 今日量 ÷ 前 5 日均量。")

if not codes:
    st.stop()

st.divider()

# ---------------------------------------------------------------- 個股明細
labels = {c: f"{c}　{name_of(c)}" for c in codes}
sel = st.selectbox("個股明細", codes, index=codes.index(st.session_state.selected),
                   format_func=lambda c: labels[c])
st.session_state.selected = sel
tw = D.is_tw(sel)

with st.spinner("載入個股資料…"):
    full = c_hist(sel, "max", TOKEN)

if full.empty:
    st.error(f"抓不到 {sel} 的價格資料，請確認代號是否正確，或稍後按「重新抓取資料」。")
    st.stop()

ind = D.add_indicators(full)
view = ind.iloc[-RANGES[rng_label]:]
last, prev = ind.iloc[-1], ind.iloc[-2]
chg = last["Close"] - prev["Close"]
hl = D.high_low_stats(full)

st.markdown(f"### {labels[sel]}　<span style='font-size:.85rem;opacity:.6'>"
            f"{'台股' if tw else '美股'} · 資料日期 {ind.index[-1]:%Y-%m-%d}</span>", unsafe_allow_html=True)

m = st.columns(6)
m[0].metric("收盤", f"{last['Close']:,.2f}", f"{chg:+,.2f} ({chg / prev['Close'] * 100:+.2f}%)",
            delta_color="inverse")
vol_ratio = last["Volume"] / ind["Volume"].iloc[-6:-1].mean()
m[1].metric("成交量", fmt_vol(last["Volume"] / 1000 if tw else last["Volume"], tw), f"量比 {vol_ratio:.2f}",
            delta_color="off")
m[2].metric("MA5 乖離", f"{last['BIAS5']:+.2f}%")
m[3].metric("MA20 乖離", f"{last['BIAS20']:+.2f}%")
m[4].metric("52 週高", f"{hl['52w']['high']:,.2f}", f"{hl['52w']['from_high']:+.2f}%", delta_color="off")
m[5].metric("歷史高", f"{hl['all']['high']:,.2f}", f"{hl['all']['from_high']:+.2f}%", delta_color="off")

tabs = st.tabs(["K線・成交量・乖離", "🔮 趨勢預測", "📋 預測檢討", "三大法人", "融資融券", "歷史高低點"])

# ---- K 線
with tabs[0]:
    st.plotly_chart(kline_fig(view, tw), width="stretch", config={"displaylogo": False})
    b = st.columns(3)
    for col, n in zip(b, (5, 20, 60)):
        v = last[f"BIAS{n}"]
        series = ind[f"BIAS{n}"].dropna().iloc[-500:]
        pct = (series < v).mean() * 100 if len(series) else float("nan")
        col.metric(f"{n}日乖離率", f"{v:+.2f}%", f"近兩年第 {pct:.0f} 百分位", delta_color="off")
    st.caption("乖離率 = (收盤 − 均線) ÷ 均線。百分位越高代表目前偏離程度在近兩年中越極端（偏多）；越低越偏空。")

# ---- 趨勢預測
with tabs[1]:
    with st.spinner("分析型態與回測中…"):
        tr, bt = c_trend(sel, TOKEN, wkey_of(sel))
    if not tr:
        st.warning("資料不足 80 個交易日，無法分析趨勢。")
    else:
        def horizon_box(col, days, sc, lab, strength, p):
            end = p["mid"].iloc[-1]
            with col.container(border=True):
                st.markdown(f"<div style='font-size:.82rem;opacity:.75'>未來 {days} 日判斷（至 {p.index[-1]:%m/%d}）</div>"
                            f"<div style='font-size:1.5rem;font-weight:700;color:{TC.score_color(sc)}'>"
                            f"{TC.badge(tr, days).split(' ')[0]} {lab}</div>", unsafe_allow_html=True)
                a, b, c_ = st.columns(3)
                a.metric("分數", f"{sc:+.1f}", f"強度 {strength}%", delta_color="off")
                b.metric("預估價", f"{end:,.2f}", f"{(end / tr['close'] - 1) * 100:+.2f}%", delta_color="inverse")
                c_.metric("可能區間", f"{p['lo'].iloc[-1]:,.1f}", f"~ {p['hi'].iloc[-1]:,.1f}", delta_color="off")

        hb = st.columns([5, 5, 2])
        horizon_box(hb[0], 5, tr["score"], tr["label"], tr["strength"], tr["proj"])
        horizon_box(hb[1], 20, tr["score20"], tr["label20"], tr["strength20"], tr["proj20"])
        with hb[2]:
            st.metric("支撐", f"{tr['support']:,.2f}" if tr["support"] else "-")
            st.metric("壓力", f"{tr['resistance']:,.2f}" if tr["resistance"] else "-")

        st.plotly_chart(TC.trend_fig(ind, tr, tw), width="stretch", config={"displaylogo": False})

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown("**觸發的訊號**")
            sig = TC.signals_table(tr)
            if sig.empty:
                st.caption("目前沒有明顯型態或訊號。")
            else:
                st.dataframe(sig.style.map(lambda v: f"color:{color_of(v)};font-weight:600", subset=["分數"])
                             .map(lambda v: f"color:{UP}" if v == "▲" else f"color:{DOWN}" if v == "▼" else "",
                                  subset=["方向"]).format({"分數": "{:+.1f}"}),
                             hide_index=True, width="stretch")
            if tr["targets"]:
                st.caption("型態目標價（量測法，可能需要超過 5 日才到）：" +
                           "、".join(f"{n} {v:,.2f}" for n, v in tr["targets"]))
        with right:
            st.markdown("**這套規則在這檔股票的過去表現**")
            if bt:
                for days in (5, 20):
                    s_ = bt.get(days)
                    if not s_:
                        st.caption(f"{days} 日：歷史資料不足。")
                        continue
                    b1, b2, b3 = st.columns(3)
                    b1.metric(f"{days}日・偏多 {s_['bull_n']} 次",
                              f"{s_['bull_hit'] * 100:.0f}%" if s_["bull_hit"] is not None else "-",
                              f"平均報酬 {s_['bull_ret'] * 100:+.2f}%" if s_["bull_ret"] is not None else "",
                              delta_color="off")
                    b2.metric(f"{days}日・偏空 {s_['bear_n']} 次",
                              f"{s_['bear_hit'] * 100:.0f}%" if s_["bear_hit"] is not None else "-",
                              f"平均報酬 {s_['bear_ret'] * 100:+.2f}%" if s_["bear_ret"] is not None else "",
                              delta_color="off")
                    b3.metric(f"{days}日・任意持有", f"{s_['base_up'] * 100:.0f}%", "上漲機率（對照組）",
                              delta_color="off")
                st.caption(f"命中率 = 發出訊號後 N 日方向猜對的比例（近 {bt[5]['days']} 個交易日）。"
                           "偏多命中率要明顯高於「任意持有」的上漲機率，訊號才算有用。"
                           "20 日的樣本彼此重疊較多，數字參考即可。")
            else:
                st.caption("歷史資料不足，無法回測。")
        with st.expander("評分規則說明"):
            st.markdown("""
| 類別 | 訊號 | 分數 |
|---|---|---|
| 反轉型態 | M頭、頭肩頂、三重頂 / W底、頭肩底、三重底 | 跌破/突破頸線 ±3，形成中 ±1.5 |
| 反轉型態 | 倒V型反轉、菱形頂 | −2 ~ −2.5 |
| 整理型態 | 上升三角形、下降楔形、上升旗形 / 下降三角形、上升楔形、下跌旗形 | 突破確認 ±3.5，形成中 ±1.2 |
| 整理型態 | 三角收斂、箱型盤整、擴散三角形、通道 | 等待突破，突破後 ±1.5 |
| 水平線 | 突破壓力 / 跌破支撐（反彈兩次以上的價位） | ±2 |
| 水平線 | 支撐反彈 / 壓力回落 | ±1 |
| 均線 | 黃金交叉 / 死亡交叉（MA5/20、MA20/60） | ±2；長均線方向相反時 0（不要買/不要賣） |
| 均線 | 多頭排列 / 空頭排列 | ±1 |
| K線 | 晨星、黃昏星 | ±2 |
| K線 | 錘頭線、吞噬、黑三兵 | ±1.5 |
| K線 | 大陽/大陰線 ±1、中陽/中陰 ±0.5、長上影 −0.5、長下影 +0.5、十字線 ±0.5 | |

**判斷**：總分 ≥ +5 強烈偏多、≥ +2 偏多、≤ −2 偏空、≤ −5 強烈偏空，其餘盤整觀望。
**5 日預估**：每日移動 = 方向 × 強度 × 0.25 ATR × 個股幅度係數；區間 = ±0.9 ATR × √天數。
**20 日預估**：K 棒與組合 K 線屬短線訊號，只計 30% 權重，型態、均線、水平線全額計分；
訊號影響每天遞減（×0.92），若有同方向的型態目標價，路徑會往目標價靠攏一半。
""")
        st.caption("⚠️ 這是依技術分析規則計算的機率傾向，不是保證，也不構成投資建議。消息面、財報、大盤急跌都可能讓型態失效。")

# ---- 預測檢討
OKC, BADC = "#3e7bfa", "#f08c00"


def review_fig(R):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("5 日預測落差（實際 ÷ 預估 − 1）", "20 日預測落差"))
    for row, h in ((1, 5), (2, 20)):
        d = R[R[f"act{h}"].notna()]
        if d.empty:
            continue
        ok = d[f"ok{h}"] == 1
        for mask, nm, col in ((ok, "成功（≤5%）", OKC), (~ok, "失敗（>5%）", BADC)):
            fig.add_trace(go.Scatter(x=d.index[mask], y=d[f"err{h}"][mask] * 100, mode="markers", name=nm,
                                     marker=dict(color=col, size=5), showlegend=row == 1,
                                     hovertemplate="%{x|%Y-%m-%d}<br>落差 %{y:+.1f}%<extra></extra>"), row, 1)
        for yv in (5, -5):
            fig.add_hline(y=yv, line=dict(color="rgba(128,128,128,.6)", dash="dash", width=1), row=row, col=1)
    fig.update_layout(height=430, margin=dict(l=8, r=8, t=36, b=8), hovermode="closest",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.1, x=0))
    fig.update_yaxes(ticksuffix="%", gridcolor="rgba(128,128,128,.15)", zeroline=False)
    fig.update_xaxes(showgrid=False)
    fig.update_annotations(font_size=11, x=0, xanchor="left")
    return fig


def cause_fig(fails):
    t = EV.cause_table(fails)
    fig = go.Figure()
    for per, col in (("5 日", OKC), ("20 日", BADC)):
        if per in t:
            fig.add_trace(go.Bar(y=t.index, x=t[per], name=per, orientation="h", marker_color=col,
                                 hovertemplate="%{y}：%{x} 筆<extra>" + per + "</extra>"))
    fig.update_layout(barmode="group", height=60 + 34 * max(len(t), 1), margin=dict(l=8, r=8, t=10, b=8),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.08, x=0), yaxis=dict(autorange="reversed"))
    fig.update_xaxes(title_text="失敗筆數", gridcolor="rgba(128,128,128,.15)")
    return fig


def fmt_summary(summ):
    t = summ.copy()
    for c_ in ("落差≤5%", "對照：股價不動", "落在預測區間", "方向命中"):
        t[c_] = t[c_].map(EV.pct)
    return t


with tabs[2]:
    with st.spinner("回溯模擬過去 500 個交易日的每日預測、分析失敗原因、校正個股權重…（第一次約 5～15 秒）"):
        rv = c_review(sel, TOKEN)
    if rv is None:
        st.warning("歷史資料不足 160 個交易日，無法檢討。")
    else:
        rec, fails, cal = rv["rec"], rv["fails"], rv["cal"]
        live = PL.load(sel, full)
        src = st.segmented_control("資料來源", ["回溯模擬（過去 500 日）", "實際紀錄（從開始使用起）"],
                                   default="回溯模擬（過去 500 日）", key=f"rv_src_{sel}") or "回溯模擬（過去 500 日）"
        is_live = src.startswith("實際")
        R = live if is_live else rec
        F = EV.failures(live, full, rv["mkt"], rv["divs"], rv["inst"]) if is_live and not live.empty else fails
        st.caption("判定：|實際收盤 ÷ 預估價 − 1| ≤ 5% 為成功。對照組「股價不動」= 假設 N 日後價格不變的命中率，"
                   "模型要高於它才有價值；「落在預測區間」為依個股波動調整的標準。")

        if R.empty:
            st.info("還沒有實際紀錄。每天打開看盤時會自動記錄當天的預測，5 個交易日後開始出現第一筆成績。")
        else:
            st.dataframe(fmt_summary(EV.summary(R)), hide_index=True, width="stretch")
            st.plotly_chart(review_fig(R), width="stretch", config={"displaylogo": False})

            with st.expander("每日預測明細（最近 60 個預測日）", expanded=False):
                dt_ = EV.daily_table(R, 60)
                st.dataframe(dt_.style.format({"5日落差": "{:+.1%}", "20日落差": "{:+.1%}",
                                               "5日預估": "{:,.2f}", "5日實際": "{:,.2f}",
                                               "20日預估": "{:,.2f}", "20日實際": "{:,.2f}", "收盤": "{:,.2f}"},
                                              na_rep="-"), hide_index=True, width="stretch")

            st.markdown("#### 失敗原因（落差 > 5%）")
            if F.empty:
                st.success("沒有落差超過 5% 的預測。")
            else:
                cl_, cr_ = st.columns([2, 3], gap="large")
                with cl_:
                    st.plotly_chart(cause_fig(F), width="stretch", config={"displaylogo": False})
                with cr_:
                    ft = F.sort_values("預測日", ascending=False).head(80).copy()
                    ft["預測日"] = ft["預測日"].dt.strftime("%Y-%m-%d")
                    st.dataframe(ft[["預測日", "週期", "落差", "主要原因", "細節"]]
                                 .style.format({"落差": "{:+.1%}"}), hide_index=True, width="stretch", height=320)

        st.markdown("#### 📝 初報告")
        md = EV.report_text(sel, name_of(sel), EV.summary(rec), fails, cal, live)
        with st.container(border=True):
            st.markdown(md.split("\n", 2)[2] if md.count("\n") >= 2 else md)

        st.markdown("#### ⚙️ 個股化預測邏輯")
        ct = EV.calib_table(cal)
        st.dataframe(ct, hide_index=True, width="stretch")
        on = st.session_state.get("use_cal", True)
        adopted = [f"{h} 日" for h in EV.HS if cal.get(h) and cal[h]["adopted"]]
        st.caption(("目前" + ("已" if on else "未") + "套用個股校正（側邊欄可切換）。") +
                   (f"通過驗證：{'、'.join(adopted)}。" if adopted else "兩個週期都沒有通過驗證，維持預設規則。") +
                   " 權重倍數 0 = 這類訊號對此股無預測力；幅度係數 < 1 = 預估移動幅度調小。")

        tables = {"預測成績": fmt_summary(EV.summary(rec)),
                  "個股化預測邏輯": ct,
                  "失敗明細（最近 100 筆）": (fails.sort_values("預測日", ascending=False).head(100)
                                          .assign(預測日=lambda x: x["預測日"].dt.strftime("%Y-%m-%d"),
                                                  落差=lambda x: x["落差"].map("{:+.1%}".format))
                                          [["預測日", "週期", "收盤", "預估價", "實際收盤", "落差", "主要原因", "細節"]]
                                          .round(2) if not fails.empty else pd.DataFrame()),
                  "每日預測明細（最近 60 日）": EV.daily_table(rec, 60).assign(
                      **{"5日落差": lambda x: x["5日落差"].map(lambda v: "-" if pd.isna(v) else f"{v:+.1%}"),
                         "20日落差": lambda x: x["20日落差"].map(lambda v: "-" if pd.isna(v) else f"{v:+.1%}")})}
        d1, d2, d3 = st.columns(3)
        d1.download_button("⬇️ 下載初報告（HTML）", EV.report_html(md, tables).encode("utf-8"),
                           file_name=f"{sel}_預測檢討_{dt.date.today():%Y%m%d}.html", mime="text/html",
                           width="stretch")
        d2.download_button("⬇️ 每日預測紀錄（CSV）",
                           rec.drop(columns=["pos"]).to_csv(encoding="utf-8-sig").encode("utf-8-sig"),
                           file_name=f"{sel}_回溯預測_{dt.date.today():%Y%m%d}.csv", mime="text/csv",
                           width="stretch")
        if d3.button("🔄 重新檢討這檔", width="stretch"):
            c_review.clear()
            st.rerun()

# ---- 三大法人
with tabs[3]:
    if not tw:
        st.info("三大法人買賣超為台股資料，美股沒有對應的公開日資料。")
    else:
        inst = c_inst(sel, TOKEN)
        if inst.empty:
            st.warning("抓不到法人資料（ETF 或剛上市個股可能沒有，或 FinMind 查詢次數已達上限，稍後再試）。")
        else:
            k = st.columns(4)
            for i, g in enumerate(("外資", "投信", "自營商", "合計")):
                today = inst[g].iloc[-1]
                s5, s20 = inst[g].iloc[-5:].sum(), inst[g].iloc[-20:].sum()
                stk = D.streak(inst[g])
                tag = f"連{'買' if stk > 0 else '賣'} {abs(stk)} 日" if stk else ""
                k[i].metric(f"{g}（{inst.index[-1]:%m/%d}）", f"{today:+,.0f} 張",
                            f"5日 {s5:+,.0f}｜20日 {s20:+,.0f}", delta_color="off")
                if tag:
                    k[i].markdown(f"<span style='color:{color_of(stk)};font-size:.85rem;font-weight:600'>"
                                  f"{tag}</span>", unsafe_allow_html=True)
            n_days = min(len(inst), RANGES[rng_label])
            st.plotly_chart(inst_fig(inst.iloc[-n_days:], full), width="stretch",
                            config={"displaylogo": False})
            with st.expander("近 20 日明細"):
                t = inst.iloc[-20:][::-1].copy()
                t.index = t.index.strftime("%Y-%m-%d")
                st.dataframe(t.style.map(lambda v: f"color:{color_of(v)}").format("{:+,.0f}"),
                             width="stretch")

# ---- 融資融券
with tabs[4]:
    if not tw:
        st.info("融資融券為台股資料，美股沒有對應的公開日資料。")
    else:
        mg = c_margin(sel, TOKEN).dropna(subset=["融資餘額"])
        if mg.empty:
            st.warning("抓不到融資融券資料（部分 ETF 或不可信用交易的個股沒有資料）。")
        else:
            lm = mg.iloc[-1]
            k = st.columns(4)
            k[0].metric("融資餘額", f"{lm['融資餘額']:,.0f} 張", f"{lm['融資增減']:+,.0f}", delta_color="inverse")
            k[1].metric("融券餘額", f"{lm['融券餘額']:,.0f} 張", f"{lm['融券增減']:+,.0f}", delta_color="inverse")
            k[2].metric("券資比", f"{lm['券資比%']:.2f}%" if pd.notna(lm["券資比%"]) else "-")
            k[3].metric("融資使用率", f"{lm['融資使用率%']:.2f}%" if pd.notna(lm["融資使用率%"]) else "-")
            n_days = min(len(mg), RANGES[rng_label])
            st.plotly_chart(margin_fig(mg.iloc[-n_days:]), width="stretch",
                            config={"displaylogo": False})
            with st.expander("近 20 日明細"):
                t = mg.iloc[-20:][::-1][["融資餘額", "融資增減", "融券餘額", "融券增減", "券資比%", "融資使用率%"]]
                t.index = t.index.strftime("%Y-%m-%d")
                st.dataframe(t.style.format("{:,.0f}", subset=["融資餘額", "融資增減", "融券餘額", "融券增減"])
                             .format("{:.2f}", subset=["券資比%", "融資使用率%"], na_rep="-"),
                             width="stretch")

# ---- 歷史高低點
with tabs[5]:
    a, y = hl["all"], hl["52w"]
    tbl = pd.DataFrame({
        "區間": ["歷史（上市以來）", "近 52 週"],
        "最高價": [a["high"], y["high"]],
        "最高日期": [f"{a['high_date']:%Y-%m-%d}", f"{y['high_date']:%Y-%m-%d}"],
        "距最高": [a["from_high"], y["from_high"]],
        "最低價": [a["low"], y["low"]],
        "最低日期": [f"{a['low_date']:%Y-%m-%d}", f"{y['low_date']:%Y-%m-%d}"],
        "距最低": [a["from_low"], y["from_low"]],
    })
    st.dataframe(tbl.style.format({"最高價": "{:,.2f}", "最低價": "{:,.2f}",
                                   "距最高": "{:+.2f}%", "距最低": "{:+.2f}%"}),
                 hide_index=True, width="stretch")
    st.plotly_chart(history_fig(full, hl, log_scale), width="stretch", config={"displaylogo": False})
    st.caption(f"歷史資料起點：{hl['first_date']:%Y-%m-%d}（Yahoo Finance 可取得的最早日期）。"
               "價格已含分割調整，未做除權息還原。")
