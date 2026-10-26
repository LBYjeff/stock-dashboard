"""趨勢預測圖：K 線 + 型態線 + 水平支撐壓力 + 未來 5 日推估區間（桌面版、手機版共用）。"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

UP, DOWN = "#e5484d", "#2f9e66"
ARROW = {1: "▲", -1: "▼", 0: "●"}


def badge(tr, days: int = 5) -> str:
    """自選股表格用的短標籤，例如 ▲ 偏多 +3.5（days=20 用中期分數）"""
    if not tr:
        return "-"
    s = tr["score"] if days == 5 else tr["score20"]
    lab = tr["label"] if days == 5 else tr["label20"]
    a = "▲" if s >= 2 else "▼" if s <= -2 else "●"
    return f"{a} {lab} {s:+.1f}"


def score_color(s) -> str:
    return UP if s >= 2 else DOWN if s <= -2 else "#8b8d98"


def badge_color(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if text.startswith("▲"):
        return f"color:{UP};font-weight:600"
    if text.startswith("▼"):
        return f"color:{DOWN};font-weight:600"
    return "color:#8b8d98"


def trend_fig(ind: pd.DataFrame, tr: dict, tw: bool, bars: int = 90, mobile: bool = False) -> go.Figure:
    win = ind.iloc[-bars:]
    idx = tr["index"]                          # analyze() 使用的日期序列（以位置對應型態線）
    n = len(idx)
    ws = n - len(win)                          # 視窗起點在 idx 中的位置
    x = list(win.index.strftime("%Y-%m-%d"))
    fut = list(tr["proj20"].index.strftime("%Y-%m-%d"))      # 20 日涵蓋 5 日
    allx = x + fut

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=x, open=win["Open"], high=win["High"], low=win["Low"], close=win["Close"], name="K線",
        increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
        decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN), showlegend=False))
    for m, c in (("MA5", "#f5a524"), ("MA20", "#3e7bfa"), ("MA60", "#9b59d0")):
        if m in win:
            fig.add_trace(go.Scatter(x=x, y=win[m], name=m, line=dict(color=c, width=1), opacity=.8))

    # 型態線（裁切到視窗內）
    seen = set()
    for ln in tr["lines"]:
        i0, i1, y0, y1 = ln["i0"], ln["i1"], ln["y0"], ln["y1"]
        if i1 < ws:
            continue
        if i0 < ws:
            slope = (y1 - y0) / max(i1 - i0, 1)
            y0, i0 = y0 + slope * (ws - i0), ws
        show = ln["label"] not in seen
        seen.add(ln["label"])
        fig.add_trace(go.Scatter(x=[allx[i0 - ws], allx[i1 - ws]], y=[y0, y1], mode="lines",
                                 name=ln["label"], showlegend=show and not mobile,
                                 line=dict(color=ln["color"], width=1.6, dash=ln["dash"])))

    # 水平支撐壓力
    for val, name, col in ((tr.get("support"), "支撐", UP), (tr.get("resistance"), "壓力", DOWN)):
        if val:
            fig.add_trace(go.Scatter(x=[allx[0], allx[-1]], y=[val, val], mode="lines", name=f"{name} {val:,.2f}",
                                     line=dict(color=col, width=1, dash="longdash"), opacity=.7,
                                     showlegend=not mobile))

    # 未來 20 日 / 5 日推估（20 日畫在底層、較淡）
    last_x, last_c = x[-1], float(win["Close"].iloc[-1])
    for key, sc, nm, alpha, dash, width in (("proj20", tr["score20"], "20日", .08, "dot", 1.6),
                                            ("proj", tr["score"], "5日", .18, "dash", 2.2)):
        p = tr[key]
        px = [last_x] + list(p.index.strftime("%Y-%m-%d"))
        hi = [last_c] + list(p["hi"]); lo = [last_c] + list(p["lo"]); mid = [last_c] + list(p["mid"])
        col = score_color(sc)
        rgba = {UP: "229,72,77", DOWN: "47,158,102"}.get(col, "139,141,152")
        fig.add_trace(go.Scatter(x=px, y=hi, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=px, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                                 fillcolor=f"rgba({rgba},{alpha})", name=f"{nm}區間", showlegend=not mobile,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=px, y=mid, mode="lines+markers", name=f"{nm}預估",
                                 line=dict(color=col, width=width, dash=dash), marker=dict(size=3 if nm == "20日" else 4),
                                 hovertemplate=f"{nm}預估 %{{y:,.2f}}<extra></extra>"))
    fig.add_vline(x=last_x, line=dict(color="rgba(128,128,128,.4)", width=1, dash="dot"))

    fig.update_layout(height=420 if mobile else 560, margin=dict(l=4 if mobile else 8, r=4 if mobile else 8, t=26, b=4),
                      hovermode="x unified", dragmode=False if mobile else "zoom",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
                      font=dict(size=10 if mobile else 11,
                                family="Microsoft JhengHei, PingFang TC, Noto Sans TC, sans-serif"))
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=allx, showgrid=False,
                     nticks=4 if mobile else 10, rangeslider_visible=False, fixedrange=mobile)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.15)", zeroline=False, fixedrange=mobile)
    return fig


def signals_table(tr: dict) -> pd.DataFrame:
    return pd.DataFrame([{"方向": ARROW[s["dir"]], "訊號": s["name"], "類別": s["kind"],
                          "分數": s["w"], "說明": s["note"]} for s in tr["signals"]])
