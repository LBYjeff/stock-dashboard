"""產生每日「看盤快照」靜態網站 → site/

  python build_site.py [--repo 帳號/儲存庫]

site/
  index.html, app.css, app.js, plotly-finance.min.js, manifest.json, icon-*.png  （從 web/ 複製）
  data/summary.json   所有清單、每檔報價與各模型預測（首頁用）
  data/<代號>.json     個股明細：圖表、預測、模型比較、檢討、初報告
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

import data as D
import evaluate as EV
import pipeline as P
import store as S
import trend as T
import trend_chart as TC

ROOT = Path(__file__).parent
WEB, SITE = ROOT / "web", ROOT / "site"
RED, GREEN, BLUE, ORANGE, PURPLE = "#FF3B30", "#34C759", "#007AFF", "#FF9500", "#AF52DE"
UP, DOWN = RED, GREEN
TC.UP, TC.DOWN = UP, DOWN
FONT = "-apple-system, BlinkMacSystemFont, 'PingFang TC', 'Helvetica Neue', 'Microsoft JhengHei', sans-serif"


# ================================================================ 工具
def clean(o):
    """轉成可 JSON 化：NaN → null、numpy → python、時間 → 字串。"""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if (math.isnan(o) or math.isinf(o)) else round(float(o), 6)
    if isinstance(o, (pd.Timestamp, dt.date, dt.datetime)):
        return str(o)[:10] if not isinstance(o, dt.datetime) or o.hour == 0 else str(o)[:16]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if o is pd.NaT:
        return None
    return o


def fig_json(fig):
    fig.update_layout(template="none")            # 不帶 Python 預設樣板，檔案小很多
    fig.update_xaxes(automargin=True)             # 刻度文字自動留邊，不會被切掉
    fig.update_yaxes(automargin=True)
    return json.loads(pio.to_json(fig, validate=False, engine="json"))


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
    x = list(df.index.strftime("%Y-%m-%d"))
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04, row_heights=[0.6, 0.18, 0.22],
                        subplot_titles=("", "成交量", "乖離率 %"))
    fig.add_trace(go.Candlestick(x=x, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="K",
                                 increasing=dict(line=dict(color=UP, width=1), fillcolor=UP),
                                 decreasing=dict(line=dict(color=DOWN, width=1), fillcolor=DOWN), showlegend=False), 1, 1)
    for m, c in (("MA5", ORANGE), ("MA20", BLUE), ("MA60", PURPLE)):
        fig.add_trace(go.Scatter(x=x, y=df[m].round(3), name=m, line=dict(color=c, width=1.2)), 1, 1)
    vol = df["Volume"] / 1000 if tw else df["Volume"]
    fig.add_trace(go.Bar(x=x, y=vol.round(0), marker_color=[UP if c >= o else DOWN for o, c in zip(df["Open"], df["Close"])],
                         marker_line_width=0, showlegend=False, name="量"), 2, 1)
    for n, c in ((5, ORANGE), (20, BLUE), (60, PURPLE)):
        fig.add_trace(go.Scatter(x=x, y=df[f"BIAS{n}"].round(3), line=dict(color=c, width=1.1), showlegend=False,
                                 name=f"乖離{n}"), 3, 1)
    fig.add_hline(y=0, line=dict(color="rgba(142,142,147,.6)", width=1, dash="dot"), row=3, col=1)
    ios_layout(fig, 500)
    fig.update_xaxes(type="category", nticks=5, rangeslider_visible=False)
    return fig


def review_fig(R):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("5 日預測落差", "20 日預測落差"))
    for row, h in ((1, 5), (2, 20)):
        d = R[R[f"act{h}"].notna()]
        if d.empty:
            continue
        ok = d[f"ok{h}"] == 1
        for mask, nm, col in ((ok, "成功 ≤5%", BLUE), (~ok, "失敗 >5%", ORANGE)):
            fig.add_trace(go.Scatter(x=list(d.index[mask].strftime("%Y-%m-%d")), y=(d[f"err{h}"][mask] * 100).round(2),
                                     mode="markers", name=nm, marker=dict(color=col, size=4), showlegend=row == 1,
                                     hovertemplate="%{x} 落差 %{y:+.1f}%<extra></extra>"), row, 1)
        for yv in (5, -5):
            fig.add_hline(y=yv, line=dict(color="rgba(142,142,147,.7)", dash="dash", width=1), row=row, col=1)
    ios_layout(fig, 400)
    fig.update_layout(hovermode="closest", margin=dict(l=6, r=6, t=48, b=6),
                      legend=dict(orientation="h", y=1.16, x=1, xanchor="right"))
    fig.update_yaxes(ticksuffix="%")
    return fig


def inst_fig(inst):
    ii = inst.iloc[-60:]
    fig = go.Figure()
    for g, col in (("外資", BLUE), ("投信", ORANGE), ("自營商", PURPLE)):
        fig.add_trace(go.Bar(x=list(ii.index.strftime("%m/%d")), y=ii[g], name=g, marker_color=col, marker_line_width=0))
    fig.update_layout(barmode="relative")
    ios_layout(fig, 280)
    fig.update_xaxes(type="category", nticks=5)
    return fig


def history_fig(px, hl):
    s = px["Close"]
    if len(s) > 1500:                      # 長歷史抽樣，縮小檔案
        s = s.iloc[::max(1, len(s) // 1200)]
    fig = go.Figure(go.Scatter(x=list(s.index.strftime("%Y-%m-%d")), y=s.round(3), line=dict(color=BLUE, width=1.3),
                               name="收盤"))
    a = hl["all"]
    for d, v, col in ((a["high_date"], a["high"], UP), (a["low_date"], a["low"], DOWN)):
        fig.add_trace(go.Scatter(x=[f"{d:%Y-%m-%d}"], y=[v], mode="markers", marker=dict(size=9, color=col),
                                 showlegend=False))
    ios_layout(fig, 260, legend=False)
    fig.update_yaxes(type="log")
    return fig


def path_df(close, p, lo, hi, dates, h):
    t = np.arange(1, h + 1)
    mid = close + (p - close) * t / h
    half = (hi - lo) / 2 * np.sqrt(t / h)
    return pd.DataFrame({"mid": mid, "hi": mid + half, "lo": mid - half}, index=dates[:h])


def md_to_html(t: str) -> str:
    out = []
    for line in t.splitlines():
        if line.startswith("# "):
            continue
        b = lambda s: EV._bold(html.escape(s))
        if line.startswith("## "):
            out.append(f"<h3>{b(line[3:])}</h3>")
        elif line.startswith("### "):
            out.append(f"<h4>{b(line[4:])}</h4>")
        elif line.strip().startswith("- "):
            ind = 14 if line.startswith("  ") else 0
            out.append(f"<div class='li' style='margin-left:{ind}px'>• {b(line.strip()[2:])}</div>")
        elif line.startswith("> "):
            out.append(f"<div class='note'>{b(line[2:])}</div>")
        elif line.strip():
            out.append(f"<div>{b(line)}</div>")
    return "".join(out)


def summ_list(rec):
    if rec is None or rec.empty:
        return []
    return clean(EV.summary(rec).to_dict("records"))


# ================================================================ 每檔股票
def stock_payload(c: str) -> dict | None:
    px = S.load_prices(c)
    if px.empty or len(px) < 160:
        return None
    tw = D.is_tw(c)
    meta = S.get_meta(c)
    ind = D.add_indicators(px)
    last = ind.iloc[-1]
    close, prev = float(px["Close"].iloc[-1]), float(px["Close"].iloc[-2])
    hl = D.high_low_stats(px)
    fcs = S.load_forecasts(c)
    stt = {h: S.load_state(c, h) for h in EV.HS}
    dates = D.next_sessions(px.attrs.get("cal"), px.index[-1], 20)

    figs = {"k3": fig_json(kline_fig(ind.iloc[-63:], tw)), "k6": fig_json(kline_fig(ind.iloc[-126:], tw)),
            "k12": fig_json(kline_fig(ind.iloc[-250:], tw)), "hist": fig_json(history_fig(px, hl))}

    # 各模型的預測圖（規則型態線 ＋ 該模型的預估路徑）
    w5, w20 = EV.weights_for(c, True)
    tr = T.analyze(px, w5=w5, w20=w20)
    signals = []
    if tr:
        signals = [{"dir": s["dir"], "name": s["name"], "kind": s["kind"], "w": s["w"], "note": s["note"]}
                   for s in tr["signals"]]
        for m, f in fcs.items():
            tr2 = dict(tr)
            tr2["proj"] = path_df(f["close"], f["p5"], f["lo5"], f["hi5"], dates, 5)
            tr2["proj20"] = path_df(f["close"], f["p20"], f["lo20"], f["hi20"], dates, 20)
            tr2["score"], tr2["score20"] = 2.5 * f.get("dir5", 0), 2.5 * f.get("dir20", 0)
            fig = TC.trend_fig(ind, tr2, tw, bars=60, mobile=True)
            fig.update_layout(font=dict(family=FONT, color="#8E8E93"))
            figs[f"trend_{m}"] = fig_json(fig)

    # 籌碼
    chips = {}
    if tw:
        inst, mg = S.load_inst(c), S.load_margin(c)
        if not inst.empty:
            chips["inst"] = [{"g": g, "v": float(inst[g].iloc[-1]), "streak": D.streak(inst[g]),
                              "s5": float(inst[g].iloc[-5:].sum())} for g in ("外資", "投信", "自營商", "合計")]
            chips["inst_date"] = f"{inst.index[-1]:%m/%d}"
            figs["inst"] = fig_json(inst_fig(inst))
        if not mg.empty:
            lm = mg.iloc[-1]
            chips["margin"] = {k: (None if pd.isna(lm[k]) else float(lm[k]))
                               for k in ("融資餘額", "融資增減", "融券餘額", "融券增減", "券資比%", "融資使用率%")}

    # 檢討：四種模型的回溯成績、冠軍的失敗原因、初報告
    review = {}
    base = S.load_rule_rec(c, px)
    if not base.empty:
        base = base.iloc[-500:]
        cal = {h: stt[h].get("calib") for h in EV.HS}
        for h in EV.HS:
            cal[f"sd{h}"] = stt[h].get("resid_sd", 1.5)
        oos = {h: S.load_ml_oos(c, h) for h in EV.HS}
        recs = {m: P.model_rec(base, m, cal, oos) for m in P.MODELS}
        recs["champion"] = P.champion_rec(recs, {h: stt[h].get("champion", "rule") for h in EV.HS})
        mkt, divs, inst = S.load_prices(D.market_symbol(c)), S.load_divs(c), S.load_inst(c)
        for m, R in recs.items():
            if R is None or R.empty:
                continue
            live = S.load_live(c, m, px)
            entry = {"summary": summ_list(R), "live": summ_list(live), "live_n": len(live)}
            if m == "champion":
                figs["review"] = fig_json(review_fig(R))
                F = EV.failures(R, px, mkt, divs, inst)
                if not F.empty:
                    ct = EV.cause_table(F)
                    entry["causes"] = [{"cause": k, **{p: int(ct.loc[k, p]) for p in ("5 日", "20 日") if p in ct}}
                                       for k in ct.index]
                    entry["fails"] = [{"date": f"{r['預測日']:%m/%d}", "h": r["週期"], "err": r["落差"],
                                       "cause": r["主要原因"], "detail": r["細節"]}
                                      for _, r in F.sort_values("預測日", ascending=False).head(20).iterrows()]
            review[m] = entry
        fails = EV.failures(base, px, mkt, divs, inst)
        text = EV.report_text(c, meta.get("name") or c, EV.summary(base), fails, {h: cal[h] for h in EV.HS})
        text = text.replace("## 四、下一步", P.model_section(stt) + "## 五、下一步")
        review["report_html"] = md_to_html(text)
        review["report_full"] = EV.report_html(text, {"個股化預測邏輯（權重校正）": EV.calib_table({h: cal[h] for h in EV.HS})})
        review["calib"] = EV.calib_table({h: cal[h] for h in EV.HS}).to_dict("records")

    return clean({
        "code": c, "name": meta.get("name") or c, "tw": tw, "date": px.index[-1], "close": close,
        "chg": close - prev, "pct": (close / prev - 1) * 100,
        "stats": {"vol": float(last["Volume"]), "vr": float(last["Volume"] / ind["Volume"].iloc[-6:-1].mean()),
                  "bias5": float(last["BIAS5"]), "bias20": float(last["BIAS20"]), "bias60": float(last["BIAS60"])},
        "hl": {k: {kk: (f"{vv:%Y-%m-%d}" if isinstance(vv, pd.Timestamp) else vv) for kk, vv in hl[k].items()}
               for k in ("all", "52w")} | {"first": f"{hl['first_date']:%Y-%m-%d}"},
        "dates": {"5": f"{dates[4]:%m/%d}", "20": f"{dates[19]:%m/%d}"},
        "forecasts": fcs,
        "states": {str(h): {"champion": stt[h].get("champion"), "metrics": stt[h].get("metrics"),
                            "importance": (stt[h].get("importance") or [])[:6], "n_train": stt[h].get("n_train")}
                   for h in EV.HS},
        "signals": signals, "chips": chips, "review": review, "figs": figs,
        "support": tr["support"] if tr else None, "resistance": tr["resistance"] if tr else None,
    })


# ================================================================ 主程式
def build(repo: str = "") -> Path:
    cfg = S.load_cfg()
    codes = S.all_codes(cfg)
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE / "data").mkdir(parents=True)
    for f in WEB.iterdir():
        if f.is_file():
            shutil.copy(f, SITE / f.name)

    stocks = {}
    for c in codes:
        try:
            p = stock_payload(c)
        except Exception as e:                       # 單檔失敗不影響整個網站
            print(f"  ✗ {c} 產生失敗：{e}")
            p = None
        if not p:
            stocks[c] = {"code": c, "name": S.get_meta(c).get("name") or c, "nodata": True}
            continue
        (SITE / "data" / f"{c}.json").write_text(json.dumps(p, ensure_ascii=False, separators=(",", ":")),
                                                 encoding="utf-8")
        px = S.load_prices(c)
        stocks[c] = clean({k: p[k] for k in ("code", "name", "tw", "date", "close", "chg", "pct", "dates")} | {
            "spark": [round(float(v), 4) for v in px["Close"].iloc[-30:]],
            "forecasts": {m: {k: f.get(k) for k in ("close", "p5", "lo5", "hi5", "p20", "lo20", "hi20", "dir5", "dir20",
                                                    "model5", "model20")} for m, f in p["forecasts"].items()},
            "champion": {h: p["states"][h]["champion"] for h in p["states"]},
            "ok5": {h: {m: (v or {}).get("ok5") for m, v in (p["states"][h]["metrics"] or {}).items()}
                    for h in p["states"]},
        })
        print(f"  ✓ {c}")

    now = dt.datetime.now()
    summary = {"updated": now.strftime("%Y-%m-%d %H:%M"), "repo": repo, "lists": cfg["watchlists"],
               "stocks": stocks, "model_names": P.MODEL_NAME}
    (SITE / "data" / "summary.json").write_text(json.dumps(clean(summary), ensure_ascii=False), encoding="utf-8")
    (SITE / ".nojekyll").write_text("")
    print(f"網站已產生：{SITE}（{len(codes)} 檔，{sum(f.stat().st_size for f in SITE.rglob('*') if f.is_file()) / 1e6:.1f} MB）")
    return SITE


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="")
    build(ap.parse_args().repo)
