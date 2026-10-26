"""預測檢討：回溯模擬每日預測 → 對答案 → 失敗原因歸類 → 個股權重校正 → 初報告。

判定標準（兩種同時計算）：
  固定 5%  ：|實際收盤 ÷ 預估價 − 1| ≤ 5% 即成功
  預測區間：實際收盤落在預估區間（±0.9 ATR×√天數）內即成功
對照組：
  股價不動：假設 N 日後收盤 = 今日收盤，用同樣 5% 標準計算的命中率
  方向命中：非盤整判斷時，預測漲跌方向與實際是否相同
"""
from __future__ import annotations

import datetime as dt
import html
import json
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import trend as T

TH = 0.05
HS = (T.HORIZON, T.HORIZON_LONG)
CALIB_FILE = Path(__file__).parent / "calib.json"
KIND_LABEL = {"K棒": "K棒", "組合K線": "組合K線", "反轉型態": "反轉型態", "整理型態": "整理型態",
              "水平線": "水平線", "均線": "均線"}


# ================================================================ 1. 回溯模擬
def simulate(df: pd.DataFrame, days: int = 500, w5=None, w20=None) -> pd.DataFrame:
    """過去 days 個交易日，每天只用『當天以前』的資料做預測，並填入之後的實際收盤。"""
    if df is None or len(df) < 140:
        return pd.DataFrame()
    c = df["Close"].values
    n, idx = len(c), df.index
    rows = []
    for t in range(max(120, n - days), n):
        r = T.analyze(df.iloc[max(0, t + 1 - T.LOOKBACK):t + 1], light=True, w5=w5, w20=w20)
        if not r:
            continue
        row = {"date": idx[t], "pos": t, "close": c[t], "atr": r["atr"],
               "score": r["score"], "score20": r["score20"],
               "pattern": r["pattern"], "pattern_dir": r["pattern_dir"]}
        for h in HS:
            row[f"p{h}"], row[f"lo{h}"], row[f"hi{h}"] = r[f"p{h}"], r[f"lo{h}"], r[f"hi{h}"]
            row[f"act{h}"] = c[t + h] if t + h < n else np.nan
            row[f"date{h}"] = idx[t + h] if t + h < n else pd.NaT
        row.update({f"sub_{k}": v for k, v in r["sub"].items()})
        rows.append(row)
    rec = pd.DataFrame(rows).set_index("date")
    return add_results(rec)


def add_results(rec: pd.DataFrame) -> pd.DataFrame:
    rec = rec.copy()
    for h in HS:
        act, p, cl = rec[f"act{h}"], rec[f"p{h}"], rec["close"]
        sc = rec["score"] if h == T.HORIZON else rec["score20"]
        done = act.notna()
        rec[f"err{h}"] = act / p - 1
        rec[f"ok{h}"] = np.where(done, (rec[f"err{h}"].abs() <= TH), np.nan)
        rec[f"band{h}"] = np.where(done, (act >= rec[f"lo{h}"]) & (act <= rec[f"hi{h}"]), np.nan)
        rec[f"naive{h}"] = np.where(done, ((act / cl - 1).abs() <= TH), np.nan)
        if f"dirsig{h}" in rec:                      # 非規則模型（校正/ML）直接給方向
            dp = rec[f"dirsig{h}"].fillna(0).values
        else:
            dp = np.where(sc.abs() >= 2, np.sign(sc), 0)
        rec[f"dirp{h}"] = dp
        rec[f"dir{h}"] = np.where(done & (dp != 0), np.sign(act - cl) == dp, np.nan)
    return rec


def summary(rec: pd.DataFrame) -> pd.DataFrame:
    out = []
    for h in HS:
        if rec.empty or f"ok{h}" not in rec:
            continue
        d = rec[rec[f"act{h}"].notna()]
        dd = d[d[f"dirp{h}"] != 0]
        pending = int(rec[f"act{h}"].isna().sum())
        mean = lambda s: float(s.astype(float).mean()) if len(s) else np.nan
        out.append({"週期": f"{h} 日", "已驗證": len(d), "待驗證": pending,
                    "落差≤5%": mean(d[f"ok{h}"]), "對照：股價不動": mean(d[f"naive{h}"]),
                    "落在預測區間": mean(d[f"band{h}"]),
                    "方向命中": mean(dd[f"dir{h}"]), "方向判斷次數": len(dd)})
    return pd.DataFrame(out)


# ================================================================ 2. 失敗原因
CAUSE_ORDER = ["除權息", "大盤連動", "跳空缺口", "法人反向", "爆量", "型態失效", "盤整誤判", "方向錯誤", "幅度低估", "其他"]
CAUSE_ADVICE = {
    "除權息": "失準其實是除權息造成的價格調整，並非預測錯誤；看盤價格未還原股息，除息日前後的預測要自行扣除。",
    "大盤連動": "這檔股票跟大盤連動高，個股型態常被大盤漲跌蓋過。建議加入「大盤趨勢濾網」：大盤偏空時降低偏多訊號的權重。",
    "跳空缺口": "多數失準來自跳空缺口，通常是財報、法說會、重大訊息或國際事件，技術面無法預知。財報公布前後應降低信心或放寬區間。",
    "法人反向": "法人籌碼主導這檔股價走勢。建議把「外資/投信連續買賣超」納入評分。",
    "爆量": "爆量常伴隨趨勢轉折或消息面，建議把量能放大（量比 > 2）納入評分，帶量突破才算數。",
    "型態失效": "型態假突破/假跌破頻繁，這檔股票的型態可靠度偏低。建議降低型態權重，或要求帶量確認。",
    "盤整誤判": "模型判斷為盤整時，實際仍大幅波動，代表這檔股票波動度高、訊號常落後。可放寬盤整區間或縮短均線週期。",
    "方向錯誤": "規則判斷的方向與實際相反，代表部分訊號對這檔股票無效甚至反向，請參考下方的權重校正結果。",
    "幅度低估": "方向判斷正確，但漲跌幅超出預估。強勢股動能延續性高，可調高預估移動幅度。",
    "其他": "找不到明顯的外部事件，屬於一般隨機波動。",
}


def failures(rec: pd.DataFrame, df: pd.DataFrame, mkt: pd.DataFrame | None = None,
             divs: pd.Series | None = None, inst: pd.DataFrame | None = None) -> pd.DataFrame:
    """針對落差 > 5% 的預測逐筆分析原因。"""
    if rec.empty:
        return pd.DataFrame()
    o, c, v = df["Open"].values, df["Close"].values, df["Volume"].values
    mkt_c = mkt["Close"] if mkt is not None and not mkt.empty else None
    inst_net = inst["合計"] if inst is not None and not inst.empty else None
    rows = []
    for h in HS:
        bad = rec[(rec[f"ok{h}"] == 0) & rec[f"act{h}"].notna()]
        for d, r in bad.iterrows():
            i, j = int(r["pos"]), int(r["pos"]) + h
            cl, p, a = r["close"], r[f"p{h}"], r[f"act{h}"]
            move, miss = a / cl - 1, a - p
            causes, details = [], []

            # 除權息
            if divs is not None and len(divs):
                dv = divs[(divs.index > d) & (divs.index <= r[f"date{h}"])].sum()
                if dv / cl >= 0.005:
                    adj = (a + dv) / p - 1
                    causes.append("除權息")
                    details.append(f"期間配息 {dv:.2f}（{dv / cl:.1%}）" +
                                   ("，還原後落差僅 {:.1%}，其實命中".format(adj) if abs(adj) <= TH else ""))
            # 大盤
            if mkt_c is not None:
                m0 = mkt_c[:d]; m1 = mkt_c[:r[f"date{h}"]]
                if len(m0) and len(m1):
                    mret = m1.iloc[-1] / m0.iloc[-1] - 1
                    th = 0.025 if h == T.HORIZON else 0.05
                    if abs(mret) >= th and np.sign(mret) == np.sign(miss):
                        causes.append("大盤連動"); details.append(f"同期大盤 {mret:+.1%}")
            # 跳空
            gaps = o[i + 1:j + 1] / c[i:j] - 1
            if len(gaps):
                g = gaps[np.argmax(np.abs(gaps))]
                if abs(g) >= max(0.03, 1.5 * r["atr"] / cl):
                    k = int(np.argmax(np.abs(gaps)))
                    causes.append("跳空缺口"); details.append(f"{df.index[i + 1 + k]:%m/%d} 跳空 {g:+.1%}")
            # 法人
            if inst_net is not None:
                win = inst_net[(inst_net.index > d) & (inst_net.index <= r[f"date{h}"])]
                prev = inst_net[inst_net.index <= d].iloc[-60:]
                if len(win) and len(prev) >= 20:
                    base = prev.abs().mean() * len(win)
                    if base > 0 and abs(win.sum()) >= 2 * base and np.sign(win.sum()) == np.sign(miss):
                        causes.append("法人反向")
                        details.append(f"三大法人合計 {win.sum():+,.0f} 張（平常的 {abs(win.sum()) / base:.1f} 倍）")
            # 爆量
            vb = v[max(0, i - 19):i + 1].mean()
            vw = v[i + 1:j + 1]
            if vb > 0 and len(vw) and vw.max() >= 2.5 * vb:
                causes.append("爆量"); details.append(f"最大量為均量 {vw.max() / vb:.1f} 倍")
            # 型態失效
            if r["pattern_dir"] and np.sign(move) == -r["pattern_dir"]:
                causes.append("型態失效"); details.append(f"{r['pattern']} 後走勢相反")
            # 方向 / 幅度
            dp = r[f"dirp{h}"]
            if dp == 0:
                causes.append("盤整誤判")
                details.append(f"預測盤整，實際 {move:+.1%}")
            elif np.sign(move) != dp:
                causes.append("方向錯誤")
                details.append(f"預測{'漲' if dp > 0 else '跌'}，實際 {move:+.1%}")
            else:
                causes.append("幅度低估")
                details.append(f"方向正確，實際 {move:+.1%} 超出預估 {p / cl - 1:+.1%}")

            primary = next((x for x in CAUSE_ORDER if x in causes), "其他")
            rows.append({"預測日": d, "週期": f"{h} 日", "收盤": cl, "預估價": p, "實際收盤": a,
                         "落差": a / p - 1, "主要原因": primary, "所有原因": "、".join(causes),
                         "細節": "；".join(details)})
    return pd.DataFrame(rows)


def cause_table(fails: pd.DataFrame) -> pd.DataFrame:
    if fails.empty:
        return pd.DataFrame()
    t = fails.pivot_table(index="主要原因", columns="週期", values="落差", aggfunc="count").fillna(0).astype(int)
    t = t.reindex([c for c in CAUSE_ORDER if c in t.index])
    for col in t.columns:
        t[f"{col}佔比"] = t[col] / t[col].sum()
    return t


# ================================================================ 3. 個股權重校正
GRID = (0.0, 0.5, 1.0, 1.5, 2.0)


def _ic(S: np.ndarray, y: np.ndarray) -> np.ndarray:
    """欄位與 y 的相關係數（Information Coefficient）。"""
    Sc = S - S.mean(axis=0)
    yc = y - y.mean()
    den = np.sqrt((Sc ** 2).sum(axis=0)) * np.sqrt((yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = (Sc.T @ yc) / den
    return np.nan_to_num(ic, nan=-1.0)


def _dir_hit(score, move):
    m = np.abs(score) >= 2
    return float((np.sign(score[m]) == np.sign(move[m])).mean()) if m.any() else np.nan, int(m.sum())


def _proj_move(score, atr_v, h, scale=1.0):
    """向量化：依分數算出 h 日後的預估移動（不含型態目標價拉力）。"""
    score = np.asarray(score, float)
    st = np.minimum(np.abs(score) / 8, 1.0)
    d = np.where(np.abs(score) >= 2, np.sign(score) * st * T.DRIFT * atr_v * scale, 0.0)
    f = h if h == T.HORIZON else float(np.sum(0.92 ** np.arange(h)))
    return d * f


def calibrate(rec: pd.DataFrame, h: int) -> dict | None:
    """個股化預測：
    1) 權重：在前 60% 期間，用網格搜尋找出讓「分數 vs 實際漲跌」相關性(IC)最高的各類訊號倍數；
    2) 幅度係數：用最小平方法調整預估移動幅度（訊號常失準的股票，幅度會自動縮小）；
    3) 在後 40% 期間驗證，「落差≤5% 命中率」比預設高、且 IC 與方向命中率沒變差才採用。"""
    d = rec[rec[f"act{h}"].notna()]
    if len(d) < 120:
        return None
    X = d[[f"sub_{k}" for k in T.KINDS]].values.astype(float)
    atr_v = d["atr"].values; cl = d["close"].values; act = d[f"act{h}"].values
    y = (act - cl) / atr_v
    move = act / cl - 1
    cut = int(len(d) * 0.6)
    tr, te = slice(0, cut), slice(cut, None)
    base = np.array([(T.DEFAULT_W5 if h == T.HORIZON else T.DEFAULT_W20)[k] for k in T.KINDS])

    # 1) 權重網格
    G = np.array(list(itertools.product(GRID, repeat=len(T.KINDS))), dtype=np.float32)
    active = np.abs(X[tr]).sum(axis=0) > 0
    G[:, ~active] = base[~active]
    G = np.unique(G, axis=0)
    ic_tr = _ic(X[tr].astype(np.float32) @ G.T, y[tr].astype(np.float32))
    ic_tr = ic_tr - 0.004 * np.abs(G - base).sum(axis=1)
    best = G[int(np.argmax(ic_tr))].astype(float)

    # 2) 幅度係數（訓練期最小平方）
    xm = _proj_move(X[tr] @ best, atr_v[tr], h)
    ym = act[tr] - cl[tr]
    scale = float(np.clip((xm @ ym) / (xm @ xm), 0.0, 2.0)) if (xm @ xm) > 0 else 1.0
    scale = round(scale, 2)

    def stats(w, sc_, sl):
        s_ = X[sl] @ w
        ic = float(_ic(s_[:, None], y[sl])[0]) if np.std(s_) > 0 else 0.0
        hit, n = _dir_hit(s_, move[sl])
        pred = cl[sl] + _proj_move(s_, atr_v[sl], h, sc_)
        ok5 = float((np.abs(act[sl] / pred - 1) <= TH).mean())
        return {"ic": ic, "dir": hit, "n_dir": n, "ok5": ok5}

    sd_tr, sb_tr = stats(base, 1.0, tr), stats(best, scale, tr)
    sd, sb = stats(base, 1.0, te), stats(best, scale, te)
    naive = float((np.abs(move[te]) <= TH).mean())
    dir_ok = (sb["n_dir"] < 20 or sd["n_dir"] < 20 or np.isnan(sd["dir"])
              or (not np.isnan(sb["dir"]) and sb["dir"] >= sd["dir"] - 0.03))
    adopted = bool(sb["ok5"] >= sd["ok5"] + 0.01 and sb["ic"] >= sd["ic"] - 0.01 and dir_ok)
    best_d = dict(zip(T.KINDS, best.round(2).tolist())); best_d["幅度"] = scale
    def_d = dict(zip(T.KINDS, base.round(2).tolist())); def_d["幅度"] = 1.0
    return {"h": h, "n_train": cut, "n_test": len(d) - cut, "default": def_d, "best": best_d,
            "train_default": sd_tr, "train_best": sb_tr, "test_default": sd, "test_best": sb,
            "test_naive_ok5": naive, "adopted": adopted}


def calib_table(cal: dict) -> pd.DataFrame:
    """權重比較表：類別 × (預設, 校正後) for 5/20 日。"""
    rows = []
    for k in list(T.KINDS) + ["幅度"]:
        r = {"項目": "預估幅度係數" if k == "幅度" else k}
        for h in HS:
            c = cal.get(h)
            r[f"{h}日 預設"] = f"×{c['default'][k]:g}" if c else "-"
            r[f"{h}日 校正後"] = f"×{c['best'][k]:g}" if c else "-"
        rows.append(r)
    for lab, key, fmt in (("測試期 落差≤5%", "ok5", "pct"), ("測試期 方向命中", "dir", "pct"),
                          ("測試期 IC", "ic", "f")):
        r = {"項目": lab}
        for h in HS:
            c = cal.get(h)
            f = (lambda v: pct(v)) if fmt == "pct" else (lambda v: "-" if v is None else f"{v:+.3f}")
            r[f"{h}日 預設"] = f(c["test_default"][key]) if c else None
            r[f"{h}日 校正後"] = f(c["test_best"][key]) if c else None
        rows.append(r)
    r = {"項目": "是否採用"}
    for h in HS:
        c = cal.get(h)
        r[f"{h}日 預設"] = ""
        r[f"{h}日 校正後"] = ("✅ 採用" if c["adopted"] else "❌ 維持預設") if c else "樣本不足"
    rows.append(r)
    return pd.DataFrame(rows).fillna("-").astype(str)


def load_calib() -> dict:
    try:
        return json.loads(CALIB_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_calib(code: str, cal5: dict | None, cal20: dict | None) -> None:
    allc = load_calib()
    allc[code] = {"updated": dt.date.today().isoformat(), "5": cal5, "20": cal20}
    try:
        CALIB_FILE.write_text(json.dumps(allc, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    except Exception:
        pass


def weights_for(code: str, enabled: bool = True):
    """回傳 (w5, w20)；沒有校正或未通過驗證時為 None（使用預設）。"""
    if not enabled:
        return None, None
    c = load_calib().get(code) or {}
    pick = lambda k: c[k]["best"] if c.get(k) and c[k].get("adopted") else None
    return pick("5"), pick("20")


# ================================================================ 4. 初報告
def pct(v):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v * 100:.0f}%"


def report_text(code: str, name: str, summ: pd.DataFrame, fails: pd.DataFrame,
                cal: dict, live: pd.DataFrame | None = None) -> str:
    """產生 Markdown 格式的初報告。"""
    L = [f"# {name}（{code}）預測檢討初報告", f"產生日期：{dt.date.today():%Y-%m-%d}", ""]
    L.append("## 一、預測成績（回溯模擬）")
    if summ.empty:
        L.append("資料不足，無法回溯模擬。")
    for _, s in summ.iterrows():
        diff = s["落差≤5%"] - s["對照：股價不動"]
        verdict = ("**優於**" if diff > 0.03 else "**不如**" if diff < -0.03 else "**接近**")
        L.append(f"- **{s['週期']}預測**（已驗證 {s['已驗證']} 筆）：落差 ≤ 5% 的比例 {pct(s['落差≤5%'])}，"
                 f"「股價不動」對照組 {pct(s['對照：股價不動'])}，模型{verdict}對照組；"
                 f"落在預測區間內 {pct(s['落在預測區間'])}；"
                 f"非盤整判斷 {s['方向判斷次數']} 次，方向命中 {pct(s['方向命中'])}。")
    if live is not None and not live.empty:
        L.append(f"- 實際紀錄（從開始記錄起）：共 {len(live)} 筆，詳見看盤內「實際紀錄」。")
    L.append("")
    L.append("## 二、預測失敗原因（落差 > 5%）")
    if fails.empty:
        L.append("沒有落差超過 5% 的預測。")
    else:
        for per, g in fails.groupby("週期", sort=False):
            cnt = g["主要原因"].value_counts()
            top = "、".join(f"{k} {v} 次（{v / len(g):.0%}）" for k, v in cnt.head(4).items())
            L.append(f"- **{per}**：共 {len(g)} 筆失敗，主要原因為 {top}。")
        L.append("")
        L.append("### 原因解讀與建議")
        main = fails["主要原因"].value_counts()
        for k in main.index[:4]:
            L.append(f"- **{k}**（{main[k]} 筆）：{CAUSE_ADVICE.get(k, '')}")
        L.append("")
        L.append("> 註：失敗筆數以「預測日」計算，同一事件會影響連續數天的預測，因此相鄰日期常是同一原因。")
    L.append("")
    L.append("## 三、個股化預測邏輯（權重與幅度校正）")
    any_cal = False
    for h in HS:
        c = cal.get(h)
        if not c:
            L.append(f"- {h} 日：已驗證樣本不足 120 筆，暫不校正。")
            continue
        any_cal = True
        changes = [("預估幅度" if k == "幅度" else k) + f" ×{c['best'][k]:g}"
                   for k in c["best"] if abs(c["best"][k] - c["default"][k]) > 1e-9]
        td, tb = c["test_default"], c["test_best"]
        L.append(f"- **{h} 日**：調整 {'、'.join(changes) if changes else '無'}。"
                 f"測試期（後 40%，{c['n_test']} 筆）落差≤5% 命中 {pct(td['ok5'])} → {pct(tb['ok5'])}"
                 f"（股價不動對照 {pct(c['test_naive_ok5'])}），方向命中 {pct(td['dir'])} → {pct(tb['dir'])}，"
                 f"IC {td['ic']:+.3f} → {tb['ic']:+.3f}。"
                 f"結論：{'✅ 採用' if c['adopted'] else '❌ 不採用，維持預設'}")
        if c["best"]["幅度"] < 0.5:
            L.append(f"  - 幅度係數只有 ×{c['best']['幅度']:g}：這檔股票的技術訊號對漲跌幅的解釋力低，"
                     "較保守（接近「股價不動」）的預估反而更準。")
        zeroed = [k for k in T.KINDS if c["best"][k] == 0 and c["default"][k] > 0]
        if zeroed:
            L.append(f"  - {'、'.join(zeroed)} 在這檔股票上沒有預測力，校正後不計分。")
    if any_cal:
        L.append("")
        L.append("> 方法：前 60% 期間用網格搜尋找出各類訊號的最佳權重倍數（0～2 倍），並以最小平方法調整預估幅度；"
                 "後 40% 期間驗證，只有「落差≤5% 命中率」提高、且 IC（分數與實際漲跌的相關係數）與方向命中率都沒變差才採用，避免背答案。")
    L.append("")
    L.append("## 四、下一步")
    L.append("- 持續累積「實際紀錄」，3 個月後再比較實際成績與回溯模擬是否一致。")
    L.append("- 第二階段：資料量足夠後，以機器學習模型（加入法人、融資、大盤、量能特徵）與規則模型比較，"
             "只有在前推驗證中勝出才採用。")
    L.append("")
    L.append("⚠️ 本報告為技術分析規則的統計檢討，不構成投資建議。")
    return "\n".join(L)


def report_html(md_text: str, tables: dict[str, pd.DataFrame]) -> str:
    """把 Markdown 初報告 + 表格組成可下載的 HTML。"""
    body = []
    for line in md_text.splitlines():
        esc = html.escape(line)
        esc = _bold(esc)
        if line.startswith("# "):
            body.append(f"<h1>{_bold(html.escape(line[2:]))}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{_bold(html.escape(line[3:]))}</h2>")
        elif line.startswith("### "):
            body.append(f"<h3>{_bold(html.escape(line[4:]))}</h3>")
        elif line.startswith("- "):
            body.append(f"<li>{_bold(html.escape(line[2:]))}</li>")
        elif line.startswith("> "):
            body.append(f"<blockquote>{_bold(html.escape(line[2:]))}</blockquote>")
        elif line.strip():
            body.append(f"<p>{esc}</p>")
    for title, t in tables.items():
        if t is not None and not t.empty:
            body.append(f"<h2>{html.escape(title)}</h2>")
            body.append(t.to_html(index=False, border=0, classes="t", na_rep="-"))
    css = """
body{font-family:'Microsoft JhengHei','PingFang TC','Noto Sans TC',sans-serif;max-width:980px;margin:24px auto;
padding:0 16px;color:#222;line-height:1.65;background:#fff}
h1{font-size:1.6rem;border-bottom:3px solid #e5484d;padding-bottom:.3rem}h2{font-size:1.2rem;margin-top:1.6rem;
color:#333;border-left:4px solid #3e7bfa;padding-left:.5rem}li{margin:.25rem 0}
blockquote{color:#666;border-left:3px solid #ccc;margin:.6rem 0;padding:.2rem .8rem;font-size:.9rem}
table.t{border-collapse:collapse;width:100%;font-size:.85rem;margin:.5rem 0}
table.t th{background:#f3f4f6;text-align:left;padding:6px 8px}table.t td{border-top:1px solid #eee;padding:5px 8px}
@media (prefers-color-scheme: dark){body{background:#16171a;color:#e6e6e6}h2{color:#ddd}
table.t th{background:#25272c}table.t td{border-color:#333}blockquote{color:#aaa}}
"""
    title = md_text.splitlines()[0].lstrip("# ") if md_text else "預測檢討"
    return (f"<!doctype html><html lang='zh-Hant'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{css}</style></head><body>{''.join(body)}</body></html>")


def _bold(s: str) -> str:
    parts = s.split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


def daily_table(rec: pd.DataFrame, last: int = 60) -> pd.DataFrame:
    """每日預測成敗明細（新到舊）。"""
    t = rec.iloc[-last:][::-1]
    mark = lambda ok: "⏳ 待驗證" if pd.isna(ok) else ("✅" if ok else "❌")
    out = pd.DataFrame({
        "預測日": t.index.strftime("%Y-%m-%d"),
        "收盤": t["close"].round(2),
        "5日判斷": [T.label_of(s) for s in t["score"]],
        "5日預估": t["p5"].round(2), "5日實際": t["act5"].round(2), "5日落差": t["err5"],
        "5日結果": [mark(x) for x in t["ok5"]],
        "20日判斷": [T.label_of(s) for s in t["score20"]],
        "20日預估": t["p20"].round(2), "20日實際": t["act20"].round(2), "20日落差": t["err20"],
        "20日結果": [mark(x) for x in t["ok20"]],
    })
    return out
